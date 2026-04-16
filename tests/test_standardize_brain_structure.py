import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import nrrd
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from src.datasets.standardize import cli as standardize_cli
from src.datasets.standardize.adapters.cfb_gbm import (
    CfbGbmAdapter,
    parse_cfb_series_path,
)
from src.datasets.standardize.adapters.brain_structure import BrainStructureAdapter
from src.datasets.standardize.adapters.remind import (
    RemindAdapter,
    is_remind_real_t1_description,
    parse_remind_mask_filename,
)
from src.datasets.standardize.adapters.ucsd_ptgbm import (
    UcsdPtgbmAdapter,
    parse_ucsd_case_dir_name,
    parse_ucsd_series_name,
)
from src.datasets.standardize.adapters.ucsf_pdgm import (
    UcsfPdgmAdapter,
    parse_ucsf_case_dir_name,
    parse_ucsf_series_name,
)
from src.datasets.standardize.adapters.upenn_gbm import (
    UpennGbmAdapter,
    parse_upenn_series_path,
)
from src.datasets.standardize.adapters.utsw_glioma import UtswGliomaAdapter
from src.datasets.standardize.adapters.vestibular_schwannoma_mc_rc2 import (
    VestibularSchwannomaMcRc2Adapter,
    parse_vestibular_series_path,
)
from src.datasets.standardize.adapters.yale_brain_mets_longitudinal import (
    YaleBrainMetsLongitudinalAdapter,
    parse_yale_series_path,
)
from src.datasets.standardize.constants import (
    CFB_GBM_PREPROC_PROFILE,
    BRAIN_STRUCTURE_PREPROC_PROFILE,
    BRAIN_STRUCTURE_SOURCE_MANIFEST_DEFAULT,
    BRAIN_STRUCTURE_STANDARDIZED_DEFAULT_ROOT,
    REMIND_PREPROC_PROFILE,
    REMIND_DATASET_KEY,
    STANDARDIZED_TASK_MANIFESTS_DEFAULT_ROOT,
    UCSD_PTGBM_PREPROC_PROFILE,
    UCSF_PDGM_PREPROC_PROFILE,
    UTSW_GLIOMA_PREPROC_PROFILE,
    UPENN_GBM_PREPROC_PROFILE,
    VESTIBULAR_SCHWANNOMA_MC_RC2_PREPROC_PROFILE,
    YALE_BRAIN_METS_LONGITUDINAL_PREPROC_PROFILE,
)
from src.datasets.standardize.pathing import (
    build_dataset_root_candidates,
    resolve_target_path,
)
from src.datasets.standardize.preprocess import (
    ClassificationViewResult,
    _command_exists,
    materialize_brain_structure_manifest,
    materialize_classification_manifest,
    preflight_synthstrip_requirements,
    write_brain_structure_cls_view,
    write_classification_view,
)
from src.datasets.standardize.registry import get_dataset_registry_entry
from src.datasets.standardize.task_manifests import build_all_task_manifests


METADATA_COLUMNS = [
    "t1_local_path",
    "split",
    "study",
    "participant_id",
    "session_id",
    "age",
    "sex",
    "clinical_diagnosis",
    "scanner_manufacturer",
    "scanner_model",
    "field_strength",
    "image_quality_rating",
    "total_intracranial_volume",
    "radiata_id",
]


def _write_metadata(root: Path, rows: list[dict[str, str]]) -> None:
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row.keys():
            if field in seen:
                continue
            seen.add(field)
            fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_nifti(path: Path, fill_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        np.full((4, 4, 4), fill_value, dtype=np.float32),
        affine=np.eye(4),
    )
    nib.save(image, str(path))


def _write_nrrd(path: Path, fill_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(path), np.full((4, 4, 4), fill_value, dtype=np.uint8))


def _write_dicom_series(
    series_dir: Path,
    *,
    series_description: str,
    num_slices: int = 4,
    base_value: int = 100,
) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    for index in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(
            str(series_dir / f"slice_{index:03d}.dcm"),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.SOPClassUID = MRImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.Modality = "MR"
        dataset.PatientID = "TEST"
        dataset.SeriesDescription = series_description
        dataset.SeriesNumber = "1"
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(index)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.Rows = 4
        dataset.Columns = 4
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 1.0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        pixels = np.full((4, 4), base_value + index, dtype=np.uint16)
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(str(series_dir / f"slice_{index:03d}.dcm"))


def _classification_row(
    *,
    dataset_key: str,
    input_path: str,
    global_case_id: str,
    subject_id: str,
    preproc_profile: str,
    source_split: str = "train",
    label_family: str = "healthy_control",
    binary_status: str = "healthy",
    label: str = "healthy",
    diagnosis_original: str = "cognitively_normal",
) -> dict[str, str]:
    return {
        "dataset_key": dataset_key,
        "subject_id": subject_id,
        "visit_id": "ses-01",
        "global_case_id": global_case_id,
        "image_path": input_path,
        "t1_path": input_path,
        "diagnosis_original": diagnosis_original,
        "binary_status": binary_status,
        "label": label,
        "label_family": label_family,
        "preproc_profile": preproc_profile,
        "source_study": "TEST",
        "source_split": source_split,
        "radiata_id": "RID-001",
        "image_quality_rating": "4",
        "total_intracranial_volume": "1234",
        "age": "70",
        "sex": "F",
        "scanner_manufacturer": "GE",
        "scanner_model": "SIGNA",
        "field_strength": "3T",
        "exclude_reason": "",
        "task_type": "classification",
        "mask_path": "",
        "brain_mask_path": "",
        "normalization_mask_method": "",
        "mask_tier": "",
    }


class BrainStructureStandardizeTests(unittest.TestCase):
    def test_build_dataset_root_candidates_accepts_windows_paths(self) -> None:
        candidates = build_dataset_root_candidates(r"Z:\dataset\brain-structure")

        self.assertEqual(str(candidates[0]), r"Z:\dataset\brain-structure")
        if os.name == "nt":
            self.assertEqual(len(candidates), 1)
        else:
            self.assertIn("/mnt/z/dataset/brain-structure", [str(path) for path in candidates])

    def test_resolve_target_path_prefers_z_mount_for_writes(self) -> None:
        resolved_copy_root = resolve_target_path(
            BRAIN_STRUCTURE_STANDARDIZED_DEFAULT_ROOT,
            default=BRAIN_STRUCTURE_STANDARDIZED_DEFAULT_ROOT,
        )
        resolved_manifest = resolve_target_path(
            BRAIN_STRUCTURE_SOURCE_MANIFEST_DEFAULT,
            default=BRAIN_STRUCTURE_SOURCE_MANIFEST_DEFAULT,
        )
        resolved_task_root = resolve_target_path(
            STANDARDIZED_TASK_MANIFESTS_DEFAULT_ROOT,
            default=STANDARDIZED_TASK_MANIFESTS_DEFAULT_ROOT,
        )

        if os.name == "nt":
            self.assertEqual(
                str(resolved_copy_root),
                r"Z:\dataset\cancervision-standardized\brain_structure",
            )
            self.assertEqual(
                str(resolved_manifest),
                r"Z:\dataset\cancervision-standardized\manifests\brain_structure_source_manifest.csv",
            )
            self.assertEqual(
                str(resolved_task_root),
                r"Z:\dataset\cancervision-standardized\task_manifests",
            )
        else:
            self.assertEqual(
                str(resolved_copy_root),
                "/mnt/z/dataset/cancervision-standardized/brain_structure",
            )
            self.assertEqual(
                str(resolved_manifest),
                "/mnt/z/dataset/cancervision-standardized/manifests/brain_structure_source_manifest.csv",
            )
            self.assertEqual(
                str(resolved_task_root),
                "/mnt/z/dataset/cancervision-standardized/task_manifests",
            )

    def test_registry_resolves_skip_and_synthstrip_policies(self) -> None:
        self.assertEqual(
            get_dataset_registry_entry("cfb_gbm").cls_skullstrip_policy,
            "synthstrip",
        )
        self.assertEqual(
            get_dataset_registry_entry("brain_structure").cls_skullstrip_policy,
            "skip",
        )
        self.assertEqual(
            get_dataset_registry_entry("UPENN-GBM").cls_skullstrip_policy,
            "skip",
        )
        self.assertEqual(
            get_dataset_registry_entry("UCSF-PDGM").cls_skullstrip_policy,
            "skip",
        )
        self.assertEqual(
            get_dataset_registry_entry("UCSD-PTGBM").cls_skullstrip_policy,
            "skip",
        )
        self.assertEqual(
            get_dataset_registry_entry("UTSW-Glioma").cls_skullstrip_policy,
            "skip",
        )
        self.assertEqual(
            get_dataset_registry_entry("ReMIND").cls_skullstrip_policy,
            "synthstrip",
        )
        self.assertEqual(
            get_dataset_registry_entry("vestibular_schwannoma_mc_rc2").cls_skullstrip_policy,
            "synthstrip",
        )

    def test_parse_remind_mask_filename_extracts_subject_phase_and_series(self) -> None:
        parsed = parse_remind_mask_filename(
            "ReMIND-001-preop-SEG-tumor-MR-3D_AX_T1_postcontrast.nrrd"
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.subject_id, "ReMIND-001")
        self.assertEqual(parsed.phase, "preop")
        self.assertEqual(parsed.label_name, "tumor")
        self.assertEqual(parsed.series_description, "3D_AX_T1_postcontrast")

    def test_is_remind_real_t1_description_identifies_precontrast_and_mp2rage(self) -> None:
        self.assertTrue(is_remind_real_t1_description("3D_AX_T1_precontrast"))
        self.assertTrue(is_remind_real_t1_description("3D_SAG_T1_MP2RAGE"))
        self.assertFalse(is_remind_real_t1_description("3D_AX_T1_postcontrast"))

    def test_parse_ucsf_case_dir_name_extracts_subject_and_visit(self) -> None:
        baseline = parse_ucsf_case_dir_name("UCSF-PDGM-0004_nifti")
        followup = parse_ucsf_case_dir_name("UCSF-PDGM-0431_FU001d_nifti")

        self.assertEqual(baseline, ("UCSF-PDGM-0004", "baseline"))
        self.assertEqual(followup, ("UCSF-PDGM-0431", "FU001d"))

    def test_parse_ucsd_case_dir_name_extracts_subject_and_visit(self) -> None:
        parsed = parse_ucsd_case_dir_name("UCSD-PTGBM-0002_01")

        self.assertEqual(parsed, ("UCSD-PTGBM-0002", "01"))

    def test_parse_cfb_series_path_extracts_subject_visit_and_modality(self) -> None:
        parsed = parse_cfb_series_path("1_t0_t1gd.nii.gz")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.subject_id, "001")
        self.assertEqual(parsed.visit_id, "t0")
        self.assertEqual(parsed.modality, "t1ce")

    def test_parse_ucsd_series_name_extracts_modalities_and_masks(self) -> None:
        self.assertEqual(
            parse_ucsd_series_name(
                "UCSD-PTGBM-0002_01_T1post.nii.gz",
                case_prefix="UCSD-PTGBM-0002_01",
            ),
            "t1ce",
        )
        self.assertEqual(
            parse_ucsd_series_name(
                "UCSD-PTGBM-0002_01_BraTS_tumor_seg.nii.gz",
                case_prefix="UCSD-PTGBM-0002_01",
            ),
            "brats_tumor_seg",
        )

    def test_parse_ucsf_series_name_extracts_modalities_and_mask(self) -> None:
        self.assertEqual(
            parse_ucsf_series_name(
                "UCSF-PDGM-0004_T1c.nii.gz",
                case_prefix="UCSF-PDGM-0004",
            ),
            "t1ce",
        )
        self.assertEqual(
            parse_ucsf_series_name(
                "UCSF-PDGM-0004_tumor_segmentation.nii.gz",
                case_prefix="UCSF-PDGM-0004",
            ),
            "tumor_segmentation",
        )

    def test_ucsf_adapter_builds_classification_and_segmentation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "UCSF-PDGM-0004_nifti"
            _write_nifti(case_dir / "UCSF-PDGM-0004_T1.nii.gz", fill_value=2.0)
            _write_nifti(case_dir / "UCSF-PDGM-0004_FLAIR.nii.gz", fill_value=3.0)
            _write_nifti(
                case_dir / "UCSF-PDGM-0004_tumor_segmentation.nii.gz",
                fill_value=1.0,
            )

            records = UcsfPdgmAdapter(root).build_records()

            self.assertEqual(len(records), 2)
            records_by_type = {record.task_type: record for record in records}
            cls_record = records_by_type["classification"]
            seg_record = records_by_type["segmentation"]

            self.assertEqual(cls_record.dataset_key, "ucsf_pdgm")
            self.assertEqual(cls_record.subject_id, "ucsf_pdgm:UCSF-PDGM-0004")
            self.assertEqual(cls_record.visit_id, "baseline")
            self.assertEqual(cls_record.global_case_id, "ucsf_pdgm__UCSF-PDGM-0004__baseline")
            self.assertEqual(cls_record.label_family, "tumor")
            self.assertEqual(cls_record.preproc_profile, UCSF_PDGM_PREPROC_PROFILE)
            self.assertEqual(cls_record.exclude_reason, "")

            self.assertTrue(seg_record.global_case_id.endswith("__seg"))
            self.assertEqual(seg_record.mask_tier, "curated")
            self.assertEqual(seg_record.exclude_reason, "")

    def test_ucsf_adapter_uses_followup_visit_and_anchor_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "UCSF-PDGM-0431_FU001d_nifti"
            t1c_path = case_dir / "UCSF-PDGM-0431_FU001d_T1c.nii.gz"
            mask_path = case_dir / "UCSF-PDGM-0431_FU001d_tumor_segmentation.nii.gz"
            _write_nifti(t1c_path, fill_value=9.0)
            _write_nifti(mask_path, fill_value=1.0)

            records = UcsfPdgmAdapter(root).build_records(include_excluded=True)

            records_by_type = {record.task_type: record for record in records}
            self.assertEqual(records_by_type["classification"].visit_id, "FU001d")
            self.assertEqual(
                records_by_type["classification"].exclude_reason,
                "missing_t1_image",
            )
            self.assertEqual(records_by_type["segmentation"].image_path, str(t1c_path))
            self.assertEqual(records_by_type["segmentation"].mask_path, str(mask_path))

    def test_ucsd_adapter_builds_classification_and_segmentation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "UCSD-PTGBM-0002_01"
            t1_path = case_dir / "UCSD-PTGBM-0002_01_T1pre.nii.gz"
            t1ce_path = case_dir / "UCSD-PTGBM-0002_01_T1post.nii.gz"
            seg_path = case_dir / "UCSD-PTGBM-0002_01_BraTS_tumor_seg.nii.gz"
            _write_nifti(t1_path, fill_value=2.0)
            _write_nifti(t1ce_path, fill_value=3.0)
            _write_nifti(seg_path, fill_value=1.0)

            records = UcsdPtgbmAdapter(root).build_records()

            self.assertEqual(len(records), 2)
            records_by_type = {record.task_type: record for record in records}
            cls_record = records_by_type["classification"]
            seg_record = records_by_type["segmentation"]

            self.assertEqual(cls_record.dataset_key, "ucsd_ptgbm")
            self.assertEqual(cls_record.subject_id, "ucsd_ptgbm:UCSD-PTGBM-0002")
            self.assertEqual(cls_record.visit_id, "01")
            self.assertEqual(cls_record.global_case_id, "ucsd_ptgbm__UCSD-PTGBM-0002__01")
            self.assertEqual(cls_record.preproc_profile, UCSD_PTGBM_PREPROC_PROFILE)
            self.assertEqual(cls_record.exclude_reason, "")

            self.assertEqual(seg_record.image_path, str(t1ce_path))
            self.assertEqual(seg_record.mask_path, str(seg_path))
            self.assertEqual(seg_record.mask_tier, "curated")
            self.assertEqual(seg_record.exclude_reason, "")

    def test_ucsd_adapter_marks_missing_t1_classification_rows_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "UCSD-PTGBM-0002_01"
            t1ce_path = case_dir / "UCSD-PTGBM-0002_01_T1post.nii.gz"
            derived_seg = case_dir / "UCSD-PTGBM-0002_01_total_cellular_tumor_seg.nii.gz"
            _write_nifti(t1ce_path, fill_value=3.0)
            _write_nifti(derived_seg, fill_value=1.0)

            records = UcsdPtgbmAdapter(root).build_records(include_excluded=True)

            records_by_type = {record.task_type: record for record in records}
            self.assertEqual(
                records_by_type["classification"].exclude_reason,
                "missing_t1_image",
            )
            self.assertEqual(records_by_type["classification"].t1_path, "")
            self.assertEqual(records_by_type["segmentation"].image_path, str(t1ce_path))
            self.assertEqual(records_by_type["segmentation"].mask_tier, "derived")

    def test_utsw_adapter_builds_classification_and_segmentation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "BT0501"
            _write_nifti(case_dir / "brain_t1.nii.gz", fill_value=2.0)
            t1ce_path = case_dir / "brain_t1ce.nii.gz"
            manual_mask_path = case_dir / "tumorseg_manual_correction.nii.gz"
            _write_nifti(t1ce_path, fill_value=3.0)
            _write_nifti(manual_mask_path, fill_value=1.0)
            _write_nifti(case_dir / "tumorseg_FeTS.nii.gz", fill_value=1.0)

            records = UtswGliomaAdapter(root).build_records()

            self.assertEqual(len(records), 2)
            records_by_type = {record.task_type: record for record in records}
            cls_record = records_by_type["classification"]
            seg_record = records_by_type["segmentation"]

            self.assertEqual(cls_record.dataset_key, "utsw_glioma")
            self.assertEqual(cls_record.subject_id, "utsw_glioma:BT0501")
            self.assertEqual(cls_record.visit_id, "baseline")
            self.assertEqual(cls_record.global_case_id, "utsw_glioma__BT0501__baseline")
            self.assertEqual(cls_record.preproc_profile, UTSW_GLIOMA_PREPROC_PROFILE)
            self.assertEqual(cls_record.exclude_reason, "")

            self.assertEqual(seg_record.image_path, str(t1ce_path))
            self.assertEqual(seg_record.mask_path, str(manual_mask_path))
            self.assertEqual(seg_record.mask_tier, "curated")
            self.assertEqual(seg_record.exclude_reason, "")

    def test_utsw_adapter_falls_back_to_fets_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "BT0502"
            _write_nifti(case_dir / "brain_t1.nii.gz", fill_value=2.0)
            _write_nifti(case_dir / "brain_t1ce.nii.gz", fill_value=3.0)
            fets_mask = case_dir / "tumorseg_FeTS.nii.gz"
            _write_nifti(fets_mask, fill_value=1.0)

            records = UtswGliomaAdapter(root).build_records()

            seg_record = {record.task_type: record for record in records}["segmentation"]
            self.assertEqual(seg_record.mask_path, str(fets_mask))
            self.assertEqual(seg_record.mask_tier, "derived")

    def test_parse_upenn_series_path_extracts_subject_visit_and_modality(self) -> None:
        parsed = parse_upenn_series_path("UPENN-GBM-00001_11_T1GD.nii.gz")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.subject_id, "UPENN-GBM-00001")
        self.assertEqual(parsed.visit_id, "11")
        self.assertEqual(parsed.modality, "t1ce")

    def test_parse_yale_series_path_extracts_subject_visit_and_modality(self) -> None:
        parsed = parse_yale_series_path(
            "YG_TEST1234_2021-03-10_17-01-09_PRE.nii.gz"
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.subject_id, "YG_TEST1234")
        self.assertEqual(parsed.visit_id, "2021-03-10")
        self.assertEqual(parsed.acquisition_time, "17-01-09")
        self.assertEqual(parsed.modality, "t1")

    def test_parse_vestibular_series_path_extracts_subject_visit_and_modality(self) -> None:
        parsed = parse_vestibular_series_path(
            "VS_MC_RC2_123_1992-08-07_T1C_seg.nii.gz"
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.subject_id, "VS_MC_RC2_123")
        self.assertEqual(parsed.visit_id, "1992-08-07")
        self.assertEqual(parsed.modality, "seg")

    def test_upenn_adapter_builds_classification_and_segmentation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            t1_path = root / "UPENN-GBM-00001_11_T1.nii.gz"
            flair_path = root / "UPENN-GBM-00001_11_FLAIR.nii.gz"
            seg_path = root / "UPENN-GBM-00001_11_SEG.nii.gz"
            _write_nifti(t1_path, fill_value=3.0)
            _write_nifti(flair_path, fill_value=4.0)
            _write_nifti(seg_path, fill_value=1.0)

            records = UpennGbmAdapter(root).build_records()

            self.assertEqual(len(records), 2)
            records_by_type = {record.task_type: record for record in records}
            cls_record = records_by_type["classification"]
            seg_record = records_by_type["segmentation"]

            self.assertEqual(cls_record.dataset_key, "upenn_gbm")
            self.assertEqual(cls_record.subject_id, "upenn_gbm:UPENN-GBM-00001")
            self.assertEqual(cls_record.visit_id, "11")
            self.assertEqual(cls_record.global_case_id, "upenn_gbm__UPENN-GBM-00001__11")
            self.assertEqual(cls_record.t1_path, str(t1_path))
            self.assertEqual(cls_record.label_family, "tumor")
            self.assertEqual(cls_record.preproc_profile, UPENN_GBM_PREPROC_PROFILE)
            self.assertEqual(cls_record.exclude_reason, "")

            self.assertEqual(seg_record.image_path, str(t1_path))
            self.assertEqual(seg_record.mask_path, str(seg_path))
            self.assertEqual(seg_record.mask_tier, "curated")
            self.assertEqual(seg_record.exclude_reason, "")

    def test_upenn_adapter_marks_missing_t1_classification_rows_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            t1ce_path = root / "UPENN-GBM-00002_11_T1GD.nii.gz"
            seg_path = root / "UPENN-GBM-00002_11_SEG.nii.gz"
            _write_nifti(t1ce_path, fill_value=7.0)
            _write_nifti(seg_path, fill_value=1.0)

            records = UpennGbmAdapter(root).build_records(include_excluded=True)

            records_by_type = {record.task_type: record for record in records}
            self.assertEqual(
                records_by_type["classification"].exclude_reason,
                "missing_t1_image",
            )
            self.assertEqual(records_by_type["classification"].t1_path, "")
            self.assertEqual(records_by_type["segmentation"].image_path, str(t1ce_path))
            self.assertEqual(records_by_type["segmentation"].exclude_reason, "")

    def test_cfb_adapter_builds_classification_and_segmentation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            visit_dir = root / "001" / "t0"
            t1_path = visit_dir / "1_t0_t1eg.nii.gz"
            t1ce_path = visit_dir / "1_t0_t1gd.nii.gz"
            seg_path = visit_dir / "1_t0_gtv.nii.gz"
            _write_nifti(t1_path, fill_value=2.0)
            _write_nifti(t1ce_path, fill_value=3.0)
            _write_nifti(seg_path, fill_value=1.0)

            records = CfbGbmAdapter(root).build_records()

            self.assertEqual(len(records), 2)
            records_by_type = {record.task_type: record for record in records}
            cls_record = records_by_type["classification"]
            seg_record = records_by_type["segmentation"]

            self.assertEqual(cls_record.dataset_key, "cfb_gbm")
            self.assertEqual(cls_record.subject_id, "cfb_gbm:001")
            self.assertEqual(cls_record.visit_id, "t0")
            self.assertEqual(cls_record.global_case_id, "cfb_gbm__001__t0")
            self.assertEqual(cls_record.preproc_profile, CFB_GBM_PREPROC_PROFILE)
            self.assertEqual(cls_record.exclude_reason, "")

            self.assertEqual(seg_record.image_path, str(t1ce_path))
            self.assertEqual(seg_record.mask_path, str(seg_path))
            self.assertEqual(seg_record.mask_tier, "curated")
            self.assertEqual(seg_record.exclude_reason, "")

    def test_cfb_adapter_marks_missing_real_t1_classification_rows_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            visit_dir = root / "002" / "t1"
            t1ce_path = visit_dir / "2_t1_t1gd.nii.gz"
            seg_path = visit_dir / "2_t1_gtv.nii.gz"
            _write_nifti(t1ce_path, fill_value=3.0)
            _write_nifti(seg_path, fill_value=1.0)

            records = CfbGbmAdapter(root).build_records(include_excluded=True)

            records_by_type = {record.task_type: record for record in records}
            self.assertEqual(
                records_by_type["classification"].exclude_reason,
                "missing_t1_image",
            )
            self.assertEqual(records_by_type["segmentation"].image_path, str(t1ce_path))
            self.assertEqual(records_by_type["segmentation"].exclude_reason, "")

    def test_brain_structure_adapter_maps_cn_and_alzheimer_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cn_path = root / "DLBS" / "P001" / "ses-01" / "t1.nii.gz"
            ad_path = root / "OASIS-2" / "P002" / "ses-02" / "t1.nii.gz"
            _write_nifti(cn_path, fill_value=1.0)
            _write_nifti(ad_path, fill_value=2.0)
            _write_metadata(
                root,
                [
                    {
                        "t1_local_path": "DLBS/P001/ses-01/t1.nii.gz",
                        "split": "train",
                        "study": "DLBS",
                        "participant_id": "P001",
                        "session_id": "ses-01",
                        "age": "70",
                        "sex": "F",
                        "clinical_diagnosis": "cognitively_normal",
                        "scanner_manufacturer": "GE",
                        "scanner_model": "SIGNA",
                        "field_strength": "3T",
                        "image_quality_rating": "4",
                        "total_intracranial_volume": "1234",
                        "radiata_id": "RID-001",
                    },
                    {
                        "t1_local_path": "OASIS-2/P002/ses-02/t1.nii.gz",
                        "split": "test",
                        "study": "OASIS-2",
                        "participant_id": "P002",
                        "session_id": "ses-02",
                        "age": "77",
                        "sex": "M",
                        "clinical_diagnosis": "alzheimers_disease",
                        "scanner_manufacturer": "Siemens",
                        "scanner_model": "Prisma",
                        "field_strength": "3T",
                        "image_quality_rating": "5",
                        "total_intracranial_volume": "1337",
                        "radiata_id": "RID-002",
                    },
                ],
            )

            records = BrainStructureAdapter(root).build_records()
            records_by_subject = {record.subject_id: record for record in records}

            cn_record = records_by_subject["brain_structure:DLBS:P001"]
            self.assertEqual(cn_record.visit_id, "ses-01")
            self.assertEqual(cn_record.global_case_id, "brain_structure__DLBS__P001__ses-01")
            self.assertEqual(cn_record.label, "healthy")
            self.assertEqual(cn_record.label_family, "healthy_control")
            self.assertEqual(cn_record.binary_status, "healthy")
            self.assertEqual(cn_record.preproc_profile, BRAIN_STRUCTURE_PREPROC_PROFILE)

            ad_record = records_by_subject["brain_structure:OASIS-2:P002"]
            self.assertEqual(ad_record.label, "unhealthy")
            self.assertEqual(ad_record.label_family, "neurodegenerative")
            self.assertEqual(ad_record.binary_status, "unhealthy")
            self.assertEqual(ad_record.source_split, "test")

    def test_brain_structure_adapter_sets_exclusion_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zero_path = root / "NKI-RS" / "P100" / "ses-01" / "t1.nii.gz"
            unknown_path = root / "IXI" / "P101" / "ses-01" / "t1.nii.gz"
            _write_nifti(zero_path, fill_value=0.0)
            _write_nifti(unknown_path, fill_value=1.0)
            (root / "data.zip").write_bytes(b"zip placeholder")
            _write_metadata(
                root,
                [
                    {
                        "t1_local_path": "NKI-RS/P100/ses-01/t1.nii.gz",
                        "split": "train",
                        "study": "NKI-RS",
                        "participant_id": "P100",
                        "session_id": "ses-01",
                        "age": "61",
                        "sex": "F",
                        "clinical_diagnosis": "cognitively_normal",
                        "scanner_manufacturer": "GE",
                        "scanner_model": "Discovery",
                        "field_strength": "3T",
                        "image_quality_rating": "3",
                        "total_intracranial_volume": "1111",
                        "radiata_id": "RID-100",
                    },
                    {
                        "t1_local_path": "IXI/P101/ses-01/t1.nii.gz",
                        "split": "val",
                        "study": "IXI",
                        "participant_id": "P101",
                        "session_id": "ses-01",
                        "age": "65",
                        "sex": "M",
                        "clinical_diagnosis": "vascular_dementia",
                        "scanner_manufacturer": "Philips",
                        "scanner_model": "Ingenia",
                        "field_strength": "3T",
                        "image_quality_rating": "2",
                        "total_intracranial_volume": "1222",
                        "radiata_id": "RID-101",
                    },
                    {
                        "t1_local_path": "DLBS/P102/ses-02/missing.nii.gz",
                        "split": "test",
                        "study": "DLBS",
                        "participant_id": "P102",
                        "session_id": "ses-02",
                        "age": "72",
                        "sex": "F",
                        "clinical_diagnosis": "cognitively_normal",
                        "scanner_manufacturer": "GE",
                        "scanner_model": "SIGNA",
                        "field_strength": "3T",
                        "image_quality_rating": "5",
                        "total_intracranial_volume": "1444",
                        "radiata_id": "RID-102",
                    },
                ],
            )

            records = BrainStructureAdapter(root).build_records(include_excluded=True)
            reasons = {record.subject_id: record.exclude_reason for record in records}

            self.assertEqual(reasons["brain_structure:NKI-RS:P100"], "empty_brain_after_load")
            self.assertEqual(
                reasons["brain_structure:IXI:P101"],
                "unmapped_brain_structure_label",
            )
            self.assertEqual(reasons["brain_structure:DLBS:P102"], "zip_only_not_extracted")

    def test_task_manifest_builders_keep_ad_separate_from_tumor_vs_cn(self) -> None:
        rows = [
            _classification_row(
                dataset_key="brain_structure",
                input_path="/tmp/cn1.nii.gz",
                global_case_id="brain_structure__DLBS__CN1__ses-01",
                subject_id="brain_structure:DLBS:CN1",
                preproc_profile="brain_structure_affine_mni_to_cls128",
            ),
            _classification_row(
                dataset_key="brain_structure",
                input_path="/tmp/ad1.nii.gz",
                global_case_id="brain_structure__OASIS-2__AD1__ses-02",
                subject_id="brain_structure:OASIS-2:AD1",
                preproc_profile="brain_structure_affine_mni_to_cls128",
                source_split="test",
                label_family="neurodegenerative",
                binary_status="unhealthy",
                label="unhealthy",
                diagnosis_original="alzheimers_disease",
            ),
            _classification_row(
                dataset_key="upenn_gbm",
                input_path="/tmp/tumor1.nii.gz",
                global_case_id="upenn_gbm__SUBJ1__v1",
                subject_id="upenn_gbm:SUBJ1",
                preproc_profile="tumor_cls128",
                label_family="tumor",
                binary_status="unhealthy",
                label="unhealthy",
                diagnosis_original="glioblastoma",
            ),
            {
                **_classification_row(
                    dataset_key="upenn_gbm",
                    input_path="/tmp/tumor1-followup.nii.gz",
                    global_case_id="upenn_gbm__SUBJ1__v2",
                    subject_id="upenn_gbm:SUBJ1",
                    preproc_profile="tumor_cls128",
                    label_family="tumor",
                    binary_status="unhealthy",
                    label="unhealthy",
                    diagnosis_original="glioblastoma",
                ),
                "visit_id": "v2",
            },
        ]

        manifests = build_all_task_manifests(rows, include_any_unhealthy=True)
        tumor_vs_cn = manifests["classification_t1_tumor_vs_cn.csv"]
        ad_vs_cn = manifests["classification_t1_ad_vs_cn.csv"]

        self.assertEqual(
            {row["subject_id"] for row in tumor_vs_cn},
            {"brain_structure:DLBS:CN1", "upenn_gbm:SUBJ1"},
        )
        self.assertNotIn(
            "brain_structure:OASIS-2:AD1",
            {row["subject_id"] for row in tumor_vs_cn},
        )

        tumor_splits = {
            row["visit_id"]: row["task_split"]
            for row in tumor_vs_cn
            if row["subject_id"] == "upenn_gbm:SUBJ1"
        }
        self.assertEqual(len(set(tumor_splits.values())), 1)

        self.assertEqual(
            {row["subject_id"] for row in ad_vs_cn},
            {"brain_structure:DLBS:CN1", "brain_structure:OASIS-2:AD1"},
        )
        self.assertEqual({row["dataset_key"] for row in ad_vs_cn}, {"brain_structure"})
        self.assertEqual(
            {
                row["task_split"]
                for row in ad_vs_cn
                if row["subject_id"] == "brain_structure:OASIS-2:AD1"
            },
            {"test"},
        )

    def test_skip_policy_avoids_synthstrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "NKI-RS" / "P200" / "ses-01" / "t1.nii.gz"
            input_path.parent.mkdir(parents=True, exist_ok=True)

            volume = np.zeros((10, 12, 14), dtype=np.float32)
            volume[2:8, 3:10, 4:11] = np.linspace(
                1.0,
                20.0,
                num=6 * 7 * 7,
            ).reshape(6, 7, 7)
            affine = np.diag([-1.0, -1.0, 1.5, 1.0])
            nib.save(nib.Nifti1Image(volume, affine), str(input_path))

            with mock.patch(
                "src.datasets.standardize.preprocess.subprocess.run",
                side_effect=AssertionError("SynthStrip should not run for skip policy"),
            ):
                result = write_brain_structure_cls_view(
                    input_path,
                    root / "outputs" / "brain_structure__NKI-RS__P200__ses-01",
                )

            self.assertEqual(result.image_shape, (128, 128, 128))
            self.assertEqual(result.normalization_mask_method, "nonzero")
            self.assertEqual(result.preproc_profile, BRAIN_STRUCTURE_PREPROC_PROFILE)
            self.assertTrue(result.image_path.is_file())
            self.assertTrue(
                result.brain_mask_path is not None and result.brain_mask_path.is_file()
            )

    def test_synthstrip_policy_runs_external_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "ReMIND" / "P201" / "ses-01" / "t1.nii.gz"
            input_path.parent.mkdir(parents=True, exist_ok=True)

            volume = np.ones((12, 12, 12), dtype=np.float32)
            volume[3:9, 3:9, 3:9] = 10.0
            nib.save(nib.Nifti1Image(volume, np.eye(4)), str(input_path))

            def fake_synthstrip_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual(cmd[0], "mri_synthstrip")
                input_arg = Path(cmd[cmd.index("-i") + 1])
                output_arg = Path(cmd[cmd.index("-o") + 1])
                mask_arg = Path(cmd[cmd.index("-m") + 1])
                input_image = nib.load(str(input_arg))
                input_volume = np.asanyarray(input_image.dataobj).astype(np.float32)
                stripped = np.zeros_like(input_volume)
                stripped[2:10, 2:10, 2:10] = input_volume[2:10, 2:10, 2:10]
                mask = (stripped != 0).astype(np.uint8)
                nib.save(nib.Nifti1Image(stripped, input_image.affine), str(output_arg))
                nib.save(nib.Nifti1Image(mask, input_image.affine), str(mask_arg))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch(
                "src.datasets.standardize.preprocess.shutil.which",
                return_value="/usr/bin/mri_synthstrip",
            ), mock.patch(
                "src.datasets.standardize.preprocess.subprocess.run",
                side_effect=fake_synthstrip_run,
            ):
                result = write_classification_view(
                    input_path,
                    root / "outputs" / "remind__P201__ses-01",
                    dataset_key=REMIND_DATASET_KEY,
                    original_preproc_profile="remind_cls128",
                )

            self.assertEqual(result.normalization_mask_method, "synthstrip")
            self.assertEqual(result.preproc_profile, "remind_cls128_synthstrip")
            self.assertTrue(result.image_path.is_file())
            self.assertTrue(
                result.brain_mask_path is not None and result.brain_mask_path.is_file()
            )

    def test_synthstrip_policy_supports_dicom_series_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            series_dir = root / "ReMIND-001" / "study1" / "series1"
            _write_dicom_series(
                series_dir,
                series_description="3D_AX_T1_precontrast",
                base_value=10,
            )

            def fake_synthstrip_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                input_arg = Path(cmd[cmd.index("-i") + 1])
                output_arg = Path(cmd[cmd.index("-o") + 1])
                mask_arg = Path(cmd[cmd.index("-m") + 1])
                input_image = nib.load(str(input_arg))
                input_volume = np.asanyarray(input_image.dataobj).astype(np.float32)
                stripped = np.zeros_like(input_volume)
                stripped[1:3, 1:3, :] = input_volume[1:3, 1:3, :]
                mask = (stripped != 0).astype(np.uint8)
                nib.save(nib.Nifti1Image(stripped, input_image.affine), str(output_arg))
                nib.save(nib.Nifti1Image(mask, input_image.affine), str(mask_arg))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch(
                "src.datasets.standardize.preprocess.shutil.which",
                return_value="/usr/bin/mri_synthstrip",
            ), mock.patch(
                "src.datasets.standardize.preprocess.subprocess.run",
                side_effect=fake_synthstrip_run,
            ):
                result = write_classification_view(
                    series_dir,
                    root / "outputs" / "remind__ReMIND-001__preop",
                    dataset_key=REMIND_DATASET_KEY,
                    original_preproc_profile=REMIND_PREPROC_PROFILE,
                )

            self.assertEqual(result.normalization_mask_method, "synthstrip")
            self.assertTrue(result.image_path.is_file())

    def test_remind_adapter_builds_classification_and_segmentation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "remind"
            mask_root = root / "masks"

            _write_dicom_series(
                image_root / "ReMIND-001" / "study-a" / "series-t1-pre",
                series_description="3D_AX_T1_precontrast",
                base_value=10,
            )
            _write_dicom_series(
                image_root / "ReMIND-001" / "study-a" / "series-t1-post",
                series_description="3D_AX_T1_postcontrast",
                base_value=20,
            )
            _write_nrrd(
                mask_root
                / "ReMIND-001"
                / "ReMIND-001-preop-SEG-tumor-MR-3D_AX_T1_postcontrast.nrrd",
                fill_value=1.0,
            )

            records = RemindAdapter(image_root, mask_root).build_records(include_excluded=True)
            records_by_type = {record.task_type: record for record in records}
            cls_record = records_by_type["classification"]
            seg_record = records_by_type["segmentation"]

            self.assertEqual(cls_record.subject_id, "remind:ReMIND-001")
            self.assertEqual(cls_record.visit_id, "preop")
            self.assertEqual(cls_record.preproc_profile, REMIND_PREPROC_PROFILE)
            self.assertIn("series-t1-pre", cls_record.t1_path)
            self.assertEqual(cls_record.exclude_reason, "")

            self.assertTrue(seg_record.global_case_id.endswith("3d_ax_t1_postcontrast"))
            self.assertIn("series-t1-post", seg_record.image_path)
            self.assertTrue(seg_record.mask_path.endswith(".nrrd"))
            self.assertEqual(seg_record.exclude_reason, "")

    def test_remind_adapter_marks_missing_real_t1_classification_rows_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "remind"
            mask_root = root / "masks"

            _write_dicom_series(
                image_root / "ReMIND-002" / "study-a" / "series-t1-post",
                series_description="3D_AX_T1_postcontrast",
                base_value=20,
            )
            _write_nrrd(
                mask_root
                / "ReMIND-002"
                / "ReMIND-002-preop-SEG-tumor-MR-3D_AX_T1_postcontrast.nrrd",
                fill_value=1.0,
            )

            records = RemindAdapter(image_root, mask_root).build_records(include_excluded=True)
            records_by_type = {record.task_type: record for record in records}
            self.assertEqual(
                records_by_type["classification"].exclude_reason,
                "missing_t1_image",
            )
            self.assertEqual(records_by_type["segmentation"].exclude_reason, "")

    def test_cli_preflight_fails_when_synthstrip_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "ReMIND" / "P210" / "ses-01" / "t1.nii.gz"
            _write_nifti(input_path, fill_value=5.0)
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _classification_row(
                        dataset_key=REMIND_DATASET_KEY,
                        input_path=str(input_path),
                        global_case_id="remind__P210__ses-01",
                        subject_id="remind:P210",
                        preproc_profile="remind_cls128",
                    )
                ],
            )

            with mock.patch(
                "src.datasets.standardize.preprocess.shutil.which",
                return_value=None,
            ):
                with self.assertRaisesRegex(RuntimeError, "SynthStrip command not found"):
                    standardize_cli.main(
                        [
                            "preprocess-brain-structure-cls",
                            "--input-manifest",
                            str(manifest_path),
                            "--output-dir",
                            str(root / "db"),
                        ]
                    )

    def test_preflight_accepts_multi_part_synthstrip_command(self) -> None:
        rows = [
            _classification_row(
                dataset_key=REMIND_DATASET_KEY,
                input_path="dummy.nii.gz",
                global_case_id="remind__P220__ses-01",
                subject_id="remind:P220",
                preproc_profile="remind_cls128",
            )
        ]
        command = f'"{sys.executable}" -c "print(123)"'
        preflight_synthstrip_requirements(rows, synthstrip_cmd=command)
        self.assertTrue(_command_exists(command))

    def test_materialize_uses_one_managed_docker_session_for_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                _classification_row(
                    dataset_key=REMIND_DATASET_KEY,
                    input_path="dummy.nii.gz",
                    global_case_id="remind__P230__ses-01",
                    subject_id="remind:P230",
                    preproc_profile="remind_cls128",
                )
            ]
            fake_output = root / "db" / "remind__P230__ses-01" / "cls" / "t1_128.nii.gz"
            fake_mask = root / "db" / "remind__P230__ses-01" / "cls" / "brain_mask_128.nii.gz"
            fake_output.parent.mkdir(parents=True, exist_ok=True)
            fake_output.write_bytes(b"")
            fake_mask.write_bytes(b"")

            command = f'"{sys.executable}" scripts/synthstrip_docker.py'
            with mock.patch(
                "src.datasets.standardize.preprocess.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ) as run_mock, mock.patch(
                "src.datasets.standardize.preprocess.write_classification_view",
                return_value=ClassificationViewResult(
                    image_path=fake_output,
                    brain_mask_path=fake_mask,
                    image_shape=(128, 128, 128),
                    normalization_mask_method="synthstrip",
                    preproc_profile="remind_cls128_synthstrip",
                ),
            ):
                materialize_classification_manifest(
                    rows,
                    root / "db",
                    synthstrip_cmd=command,
                )

            command_calls = [call.args[0] for call in run_mock.call_args_list]
            self.assertTrue(any("--start-session" in call for call in command_calls))
            self.assertTrue(any("--stop-session" in call for call in command_calls))

    def test_materialize_reuses_existing_outputs_without_reprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                _classification_row(
                    dataset_key=REMIND_DATASET_KEY,
                    input_path="dummy.nii.gz",
                    global_case_id="remind__P240__ses-01",
                    subject_id="remind:P240",
                    preproc_profile="remind_cls128",
                )
            ]
            case_root = root / "db" / "remind__P240__ses-01"
            image_path = case_root / "cls" / "t1_128.nii.gz"
            mask_path = case_root / "cls" / "brain_mask_128.nii.gz"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(
                nib.Nifti1Image(
                    np.ones((128, 128, 128), dtype=np.float32),
                    np.eye(4),
                ),
                str(image_path),
            )
            nib.save(
                nib.Nifti1Image(
                    np.ones((128, 128, 128), dtype=np.uint8),
                    np.eye(4),
                ),
                str(mask_path),
            )

            with mock.patch(
                "src.datasets.standardize.preprocess.write_classification_view",
                side_effect=AssertionError("should reuse existing outputs"),
            ):
                materialized_rows, manifest_path = materialize_classification_manifest(
                    rows,
                    root / "db",
                    synthstrip_cmd="definitely_missing_synthstrip",
                )

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(materialized_rows), 1)
            row = materialized_rows[0]
            self.assertEqual(row["image_path"], str(image_path))
            self.assertEqual(row["t1_path"], str(image_path))
            self.assertEqual(row["brain_mask_path"], str(mask_path))
            self.assertEqual(row["normalization_mask_method"], "synthstrip")
            self.assertEqual(row["exclude_reason"], "")

    def test_materialize_reprocesses_invalid_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                _classification_row(
                    dataset_key=REMIND_DATASET_KEY,
                    input_path="dummy.nii.gz",
                    global_case_id="remind__P250__ses-01",
                    subject_id="remind:P250",
                    preproc_profile="remind_cls128",
                )
            ]
            case_root = root / "db" / "remind__P250__ses-01"
            image_path = case_root / "cls" / "t1_128.nii.gz"
            mask_path = case_root / "cls" / "brain_mask_128.nii.gz"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"bad")
            mask_path.write_bytes(b"bad")

            command = f'"{sys.executable}" -c "print(123)"'
            with mock.patch(
                "src.datasets.standardize.preprocess.write_classification_view",
                return_value=ClassificationViewResult(
                    image_path=image_path,
                    brain_mask_path=mask_path,
                    image_shape=(128, 128, 128),
                    normalization_mask_method="synthstrip",
                    preproc_profile="remind_cls128_synthstrip",
                ),
            ) as write_mock:
                materialize_classification_manifest(
                    rows,
                    root / "db",
                    synthstrip_cmd=command,
                )

            self.assertEqual(write_mock.call_count, 1)

    def test_materialize_reuses_existing_failed_rows_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                _classification_row(
                    dataset_key=REMIND_DATASET_KEY,
                    input_path="dummy.nii.gz",
                    global_case_id="remind__P255__ses-01",
                    subject_id="remind:P255",
                    preproc_profile="remind_cls128",
                )
            ]
            manifest_path = root / "db" / "classification_materialized_manifest.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(rows[0].keys()),
                )
                writer.writeheader()
                failed_row = dict(rows[0])
                failed_row["exclude_reason"] = "low_signal_image"
                failed_row["image_path"] = ""
                failed_row["t1_path"] = ""
                writer.writerow(failed_row)

            with mock.patch(
                "src.datasets.standardize.preprocess.write_classification_view",
                side_effect=AssertionError("should reuse failed row"),
            ):
                materialized_rows, _ = materialize_classification_manifest(
                    rows,
                    root / "db",
                    synthstrip_cmd="definitely_missing_synthstrip",
                )

            self.assertEqual(len(materialized_rows), 1)
            self.assertEqual(materialized_rows[0]["exclude_reason"], "low_signal_image")
            self.assertEqual(materialized_rows[0]["t1_path"], "")

    def test_cli_build_upenn_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_nifti(root / "UPENN-GBM-00003_11_T1.nii.gz", fill_value=2.0)
            _write_nifti(root / "UPENN-GBM-00003_11_SEG.nii.gz", fill_value=1.0)
            output_csv = root / "out" / "upenn.csv"

            standardize_cli.main(
                [
                    "build-upenn-gbm-manifest",
                    "--upenn-gbm-root",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["task_type"] for row in rows}, {"classification", "segmentation"})

    def test_cli_build_cfb_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            visit_dir = root / "001" / "t0"
            _write_nifti(visit_dir / "1_t0_t1eg.nii.gz", fill_value=2.0)
            _write_nifti(visit_dir / "1_t0_t1gd.nii.gz", fill_value=3.0)
            _write_nifti(visit_dir / "1_t0_gtv.nii.gz", fill_value=1.0)
            output_csv = root / "out" / "cfb.csv"

            standardize_cli.main(
                [
                    "build-cfb-gbm-manifest",
                    "--cfb-gbm-root",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["task_type"] for row in rows}, {"classification", "segmentation"})

    def test_cli_build_ucsf_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "UCSF-PDGM-0004_nifti"
            _write_nifti(case_dir / "UCSF-PDGM-0004_T1.nii.gz", fill_value=2.0)
            _write_nifti(
                case_dir / "UCSF-PDGM-0004_tumor_segmentation.nii.gz",
                fill_value=1.0,
            )
            output_csv = root / "out" / "ucsf.csv"

            standardize_cli.main(
                [
                    "build-ucsf-pdgm-manifest",
                    "--ucsf-pdgm-root",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["task_type"] for row in rows}, {"classification", "segmentation"})

    def test_cli_build_ucsd_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "UCSD-PTGBM-0002_01"
            _write_nifti(case_dir / "UCSD-PTGBM-0002_01_T1pre.nii.gz", fill_value=2.0)
            _write_nifti(case_dir / "UCSD-PTGBM-0002_01_T1post.nii.gz", fill_value=3.0)
            _write_nifti(
                case_dir / "UCSD-PTGBM-0002_01_BraTS_tumor_seg.nii.gz",
                fill_value=1.0,
            )
            output_csv = root / "out" / "ucsd.csv"

            standardize_cli.main(
                [
                    "build-ucsd-ptgbm-manifest",
                    "--ucsd-ptgbm-root",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["task_type"] for row in rows}, {"classification", "segmentation"})

    def test_cli_build_utsw_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_dir = root / "BT0501"
            _write_nifti(case_dir / "brain_t1.nii.gz", fill_value=2.0)
            _write_nifti(case_dir / "brain_t1ce.nii.gz", fill_value=3.0)
            _write_nifti(case_dir / "tumorseg_manual_correction.nii.gz", fill_value=1.0)
            output_csv = root / "out" / "utsw.csv"

            standardize_cli.main(
                [
                    "build-utsw-glioma-manifest",
                    "--utsw-glioma-root",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["task_type"] for row in rows}, {"classification", "segmentation"})

    def test_yale_adapter_builds_classification_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_nifti(
                root / "YG_TEST1234" / "2021-03-10" / "YG_TEST1234_2021-03-10_17-01-09_PRE.nii.gz",
                fill_value=2.0,
            )
            _write_nifti(
                root / "YG_TEST1234" / "2021-03-10" / "YG_TEST1234_2021-03-10_17-01-09_POST.nii.gz",
                fill_value=3.0,
            )

            records = YaleBrainMetsLongitudinalAdapter(root).build_records()

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(
                record.subject_id,
                "yale_brain_mets_longitudinal:YG_TEST1234",
            )
            self.assertEqual(record.visit_id, "2021-03-10")
            self.assertEqual(
                record.preproc_profile,
                YALE_BRAIN_METS_LONGITUDINAL_PREPROC_PROFILE,
            )
            self.assertTrue(record.t1_path.endswith("_PRE.nii.gz"))
            self.assertEqual(record.exclude_reason, "")

    def test_yale_adapter_marks_missing_pre_image_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_nifti(
                root / "YG_TEST1234" / "2021-03-10" / "YG_TEST1234_2021-03-10_17-01-09_POST.nii.gz",
                fill_value=3.0,
            )

            records = YaleBrainMetsLongitudinalAdapter(root).build_records(
                include_excluded=True
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].exclude_reason, "missing_t1_image")

    def test_yale_adapter_marks_invalid_pre_image_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            invalid_pre = (
                root
                / "YG_TEST1234"
                / "2021-03-10"
                / "YG_TEST1234_2021-03-10_17-01-09_PRE.nii.gz"
            )
            invalid_pre.parent.mkdir(parents=True, exist_ok=True)
            invalid_pre.write_bytes(b"")

            records = YaleBrainMetsLongitudinalAdapter(root).build_records(
                include_excluded=True
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].exclude_reason, "invalid_image_file")

    def test_vestibular_adapter_builds_classification_and_segmentation_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            t1_path = root / "VS_MC_RC2_123_1992-08-07_T1.nii.gz"
            t1ce_path = root / "VS_MC_RC2_123_1992-08-07_T1C.nii.gz"
            seg_path = root / "VS_MC_RC2_123_1992-08-07_T1C_seg.nii.gz"
            _write_nifti(t1_path, fill_value=3.0)
            _write_nifti(t1ce_path, fill_value=4.0)
            _write_nifti(seg_path, fill_value=1.0)

            records = VestibularSchwannomaMcRc2Adapter(root).build_records()

            self.assertEqual(len(records), 2)
            records_by_type = {record.task_type: record for record in records}
            cls_record = records_by_type["classification"]
            seg_record = records_by_type["segmentation"]

            self.assertEqual(cls_record.dataset_key, "vestibular_schwannoma_mc_rc2")
            self.assertEqual(
                cls_record.subject_id,
                "vestibular_schwannoma_mc_rc2:VS_MC_RC2_123",
            )
            self.assertEqual(cls_record.visit_id, "1992-08-07")
            self.assertEqual(
                cls_record.global_case_id,
                "vestibular_schwannoma_mc_rc2__VS_MC_RC2_123__1992-08-07",
            )
            self.assertEqual(
                cls_record.preproc_profile,
                VESTIBULAR_SCHWANNOMA_MC_RC2_PREPROC_PROFILE,
            )
            self.assertEqual(cls_record.exclude_reason, "")

            self.assertEqual(seg_record.image_path, str(t1ce_path))
            self.assertEqual(seg_record.mask_path, str(seg_path))
            self.assertEqual(seg_record.mask_tier, "curated")
            self.assertEqual(seg_record.exclude_reason, "")

    def test_vestibular_adapter_marks_missing_t1_classification_rows_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            t1ce_path = root / "VS_MC_RC2_124_1993-12-28_T1C.nii.gz"
            seg_path = root / "VS_MC_RC2_124_1993-12-28_T1C_seg.nii.gz"
            _write_nifti(t1ce_path, fill_value=7.0)
            _write_nifti(seg_path, fill_value=1.0)

            records = VestibularSchwannomaMcRc2Adapter(root).build_records(
                include_excluded=True
            )

            records_by_type = {record.task_type: record for record in records}
            self.assertEqual(
                records_by_type["classification"].exclude_reason,
                "missing_t1_image",
            )
            self.assertEqual(records_by_type["classification"].t1_path, "")
            self.assertEqual(records_by_type["segmentation"].image_path, str(t1ce_path))
            self.assertEqual(records_by_type["segmentation"].exclude_reason, "")

    def test_cli_build_remind_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "remind"
            mask_root = root / "masks"
            _write_dicom_series(
                image_root / "ReMIND-003" / "study-a" / "series-t1-pre",
                series_description="3D_AX_T1_precontrast",
                base_value=10,
            )
            _write_dicom_series(
                image_root / "ReMIND-003" / "study-a" / "series-t2",
                series_description="3D_AX_T2_SPACE",
                base_value=30,
            )
            _write_nrrd(
                mask_root
                / "ReMIND-003"
                / "ReMIND-003-preop-SEG-tumor-MR-3D_AX_T2_SPACE.nrrd",
                fill_value=1.0,
            )
            output_csv = root / "out" / "remind.csv"

            standardize_cli.main(
                [
                    "build-remind-manifest",
                    "--remind-root",
                    str(image_root),
                    "--remind-mask-root",
                    str(mask_root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["task_type"] for row in rows}, {"classification", "segmentation"})

    def test_cli_build_yale_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_nifti(
                root / "YG_TEST1234" / "2021-03-10" / "YG_TEST1234_2021-03-10_17-01-09_PRE.nii.gz",
                fill_value=2.0,
            )
            output_csv = root / "out" / "yale.csv"

            standardize_cli.main(
                [
                    "build-yale-brain-mets-longitudinal-manifest",
                    "--yale-root",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["dataset_key"],
                "yale_brain_mets_longitudinal",
            )

    def test_cli_build_vestibular_manifest_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_nifti(root / "VS_MC_RC2_123_1992-08-07_T1.nii.gz", fill_value=1.0)
            _write_nifti(root / "VS_MC_RC2_123_1992-08-07_T1C.nii.gz", fill_value=2.0)
            _write_nifti(
                root / "VS_MC_RC2_123_1992-08-07_T1C_seg.nii.gz",
                fill_value=1.0,
            )
            output_csv = root / "out" / "vestibular.csv"

            standardize_cli.main(
                [
                    "build-vestibular-schwannoma-mc-rc2-manifest",
                    "--vestibular-root",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertTrue(output_csv.is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["task_type"] for row in rows},
                {"classification", "segmentation"},
            )

    def test_materialized_manifest_points_to_new_database_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "IXI" / "P300" / "ses-01" / "t1.nii.gz"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            _write_nifti(input_path, fill_value=5.0)

            rows = [
                _classification_row(
                    dataset_key="brain_structure",
                    input_path=str(input_path),
                    global_case_id="brain_structure__IXI__P300__ses-01",
                    subject_id="brain_structure:IXI:P300",
                    preproc_profile="brain_structure_affine_mni_to_cls128",
                )
            ]

            materialized_rows, manifest_path = materialize_brain_structure_manifest(
                rows,
                root / "db",
            )
            input_path.unlink()

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(materialized_rows), 1)
            materialized_row = materialized_rows[0]
            self.assertNotEqual(materialized_row["t1_path"], str(input_path))
            self.assertEqual(materialized_row["normalization_mask_method"], "nonzero")
            self.assertTrue(Path(materialized_row["t1_path"]).is_file())
            self.assertTrue(Path(materialized_row["brain_mask_path"]).is_file())
            self.assertEqual(materialized_row["mask_path"], "")

            reloaded = np.asanyarray(nib.load(materialized_row["t1_path"]).dataobj)
            self.assertEqual(reloaded.shape, (128, 128, 128))

    def test_materialized_manifest_records_synthstrip_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "ReMIND" / "P400" / "ses-01" / "t1.nii.gz"
            _write_nifti(input_path, fill_value=3.0)

            rows = [
                _classification_row(
                    dataset_key=REMIND_DATASET_KEY,
                    input_path=str(input_path),
                    global_case_id="remind__P400__ses-01",
                    subject_id="remind:P400",
                    preproc_profile="remind_cls128",
                )
            ]

            with mock.patch(
                "src.datasets.standardize.preprocess.shutil.which",
                return_value="/usr/bin/mri_synthstrip",
            ), mock.patch(
                "src.datasets.standardize.preprocess.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["mri_synthstrip"],
                    1,
                    "",
                    "boom",
                ),
            ):
                materialized_rows, manifest_path = materialize_classification_manifest(
                    rows,
                    root / "db",
                )

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(materialized_rows), 1)
            failed_row = materialized_rows[0]
            self.assertEqual(failed_row["exclude_reason"], "synthstrip_failed")
            self.assertEqual(failed_row["t1_path"], "")
            self.assertEqual(failed_row["brain_mask_path"], "")

    def test_materialized_manifest_maps_nan_synthstrip_failures_to_low_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "ReMIND" / "P401" / "ses-01" / "t1.nii.gz"
            _write_nifti(input_path, fill_value=3.0)

            rows = [
                _classification_row(
                    dataset_key=REMIND_DATASET_KEY,
                    input_path=str(input_path),
                    global_case_id="remind__P401__ses-01",
                    subject_id="remind:P401",
                    preproc_profile="remind_cls128",
                )
            ]

            stderr = (
                "RuntimeWarning: divide by zero encountered in divide\n"
                "ValueError: cannot convert float NaN to integer"
            )
            with mock.patch(
                "src.datasets.standardize.preprocess.shutil.which",
                return_value="/usr/bin/mri_synthstrip",
            ), mock.patch(
                "src.datasets.standardize.preprocess.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["mri_synthstrip"],
                    1,
                    "",
                    stderr,
                ),
            ):
                materialized_rows, _ = materialize_classification_manifest(
                    rows,
                    root / "db",
                )

            self.assertEqual(materialized_rows[0]["exclude_reason"], "low_signal_image")


if __name__ == "__main__":
    unittest.main()
