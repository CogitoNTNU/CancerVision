import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.datasets.brats_paths import (
    build_brats_data_dicts,
    detect_brats_layout,
    resolve_brats_data_dir,
)
from src.models.dynnet_data import (
    apply_path_prefix_maps,
    infer_cancervision_path_prefix_maps,
)


class BratsPathTests(unittest.TestCase):
    def test_detect_brats_layout_supports_2020_and_gli_names(self):
        self.assertEqual(detect_brats_layout("BraTS20_Training_001").name, "brats2020")
        self.assertEqual(detect_brats_layout("BraTS-GLI-00000-000").name, "brats_gli")
        self.assertIsNone(detect_brats_layout("something_else"))

    def test_build_brats_data_dicts_supports_brats2020_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            patient_dir = root / "BraTS20_Training_001"
            patient_dir.mkdir()
            for suffix in ("flair", "t1", "t1ce", "t2", "seg"):
                (patient_dir / f"BraTS20_Training_001_{suffix}.nii.gz").write_bytes(
                    b""
                )

            rows = build_brats_data_dicts(root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["image"],
                [
                    str(patient_dir / "BraTS20_Training_001_flair.nii.gz"),
                    str(patient_dir / "BraTS20_Training_001_t1.nii.gz"),
                    str(patient_dir / "BraTS20_Training_001_t1ce.nii.gz"),
                    str(patient_dir / "BraTS20_Training_001_t2.nii.gz"),
                ],
            )
            self.assertEqual(
                rows[0]["label"],
                str(patient_dir / "BraTS20_Training_001_seg.nii.gz"),
            )

    def test_build_brats_data_dicts_supports_nested_brats2023_2024_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wrapper_dir = root / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
            patient_dir = wrapper_dir / "BraTS-GLI-00000-000"
            patient_dir.mkdir(parents=True)
            for suffix in ("t2f", "t1n", "t1c", "t2w", "seg"):
                (patient_dir / f"BraTS-GLI-00000-000-{suffix}.nii").write_bytes(b"")

            resolved = resolve_brats_data_dir(root)
            rows = build_brats_data_dicts(root)

            self.assertEqual(resolved, wrapper_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["image"],
                [
                    str(patient_dir / "BraTS-GLI-00000-000-t2f.nii"),
                    str(patient_dir / "BraTS-GLI-00000-000-t1n.nii"),
                    str(patient_dir / "BraTS-GLI-00000-000-t1c.nii"),
                    str(patient_dir / "BraTS-GLI-00000-000-t2w.nii"),
                ],
            )
            self.assertEqual(
                rows[0]["label"],
                str(patient_dir / "BraTS-GLI-00000-000-seg.nii"),
            )

    def test_resolve_brats_data_dir_finds_training_wrapper_with_siblings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "BraTS2020_ValidationData").mkdir()
            wrapper_dir = root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
            patient_dir = wrapper_dir / "BraTS20_Training_001"
            patient_dir.mkdir(parents=True)
            for suffix in ("flair", "t1", "t1ce", "t2", "seg"):
                (patient_dir / f"BraTS20_Training_001_{suffix}.nii").write_bytes(b"")

            resolved = resolve_brats_data_dir(root)
            rows = build_brats_data_dicts(root)

            self.assertEqual(resolved, wrapper_dir)
            self.assertEqual(len(rows), 1)

    def test_infer_cancervision_path_prefix_maps_handles_cluster_brats2020_layout(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_root = repo_root / "res" / "dataset"
        standardized_root = dataset_root / "cancervision-standardized"
        brats_root = dataset_root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"

        if not standardized_root.is_dir() or not brats_root.is_dir():
            self.skipTest("Cluster-style dataset layout is not available in this workspace")

        with mock.patch(
            "src.models.dynnet_data.DEFAULT_CANCERVISION_DATASET_ROOT",
            standardized_root,
        ):
            mappings = infer_cancervision_path_prefix_maps()

        self.assertIn(rf"Z:\dataset\brats2020={dataset_root}", mappings)

        remapped = apply_path_prefix_maps(
            r"Z:\dataset\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii",
            mappings,
        )
        self.assertEqual(
            os.path.normpath(remapped),
            os.path.normpath(
                str(
                    brats_root
                    / "BraTS20_Training_001"
                    / "BraTS20_Training_001_t1ce.nii"
                )
            ),
        )

    def test_infer_cancervision_path_prefix_maps_handles_standardized_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        standardized_root = repo_root / "res" / "dataset" / "cancervision-standardized"

        if not standardized_root.is_dir():
            self.skipTest("CancerVision standardized dataset root is not available in this workspace")

        with mock.patch(
            "src.models.dynnet_data.DEFAULT_CANCERVISION_DATASET_ROOT",
            standardized_root,
        ):
            mappings = infer_cancervision_path_prefix_maps()

        self.assertIn(
            rf"Z:\dataset\cancervision-standardized={standardized_root}",
            mappings,
        )

        remapped = apply_path_prefix_maps(
            r"Z:\dataset\cancervision-standardized\task_manifests\segmentation_binary_curated.csv",
            mappings,
        )
        self.assertEqual(
            os.path.normpath(remapped),
            os.path.normpath(
                str(
                    standardized_root
                    / "task_manifests"
                    / "segmentation_binary_curated.csv"
                )
            ),
        )
