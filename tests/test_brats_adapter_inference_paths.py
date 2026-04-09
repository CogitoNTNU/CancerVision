from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.adapters.brats import BratsAdapter


def _write_nifti(path: Path, value: float) -> None:
    data = np.full((4, 4, 4), value, dtype=np.float32)
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii, str(path))


def test_load_inference_image_accepts_patient_directory(tmp_path: Path):
    patient_name = "BraTS20_Validation_009"
    patient_dir = tmp_path / patient_name
    patient_dir.mkdir()

    for idx, suffix in enumerate(("flair", "t1", "t1ce", "t2"), start=1):
        _write_nifti(patient_dir / f"{patient_name}_{suffix}.nii", float(idx))

    adapter = BratsAdapter()
    image = adapter.load_inference_image(str(patient_dir))

    assert tuple(image.shape) == (4, 4, 4, 4)


def test_load_inference_image_accepts_modality_file_path(tmp_path: Path):
    patient_name = "BraTS20_Validation_010"
    patient_dir = tmp_path / patient_name
    patient_dir.mkdir()

    for idx, suffix in enumerate(("flair", "t1", "t1ce", "t2"), start=1):
        _write_nifti(patient_dir / f"{patient_name}_{suffix}.nii", float(idx))

    adapter = BratsAdapter()
    image = adapter.load_inference_image(str(patient_dir / f"{patient_name}_flair.nii"))

    assert tuple(image.shape) == (4, 4, 4, 4)
