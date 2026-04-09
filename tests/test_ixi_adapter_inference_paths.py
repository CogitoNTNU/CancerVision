from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.adapters.ixi import IxiAdapter


def _write_nifti(path: Path, value: float) -> None:
    data = np.full((4, 4, 4), value, dtype=np.float32)
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii, str(path))


def test_ixi_adapter_supports_directory_with_ixi_t2_file(tmp_path: Path):
    sample_path = tmp_path / "IXI002-Guys-0828-T2.nii.gz"
    _write_nifti(sample_path, 2.0)

    adapter = IxiAdapter()
    assert adapter.supports_path(str(tmp_path))


def test_ixi_adapter_load_inference_image_accepts_file(tmp_path: Path):
    sample_path = tmp_path / "IXI012-HH-1211-T2.nii.gz"
    _write_nifti(sample_path, 1.5)

    adapter = IxiAdapter()
    image = adapter.load_inference_image(str(sample_path))

    assert tuple(image.shape) == (4, 4, 4, 4)
    assert np.allclose(image[0].numpy(), image[1].numpy())
    assert np.allclose(image[2].numpy(), image[3].numpy())


def test_ixi_adapter_load_inference_image_accepts_directory(tmp_path: Path):
    sample_path = tmp_path / "IXI013-HH-1212-T2.nii.gz"
    _write_nifti(sample_path, 3.0)

    adapter = IxiAdapter()
    image = adapter.load_inference_image(str(tmp_path))

    assert tuple(image.shape) == (4, 4, 4, 4)
