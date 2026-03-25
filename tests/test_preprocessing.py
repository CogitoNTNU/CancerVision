import torch

from src.data.preprocess import preprocess_image_volume


def test_preprocess_image_volume_sanitizes_nan_and_inf():
    image = torch.zeros(4, 8, 8, 8)
    image[0, 0, 0, 0] = float("nan")
    image[1, 0, 0, 0] = float("inf")
    image[2, 0, 0, 0] = float("-inf")
    image[3, 2:6, 2:6, 2:6] = 10.0

    out = preprocess_image_volume(image)

    assert torch.isfinite(out).all()


def test_preprocess_image_volume_preserves_shape_and_dtype():
    image = torch.randn(4, 12, 10, 6)
    out = preprocess_image_volume(image)

    assert out.shape == image.shape
    assert out.dtype == torch.float32
