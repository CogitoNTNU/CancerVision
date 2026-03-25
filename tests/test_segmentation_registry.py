from src.segmentation.registry import get_segmentation_backend, list_segmentation_backends


def test_monai_backend_is_registered():
    assert "monai_unet" in list_segmentation_backends()
    backend = get_segmentation_backend("monai_unet")
    assert backend.name == "monai_unet"


def test_nnunet_backend_placeholder_registered():
    assert "nnunet" in list_segmentation_backends()
    backend = get_segmentation_backend("nnunet")
    assert backend.name == "nnunet"
