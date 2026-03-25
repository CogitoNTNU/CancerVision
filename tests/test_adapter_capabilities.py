from src.data.registry import get_dataset_adapter


def test_brats_adapter_channel_contract():
    adapter = get_dataset_adapter("brats")
    assert adapter.get_input_channels() == 4
    assert adapter.get_output_channels() == 3


def test_brats_adapter_has_label_transform():
    adapter = get_dataset_adapter("brats")
    transform = adapter.get_segmentation_label_transform()
    assert transform is not None
