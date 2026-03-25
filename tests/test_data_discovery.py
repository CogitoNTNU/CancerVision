from src.data.discovery import DatasetMatch
from src.data.registry import get_dataset_adapter, list_dataset_types


def test_dataset_match_dataclass_fields():
    match = DatasetMatch(dataset_type="brats", path="/tmp/example")
    assert match.dataset_type == "brats"
    assert match.path == "/tmp/example"


def test_brats_adapter_is_registered():
    assert "brats" in list_dataset_types()
    adapter = get_dataset_adapter("brats")
    assert adapter.name == "brats"
