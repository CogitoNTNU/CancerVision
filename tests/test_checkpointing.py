from pathlib import Path

import torch

from src.core.checkpointing import load_checkpoint


def test_load_checkpoint_strips_orig_mod_and_module_prefixes(tmp_path: Path):
    checkpoint_path = tmp_path / "compiled_checkpoint.pth"

    payload = {
        "model_state_dict": {
            "_orig_mod.stem.weight": torch.randn(2, 2, 3, 3, 3),
            "module.head.bias": torch.randn(2),
        },
        "metadata": {"model_backend": "torch_unet"},
    }
    torch.save(payload, checkpoint_path)

    checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")

    assert "stem.weight" in checkpoint["model_state_dict"]
    assert "head.bias" in checkpoint["model_state_dict"]
    assert "_orig_mod.stem.weight" not in checkpoint["model_state_dict"]
    assert "module.head.bias" not in checkpoint["model_state_dict"]


def test_load_checkpoint_supports_state_dict_key(tmp_path: Path):
    checkpoint_path = tmp_path / "state_dict_checkpoint.pth"

    payload = {
        "state_dict": {
            "_orig_mod.layer.weight": torch.randn(4, 4),
        },
        "metadata": {"foo": "bar"},
    }
    torch.save(payload, checkpoint_path)

    checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")

    assert "layer.weight" in checkpoint["model_state_dict"]
    assert checkpoint["metadata"] == {"foo": "bar"}
