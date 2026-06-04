from __future__ import annotations

from importlib import import_module

__all__ = [
    "BraTSH5Dataset",
    "BinarizeLabeld",
    "BrainStructureAdapter",
    "ConvertToMultiChannelBasedOnBratsClassesd",
    "EnsureFloatLabeld",
    "StandardizedRecord",
    "build_brats_data_dicts",
    "default_brats_data_dir",
    "resolve_brats_data_dir",
]


def __getattr__(name: str):
    if name in {
        "build_brats_data_dicts",
        "default_brats_data_dir",
        "resolve_brats_data_dir",
    }:
        module = import_module(".brats_paths", __name__)
        return getattr(module, name)

    if name == "BraTSH5Dataset":
        module = import_module(".brats_h5", __name__)
        return getattr(module, name)

    if name in {
        "BinarizeLabeld",
        "ConvertToMultiChannelBasedOnBratsClassesd",
        "EnsureFloatLabeld",
    }:
        module = import_module(".brats_transforms", __name__)
        return getattr(module, name)

    if name in {"BrainStructureAdapter", "StandardizedRecord"}:
        module = import_module(".standardize", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
