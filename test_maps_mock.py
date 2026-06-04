import os
import re
from pathlib import Path
from typing import Sequence

WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
DEFAULT_CANCERVISION_WINDOWS_ROOT = r"Z:\dataset\cancervision-standardized"

def parse_path_prefix_map(mapping_text: str) -> tuple[str, str]:
    if "=" not in mapping_text:
        raise ValueError(f"Invalid map: {mapping_text}")
    source_prefix, target_prefix = mapping_text.split("=", 1)
    return source_prefix.strip(), target_prefix.strip()

def apply_path_prefix_maps(raw_path: str, path_prefix_maps: Sequence[str] | None = None) -> str:
    if not raw_path: return raw_path
    for mapping_text in path_prefix_maps or []:
        source_prefix, target_prefix = parse_path_prefix_map(mapping_text)
        if raw_path.startswith(source_prefix):
            suffix = raw_path[len(source_prefix) :].lstrip("\\/")
            suffix_parts = [part for part in re.split(r"[\\/]+", suffix) if part]
            return os.path.normpath(str(Path(target_prefix).joinpath(*suffix_parts)))
    return raw_path

def infer_cancervision_path_prefix_maps() -> list[str]:
    dataset_root = Path(r"C:\Users\eld\Documents\GitHub\CancerVision\res\dataset\cancervision-standardized")
    mappings: list[str] = []
    if dataset_root.is_dir():
        mappings.append(f"{DEFAULT_CANCERVISION_WINDOWS_ROOT}={dataset_root}")
    dataset_parent = dataset_root.parent
    if dataset_parent.is_dir():
        windows_base = r"Z:\dataset"
        mappings.append(f"{windows_base}={dataset_parent}")
    return mappings

raw = r'Z:\dataset\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii'
maps = infer_cancervision_path_prefix_maps()
print("Inferred mappings:", maps)
print("Original path:", raw)
print("Mapped path:", apply_path_prefix_maps(raw, maps))
