import os
from pathlib import Path
from src.models.dynnet_config import DEFAULT_CANCERVISION_DATASET_ROOT
from src.models.dynnet_data import apply_path_prefix_maps, infer_cancervision_path_prefix_maps

raw = r'Z:\dataset\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_t1ce.nii'
maps = infer_cancervision_path_prefix_maps()
print("Inferred mappings:")
for m in maps:
    print(f"  {m}")

mapped = apply_path_prefix_maps(raw, maps)
print("\nOriginal raw path:", raw)
print("Mapped path:", mapped)

# Test existing resolution
from src.models.dynnet_data import _resolve_manifest_data_path

try:
    resolved = _resolve_manifest_data_path(
        raw, 
        manifest_dir=Path('.'), 
        field_name='image_path', 
        case_id='test-1', 
        path_prefix_maps=maps
    )
    print("\nResolved existing path:", resolved)
except Exception as e:
    print("\nResolution failed (expected if the file doesn't exist locally):")
    print(e)
