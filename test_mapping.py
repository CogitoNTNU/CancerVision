import tempfile
from pathlib import Path
from src.models import dynnet_data

tmp = tempfile.TemporaryDirectory()
root = Path(tmp.name) / "cancervision-standardized"
case_id = "brats2023__BraTS-GLI-00008-001__baseline__seg"

seg_dir = root / "segmentation_native" / case_id / "seg"
seg_dir.mkdir(parents=True, exist_ok=True)
(seg_dir / "image.nii.gz").write_text("x")
(seg_dir / "mask.nii.gz").write_text("x")

dynnet_data.DEFAULT_CANCERVISION_DATASET_ROOT = root

raw = r"Z:\dataset\brats2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00008-001\BraTS-GLI-00008-001-t1c.nii"

img = dynnet_data._resolve_manifest_data_path(
    raw,
    manifest_dir=Path(tmp.name),
    field_name="image_path",
    case_id=case_id,
)
print("SUCCESS: Image mapped to", img)
