"""BraTS dataset path discovery for 2020, 2023, and 2024 layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BraTSLayout:
    name: str
    patient_prefix: str
    separator: str
    image_suffixes: tuple[str, str, str, str]
    label_suffix: str


BRATS_LAYOUTS: tuple[BraTSLayout, ...] = (
    BraTSLayout(
        name="brats2020",
        patient_prefix="BraTS20_Training_",
        separator="_",
        image_suffixes=("flair", "t1", "t1ce", "t2"),
        label_suffix="seg",
    ),
    BraTSLayout(
        name="brats_gli",
        patient_prefix="BraTS-GLI-",
        separator="-",
        image_suffixes=("t2f", "t1n", "t1c", "t2w"),
        label_suffix="seg",
    ),
)


def find_nifti(directory: str | Path, pattern: str) -> str:
    """Find first matching NIfTI path for pattern stem inside directory."""

    source = Path(directory)
    for ext in (".nii", ".nii.gz"):
        candidate = source / f"{pattern}{ext}"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {source}"
    )


def detect_brats_layout(patient_name: str) -> BraTSLayout | None:
    """Return matching BraTS layout from patient directory name."""

    for layout in BRATS_LAYOUTS:
        if patient_name.startswith(layout.patient_prefix):
            return layout
    return None


def _has_patient_dirs(root: Path) -> bool:
    return any(
        path.is_dir() and detect_brats_layout(path.name) is not None
        for path in root.iterdir()
    )


def _find_nested_patient_dir(root: Path, *, max_depth: int = 4) -> Path | None:
    current_level = [root]
    for _ in range(max_depth + 1):
        next_level: list[Path] = []
        for candidate in current_level:
            if _has_patient_dirs(candidate):
                return candidate
            next_level.extend(
                path for path in sorted(candidate.iterdir()) if path.is_dir()
            )
        current_level = next_level
    return None


def resolve_brats_data_dir(data_dir: str | Path) -> Path:
    """Resolve dataset root, descending wrapper folders when needed."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    resolved = _find_nested_patient_dir(root)
    return root if resolved is None else resolved


def build_brats_data_dicts(data_dir: str | Path) -> list[dict[str, list[str] | str]]:
    """Scan BraTS 2020/2023/2024 patient folders into MONAI data dicts."""

    resolved_root = resolve_brats_data_dir(data_dir)
    data_dicts: list[dict[str, list[str] | str]] = []

    patient_dirs = [
        path
        for path in sorted(resolved_root.iterdir())
        if path.is_dir() and detect_brats_layout(path.name) is not None
    ]

    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        layout = detect_brats_layout(patient_name)
        if layout is None:
            continue

        try:
            image_paths = [
                find_nifti(patient_dir, f"{patient_name}{layout.separator}{suffix}")
                for suffix in layout.image_suffixes
            ]
            label_path = find_nifti(
                patient_dir,
                f"{patient_name}{layout.separator}{layout.label_suffix}",
            )
        except FileNotFoundError as exc:
            print(f"WARNING: skipping {patient_name} -- {exc}", flush=True)
            continue

        data_dicts.append({"image": image_paths, "label": label_path})

    if not data_dicts:
        raise FileNotFoundError(
            f"No valid BraTS patient folders found in {resolved_root}"
        )

    return data_dicts


def default_brats_data_dir(repo_root: Path) -> Path:
    """Choose first existing common BraTS root, else keep legacy 2020 default."""

    candidates = (
        repo_root
        / "res"
        / "dataset"
        / "BraTS2020_TrainingData"
        / "MICCAI_BraTS2020_TrainingData",
        Path(r"Z:\dataset\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"),
        Path(r"Z:\dataset\brats2020"),
        Path(r"Z:\dataset\brats2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
        Path(r"Z:\dataset\brats2024\BraTS2024_small_dataset"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]
