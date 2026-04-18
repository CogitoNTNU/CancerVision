"""Preprocessing utilities for standardized classification and segmentation views."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Literal
import uuid

import nibabel as nib
import numpy as np
import pydicom
import nrrd
from scipy.ndimage import zoom

from .constants import (
    BRAIN_STRUCTURE_CLASSIFICATION_OUTPUT,
    BRAIN_STRUCTURE_DATASET_KEY,
    BRAIN_STRUCTURE_MASK_OUTPUT,
    BRAIN_STRUCTURE_PREPROC_PROFILE,
)
from .io import read_csv_rows, write_csv_rows
from .models import standardized_manifest_fieldnames
from .registry import (
    ClsSkullstripPolicy,
    get_dataset_registry_entry,
    normalize_dataset_key,
)

NormalizationMaskMethod = Literal["nonzero", "synthstrip"]
SYNTHSTRIP_DOCKER_HELPER = "synthstrip_docker.py"
SYNTHSTRIP_SESSION_ENV = "CANCERVISION_SYNTHSTRIP_SESSION_NAME"
SYNTHSTRIP_WORKSPACE_ENV = "CANCERVISION_SYNTHSTRIP_WORKSPACE_ROOT"


class ClassificationPreprocessError(RuntimeError):
    """Processing error that maps to manifest exclusion reason."""

    def __init__(self, exclude_reason: str, message: str) -> None:
        super().__init__(message)
        self.exclude_reason = exclude_reason


@dataclass(frozen=True, slots=True)
class ClassificationViewResult:
    """Materialized classification view outputs."""

    image_path: Path
    brain_mask_path: Path | None
    image_shape: tuple[int, int, int]
    normalization_mask_method: NormalizationMaskMethod
    preproc_profile: str

    @property
    def mask_path(self) -> Path | None:
        """Backward-compatible alias for brain mask path."""

        return self.brain_mask_path


@dataclass(frozen=True, slots=True)
class SegmentationPairResult:
    """Materialized native-scale segmentation outputs."""

    image_path: Path
    mask_path: Path
    brain_mask_path: Path | None
    normalization_mask_method: str
    preproc_profile: str


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _ensure_nonempty_mask(
    mask: np.ndarray,
    *,
    exclude_reason: str,
) -> np.ndarray:
    if not np.any(mask):
        raise ClassificationPreprocessError(
            exclude_reason,
            f"Mask empty after preprocessing ({exclude_reason}).",
        )
    return mask


def _nonzero_mask(volume: np.ndarray) -> np.ndarray:
    return _ensure_nonempty_mask(
        volume != 0,
        exclude_reason="empty_brain_after_load",
    )


def _crop_slices(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask)
    mins = np.maximum(coordinates.min(axis=0) - margin, 0)
    maxs = np.minimum(coordinates.max(axis=0) + margin + 1, np.array(mask.shape))
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(mins, maxs, strict=True)
    )


def _normalize_in_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_values = volume[mask]
    lower, upper = np.percentile(masked_values, [0.5, 99.5])
    clipped = volume.astype(np.float32, copy=True)
    clipped[mask] = np.clip(masked_values, lower, upper)

    normalized = np.zeros_like(clipped, dtype=np.float32)
    centered = clipped[mask] - float(clipped[mask].mean())
    std = float(clipped[mask].std())
    if std < 1e-6:
        std = 1.0
    normalized[mask] = centered / std
    return normalized


def _resize_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    order: int,
) -> np.ndarray:
    zoom_factors = tuple(
        target / current
        for target, current in zip(target_shape, volume.shape, strict=True)
    )
    return zoom(volume, zoom=zoom_factors, order=order)


def _dicom_sort_key(dataset: pydicom.Dataset) -> tuple[float, float, float, float]:
    image_position = getattr(dataset, "ImagePositionPatient", None)
    if image_position is not None and len(image_position) >= 3:
        return (
            float(image_position[0]),
            float(image_position[1]),
            float(image_position[2]),
            float(getattr(dataset, "InstanceNumber", 0)),
        )
    return (0.0, 0.0, float(getattr(dataset, "InstanceNumber", 0)), 0.0)


def _load_dicom_series(input_path: str | Path) -> np.ndarray:
    datasets = _load_dicom_datasets(input_path)
    return _stack_dicom_datasets(datasets)


def _load_dicom_datasets(input_path: str | Path) -> list[pydicom.Dataset]:
    source = Path(input_path)
    series_dir = source if source.is_dir() else source.parent
    dicom_files = sorted(path for path in series_dir.iterdir() if path.is_file() and path.suffix.lower() == ".dcm")
    if not dicom_files:
        raise ClassificationPreprocessError(
            "missing_dicom_series",
            f"No DICOM files found in {series_dir}",
        )

    datasets = [pydicom.dcmread(str(path)) for path in dicom_files]
    datasets.sort(key=_dicom_sort_key)
    return datasets


def _stack_dicom_datasets(datasets: list[pydicom.Dataset]) -> np.ndarray:
    try:
        slices = [dataset.pixel_array.astype(np.float32) for dataset in datasets]
    except Exception as exc:  # pragma: no cover - exact decoder depends on DICOM transfer syntax
        raise ClassificationPreprocessError(
            "dicom_decode_failed",
            f"Could not decode DICOM series {series_dir}: {exc}",
        ) from exc

    volume = np.stack(slices, axis=-1)
    if volume.ndim != 3:
        raise ClassificationPreprocessError(
            "invalid_dicom_volume",
            f"DICOM volume must be 3D after stacking, got shape {volume.shape}",
        )
    return volume


def _build_dicom_affine(datasets: list[pydicom.Dataset]) -> np.ndarray:
    first = datasets[0]
    orientation = getattr(first, "ImageOrientationPatient", None)
    image_position = getattr(first, "ImagePositionPatient", None)
    pixel_spacing = getattr(first, "PixelSpacing", None)
    if (
        orientation is None
        or image_position is None
        or pixel_spacing is None
        or len(orientation) < 6
        or len(image_position) < 3
        or len(pixel_spacing) < 2
    ):
        return np.diag([1.0, 1.0, 1.0, 1.0])

    row_direction = np.asarray(orientation[:3], dtype=np.float64)
    column_direction = np.asarray(orientation[3:6], dtype=np.float64)
    slice_direction = np.cross(row_direction, column_direction)

    row_spacing = float(pixel_spacing[0])
    column_spacing = float(pixel_spacing[1])
    origin = np.asarray(image_position[:3], dtype=np.float64)

    if len(datasets) > 1:
        second_position = getattr(datasets[1], "ImagePositionPatient", None)
        if second_position is not None and len(second_position) >= 3:
            slice_vector = np.asarray(second_position[:3], dtype=np.float64) - origin
        else:
            slice_thickness = float(
                getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0))
            )
            slice_vector = slice_direction * slice_thickness
    else:
        slice_thickness = float(
            getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0))
        )
        slice_vector = slice_direction * slice_thickness

    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = row_direction * row_spacing
    affine[:3, 1] = column_direction * column_spacing
    affine[:3, 2] = slice_vector
    affine[:3, 3] = origin
    return affine


def _build_nrrd_affine(header: dict[str, object], ndim: int) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)

    space_directions = header.get("space directions")
    if isinstance(space_directions, np.ndarray):
        directions = space_directions.tolist()
    else:
        directions = list(space_directions) if isinstance(space_directions, (list, tuple)) else []
    if len(directions) >= 3:
        valid_directions = True
        for axis, direction in enumerate(directions[:3]):
            if direction is None:
                valid_directions = False
                break
            direction_array = np.asarray(direction, dtype=np.float64)
            if direction_array.shape != (3,):
                valid_directions = False
                break
            affine[:3, axis] = direction_array
        if valid_directions:
            origin = header.get("space origin")
            if origin is not None:
                origin_array = np.asarray(origin, dtype=np.float64)
                if origin_array.shape == (3,):
                    affine[:3, 3] = origin_array
            return affine

    spacings = header.get("spacings")
    if isinstance(spacings, np.ndarray):
        spacings_list = spacings.tolist()
    else:
        spacings_list = list(spacings) if isinstance(spacings, (list, tuple)) else []
    if len(spacings_list) >= min(ndim, 3):
        for axis, spacing in enumerate(spacings_list[:3]):
            try:
                affine[axis, axis] = float(spacing)
            except (TypeError, ValueError):
                affine[axis, axis] = 1.0
    return affine


def _build_nifti_from_source(
    input_path: str | Path,
    *,
    exclude_reason: str,
) -> nib.Nifti1Image:
    source = Path(input_path)
    if source.is_dir() or source.suffix.lower() == ".dcm":
        datasets = _load_dicom_datasets(source)
        volume = _stack_dicom_datasets(datasets)
        affine = _build_dicom_affine(datasets)
        return nib.Nifti1Image(volume.astype(np.float32), affine)

    if source.name.lower().endswith((".nrrd", ".nhdr")):
        try:
            data, header = nrrd.read(str(source))
        except Exception as exc:
            raise ClassificationPreprocessError(
                exclude_reason,
                f"Could not read NRRD file {source}: {exc}",
            ) from exc
        array = np.asarray(data)
        if array.ndim < 3:
            raise ClassificationPreprocessError(
                exclude_reason,
                f"NRRD volume must be at least 3D, got shape {array.shape}",
            )
        affine = _build_nrrd_affine(header, array.ndim)
        return nib.Nifti1Image(array, affine)

    try:
        image = nib.load(str(source))
    except Exception as exc:
        raise ClassificationPreprocessError(
            exclude_reason,
            f"Could not read image file {source}: {exc}",
        ) from exc
    data = np.asanyarray(image.dataobj)
    return nib.Nifti1Image(data, image.affine, image.header)


def _load_input_volume(input_path: str | Path) -> np.ndarray:
    source = Path(input_path)
    if source.is_dir() or source.suffix.lower() == ".dcm":
        return _load_dicom_series(source)
    canonical_image = nib.as_closest_canonical(nib.load(str(source)))
    return np.asanyarray(canonical_image.dataobj).astype(np.float32)


def _split_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=os.name != "nt")
    if not parts:
        raise RuntimeError("SynthStrip command is empty.")
    normalized_parts: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
            normalized_parts.append(part[1:-1])
        else:
            normalized_parts.append(part)
    return normalized_parts


def _command_exists(command: str) -> bool:
    executable = _split_command(command)[0]
    executable_path = Path(executable)
    if executable_path.is_file():
        return True
    return shutil.which(executable) is not None


def _uses_managed_synthstrip_session(command: str) -> bool:
    return any(
        Path(part).name.lower() == SYNTHSTRIP_DOCKER_HELPER
        for part in _split_command(command)
    )


def _synthstrip_workspace_root() -> Path:
    configured_root = os.environ.get(SYNTHSTRIP_WORKSPACE_ENV)
    if configured_root:
        root = Path(configured_root)
    else:
        root = Path(tempfile.gettempdir()) / "cancervision-synthstrip"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_synthstrip_session_command(
    synthstrip_cmd: str,
    session_action: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            *_split_command(synthstrip_cmd),
            session_action,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _classify_synthstrip_failure(detail: str) -> str:
    normalized = detail.lower()
    if (
        "cannot convert float nan to integer" in normalized
        or "divide by zero encountered in divide" in normalized
        or "invalid value encountered in divide" in normalized
    ):
        return "low_signal_image"
    return "synthstrip_failed"


@contextmanager
def _managed_synthstrip_session(
    rows: list[dict[str, str]],
    *,
    synthstrip_cmd: str,
):
    if not rows_requiring_synthstrip(rows) or not _uses_managed_synthstrip_session(
        synthstrip_cmd
    ):
        yield
        return

    previous_session_name = os.environ.get(SYNTHSTRIP_SESSION_ENV)
    previous_workspace_root = os.environ.get(SYNTHSTRIP_WORKSPACE_ENV)
    session_name = f"cancervision-synthstrip-{uuid.uuid4().hex[:8]}"
    workspace_root = str(_synthstrip_workspace_root())
    os.environ[SYNTHSTRIP_SESSION_ENV] = session_name
    os.environ[SYNTHSTRIP_WORKSPACE_ENV] = workspace_root
    print(
        f"Starting persistent SynthStrip container: {session_name}",
        flush=True,
    )

    try:
        start_result = _run_synthstrip_session_command(
            synthstrip_cmd,
            "--start-session",
        )
        if start_result.returncode != 0:
            detail = start_result.stderr.strip() or start_result.stdout.strip() or (
                "Persistent SynthStrip container failed to start."
            )
            raise RuntimeError(detail)
        yield
    finally:
        print(
            f"Stopping persistent SynthStrip container: {session_name}",
            flush=True,
        )
        stop_result = _run_synthstrip_session_command(
            synthstrip_cmd,
            "--stop-session",
        )
        if stop_result.returncode != 0:
            detail = stop_result.stderr.strip() or stop_result.stdout.strip() or (
                "Persistent SynthStrip container failed to stop cleanly."
            )
            print(detail, flush=True)

        if previous_session_name is None:
            os.environ.pop(SYNTHSTRIP_SESSION_ENV, None)
        else:
            os.environ[SYNTHSTRIP_SESSION_ENV] = previous_session_name

        if previous_workspace_root is None:
            os.environ.pop(SYNTHSTRIP_WORKSPACE_ENV, None)
        else:
            os.environ[SYNTHSTRIP_WORKSPACE_ENV] = previous_workspace_root


def _run_synthstrip(
    canonical_image: nib.Nifti1Image,
    *,
    synthstrip_cmd: str,
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(
        dir=_synthstrip_workspace_root(),
        prefix="case-",
    ) as tmpdir:
        tmp_root = Path(tmpdir)
        input_path = tmp_root / "input.nii.gz"
        stripped_path = tmp_root / "stripped.nii.gz"
        mask_path = tmp_root / "mask.nii.gz"
        nib.save(canonical_image, str(input_path))

        try:
            result = subprocess.run(
                [
                    *_split_command(synthstrip_cmd),
                    "-i",
                    str(input_path),
                    "-o",
                    str(stripped_path),
                    "-m",
                    str(mask_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"SynthStrip command not found: {synthstrip_cmd}"
            ) from exc

        if result.returncode != 0 or not stripped_path.is_file() or not mask_path.is_file():
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "SynthStrip did not create expected outputs."
            raise ClassificationPreprocessError(
                _classify_synthstrip_failure(detail),
                f"SynthStrip failed: {detail}",
            )

        stripped_image = nib.load(str(stripped_path))
        mask_image = nib.load(str(mask_path))
        stripped_volume = np.asanyarray(stripped_image.dataobj).astype(np.float32)
        brain_mask = np.asanyarray(mask_image.dataobj) > 0
        _ensure_nonempty_mask(
            brain_mask,
            exclude_reason="empty_brain_after_synthstrip",
        )
        return stripped_volume, brain_mask.astype(np.uint8)


def _finalize_classification_view(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    crop_margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalize_in_mask(volume, mask)
    crop = _crop_slices(mask, crop_margin)

    cropped_volume = normalized[crop]
    cropped_mask = mask[crop].astype(np.float32)
    resized_volume = _resize_volume(cropped_volume, target_shape, order=1).astype(
        np.float32
    )
    resized_mask = _resize_volume(cropped_mask, target_shape, order=0) > 0.5
    return resized_volume, resized_mask.astype(np.uint8)


def _resolved_preproc_profile(
    dataset_key: str,
    original_preproc_profile: str,
    *,
    skullstrip_policy: ClsSkullstripPolicy,
) -> str:
    base_profile = _text(original_preproc_profile) or (
        f"{normalize_dataset_key(dataset_key)}_cls128"
    )
    if skullstrip_policy == "synthstrip" and not base_profile.endswith("_synthstrip"):
        return f"{base_profile}_synthstrip"
    return base_profile


def build_classification_view(
    input_path: str | Path,
    *,
    dataset_key: str,
    synthstrip_cmd: str = "mri_synthstrip",
    target_shape: tuple[int, int, int] = (128, 128, 128),
    crop_margin: int = 4,
) -> tuple[np.ndarray, np.ndarray, NormalizationMaskMethod]:
    """Build fixed-size classification tensor + brain mask from source NIfTI."""

    registry_entry = get_dataset_registry_entry(dataset_key)
    source = Path(input_path)

    if registry_entry.cls_skullstrip_policy == "synthstrip":
        if source.is_dir() or source.suffix.lower() == ".dcm":
            source_volume = _load_input_volume(source)
            canonical_image = nib.Nifti1Image(source_volume, np.eye(4))
            stripped_volume, brain_mask = _run_synthstrip(
                canonical_image,
                synthstrip_cmd=synthstrip_cmd,
            )
        else:
            canonical_image = nib.as_closest_canonical(nib.load(str(source)))
            stripped_volume, brain_mask = _run_synthstrip(
                canonical_image,
                synthstrip_cmd=synthstrip_cmd,
            )
        image_array, mask_array = _finalize_classification_view(
            stripped_volume,
            brain_mask.astype(bool),
            target_shape=target_shape,
            crop_margin=crop_margin,
        )
        return image_array, mask_array, "synthstrip"

    volume = _load_input_volume(source)
    brain_mask = _nonzero_mask(volume)
    image_array, mask_array = _finalize_classification_view(
        volume,
        brain_mask.astype(bool),
        target_shape=target_shape,
        crop_margin=crop_margin,
    )
    return image_array, mask_array, "nonzero"


def write_classification_view(
    input_path: str | Path,
    case_output_root: str | Path,
    *,
    dataset_key: str,
    original_preproc_profile: str = "",
    synthstrip_cmd: str = "mri_synthstrip",
    target_shape: tuple[int, int, int] = (128, 128, 128),
    crop_margin: int = 4,
    save_mask: bool = True,
) -> ClassificationViewResult:
    """Write materialized classification view plus optional brain mask."""

    registry_entry = get_dataset_registry_entry(dataset_key)
    image_array, mask_array, normalization_mask_method = build_classification_view(
        input_path,
        dataset_key=dataset_key,
        synthstrip_cmd=synthstrip_cmd,
        target_shape=target_shape,
        crop_margin=crop_margin,
    )

    output_root = Path(case_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / BRAIN_STRUCTURE_CLASSIFICATION_OUTPUT
    image_path.parent.mkdir(parents=True, exist_ok=True)
    brain_mask_path = output_root / BRAIN_STRUCTURE_MASK_OUTPUT

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(image_array, affine), str(image_path))
    if save_mask:
        nib.save(nib.Nifti1Image(mask_array, affine), str(brain_mask_path))
    else:
        brain_mask_path = None

    return ClassificationViewResult(
        image_path=image_path,
        brain_mask_path=brain_mask_path,
        image_shape=tuple(int(axis) for axis in image_array.shape),
        normalization_mask_method=normalization_mask_method,
        preproc_profile=_resolved_preproc_profile(
            dataset_key,
            original_preproc_profile,
            skullstrip_policy=registry_entry.cls_skullstrip_policy,
        ),
    )


def build_brain_structure_cls_view(
    input_path: str | Path,
    *,
    target_shape: tuple[int, int, int] = (128, 128, 128),
    crop_margin: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible brain-structure wrapper for cls view build."""

    image_array, mask_array, _ = build_classification_view(
        input_path,
        dataset_key=BRAIN_STRUCTURE_DATASET_KEY,
        target_shape=target_shape,
        crop_margin=crop_margin,
    )
    return image_array, mask_array


def write_brain_structure_cls_view(
    input_path: str | Path,
    case_output_root: str | Path,
    *,
    target_shape: tuple[int, int, int] = (128, 128, 128),
    crop_margin: int = 4,
    save_mask: bool = True,
) -> ClassificationViewResult:
    """Backward-compatible brain-structure wrapper for cls view write."""

    return write_classification_view(
        input_path,
        case_output_root,
        dataset_key=BRAIN_STRUCTURE_DATASET_KEY,
        original_preproc_profile=BRAIN_STRUCTURE_PREPROC_PROFILE,
        target_shape=target_shape,
        crop_margin=crop_margin,
        save_mask=save_mask,
    )


def rows_requiring_synthstrip(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return manifest rows that require SynthStrip."""

    required_rows: list[dict[str, str]] = []
    for row in rows:
        task_type = _text(row.get("task_type") or "classification")
        if task_type not in {"classification", "segmentation"}:
            continue
        if _text(row.get("exclude_reason")):
            continue
        entry = get_dataset_registry_entry(_text(row.get("dataset_key")))
        if entry.cls_skullstrip_policy == "synthstrip":
            required_rows.append(row)
    return required_rows


def preflight_synthstrip_requirements(
    rows: list[dict[str, str]],
    *,
    synthstrip_cmd: str = "mri_synthstrip",
) -> None:
    """Fail fast when selected rows need SynthStrip but command is unavailable."""

    required_rows = rows_requiring_synthstrip(rows)
    if required_rows and not _command_exists(synthstrip_cmd):
        raise RuntimeError(
            f"SynthStrip command not found: {synthstrip_cmd}. "
            f"Required for {len(required_rows)} classification rows."
        )


def _extra_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    base_fields = standardized_manifest_fieldnames()
    seen = set(base_fields)
    extras: list[str] = []
    for row in rows:
        for field in row.keys():
            if field in seen:
                continue
            seen.add(field)
            extras.append(field)
    return [*base_fields, *extras]


def _blank_materialized_paths(
    row: dict[str, str],
    *,
    clear_mask: bool = False,
) -> dict[str, str]:
    materialized_row = dict(row)
    materialized_row["image_path"] = ""
    materialized_row["t1_path"] = ""
    if clear_mask:
        materialized_row["mask_path"] = ""
    materialized_row["brain_mask_path"] = ""
    materialized_row["normalization_mask_method"] = ""
    return materialized_row


def _build_resumed_failed_row(
    row: dict[str, str],
    *,
    existing_manifest_row: dict[str, str],
    clear_mask: bool = False,
) -> dict[str, str] | None:
    existing_exclude_reason = _text(existing_manifest_row.get("exclude_reason"))
    if not existing_exclude_reason:
        return None
    resumed_row = _blank_materialized_paths(
        dict(row),
        clear_mask=clear_mask,
    )
    resumed_row["exclude_reason"] = existing_exclude_reason
    existing_preproc_profile = _text(existing_manifest_row.get("preproc_profile"))
    if existing_preproc_profile:
        resumed_row["preproc_profile"] = existing_preproc_profile
    return resumed_row


def _materialized_paths_for_case(
    case_output_root: str | Path,
) -> tuple[Path, Path]:
    root = Path(case_output_root)
    return (
        root / BRAIN_STRUCTURE_CLASSIFICATION_OUTPUT,
        root / BRAIN_STRUCTURE_MASK_OUTPUT,
    )


def _segmentation_materialized_paths_for_case(
    image_source: str | Path,
    mask_source: str | Path,
    case_output_root: str | Path,
) -> tuple[Path, Path]:
    root = Path(case_output_root) / "seg"
    return root / "image.nii.gz", root / "mask.nii.gz"


def _segmentation_brain_mask_path_for_case(case_output_root: str | Path) -> Path:
    return Path(case_output_root) / "seg" / "brain_mask.nii.gz"


def _is_valid_materialized_nifti(
    image_path: Path,
    *,
    expected_shape: tuple[int, int, int] = (128, 128, 128),
) -> bool:
    if not image_path.is_file():
        return False
    try:
        image = nib.load(str(image_path))
    except Exception:
        return False
    return tuple(int(axis) for axis in image.shape) == expected_shape


def _is_valid_materialized_segmentation_nifti(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        image = nib.load(str(path))
    except Exception:
        return False
    return len(image.shape) >= 3 and all(int(axis) > 0 for axis in image.shape[:3])


def _load_existing_materialized_rows(
    manifest_path: str | Path,
) -> dict[str, dict[str, str]]:
    source = Path(manifest_path)
    if not source.is_file():
        return {}
    rows_by_case_id: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(source):
        global_case_id = _text(row.get("global_case_id"))
        if global_case_id:
            rows_by_case_id[global_case_id] = row
    return rows_by_case_id


def _build_resumed_materialized_row(
    row: dict[str, str],
    *,
    dataset_key: str,
    case_output_root: Path,
    save_mask: bool,
    existing_manifest_row: dict[str, str] | None,
) -> dict[str, str] | None:
    image_path, default_mask_path = _materialized_paths_for_case(case_output_root)
    if not _is_valid_materialized_nifti(image_path):
        return None

    registry_entry = get_dataset_registry_entry(dataset_key)
    materialized_row = dict(row)
    materialized_row.setdefault("brain_mask_path", "")
    materialized_row.setdefault("normalization_mask_method", "")
    materialized_row["exclude_reason"] = ""
    materialized_row["image_path"] = str(image_path)
    materialized_row["t1_path"] = str(image_path)

    existing_mask_path_text = ""
    existing_mask_method = ""
    existing_preproc_profile = ""
    if existing_manifest_row is not None:
        existing_mask_path_text = _text(existing_manifest_row.get("brain_mask_path"))
        existing_mask_method = _text(
            existing_manifest_row.get("normalization_mask_method")
        )
        existing_preproc_profile = _text(existing_manifest_row.get("preproc_profile"))

    resolved_mask_path = (
        Path(existing_mask_path_text) if existing_mask_path_text else default_mask_path
    )
    if save_mask:
        if _is_valid_materialized_nifti(resolved_mask_path):
            materialized_row["brain_mask_path"] = str(resolved_mask_path)
        elif _is_valid_materialized_nifti(default_mask_path):
            materialized_row["brain_mask_path"] = str(default_mask_path)
        else:
            return None
    else:
        materialized_row["brain_mask_path"] = ""

    materialized_row["normalization_mask_method"] = (
        existing_mask_method
        or (
            "synthstrip"
            if registry_entry.cls_skullstrip_policy == "synthstrip"
            else "nonzero"
        )
    )
    materialized_row["preproc_profile"] = (
        existing_preproc_profile
        or _resolved_preproc_profile(
            dataset_key,
            _text(row.get("preproc_profile")) or registry_entry.preproc_profile,
            skullstrip_policy=registry_entry.cls_skullstrip_policy,
        )
    )
    return materialized_row


def _build_resumed_segmentation_materialized_row(
    row: dict[str, str],
    *,
    case_output_root: Path,
    existing_manifest_row: dict[str, str] | None,
) -> dict[str, str] | None:
    default_image_path, default_mask_path = _segmentation_materialized_paths_for_case(
        row["image_path"],
        row["mask_path"],
        case_output_root,
    )
    existing_image_path_text = ""
    existing_mask_path_text = ""
    existing_brain_mask_path_text = ""
    existing_normalization_mask_method = ""
    existing_preproc_profile = ""
    if existing_manifest_row is not None:
        existing_image_path_text = _text(existing_manifest_row.get("image_path"))
        existing_mask_path_text = _text(existing_manifest_row.get("mask_path"))
        existing_brain_mask_path_text = _text(
            existing_manifest_row.get("brain_mask_path")
        )
        existing_normalization_mask_method = _text(
            existing_manifest_row.get("normalization_mask_method")
        )
        existing_preproc_profile = _text(existing_manifest_row.get("preproc_profile"))

    resolved_image_path = (
        Path(existing_image_path_text) if existing_image_path_text else default_image_path
    )
    resolved_mask_path = (
        Path(existing_mask_path_text) if existing_mask_path_text else default_mask_path
    )
    if not _is_valid_materialized_segmentation_nifti(
        resolved_image_path
    ) or not _is_valid_materialized_segmentation_nifti(resolved_mask_path):
        if not _is_valid_materialized_segmentation_nifti(
            default_image_path
        ) or not _is_valid_materialized_segmentation_nifti(default_mask_path):
            return None
        resolved_image_path = default_image_path
        resolved_mask_path = default_mask_path

    materialized_row = dict(row)
    materialized_row["exclude_reason"] = ""
    materialized_row["image_path"] = str(resolved_image_path)
    materialized_row["mask_path"] = str(resolved_mask_path)
    materialized_row["t1_path"] = ""
    registry_entry = get_dataset_registry_entry(_text(row.get("dataset_key")))
    default_brain_mask_path = _segmentation_brain_mask_path_for_case(case_output_root)
    resolved_brain_mask_path = (
        Path(existing_brain_mask_path_text)
        if existing_brain_mask_path_text
        else default_brain_mask_path
    )
    if existing_normalization_mask_method == "synthstrip" or (
        not existing_normalization_mask_method
        and registry_entry.cls_skullstrip_policy == "synthstrip"
    ):
        materialized_row["normalization_mask_method"] = "synthstrip"
        materialized_row["brain_mask_path"] = (
            str(resolved_brain_mask_path)
            if _is_valid_materialized_segmentation_nifti(resolved_brain_mask_path)
            else ""
        )
    else:
        materialized_row["brain_mask_path"] = ""
        materialized_row["normalization_mask_method"] = ""
    if existing_preproc_profile:
        materialized_row["preproc_profile"] = existing_preproc_profile
    return materialized_row


def materialize_classification_manifest(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    *,
    save_mask: bool = True,
    output_manifest_path: str | Path | None = None,
    synthstrip_cmd: str = "mri_synthstrip",
    progress_interval: int = 100,
) -> tuple[list[dict[str, str]], Path]:
    """Write classification views and emit manifest pointing to copied outputs.

    Existing valid outputs are reused automatically so interrupted runs can resume.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        destination / "classification_materialized_manifest.csv"
        if output_manifest_path is None
        else Path(output_manifest_path)
    )

    classification_rows = [
        row
        for row in rows
        if _text(row.get("task_type") or "classification") == "classification"
    ]
    total_rows = len(classification_rows)
    skipped_rows = sum(1 for row in classification_rows if _text(row.get("exclude_reason")))
    print(
        f"Materializing {total_rows} classification rows to {destination}",
        flush=True,
    )
    if skipped_rows:
        print(
            f"Rows already excluded in source manifest: {skipped_rows}",
            flush=True,
        )

    existing_rows_by_case_id = _load_existing_materialized_rows(manifest_path)
    materialized_rows: list[dict[str, str]] = []
    rows_to_process: list[dict[str, str]] = []
    processed_count = 0
    success_count = 0
    failed_count = 0
    resumed_count = 0
    resumed_failed_count = 0

    for row in classification_rows:
        dataset_key = _text(row.get("dataset_key"))
        materialized_row = dict(row)
        materialized_row.setdefault("brain_mask_path", "")
        materialized_row.setdefault("normalization_mask_method", "")

        if _text(row.get("exclude_reason")):
            materialized_rows.append(_blank_materialized_paths(materialized_row))
            processed_count += 1
            continue

        resumed_row = _build_resumed_materialized_row(
            row,
            dataset_key=dataset_key,
            case_output_root=destination / row["global_case_id"],
            save_mask=save_mask,
            existing_manifest_row=existing_rows_by_case_id.get(
                _text(row.get("global_case_id"))
            ),
        )
        if resumed_row is not None:
            materialized_rows.append(resumed_row)
            processed_count += 1
            success_count += 1
            resumed_count += 1
            continue

        resumed_failed_row = _build_resumed_failed_row(
            row,
            existing_manifest_row=existing_rows_by_case_id.get(
                _text(row.get("global_case_id"))
            )
            or {},
        )
        if resumed_failed_row is not None:
            materialized_rows.append(resumed_failed_row)
            processed_count += 1
            failed_count += 1
            resumed_failed_count += 1
            continue

        rows_to_process.append(row)

    if resumed_count:
        print(
            f"Reusing existing standardized rows: {resumed_count}",
            flush=True,
        )
    if resumed_failed_count:
        print(
            f"Reusing existing failed rows: {resumed_failed_count}",
            flush=True,
        )

    preflight_synthstrip_requirements(
        rows_to_process,
        synthstrip_cmd=synthstrip_cmd,
    )

    with _managed_synthstrip_session(
        rows_to_process,
        synthstrip_cmd=synthstrip_cmd,
    ):
        for row in rows_to_process:
            dataset_key = _text(row.get("dataset_key"))
            registry_entry = get_dataset_registry_entry(dataset_key)
            materialized_row = dict(row)
            materialized_row.setdefault("brain_mask_path", "")
            materialized_row.setdefault("normalization_mask_method", "")

            try:
                result = write_classification_view(
                    row["t1_path"],
                    destination / row["global_case_id"],
                    dataset_key=dataset_key,
                    original_preproc_profile=_text(row.get("preproc_profile"))
                    or registry_entry.preproc_profile,
                    synthstrip_cmd=synthstrip_cmd,
                    save_mask=save_mask,
                )
            except ClassificationPreprocessError as exc:
                failed_row = _blank_materialized_paths(materialized_row)
                failed_row["exclude_reason"] = exc.exclude_reason
                failed_row["preproc_profile"] = _resolved_preproc_profile(
                    dataset_key,
                    _text(row.get("preproc_profile")) or registry_entry.preproc_profile,
                    skullstrip_policy=registry_entry.cls_skullstrip_policy,
                )
                materialized_rows.append(failed_row)
                processed_count += 1
                failed_count += 1
                if failed_count <= 10:
                    print(
                        f"[{processed_count}/{total_rows}] failed {row.get('global_case_id', '<unknown>')} -> {exc.exclude_reason}",
                        flush=True,
                    )
                continue

            materialized_row["image_path"] = str(result.image_path)
            materialized_row["t1_path"] = str(result.image_path)
            materialized_row["brain_mask_path"] = (
                "" if result.brain_mask_path is None else str(result.brain_mask_path)
            )
            materialized_row["normalization_mask_method"] = (
                result.normalization_mask_method
            )
            materialized_row["preproc_profile"] = result.preproc_profile
            materialized_rows.append(materialized_row)
            processed_count += 1
            success_count += 1
            if processed_count == 1 or processed_count % max(progress_interval, 1) == 0:
                print(
                    f"[{processed_count}/{total_rows}] standardized rows: ok={success_count}, failed={failed_count}, resumed={resumed_count}",
                    flush=True,
                )

    write_csv_rows(manifest_path, materialized_rows, _extra_fieldnames(materialized_rows))
    print(
        f"Finished materialization: ok={success_count}, failed={failed_count}, total={total_rows}",
        flush=True,
    )
    return materialized_rows, manifest_path


class SegmentationMaterializationError(RuntimeError):
    """Segmentation materialization error that maps to manifest exclusion reason."""

    def __init__(self, exclude_reason: str, message: str) -> None:
        super().__init__(message)
        self.exclude_reason = exclude_reason


def write_segmentation_pair(
    image_path: str | Path,
    mask_path: str | Path,
    case_output_root: str | Path,
    *,
    original_preproc_profile: str = "",
    dataset_key: str,
    synthstrip_cmd: str = "mri_synthstrip",
) -> SegmentationPairResult:
    """Convert native-scale segmentation image/mask pair into `.nii.gz` layout."""

    source_image_path = Path(image_path)
    source_mask_path = Path(mask_path)
    if not source_image_path.exists():
        raise SegmentationMaterializationError(
            "missing_anchor_image",
            f"Segmentation anchor image not found: {source_image_path}",
        )
    if not source_mask_path.exists():
        raise SegmentationMaterializationError(
            "missing_segmentation_mask",
            f"Segmentation mask not found: {source_mask_path}",
        )

    materialized_image_path, materialized_mask_path = _segmentation_materialized_paths_for_case(
        source_image_path,
        source_mask_path,
        case_output_root,
    )
    registry_entry = get_dataset_registry_entry(dataset_key)
    normalization_mask_method = ""
    brain_mask_path: Path | None = None
    try:
        materialized_image_path.parent.mkdir(parents=True, exist_ok=True)
        image = _build_nifti_from_source(
            source_image_path,
            exclude_reason="invalid_image_file",
        )
        mask = _build_nifti_from_source(
            source_mask_path,
            exclude_reason="invalid_segmentation_mask",
        )
        if registry_entry.cls_skullstrip_policy == "synthstrip":
            stripped_volume, brain_mask = _run_synthstrip(
                image,
                synthstrip_cmd=synthstrip_cmd,
            )
            image = nib.Nifti1Image(stripped_volume, image.affine, image.header)
            brain_mask_path = _segmentation_brain_mask_path_for_case(case_output_root)
            nib.save(
                nib.Nifti1Image(
                    brain_mask.astype(np.uint8),
                    image.affine,
                ),
                str(brain_mask_path),
            )
            normalization_mask_method = "synthstrip"
        nib.save(image, str(materialized_image_path))
        nib.save(mask, str(materialized_mask_path))
    except ClassificationPreprocessError as exc:
        mapped_reason = exc.exclude_reason
        if mapped_reason == "invalid_image_file":
            mapped_reason = "invalid_anchor_image"
        elif mapped_reason == "invalid_segmentation_mask":
            mapped_reason = "invalid_segmentation_mask"
        raise SegmentationMaterializationError(
            mapped_reason,
            str(exc),
        ) from exc
    except OSError as exc:
        raise SegmentationMaterializationError(
            "segmentation_materialization_failed",
            (
                f"Could not materialize segmentation pair "
                f"{source_image_path} / {source_mask_path}: {exc}"
            ),
        ) from exc

    return SegmentationPairResult(
        image_path=materialized_image_path,
        mask_path=materialized_mask_path,
        brain_mask_path=brain_mask_path,
        normalization_mask_method=normalization_mask_method,
        preproc_profile=_resolved_preproc_profile(
            dataset_key,
            original_preproc_profile,
            skullstrip_policy=registry_entry.cls_skullstrip_policy,
        ),
    )


def materialize_segmentation_manifest(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    *,
    output_manifest_path: str | Path | None = None,
    synthstrip_cmd: str = "mri_synthstrip",
    progress_interval: int = 100,
) -> tuple[list[dict[str, str]], Path]:
    """Convert native-scale segmentation pairs and emit manifest pointing to copies."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        destination / "segmentation_materialized_manifest.csv"
        if output_manifest_path is None
        else Path(output_manifest_path)
    )

    segmentation_rows = [
        row
        for row in rows
        if _text(row.get("task_type")) == "segmentation"
    ]
    total_rows = len(segmentation_rows)
    skipped_rows = sum(1 for row in segmentation_rows if _text(row.get("exclude_reason")))
    print(
        f"Materializing {total_rows} segmentation rows to {destination}",
        flush=True,
    )
    if skipped_rows:
        print(
            f"Rows already excluded in source manifest: {skipped_rows}",
            flush=True,
        )

    existing_rows_by_case_id = _load_existing_materialized_rows(manifest_path)
    materialized_rows: list[dict[str, str]] = []
    rows_to_process: list[dict[str, str]] = []
    processed_count = 0
    success_count = 0
    failed_count = 0
    resumed_count = 0
    resumed_failed_count = 0

    for row in segmentation_rows:
        materialized_row = dict(row)
        materialized_row.setdefault("brain_mask_path", "")
        materialized_row.setdefault("normalization_mask_method", "")

        if _text(row.get("exclude_reason")):
            materialized_rows.append(
                _blank_materialized_paths(
                    materialized_row,
                    clear_mask=True,
                )
            )
            processed_count += 1
            continue

        resumed_row = _build_resumed_segmentation_materialized_row(
            row,
            case_output_root=destination / row["global_case_id"],
            existing_manifest_row=existing_rows_by_case_id.get(
                _text(row.get("global_case_id"))
            ),
        )
        if resumed_row is not None:
            materialized_rows.append(resumed_row)
            processed_count += 1
            success_count += 1
            resumed_count += 1
            continue

        resumed_failed_row = _build_resumed_failed_row(
            row,
            existing_manifest_row=existing_rows_by_case_id.get(
                _text(row.get("global_case_id"))
            )
            or {},
            clear_mask=True,
        )
        if resumed_failed_row is not None:
            materialized_rows.append(resumed_failed_row)
            processed_count += 1
            failed_count += 1
            resumed_failed_count += 1
            continue

        rows_to_process.append(row)

    if resumed_count:
        print(
            f"Reusing existing standardized rows: {resumed_count}",
            flush=True,
        )
    if resumed_failed_count:
        print(
            f"Reusing existing failed rows: {resumed_failed_count}",
            flush=True,
        )

    preflight_synthstrip_requirements(
        rows_to_process,
        synthstrip_cmd=synthstrip_cmd,
    )

    with _managed_synthstrip_session(
        rows_to_process,
        synthstrip_cmd=synthstrip_cmd,
    ):
        for row in rows_to_process:
            materialized_row = dict(row)
            materialized_row.setdefault("brain_mask_path", "")
            materialized_row.setdefault("normalization_mask_method", "")

            try:
                result = write_segmentation_pair(
                    row["image_path"],
                    row["mask_path"],
                    destination / row["global_case_id"],
                    original_preproc_profile=_text(row.get("preproc_profile")),
                    dataset_key=_text(row.get("dataset_key")),
                    synthstrip_cmd=synthstrip_cmd,
                )
            except SegmentationMaterializationError as exc:
                failed_row = _blank_materialized_paths(
                    materialized_row,
                    clear_mask=True,
                )
                failed_row["exclude_reason"] = exc.exclude_reason
                materialized_rows.append(failed_row)
                processed_count += 1
                failed_count += 1
                if failed_count <= 10:
                    print(
                        f"[{processed_count}/{total_rows}] failed {row.get('global_case_id', '<unknown>')} -> {exc.exclude_reason}",
                        flush=True,
                    )
                continue

            materialized_row["image_path"] = str(result.image_path)
            materialized_row["mask_path"] = str(result.mask_path)
            materialized_row["t1_path"] = ""
            materialized_row["brain_mask_path"] = (
                "" if result.brain_mask_path is None else str(result.brain_mask_path)
            )
            materialized_row["normalization_mask_method"] = (
                result.normalization_mask_method
            )
            materialized_row["preproc_profile"] = result.preproc_profile
            materialized_rows.append(materialized_row)
            processed_count += 1
            success_count += 1
            if processed_count == 1 or processed_count % max(progress_interval, 1) == 0:
                print(
                    f"[{processed_count}/{total_rows}] standardized rows: ok={success_count}, failed={failed_count}, resumed={resumed_count}",
                    flush=True,
                )

    write_csv_rows(manifest_path, materialized_rows, _extra_fieldnames(materialized_rows))
    print(
        f"Finished materialization: ok={success_count}, failed={failed_count}, total={total_rows}",
        flush=True,
    )
    return materialized_rows, manifest_path


def materialize_brain_structure_manifest(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    *,
    save_mask: bool = True,
    output_manifest_path: str | Path | None = None,
    synthstrip_cmd: str = "mri_synthstrip",
) -> tuple[list[dict[str, str]], Path]:
    """Backward-compatible wrapper for brain-structure-only cls materialization."""

    filtered_rows = [
        row for row in rows if _text(row.get("dataset_key")) == BRAIN_STRUCTURE_DATASET_KEY
    ]
    resolved_manifest_path = output_manifest_path
    if resolved_manifest_path is None:
        resolved_manifest_path = Path(output_dir) / "brain_structure_cls_manifest.csv"
    return materialize_classification_manifest(
        filtered_rows,
        output_dir,
        save_mask=save_mask,
        output_manifest_path=resolved_manifest_path,
        synthstrip_cmd=synthstrip_cmd,
    )
