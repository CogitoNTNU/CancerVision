"""Rule-based classification from BraTS-style segmentation maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TumorFeatureProfile:
    """Volumetric features extracted from one case segmentation."""

    total_voxels: int
    wt_voxels: int
    tc_voxels: int
    et_voxels: int
    wt_ratio: float
    tc_ratio: float
    et_ratio: float
    et_to_wt_ratio: float
    tc_to_wt_ratio: float


@dataclass(frozen=True)
class ClassificationThresholds:
    """Thresholds controlling rule-based class assignment."""

    min_tumor_voxels: int = 16
    enhancing_ratio_for_aggressive: float = 0.20
    core_ratio_for_compact: float = 0.70


def _to_bool_masks(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wt = np.isin(label_map, (1, 2, 4))
    tc = np.isin(label_map, (1, 4))
    et = label_map == 4
    return wt, tc, et


def extract_tumor_features(label_map: np.ndarray) -> TumorFeatureProfile:
    """Extract volumetric features from a BraTS label map (0, 1, 2, 4)."""
    if label_map.ndim != 3:
        raise ValueError("Expected 3D label map for tumor classification")

    total_voxels = int(label_map.size)
    wt_mask, tc_mask, et_mask = _to_bool_masks(label_map)
    wt_voxels = int(wt_mask.sum())
    tc_voxels = int(tc_mask.sum())
    et_voxels = int(et_mask.sum())

    if total_voxels < 1:
        raise ValueError("Label map must contain at least one voxel")

    wt_ratio = wt_voxels / total_voxels
    tc_ratio = tc_voxels / total_voxels
    et_ratio = et_voxels / total_voxels

    if wt_voxels > 0:
        et_to_wt_ratio = et_voxels / wt_voxels
        tc_to_wt_ratio = tc_voxels / wt_voxels
    else:
        et_to_wt_ratio = 0.0
        tc_to_wt_ratio = 0.0

    return TumorFeatureProfile(
        total_voxels=total_voxels,
        wt_voxels=wt_voxels,
        tc_voxels=tc_voxels,
        et_voxels=et_voxels,
        wt_ratio=wt_ratio,
        tc_ratio=tc_ratio,
        et_ratio=et_ratio,
        et_to_wt_ratio=et_to_wt_ratio,
        tc_to_wt_ratio=tc_to_wt_ratio,
    )


def classify_profile(
    profile: TumorFeatureProfile,
    thresholds: ClassificationThresholds,
) -> str:
    """Assign one clinically-oriented category from segmentation-derived features."""
    if profile.wt_voxels < thresholds.min_tumor_voxels:
        return "no_tumor"

    if profile.et_to_wt_ratio >= thresholds.enhancing_ratio_for_aggressive:
        return "enhancing_dominant"

    if profile.tc_to_wt_ratio >= thresholds.core_ratio_for_compact:
        return "core_dominant"

    return "edema_dominant"
