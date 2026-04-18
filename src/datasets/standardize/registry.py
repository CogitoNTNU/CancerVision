"""Dataset registry for standardization adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from .adapters import (
    BraTSAdapter,
    BraTSDatasetSpec,
    CfbGbmAdapter,
    BrainStructureAdapter,
    RemindAdapter,
    UcsdPtgbmAdapter,
    UcsfPdgmAdapter,
    UpennGbmAdapter,
    UtswGliomaAdapter,
    VestibularSchwannomaMcRc2Adapter,
    YaleBrainMetsLongitudinalAdapter,
)
from .constants import (
    CFB_GBM_DATASET_KEY,
    CFB_GBM_DEFAULT_ROOT,
    CFB_GBM_PREPROC_PROFILE,
    BRATS2020_DATASET_KEY,
    BRATS2020_DEFAULT_ROOT,
    BRATS2020_PREPROC_PROFILE,
    BRATS2020_SEG_PREPROC_PROFILE,
    BRATS2023_DATASET_KEY,
    BRATS2023_DEFAULT_ROOT,
    BRATS2023_PREPROC_PROFILE,
    BRATS2023_SEG_PREPROC_PROFILE,
    BRATS2024_DATASET_KEY,
    BRATS2024_DEFAULT_ROOT,
    BRATS2024_PREPROC_PROFILE,
    BRATS2024_SEG_PREPROC_PROFILE,
    BRAIN_STRUCTURE_DATASET_KEY,
    BRAIN_STRUCTURE_DEFAULT_ROOT,
    BRAIN_STRUCTURE_PREPROC_PROFILE,
    REMIND_DEFAULT_ROOT,
    REMIND_DATASET_KEY,
    REMIND_MASKS_DEFAULT_ROOT,
    REMIND_PREPROC_PROFILE,
    UCSD_PTGBM_DATASET_KEY,
    UCSD_PTGBM_DEFAULT_ROOT,
    UCSD_PTGBM_PREPROC_PROFILE,
    UCSF_PDGM_DEFAULT_ROOT,
    UCSF_PDGM_DATASET_KEY,
    UCSF_PDGM_PREPROC_PROFILE,
    UTSW_GLIOMA_DATASET_KEY,
    UTSW_GLIOMA_DEFAULT_ROOT,
    UTSW_GLIOMA_PREPROC_PROFILE,
    UPENN_GBM_DEFAULT_ROOT,
    UPENN_GBM_DATASET_KEY,
    UPENN_GBM_PREPROC_PROFILE,
    VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY,
    VESTIBULAR_SCHWANNOMA_MC_RC2_DEFAULT_ROOT,
    VESTIBULAR_SCHWANNOMA_MC_RC2_PREPROC_PROFILE,
    YALE_BRAIN_METS_LONGITUDINAL_DEFAULT_ROOT,
    YALE_BRAIN_METS_LONGITUDINAL_DATASET_KEY,
    YALE_BRAIN_METS_LONGITUDINAL_PREPROC_PROFILE,
)
from .pathing import resolve_dataset_root

ClsSkullstripPolicy = Literal["skip", "synthstrip"]


@dataclass(frozen=True, slots=True)
class DatasetRegistryEntry:
    """Static config for one standardized dataset source."""

    key: str
    default_root: str
    preproc_profile: str
    cls_skullstrip_policy: ClsSkullstripPolicy


def normalize_dataset_key(dataset_key: str) -> str:
    """Normalize dataset keys from manifests into registry keys."""

    normalized = re.sub(r"[^a-z0-9]+", "_", dataset_key.strip().lower())
    return normalized.strip("_")


STANDARDIZE_DATASET_REGISTRY = {
    BRATS2020_DATASET_KEY: DatasetRegistryEntry(
        key=BRATS2020_DATASET_KEY,
        default_root=BRATS2020_DEFAULT_ROOT,
        preproc_profile=BRATS2020_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    BRATS2023_DATASET_KEY: DatasetRegistryEntry(
        key=BRATS2023_DATASET_KEY,
        default_root=BRATS2023_DEFAULT_ROOT,
        preproc_profile=BRATS2023_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    BRATS2024_DATASET_KEY: DatasetRegistryEntry(
        key=BRATS2024_DATASET_KEY,
        default_root=BRATS2024_DEFAULT_ROOT,
        preproc_profile=BRATS2024_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    CFB_GBM_DATASET_KEY: DatasetRegistryEntry(
        key=CFB_GBM_DATASET_KEY,
        default_root=CFB_GBM_DEFAULT_ROOT,
        preproc_profile=CFB_GBM_PREPROC_PROFILE,
        cls_skullstrip_policy="synthstrip",
    ),
    BRAIN_STRUCTURE_DATASET_KEY: DatasetRegistryEntry(
        key=BRAIN_STRUCTURE_DATASET_KEY,
        default_root=BRAIN_STRUCTURE_DEFAULT_ROOT,
        preproc_profile=BRAIN_STRUCTURE_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    UPENN_GBM_DATASET_KEY: DatasetRegistryEntry(
        key=UPENN_GBM_DATASET_KEY,
        default_root=UPENN_GBM_DEFAULT_ROOT,
        preproc_profile=UPENN_GBM_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    UCSF_PDGM_DATASET_KEY: DatasetRegistryEntry(
        key=UCSF_PDGM_DATASET_KEY,
        default_root=UCSF_PDGM_DEFAULT_ROOT,
        preproc_profile=UCSF_PDGM_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    UTSW_GLIOMA_DATASET_KEY: DatasetRegistryEntry(
        key=UTSW_GLIOMA_DATASET_KEY,
        default_root=UTSW_GLIOMA_DEFAULT_ROOT,
        preproc_profile=UTSW_GLIOMA_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    UCSD_PTGBM_DATASET_KEY: DatasetRegistryEntry(
        key=UCSD_PTGBM_DATASET_KEY,
        default_root=UCSD_PTGBM_DEFAULT_ROOT,
        preproc_profile=UCSD_PTGBM_PREPROC_PROFILE,
        cls_skullstrip_policy="skip",
    ),
    REMIND_DATASET_KEY: DatasetRegistryEntry(
        key=REMIND_DATASET_KEY,
        default_root=REMIND_DEFAULT_ROOT,
        preproc_profile=REMIND_PREPROC_PROFILE,
        cls_skullstrip_policy="synthstrip",
    ),
    YALE_BRAIN_METS_LONGITUDINAL_DATASET_KEY: DatasetRegistryEntry(
        key=YALE_BRAIN_METS_LONGITUDINAL_DATASET_KEY,
        default_root=YALE_BRAIN_METS_LONGITUDINAL_DEFAULT_ROOT,
        preproc_profile=YALE_BRAIN_METS_LONGITUDINAL_PREPROC_PROFILE,
        cls_skullstrip_policy="synthstrip",
    ),
    VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY: DatasetRegistryEntry(
        key=VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY,
        default_root=VESTIBULAR_SCHWANNOMA_MC_RC2_DEFAULT_ROOT,
        preproc_profile=VESTIBULAR_SCHWANNOMA_MC_RC2_PREPROC_PROFILE,
        cls_skullstrip_policy="synthstrip",
    ),
}


def get_dataset_registry_entry(dataset_key: str) -> DatasetRegistryEntry:
    """Resolve dataset registry entry from manifest dataset key."""

    normalized_key = normalize_dataset_key(dataset_key)
    try:
        return STANDARDIZE_DATASET_REGISTRY[normalized_key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset_key '{dataset_key}'. Add explicit registry config first."
        ) from exc


def get_brain_structure_adapter(
    root: str | Path | None = None,
    *,
    metadata_filename: str = "metadata.csv",
) -> BrainStructureAdapter:
    """Create adapter with default root fallback and Windows-path support."""

    resolved_root = resolve_dataset_root(root, default=BRAIN_STRUCTURE_DEFAULT_ROOT)
    return BrainStructureAdapter(resolved_root, metadata_filename=metadata_filename)


def _get_brats_adapter(
    root: str | Path | None,
    *,
    dataset_key: str,
    default_root: str,
    source_study: str,
    classification_preproc_profile: str,
    segmentation_preproc_profile: str,
) -> BraTSAdapter:
    resolved_root = resolve_dataset_root(root, default=default_root)
    return BraTSAdapter(
        resolved_root,
        spec=BraTSDatasetSpec(
            dataset_key=dataset_key,
            source_study=source_study,
            classification_preproc_profile=classification_preproc_profile,
            segmentation_preproc_profile=segmentation_preproc_profile,
        ),
    )


def get_brats2020_adapter(
    root: str | Path | None = None,
) -> BraTSAdapter:
    """Create BraTS 2020 adapter with default root fallback and path support."""

    return _get_brats_adapter(
        root,
        dataset_key=BRATS2020_DATASET_KEY,
        default_root=BRATS2020_DEFAULT_ROOT,
        source_study="BraTS2020",
        classification_preproc_profile=BRATS2020_PREPROC_PROFILE,
        segmentation_preproc_profile=BRATS2020_SEG_PREPROC_PROFILE,
    )


def get_brats2023_adapter(
    root: str | Path | None = None,
) -> BraTSAdapter:
    """Create BraTS 2023 adapter with default root fallback and path support."""

    return _get_brats_adapter(
        root,
        dataset_key=BRATS2023_DATASET_KEY,
        default_root=BRATS2023_DEFAULT_ROOT,
        source_study="BraTS2023",
        classification_preproc_profile=BRATS2023_PREPROC_PROFILE,
        segmentation_preproc_profile=BRATS2023_SEG_PREPROC_PROFILE,
    )


def get_brats2024_adapter(
    root: str | Path | None = None,
) -> BraTSAdapter:
    """Create BraTS 2024 adapter with default root fallback and path support."""

    return _get_brats_adapter(
        root,
        dataset_key=BRATS2024_DATASET_KEY,
        default_root=BRATS2024_DEFAULT_ROOT,
        source_study="BraTS2024",
        classification_preproc_profile=BRATS2024_PREPROC_PROFILE,
        segmentation_preproc_profile=BRATS2024_SEG_PREPROC_PROFILE,
    )


def get_cfb_gbm_adapter(
    root: str | Path | None = None,
) -> CfbGbmAdapter:
    """Create CFB-GBM adapter with default root fallback and path support."""

    resolved_root = resolve_dataset_root(root, default=CFB_GBM_DEFAULT_ROOT)
    return CfbGbmAdapter(resolved_root)


def get_upenn_gbm_adapter(
    root: str | Path | None = None,
) -> UpennGbmAdapter:
    """Create UPENN-GBM adapter with default root fallback and path support."""

    resolved_root = resolve_dataset_root(root, default=UPENN_GBM_DEFAULT_ROOT)
    return UpennGbmAdapter(resolved_root)


def get_ucsf_pdgm_adapter(
    root: str | Path | None = None,
) -> UcsfPdgmAdapter:
    """Create UCSF-PDGM adapter with default root fallback and path support."""

    resolved_root = resolve_dataset_root(root, default=UCSF_PDGM_DEFAULT_ROOT)
    return UcsfPdgmAdapter(resolved_root)


def get_utsw_glioma_adapter(
    root: str | Path | None = None,
) -> UtswGliomaAdapter:
    """Create UTSW-Glioma adapter with default root fallback and path support."""

    resolved_root = resolve_dataset_root(root, default=UTSW_GLIOMA_DEFAULT_ROOT)
    return UtswGliomaAdapter(resolved_root)


def get_ucsd_ptgbm_adapter(
    root: str | Path | None = None,
) -> UcsdPtgbmAdapter:
    """Create UCSD-PTGBM adapter with default root fallback and path support."""

    resolved_root = resolve_dataset_root(root, default=UCSD_PTGBM_DEFAULT_ROOT)
    return UcsdPtgbmAdapter(resolved_root)


def get_remind_adapter(
    image_root: str | Path | None = None,
    *,
    mask_root: str | Path | None = None,
) -> RemindAdapter:
    """Create ReMIND adapter with default roots and Windows-path support."""

    resolved_image_root = resolve_dataset_root(image_root, default=REMIND_DEFAULT_ROOT)
    resolved_mask_root = resolve_dataset_root(mask_root, default=REMIND_MASKS_DEFAULT_ROOT)
    return RemindAdapter(resolved_image_root, resolved_mask_root)


def get_yale_brain_mets_longitudinal_adapter(
    root: str | Path | None = None,
) -> YaleBrainMetsLongitudinalAdapter:
    """Create Yale Brain Mets adapter with default root fallback and path support."""

    resolved_root = resolve_dataset_root(
        root,
        default=YALE_BRAIN_METS_LONGITUDINAL_DEFAULT_ROOT,
    )
    return YaleBrainMetsLongitudinalAdapter(resolved_root)


def get_vestibular_schwannoma_mc_rc2_adapter(
    root: str | Path | None = None,
) -> VestibularSchwannomaMcRc2Adapter:
    """Create Vestibular Schwannoma adapter with default root fallback."""

    resolved_root = resolve_dataset_root(
        root,
        default=VESTIBULAR_SCHWANNOMA_MC_RC2_DEFAULT_ROOT,
    )
    return VestibularSchwannomaMcRc2Adapter(resolved_root)
