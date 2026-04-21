#!/usr/bin/env python
"""Case-level classification of BraTS predicted segmentations.

Scans a directory of predicted NIfTI masks and writes a CSV row per case with:

    * binary cancer/no-cancer label ("tumor" / "no_tumor") using a voxel-count threshold
    * coarse phenotype label ("enhancing_dominant" / "core_dominant" / "edema_dominant")
    * raw TC/WT/ET voxel counts and ratios

Classification is purely derived from the segmentation mask, so upstream
segmentation quality dominates the output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from .rules import (
    ClassificationThresholds,
    DEFAULT_MIN_TUMOR_VOXELS,
    classify_profile,
    extract_tumor_features,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify predicted BraTS segmentation masks (cancer vs non-cancer + phenotype)."
    )
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument(
        "--glob",
        type=str,
        default="*_pred.nii.gz",
        help="Glob pattern used to discover prediction masks",
    )
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument(
        "--min-tumor-voxels",
        type=int,
        default=DEFAULT_MIN_TUMOR_VOXELS,
        help="Whole-tumor voxel count required to classify a case as cancerous",
    )
    parser.add_argument("--enhancing-ratio", type=float, default=0.20)
    parser.add_argument("--core-ratio", type=float, default=0.70)
    return parser.parse_args(argv)


def _case_id(path: Path) -> str:
    name = path.name
    for suffix in ("_pred.nii.gz", "_pred.nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def classify_predictions(
    *,
    input_root: Path,
    glob_pattern: str,
    output_csv: Path,
    thresholds: ClassificationThresholds,
) -> Path:
    prediction_files = sorted(input_root.glob(glob_pattern))
    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction files found in {input_root} matching '{glob_pattern}'"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "prediction_path",
                "binary_label",
                "phenotype_label",
                "wt_voxels",
                "tc_voxels",
                "et_voxels",
                "wt_ratio",
                "tc_ratio",
                "et_ratio",
                "et_to_wt_ratio",
                "tc_to_wt_ratio",
            ],
        )
        writer.writeheader()

        for prediction_path in prediction_files:
            volume = nib.load(str(prediction_path)).get_fdata(dtype=np.float32)
            label_map = np.asarray(volume, dtype=np.uint8)
            profile = extract_tumor_features(label_map)
            phenotype = classify_profile(profile, thresholds)
            binary = "no_tumor" if profile.wt_voxels < thresholds.min_tumor_voxels else "tumor"

            writer.writerow(
                {
                    "case_id": _case_id(prediction_path),
                    "prediction_path": str(prediction_path),
                    "binary_label": binary,
                    "phenotype_label": phenotype,
                    "wt_voxels": profile.wt_voxels,
                    "tc_voxels": profile.tc_voxels,
                    "et_voxels": profile.et_voxels,
                    "wt_ratio": f"{profile.wt_ratio:.8f}",
                    "tc_ratio": f"{profile.tc_ratio:.8f}",
                    "et_ratio": f"{profile.et_ratio:.8f}",
                    "et_to_wt_ratio": f"{profile.et_to_wt_ratio:.8f}",
                    "tc_to_wt_ratio": f"{profile.tc_to_wt_ratio:.8f}",
                }
            )

    return output_csv


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_csv = Path(args.output_csv).resolve()
    thresholds = ClassificationThresholds(
        min_tumor_voxels=args.min_tumor_voxels,
        enhancing_ratio_for_aggressive=args.enhancing_ratio,
        core_ratio_for_compact=args.core_ratio,
    )

    print(f"Input root : {input_root}", flush=True)
    print(f"Pattern    : {args.glob}", flush=True)
    print(f"Output csv : {output_csv}", flush=True)

    saved = classify_predictions(
        input_root=input_root,
        glob_pattern=args.glob,
        output_csv=output_csv,
        thresholds=thresholds,
    )
    print(f"Saved classification report: {saved}", flush=True)


if __name__ == "__main__":
    main()
