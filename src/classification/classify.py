#!/usr/bin/env python
"""Classify brain tumor cases from segmentation predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from .classifier_registry import ClassifierRegistry, resolve_repo_root
from .rules import classify_profile, extract_tumor_features


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify tumor phenotype from predicted segmentation masks."
    )
    parser.add_argument(
        "--classifier-id",
        type=str,
        default="brats_rule_based_v1",
        help="Classifier id from res/classification/classifier_registry.json",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Optional path to classifier registry JSON",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Directory containing predicted segmentation masks",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_pred.nii.gz",
        help="Glob pattern used to discover prediction masks",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path for case-level classification report",
    )
    return parser.parse_args(argv)


def _case_id_from_prediction_file(path: Path) -> str:
    stem = path.name
    for suffix in ("_pred.nii.gz", "_pred.nii"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return path.stem


def classify_predictions(
    classifier_id: str,
    registry_path: str | None,
    input_root: Path,
    glob_pattern: str,
    output_csv: Path,
) -> Path:
    registry = ClassifierRegistry(
        repo_root=Path(__file__).resolve(),
        registry_path=Path(registry_path) if registry_path else None,
    )
    spec = registry.get(classifier_id)

    if spec.classifier_type != "rule_based":
        raise ValueError(
            f"Classifier '{classifier_id}' has unsupported type '{spec.classifier_type}'."
            " Supported: rule_based"
        )

    prediction_files = sorted(input_root.glob(glob_pattern))
    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction files found in {input_root} with pattern '{glob_pattern}'"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "prediction_path",
                "class_label",
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
            data = nib.load(str(prediction_path)).get_fdata(dtype=np.float32)
            label_map = np.asarray(data, dtype=np.uint8)
            profile = extract_tumor_features(label_map)
            class_label = classify_profile(profile, spec.thresholds)

            writer.writerow(
                {
                    "case_id": _case_id_from_prediction_file(prediction_path),
                    "prediction_path": str(prediction_path),
                    "class_label": class_label,
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

    default_csv = (
        resolve_repo_root(Path(__file__).resolve())
        / "res"
        / "classification"
        / "reports"
        / f"{args.classifier_id}_predictions.csv"
    )
    output_csv = Path(args.output_csv).resolve() if args.output_csv else default_csv

    print(f"Classifier id : {args.classifier_id}", flush=True)
    print(f"Input root    : {input_root}", flush=True)
    print(f"Pattern       : {args.glob}", flush=True)
    print(f"Output csv    : {output_csv}", flush=True)

    saved = classify_predictions(
        classifier_id=args.classifier_id,
        registry_path=args.registry,
        input_root=input_root,
        glob_pattern=args.glob,
        output_csv=output_csv,
    )
    print(f"Saved classification report: {saved}", flush=True)


if __name__ == "__main__":
    main()
