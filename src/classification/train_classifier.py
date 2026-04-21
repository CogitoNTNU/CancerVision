#!/usr/bin/env python
"""Train a case-level tumor classifier from tabular features."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random forest classifier from a feature CSV file."
    )
    parser.add_argument("--features-csv", type=str, required=True, help="Input CSV")
    parser.add_argument(
        "--target-column",
        type=str,
        default="class_label",
        help="Target label column name",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of feature columns",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        required=True,
        help="Path to save pickled classifier artifact",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to save metrics JSON",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args(argv)


def _load_feature_table(
    csv_path: Path,
    target_column: str,
    feature_columns: list[str] | None,
) -> tuple[list[list[float]], list[str], list[str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Feature CSV has no rows: {csv_path}")

    if target_column not in rows[0]:
        raise KeyError(f"Target column '{target_column}' missing in {csv_path}")

    if feature_columns is None:
        excluded = {target_column, "case_id", "prediction_path"}
        feature_columns = [c for c in rows[0].keys() if c not in excluded]

    if not feature_columns:
        raise ValueError("No feature columns selected for classifier training")

    x_values: list[list[float]] = []
    y_values: list[str] = []
    for row in rows:
        x_values.append([float(row[name]) for name in feature_columns])
        y_values.append(row[target_column])

    return x_values, y_values, feature_columns


def train_classifier(
    features_csv: Path,
    target_column: str,
    feature_columns: list[str] | None,
    model_output: Path,
    metrics_output: Path | None,
    test_size: float,
    seed: int,
) -> tuple[Path, float]:
    x_values, y_values, used_features = _load_feature_table(
        features_csv,
        target_column,
        feature_columns,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=test_size,
        random_state=seed,
        stratify=y_values if len(set(y_values)) > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "target_column": target_column,
                "feature_columns": used_features,
            },
            handle,
        )

    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with metrics_output.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "accuracy": accuracy,
                    "classification_report": report,
                    "n_train": len(x_train),
                    "n_test": len(x_test),
                    "feature_columns": used_features,
                },
                handle,
                indent=2,
            )

    return model_output, accuracy


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    model_output, accuracy = train_classifier(
        features_csv=Path(args.features_csv).resolve(),
        target_column=args.target_column,
        feature_columns=args.feature_columns,
        model_output=Path(args.model_output).resolve(),
        metrics_output=Path(args.metrics_output).resolve() if args.metrics_output else None,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"Saved classifier model: {model_output}", flush=True)
    print(f"Validation accuracy : {accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
