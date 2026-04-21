"""CSV helpers for manifest IO."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .models import StandardizedRecord, standardized_manifest_fieldnames


def write_csv_rows(
    output_path: str | Path,
    rows: Iterable[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Write rows to CSV, always emitting header."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_standardized_manifest(
    output_path: str | Path,
    records: Iterable[StandardizedRecord],
) -> None:
    """Write standardized records to CSV."""

    write_csv_rows(
        output_path,
        (record.to_row() for record in records),
        standardized_manifest_fieldnames(),
    )


def read_csv_rows(input_path: str | Path) -> list[dict[str, str]]:
    """Read CSV file into stripped string rows."""

    source = Path(input_path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
