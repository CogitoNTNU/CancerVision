from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.standardize.io import read_csv_rows, write_csv_rows
from src.datasets.standardize.models import standardized_manifest_fieldnames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one or more standardized manifest CSV files."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged manifest CSV path.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input standardized manifest CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    extras: list[str] = []
    seen_extra: set[str] = set()

    for input_path in args.inputs:
        for row in read_csv_rows(input_path):
            rows.append(row)
            for key in row.keys():
                if key in seen_extra or key in standardized_manifest_fieldnames():
                    continue
                seen_extra.add(key)
                extras.append(key)

    fieldnames = [*standardized_manifest_fieldnames(), *extras]
    write_csv_rows(Path(args.output), rows, fieldnames)
    print(f"Merged {len(args.inputs)} manifests with {len(rows)} rows into {args.output}")


if __name__ == "__main__":
    main()
