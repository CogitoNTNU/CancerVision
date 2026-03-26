#!/usr/bin/env python
"""Compatibility wrapper for the DynUNet BraTS trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import dynnet  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    print(
        "src.models.train_brats is deprecated; forwarding to src.models.dynnet.",
        flush=True,
    )
    dynnet.main(argv)


if __name__ == "__main__":
    main()
