#!/usr/bin/env python
"""Compatibility wrapper for legacy BraTS training entrypoint.

Use `src.training.train_segmentation` for new development.
"""

import os
import sys

_project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.training.train_segmentation import main


if __name__ == "__main__":
    main()
