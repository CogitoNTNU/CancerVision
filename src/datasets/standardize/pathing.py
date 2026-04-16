"""Cross-platform dataset root resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

WINDOWS_DRIVE_PATTERN = re.compile(r"^(?P<drive>[a-zA-Z]):[\\/](?P<rest>.*)$")


def build_dataset_root_candidates(raw_path: str | Path) -> list[Path]:
    """Return raw path plus common mounted-path fallback for Windows drives."""

    text = str(raw_path).strip().strip('"')
    if not text:
        raise ValueError("Dataset root path cannot be empty.")

    candidates: list[Path] = [Path(text)]
    match = WINDOWS_DRIVE_PATTERN.match(text)
    if match and os.name != "nt":
        windows_path = PureWindowsPath(text)
        mounted = Path("/mnt") / match.group("drive").lower()
        for part in windows_path.parts[1:]:
            mounted /= part
        candidates.append(mounted)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_dataset_root(
    raw_path: str | Path | None,
    *,
    default: str | Path,
) -> Path:
    """Resolve dataset root from explicit or default path."""

    chosen = default if raw_path is None else raw_path
    candidates = build_dataset_root_candidates(chosen)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Dataset root not found. Checked: {checked}")


def resolve_existing_path(raw_path: str | Path) -> Path:
    """Resolve existing file or directory from Windows or mounted path."""

    candidates = build_dataset_root_candidates(raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Path not found. Checked: {checked}")


def resolve_target_path(raw_path: str | Path | None, *, default: str | Path) -> Path:
    """Resolve write target path, preferring mounted `/mnt/<drive>` on non-Windows."""

    chosen = default if raw_path is None else raw_path
    candidates = build_dataset_root_candidates(chosen)
    if os.name == "nt":
        return candidates[0]
    if len(candidates) > 1:
        return candidates[1]
    return candidates[0]
