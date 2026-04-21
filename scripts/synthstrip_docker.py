#!/usr/bin/env python3
"""Run the official freesurfer/synthstrip Docker image efficiently."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


IMAGE = os.environ.get("SYNTHSTRIP_DOCKER_IMAGE", "freesurfer/synthstrip:1.8")
DEFAULT_SESSION_NAME = "cancervision-synthstrip"
SESSION_ENV = "CANCERVISION_SYNTHSTRIP_SESSION_NAME"
WORKSPACE_ENV = "CANCERVISION_SYNTHSTRIP_WORKSPACE_ROOT"
PATH_FLAGS = {"-i", "--input", "-o", "--output", "-m", "--mask", "-d", "--sdt", "--model"}
DOCKER_BACKEND_ENV = "CANCERVISION_DOCKER_BACKEND"


def _workspace_root() -> Path:
    configured = os.environ.get(WORKSPACE_ENV)
    if configured:
        root = Path(configured)
    else:
        root = Path(tempfile.gettempdir()) / "cancervision-synthstrip"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _session_name() -> str:
    return os.environ.get(SESSION_ENV, DEFAULT_SESSION_NAME)


def _windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise RuntimeError(f"Cannot convert path to WSL mount: {resolved}")
    tail = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _detect_docker_backend() -> str:
    configured = os.environ.get(DOCKER_BACKEND_ENV, "").strip().lower()
    if configured in {"native", "wsl"}:
        return configured

    if os.name != "nt":
        return "native"

    native = subprocess.run(
        ["docker", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if native.returncode == 0:
        return "native"

    wsl = subprocess.run(
        ["wsl.exe", "docker", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if wsl.returncode == 0:
        return "wsl"

    return "native"


def _docker_host_path(path: Path) -> str:
    if _detect_docker_backend() == "wsl":
        return _windows_to_wsl_path(path)
    return str(path)


def _docker(*args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    backend = _detect_docker_backend()
    command = ["docker", *args]
    if backend == "wsl":
        command = ["wsl.exe", "docker", *args]
    return subprocess.run(
        command,
        check=False,
        capture_output=capture_output,
        text=True,
    )


def _session_running(name: str) -> bool:
    result = _docker("inspect", "-f", "{{.State.Running}}", name, capture_output=True)
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _session_exists(name: str) -> bool:
    result = _docker("inspect", name, capture_output=True)
    return result.returncode == 0


def _start_session() -> int:
    if shutil.which("docker") is None and not (
        os.name == "nt" and shutil.which("wsl.exe") is not None
    ):
        print("Cannot find docker in PATH.", file=sys.stderr)
        return 1

    name = _session_name()
    workspace_root = _workspace_root()

    if _session_running(name):
        return 0

    if _session_exists(name):
        remove_result = _docker("rm", "-f", name, capture_output=True)
        if remove_result.returncode != 0:
            detail = remove_result.stderr.strip() or remove_result.stdout.strip()
            if detail:
                print(detail, file=sys.stderr)
            return int(remove_result.returncode)

    result = _docker(
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "--entrypoint",
        "/bin/sh",
        "-v",
        f"{_docker_host_path(workspace_root)}:/workspace",
        "-w",
        "/workspace",
        IMAGE,
        "-lc",
        "while true; do /bin/sleep 3600; done",
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if detail:
            print(detail, file=sys.stderr)
        return int(result.returncode)

    if not _session_running(name):
        inspect_result = _docker("logs", name, capture_output=True)
        detail = inspect_result.stderr.strip() or inspect_result.stdout.strip()
        if detail:
            print(detail, file=sys.stderr)
        else:
            print("Persistent SynthStrip container failed to stay running.", file=sys.stderr)
        return 1

    return 0


def _stop_session() -> int:
    if shutil.which("docker") is None and not (
        os.name == "nt" and shutil.which("wsl.exe") is not None
    ):
        print("Cannot find docker in PATH.", file=sys.stderr)
        return 1

    name = _session_name()
    if not _session_exists(name):
        return 0

    result = _docker("rm", "-f", name, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if detail:
            print(detail, file=sys.stderr)
    return int(result.returncode)


def _containerize_paths(argv: list[str], workspace_root: Path) -> list[str]:
    args: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        args.append(arg)
        if arg in PATH_FLAGS:
            idx += 1
            if idx >= len(argv):
                raise SystemExit(f"Missing value after {arg}")
            host_path = Path(argv[idx]).expanduser().resolve()
            try:
                relative_path = host_path.relative_to(workspace_root)
            except ValueError as exc:
                raise SystemExit(
                    f"Path {host_path} is outside shared SynthStrip workspace {workspace_root}."
                ) from exc
            args.append((Path("/workspace") / relative_path).as_posix())
        idx += 1
    return args


def _run_synthstrip(argv: list[str]) -> int:
    if shutil.which("docker") is None and not (
        os.name == "nt" and shutil.which("wsl.exe") is not None
    ):
        print("Cannot find docker in PATH.", file=sys.stderr)
        return 1

    workspace_root = _workspace_root()
    name = _session_name()
    persistent_session = SESSION_ENV in os.environ
    auto_started = False

    if not _session_running(name):
        start_code = _start_session()
        if start_code != 0:
            return start_code
        auto_started = True

    container_args = _containerize_paths(argv, workspace_root)
    result = _docker("exec", name, "/freesurfer/mri_synthstrip", *container_args)

    if auto_started and not persistent_session:
        _stop_session()

    return int(result.returncode)


def main(argv: list[str]) -> int:
    if argv == ["--start-session"]:
        return _start_session()
    if argv == ["--stop-session"]:
        return _stop_session()
    return _run_synthstrip(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
