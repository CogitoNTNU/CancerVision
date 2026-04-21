"""Training entrypoint and runtime helpers."""

from .distributed import RuntimeContext, cleanup, setup_runtime

__all__ = ["RuntimeContext", "cleanup", "setup_runtime"]
