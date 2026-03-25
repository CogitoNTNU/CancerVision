"""Training entrypoints."""

from .train_classifier import main as train_classifier_main
from .train_segmentation_h5 import main as train_segmentation_h5_main
from .train_segmentation import main as train_segmentation_main

__all__ = [
	"train_classifier_main",
	"train_segmentation_main",
	"train_segmentation_h5_main",
]
