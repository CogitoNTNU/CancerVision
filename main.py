"""Project CLI entrypoint."""

from __future__ import annotations

import argparse
import os
import sys

from src.classifier import HeuristicTumorClassifier, load_torch_classifier
from src.data.discovery import discover_datasets
from src.data.registry import get_dataset_adapter, list_dataset_types
from src.inference import InferenceService, SegmentationInferer
from src.segmentation import list_segmentation_backends
from src.training.train_classifier import main as train_classifier_main
from src.training.train_segmentation_h5 import main as train_segmentation_h5_main
from src.training.train_segmentation import main as train_segmentation_main


def _forward_args(target, forwarded_args: list[str]) -> None:
    """Forward arguments to module-level main() parsers."""
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *forwarded_args]
        target()
    finally:
        sys.argv = old_argv


def _parse_main_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="CancerVision pipeline commands",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "train-segmentation",
        help="Train the segmentation model",
    )
    subparsers.add_parser(
        "train-classifier",
        help="Train the tumor presence classifier",
    )
    subparsers.add_parser(
        "train-segmentation-h5",
        help="Train 2D segmentation model on BraTS archive H5 slices",
    )

    datasets_parser = subparsers.add_parser(
        "datasets",
        help="List supported dataset types or discover dataset roots",
    )
    datasets_parser.add_argument(
        "--search-dir",
        default=None,
        help="Optional directory to scan for known datasets",
    )
    datasets_parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Max folder depth for dataset discovery",
    )

    subparsers.add_parser(
        "models",
        help="List supported segmentation model backends",
    )

    subparsers.add_parser(
        "web",
        help="Run the browser-based inference interface",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run classifier-gated segmentation inference",
    )
    infer_parser.add_argument(
        "--dataset",
        default="brats",
        help="Dataset adapter name",
    )
    infer_parser.add_argument(
        "--sample-path",
        required=True,
        help="Path to one dataset sample directory/file understood by the adapter",
    )
    infer_parser.add_argument(
        "--segmentation-checkpoint",
        default=os.path.join("res", "models", "best_segmentation_model.pth"),
        help="Path to segmentation checkpoint",
    )
    infer_parser.add_argument(
        "--model-backend",
        default=None,
        help="Optional segmentation backend override for loading legacy checkpoints",
    )
    infer_parser.add_argument(
        "--classifier-checkpoint",
        default=None,
        help="Optional classifier checkpoint; falls back to heuristic classifier",
    )
    infer_parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.5,
        help="Threshold for deciding if segmentation should run",
    )
    infer_parser.add_argument(
        "--save-prediction-path",
        default=None,
        help="Optional path to save predicted segmentation mask in adapter format",
    )

    return parser.parse_known_args()


def _run_inference(args: argparse.Namespace) -> None:
    adapter = get_dataset_adapter(args.dataset)

    if args.classifier_checkpoint:
        classifier = load_torch_classifier(args.classifier_checkpoint)
        classifier_name = "torch"
    else:
        classifier = HeuristicTumorClassifier()
        classifier_name = "heuristic"

    segmenter = SegmentationInferer.from_checkpoint(
        args.segmentation_checkpoint,
        model_backend=args.model_backend,
        in_channels=adapter.get_input_channels(),
        out_channels=adapter.get_output_channels(),
    )
    service = InferenceService(
        classifier=classifier,
        segmenter=segmenter,
        classifier_threshold=args.classifier_threshold,
    )

    result = service.run_sample(adapter, args.sample_path)

    print(f"Dataset type         : {args.dataset}")
    print(f"Classifier type      : {classifier_name}")
    print(f"Tumor probability    : {result.tumor_probability:.4f}")
    print(f"Tumor detected       : {result.has_tumor}")

    if result.segmentation_mask is None:
        print("Segmentation skipped : classifier below threshold")
        return

    print("Segmentation skipped : no")
    print(f"Mask shape           : {tuple(result.segmentation_mask.shape)}")
    print(f"Positive voxels      : {int(result.segmentation_mask.sum().item())}")

    if args.save_prediction_path:
        service.save_prediction(
            dataset_adapter=adapter,
            sample_path=args.sample_path,
            result=result,
            output_path=args.save_prediction_path,
        )
        print(f"Prediction saved     : {args.save_prediction_path}")


def _run_datasets_command(args: argparse.Namespace) -> None:
    available = list_dataset_types()
    print("Supported dataset adapters:")
    for dataset_type in available:
        print(f"- {dataset_type}")

    if args.search_dir:
        matches = discover_datasets(args.search_dir, max_depth=args.max_depth)
        print(f"\nDiscovered datasets under: {args.search_dir}")
        if not matches:
            print("- none found")
            return
        for match in matches:
            print(f"- {match.dataset_type}: {match.path}")


def _run_models_command() -> None:
    print("Supported segmentation backends:")
    for backend_name in list_segmentation_backends():
        print(f"- {backend_name}")


def main() -> None:
    args, forwarded_args = _parse_main_args()

    if args.command is None:
        print("No command specified. Use one of: datasets, models, train-segmentation, train-classifier, infer")
        return

    if args.command == "train-segmentation":
        _forward_args(train_segmentation_main, forwarded_args)
        return

    if args.command == "train-classifier":
        _forward_args(train_classifier_main, forwarded_args)
        return

    if args.command == "train-segmentation-h5":
        _forward_args(train_segmentation_h5_main, forwarded_args)
        return

    if args.command == "datasets":
        _run_datasets_command(args)
        return

    if args.command == "models":
        _run_models_command()
        return

    if args.command == "web":
        from src.web.app import main as web_main

        _forward_args(web_main, forwarded_args)
        return

    if args.command == "infer":
        _run_inference(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()