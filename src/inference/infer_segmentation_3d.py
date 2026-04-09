"""Simple 3D segmentation inference script for checkpoints from train_segmentation.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from src.core import resolve_device
from src.data.preprocess import preprocess_image_volume
from src.data.registry import get_dataset_adapter
from src.inference.segmenter import SegmentationInferer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple 3D segmentation inference")
    parser.add_argument(
        "--sample-path",
        required=True,
        help="Path to one sample recognized by the dataset adapter",
    )
    parser.add_argument(
        "--checkpoint",
        default="res/models/best_segmentation_model.pth",
        help="Path to checkpoint produced by train_segmentation.py",
    )
    parser.add_argument(
        "--dataset",
        default="brats",
        help="Dataset adapter name (default: brats)",
    )
    parser.add_argument(
        "--model-backend",
        default=None,
        help="Optional backend override (for legacy checkpoints)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary mask",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Sliding window ROI size",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=4,
        help="Sliding window mini-batch size",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device",
    )
    parser.add_argument(
        "--save-prediction-path",
        default=None,
        help="Optional output path for adapter-specific saved mask",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    adapter = get_dataset_adapter(args.dataset)
    device = resolve_device(args.device)

    segmenter = SegmentationInferer.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_backend=args.model_backend,
        in_channels=adapter.get_input_channels(),
        out_channels=adapter.get_output_channels(),
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        device=device,
    )

    image = adapter.load_inference_image(args.sample_path)
    preprocessed = preprocess_image_volume(image)
    mask = segmenter.predict_mask(preprocessed, threshold=args.threshold)

    print(f"Dataset            : {args.dataset}")
    print(f"Checkpoint         : {checkpoint_path}")
    print(f"Device             : {device}")
    print(f"Backend            : {segmenter.backend_name}")
    print(f"Mask shape         : {tuple(mask.shape)}")
    print(f"Positive voxels    : {int(mask.sum().item())}")

    if args.save_prediction_path:
        adapter.save_prediction_mask(args.sample_path, mask, args.save_prediction_path)
        print(f"Prediction saved   : {args.save_prediction_path}")


if __name__ == "__main__":
    main()
