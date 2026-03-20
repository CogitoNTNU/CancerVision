#!/usr/bin/env python
"""Standalone BraTS2020 3D U-Net training script for SLURM / SSH execution.

Trains a 3D U-Net on BraTS2020 NIfTI volumes (flair, t1, t1ce, t2 + seg).
Uses MONAI transforms, sliding-window inference, and per-channel Dice tracking.

Usage:
    python train_brats.py [OPTIONS]
"""

import argparse
import logging
import os
import sys
import time

import torch
import wandb
from sklearn.model_selection import train_test_split

from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.utils import set_determinism

# ---------------------------------------------------------------------------
# Ensure the src package is importable (needed for custom transforms)
# ---------------------------------------------------------------------------
_src_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from datasets import ConvertToMultiChannelBasedOnBratsClassesd  # noqa: E402

# ---------------------------------------------------------------------------
# Load config from .env if available
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

_project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_dotenv_path = os.path.join(_project_root, ".env")
_dotenv_exists = os.path.isfile(_dotenv_path)
if _dotenv_exists:
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb_mode = (os.getenv("WANDB_MODE") or "").strip().lower()
if wandb_mode not in {"", "online", "offline", "disabled"}:
    raise ValueError(
        "Invalid WANDB_MODE. Use one of: online, offline, disabled."
    )


def configure_logging() -> logging.Logger:
    """Configure root logging once and return a module logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def init_wandb(args, logger: logging.Logger) -> bool:
    """Initialize W&B safely.

    Returns True if W&B is active for this run, False otherwise.
    """
    if not wandb_api_key and wandb_mode == "":
        effective_wandb_mode = "disabled"
    else:
        effective_wandb_mode = wandb_mode or "online"

    if effective_wandb_mode == "disabled":
        logger.info("W&B disabled for this run.")
        return False

    try:
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            # Allow anonymous logging when no key is provided but W&B is enabled.
            wandb.login(anonymous="allow")
    except Exception as exc:
        logger.warning(
            "W&B login failed (%s). Continuing with W&B disabled.", exc
        )
        return False

    try:
        wandb.init(
            project="cancervision",
            entity="dev_cogito_ntnu",
            mode=effective_wandb_mode,
            config={
                "max_epochs": args.max_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "val_interval": args.val_interval,
                "seed": args.seed,
                "roi_size": args.roi_size,
                "num_samples": args.num_samples,
                "test_size": args.test_size,
            },
        )
        logger.info("W&B initialized in %s mode.", effective_wandb_mode)
        return True
    except Exception as exc:
        logger.warning(
            "W&B init failed (%s). Continuing with W&B disabled.", exc
        )
        return False


def log_wandb(wandb_enabled: bool, payload: dict) -> None:
    """Log to W&B only when initialized."""
    if wandb_enabled:
        wandb.log(payload)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_nifti(directory: str, pattern: str) -> str:
    """Find a NIfTI file matching *pattern* inside *directory*.

    Checks for both .nii and .nii.gz extensions and returns the first match.
    """
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(directory, pattern + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {directory}"
    )


def build_data_dicts(data_dir: str):
    """Scan BraTS2020 patient folders and return a list of data dicts.

    Each dict has keys: "image" (list of 4 modality paths) and "label" (seg path).
    """
    data_dicts = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    patient_dirs = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("BraTS20_Training_")
    )

    for patient_name in patient_dirs:
        patient_path = os.path.join(data_dir, patient_name)
        try:
            flair = find_nifti(patient_path, f"{patient_name}_flair")
            t1 = find_nifti(patient_path, f"{patient_name}_t1")
            t1ce = find_nifti(patient_path, f"{patient_name}_t1ce")
            t2 = find_nifti(patient_path, f"{patient_name}_t2")
            seg = find_nifti(patient_path, f"{patient_name}_seg")
        except FileNotFoundError as exc:
            print(f"WARNING: skipping {patient_name} -- {exc}")
            continue

        data_dicts.append(
            {
                "image": [flair, t1, t1ce, t2],
                "label": seg,
            }
        )

    if len(data_dicts) == 0:
        raise FileNotFoundError(
            f"No valid BraTS patient folders found in {data_dir}"
        )

    return data_dicts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Train 3D U-Net on BraTS2020 NIfTI volumes"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.normpath(
            os.path.join(
                script_dir,
                "..",
                "..",
                "res",
                "data",
                "dataset",
                # "archive",
                "BraTS2020_TrainingData",
                "MICCAI_BraTS2020_TrainingData",
            )
        ),
        help="Path to MICCAI_BraTS2020_TrainingData folder containing patient dirs",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.normpath(
            os.path.join(script_dir, "..", "..", "res", "models")
        ),
        help="Directory to save the best model checkpoint",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Training batch size (1 for 3D)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--val-interval", type=int, default=1, help="Validate every N epochs"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (0 = main process)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of patients for validation",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Random crop ROI size (3 ints, e.g. 128 128 128)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of random crops per volume per iteration",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transforms(roi_size, num_samples):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    logger = configure_logging()
    print_config()

    logger.info(".env path      : %s", _dotenv_path)
    logger.info(".env exists    : %s", _dotenv_exists)
    logger.info("WANDB_API_KEY  : %s", "present" if wandb_api_key else "missing")
    logger.info("WANDB_MODE     : %s", wandb_mode or "<unset>")

    # W&B experiment tracking (safe initialization)
    wandb_enabled = init_wandb(args, logger)

    # Reproducibility
    set_determinism(seed=args.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = os.path.normpath(args.data_dir)
    logger.info("Data directory : %s", data_dir)
    logger.info("Exists         : %s", os.path.isdir(data_dir))

    data_dicts = build_data_dicts(data_dir)
    logger.info("Total patients : %d", len(data_dicts))

    # 80/20 patient-level split
    train_dicts, val_dicts = train_test_split(
        data_dicts, test_size=args.test_size, random_state=42
    )
    logger.info("Train patients : %d", len(train_dicts))
    logger.info("Val patients   : %d", len(val_dicts))

    roi_size = tuple(args.roi_size)
    train_transform = get_train_transforms(roi_size, args.num_samples)
    val_transform = get_val_transforms()

    train_ds = Dataset(data=train_dicts, transform=train_transform)
    val_ds = Dataset(data=val_dicts, transform=val_transform)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    logger.info("DataLoader num_workers: %d", num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # ------------------------------------------------------------------
    # Model / loss / optimizer
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    ).to(device)

    loss_function = DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    save_dir = os.path.normpath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_metric = -1.0
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        logger.info("%s", "-" * 40)
        logger.info("Epoch %d/%d", epoch + 1, args.max_epochs)

        # ---- train ----
        model.train()
        epoch_loss = 0.0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                logger.info(
                    "  step %d/%d  train_loss: %.4f  step_time: %.4fs",
                    step,
                    len(train_ds) // train_loader.batch_size,
                    loss.item(),
                    time.time() - step_start,
                )

        lr_scheduler.step()
        epoch_loss /= max(step, 1)
        epoch_loss_values.append(epoch_loss)
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.info("  avg loss: %.4f  lr: %.2e", epoch_loss, current_lr)

        log_wandb(
            wandb_enabled,
            {
                "train/loss": epoch_loss,
                "train/lr": current_lr,
                "epoch": epoch + 1,
            },
        )

        # ---- validate ----
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = sliding_window_inference(
                        val_inputs,
                        roi_size=roi_size,
                        sw_batch_size=4,
                        predictor=model,
                        overlap=0.5,
                    )
                    val_outputs = [
                        post_trans(i) for i in decollate_batch(val_outputs)
                    ]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
                metric_values_tc.append(metric_tc)
                metric_values_wt.append(metric_wt)
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    ckpt_path = os.path.join(save_dir, "best_metric_model.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info("  -> saved new best model to %s", ckpt_path)

                log_wandb(
                    wandb_enabled,
                    {
                        "val/dice_mean": metric,
                        "val/dice_tc": metric_tc,
                        "val/dice_wt": metric_wt,
                        "val/dice_et": metric_et,
                        "val/best_dice": best_metric,
                        "epoch": epoch + 1,
                    },
                )

                logger.info(
                    "  val dice: %.4f  (TC=%.4f  WT=%.4f  ET=%.4f)",
                    metric,
                    metric_tc,
                    metric_wt,
                    metric_et,
                )
                logger.info(
                    "  best dice: %.4f @ epoch %d", best_metric, best_metric_epoch
                )

        logger.info("  epoch time: %.1fs", time.time() - epoch_start)

    total_time = time.time() - total_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("%s", "=" * 60)
    logger.info("Training complete")
    logger.info("  Best mean Dice : %.4f @ epoch %d", best_metric, best_metric_epoch)
    if metric_values_tc:
        logger.info("  Best TC Dice   : %.4f", max(metric_values_tc))
        logger.info("  Best WT Dice   : %.4f", max(metric_values_wt))
        logger.info("  Best ET Dice   : %.4f", max(metric_values_et))
    logger.info("  Total time     : %.1fs (%.2fh)", total_time, total_time / 3600)
    logger.info("  Checkpoint     : %s", os.path.join(save_dir, "best_metric_model.pth"))
    logger.info("%s", "=" * 60)

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
