"""Train 2D segmentation on BraTS2020 archive H5 slices."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType

import h5py
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.core import resolve_device, save_checkpoint, set_reproducible
from src.data.preprocess import preprocess_image_volume


def _load_env_if_available() -> None:
    """Load project .env so training can run with plain python commands."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    env_path = os.path.join(project_root, ".env")
    if not os.path.isfile(env_path):
        return

    try:
        from dotenv import load_dotenv
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        print(f"WARNING: could not import python-dotenv ({exc}); skipping .env load")
        return

    load_dotenv(dotenv_path=env_path, override=False)


def _channel_metric_names(out_channels: int) -> list[str]:
    if out_channels == 3:
        return ["tc", "wt", "et"]
    return [f"class_{i}" for i in range(out_channels)]


def _reduce_dims_for_channel_first(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(i for i in range(tensor.ndim) if i != 1)


def _confusion_counts_by_channel(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reduce_dims = _reduce_dims_for_channel_first(prediction)
    tp = (prediction * target).sum(dim=reduce_dims)
    fp = (prediction * (1.0 - target)).sum(dim=reduce_dims)
    fn = ((1.0 - prediction) * target).sum(dim=reduce_dims)
    return tp, fp, fn

def _get_wandb_module() -> ModuleType | None:
    required = ("init", "log", "finish")

    try:
        wandb_module = importlib.import_module("wandb")
        if all(hasattr(wandb_module, attr) for attr in required):
            return wandb_module
    except Exception:
        pass

    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    removed_paths: list[tuple[int, str]] = []
    for idx in range(len(sys.path) - 1, -1, -1):
        path = sys.path[idx]
        normalized = os.path.normpath(path) if path else path
        if normalized in ("", repo_root):
            removed_paths.append((idx, path))
            sys.path.pop(idx)

    try:
        sys.modules.pop("wandb", None)
        wandb_module = importlib.import_module("wandb")
        if all(hasattr(wandb_module, attr) for attr in required):
            return wandb_module
    except Exception:
        pass
    finally:
        for idx, path in sorted(removed_paths, key=lambda x: x[0]):
            sys.path.insert(idx, path)

    print("WARNING: Failed to import functional 'wandb'; disabling W&B logging.")
    return None


class BratsH5SliceDataset(Dataset):
    """Dataset over H5 slices with keys image(H,W,4), mask(H,W,3)."""

    def __init__(self, files: list[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]
        with h5py.File(file_path, "r") as h:
            image = np.array(h["image"], dtype=np.float32)  # (H,W,4)
            mask = np.array(h["mask"], dtype=np.float32)    # (H,W,3)

        image_t = torch.from_numpy(image).permute(2, 0, 1)  # (4,H,W)
        # Reuse 3D preprocessing pipeline by adding dummy depth dimension.
        image_t = preprocess_image_volume(image_t.unsqueeze(-1)).squeeze(-1)

        mask_t = torch.from_numpy(mask).permute(2, 0, 1)  # (3,H,W)
        return image_t, mask_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D UNet on BraTS H5 slices")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="res/datasets/archive/BraTS2020_training_data",
        help="Directory containing volume_*_slice_*.h5 files",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="res/models",
        help="Where to save checkpoints",
    )
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-train-steps", type=int, default=300)
    parser.add_argument("--max-val-steps", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="cancervision-h5")
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY", "cancervision"))
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def _start_wandb(args: argparse.Namespace) -> tuple[bool, ModuleType | None]:
    wandb_module = _get_wandb_module()
    if args.disable_wandb:
        return False, None

    if wandb_module is None:
        return False, None

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("WARNING: WANDB_API_KEY not set; disabling W&B logging.")
        return False, None

    wandb_module.login(key=api_key)

    wandb_module.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    return True, wandb_module


def main() -> None:
    _load_env_if_available()
    args = parse_args()
    set_reproducible(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(str(p) for p in data_dir.glob("volume_*_slice_*.h5"))
    if not files:
        raise FileNotFoundError(f"No H5 slices found under: {data_dir}")

    train_files, val_files = train_test_split(files, test_size=args.test_size, random_state=args.seed)

    train_ds = BratsH5SliceDataset(train_files)
    val_ds = BratsH5SliceDataset(val_files)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceLoss(squared_pred=True, to_onehot_y=False, sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    channel_names = _channel_metric_names(3)

    use_wandb, wandb_module = _start_wandb(args)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = -1.0
    best_path = os.path.join(args.save_dir, "best_segmentation_h5_2d.pth")

    for epoch in range(args.max_epochs):
        epoch_start = os.times().elapsed
        model.train()
        train_loss = 0.0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if step >= args.max_train_steps:
                break

        avg_train_loss = train_loss / max(min(len(train_loader), args.max_train_steps), 1)

        model.eval()
        dice_metric.reset()
        dice_metric_batch.reset()
        val_loss = 0.0
        val_batches = 0
        tp_sum = torch.zeros(3, device=device)
        fp_sum = torch.zeros(3, device=device)
        fn_sum = torch.zeros(3, device=device)
        with torch.no_grad():
            for step, (x, y) in enumerate(val_loader, start=1):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = (torch.sigmoid(logits) >= 0.5).float()
                dice_metric(y_pred=pred, y=y)
                dice_metric_batch(y_pred=pred, y=y)
                val_loss += loss_fn(logits, y).item()
                val_batches += 1

                tp, fp, fn = _confusion_counts_by_channel(pred, y)
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
                if step >= args.max_val_steps:
                    break

        val_dice = float(dice_metric.aggregate().item())
        val_dice_batch = dice_metric_batch.aggregate().detach().cpu()
        avg_val_loss = val_loss / max(val_batches, 1)
        precision = (tp_sum / (tp_sum + fp_sum + 1e-8)).detach().cpu()
        recall = (tp_sum / (tp_sum + fn_sum + 1e-8)).detach().cpu()

        summary = [
            f"Epoch {epoch + 1}/{args.max_epochs}",
            f"train_loss={avg_train_loss:.4f}",
            f"val_loss={avg_val_loss:.4f}",
            f"val_dice={val_dice:.4f}",
        ]
        for idx, name in enumerate(channel_names):
            summary.append(f"val_dice_{name}={val_dice_batch[idx].item():.4f}")
            summary.append(f"val_precision_{name}={precision[idx].item():.4f}")
            summary.append(f"val_recall_{name}={recall[idx].item():.4f}")
        summary.append(f"epoch_time={os.times().elapsed - epoch_start:.1f}s")
        print(" ".join(summary))

        if use_wandb:
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/dice_mean": val_dice,
            }
            for idx, name in enumerate(channel_names):
                log_payload[f"val/dice_{name}"] = val_dice_batch[idx].item()
                log_payload[f"val/precision_{name}"] = precision[idx].item()
                log_payload[f"val/recall_{name}"] = recall[idx].item()
            wandb_module.log(log_payload)

        if val_dice > best_val:
            best_val = val_dice
            save_checkpoint(
                best_path,
                model=model,
                metadata={
                    "task": "segmentation_h5_2d",
                    "dataset": "brats2020_archive_h5",
                    "model_backend": "monai_unet_2d",
                    "in_channels": 4,
                    "out_channels": 3,
                    "best_val_dice": best_val,
                    "args": vars(args),
                },
            )
            print(f"Saved checkpoint: {best_path}")

    if use_wandb:
        wandb_module.finish()


if __name__ == "__main__":
    main()
