"""Production-grade segmentation training entrypoint."""

from __future__ import annotations

import argparse
import importlib
from contextlib import nullcontext
import os
import sys
import time
from types import ModuleType

import torch
from torch.amp import GradScaler
from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.model_selection import train_test_split

from src.core import resolve_device, save_checkpoint, set_reproducible
from src.data.registry import get_dataset_adapter
from src.segmentation.registry import get_segmentation_backend


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


def _channel_metric_names(dataset_name: str, out_channels: int) -> list[str]:
    if dataset_name.lower() == "brats" and out_channels == 3:
        return ["tc", "wt", "et"]
    return [f"class_{i}" for i in range(out_channels)]


def _reduce_dims_for_channel_first(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(i for i in range(tensor.ndim) if i != 1)


def _confusion_counts_by_channel(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TP/FP/FN vectors per channel for binary multi-label outputs."""
    reduce_dims = _reduce_dims_for_channel_first(prediction)
    tp = (prediction * target).sum(dim=reduce_dims)
    fp = (prediction * (1.0 - target)).sum(dim=reduce_dims)
    fn = ((1.0 - prediction) * target).sum(dim=reduce_dims)
    return tp, fp, fn


def _resolve_amp_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16

    # auto
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _configure_cuda_performance(device: torch.device, enable_tf32: bool) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.set_float32_matmul_precision("high")

def _get_wandb_module() -> ModuleType | None:
    """Return a functional wandb module or None if unavailable/shadowed."""
    required = ("init", "log", "finish")

    try:
        wandb_module = importlib.import_module("wandb")
        if all(hasattr(wandb_module, attr) for attr in required):
            return wandb_module
    except Exception:
        pass

    # Recover from local-folder shadowing, e.g. repo-level ./wandb run directory.
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


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))

    default_dataset = "brats"
    default_data_dir = get_dataset_adapter(default_dataset).default_data_dir(project_root)

    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--dataset", type=str, default=default_dataset)
    parser.add_argument("--model-backend", type=str, default="monai_unet")
    parser.add_argument("--data-dir", type=str, default=default_data_dir)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(project_root, "res", "models"),
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max-val-steps", type=int, default=0, help="0 means full validation")
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    parser.add_argument("--compile-model", action="store_true", help="Enable torch.compile for the segmentation model")
    parser.add_argument("--compile-mode", type=str, default="max-autotune", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--enable-tf32", action="store_true", help="Enable TF32 acceleration on supported NVIDIA GPUs")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic settings (slower, reproducible)")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cancervision")
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY", "cancervision"))
    return parser.parse_args()


def _maybe_start_wandb(args: argparse.Namespace) -> tuple[bool, ModuleType | None]:
    wandb_module = _get_wandb_module()
    if args.disable_wandb:
        return False, None

    if wandb_module is None:
        return False, None

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("WARNING: WANDB_API_KEY not set; disabling W&B logging.")
        return False, None

    wandb_module.login(key=wandb_api_key)

    wandb_module.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
    )
    return True, wandb_module


def main() -> None:
    _load_env_if_available()
    args = parse_args()
    print_config()

    use_wandb, wandb_module = _maybe_start_wandb(args)
    device = resolve_device(args.device)
    if args.deterministic:
        set_reproducible(args.seed)
    _configure_cuda_performance(device=device, enable_tf32=args.enable_tf32)

    adapter = get_dataset_adapter(args.dataset)
    backend = get_segmentation_backend(args.model_backend)
    amp_dtype = _resolve_amp_dtype(args.precision, device)
    use_amp = amp_dtype is not None
    use_channels_last_3d = device.type == "cuda"

    print(f"Dataset adapter  : {args.dataset}")
    print(f"Model backend    : {args.model_backend}")
    print(f"Data directory   : {args.data_dir}")
    print(f"Device           : {device}")
    print(f"Precision        : {args.precision} ({'AMP' if use_amp else 'fp32'})")

    records = adapter.build_training_records(args.data_dir)
    train_records, val_records = train_test_split(
        records,
        test_size=args.test_size,
        random_state=args.seed,
    )

    roi_size = tuple(args.roi_size)
    train_transform = backend.get_train_transforms(adapter, roi_size=roi_size, num_samples=args.num_samples)
    val_transform = backend.get_val_transforms(adapter)

    train_ds = Dataset(data=train_records, transform=train_transform)
    val_ds = Dataset(data=val_records, transform=val_transform)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
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

    model = backend.build_model(
        in_channels=adapter.get_input_channels(),
        out_channels=adapter.get_output_channels(),
    ).to(device)
    if use_channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    if args.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"Model compiled   : yes ({args.compile_mode})")
        except Exception as exc:
            print(f"WARNING: torch.compile failed ({exc}); continuing without compile.")

    grad_scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and use_amp and amp_dtype == torch.float16),
    )
    channel_names = _channel_metric_names(args.dataset, adapter.get_output_channels())

    loss_function = DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    os.makedirs(args.save_dir, exist_ok=True)
    best_metric = -1.0
    best_metric_epoch = -1

    total_start = time.time()
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        print("-" * 60)
        print(f"Epoch {epoch + 1}/{args.max_epochs}")

        model.train()
        epoch_loss = 0.0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            if use_channels_last_3d:
                inputs = inputs.to(memory_format=torch.channels_last_3d)

            optimizer.zero_grad()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_loss += loss.item()

            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

        lr_scheduler.step()
        epoch_loss /= max(step, 1)
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"train/loss={epoch_loss:.4f} train/lr={current_lr:.2e}")

        if use_wandb:
            wandb_module.log(
                {
                    "train/loss": epoch_loss,
                    "train/lr": current_lr,
                    "train/steps": step,
                    "epoch": epoch + 1,
                }
            )

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_start = time.time()
            val_loss = 0.0
            val_batches = 0
            tp_sum = torch.zeros(adapter.get_output_channels(), device=device)
            fp_sum = torch.zeros(adapter.get_output_channels(), device=device)
            fn_sum = torch.zeros(adapter.get_output_channels(), device=device)
            with torch.no_grad():
                val_step = 0
                for val_data in val_loader:
                    val_step += 1
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    if use_channels_last_3d:
                        val_inputs = val_inputs.to(memory_format=torch.channels_last_3d)
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if use_amp and device.type == "cuda"
                        else nullcontext()
                    )
                    with autocast_ctx:
                        val_logits = sliding_window_inference(
                            val_inputs,
                            roi_size=roi_size,
                            sw_batch_size=4,
                            predictor=model,
                            overlap=0.5,
                        )
                    val_pred = (torch.sigmoid(val_logits) >= 0.5).float()
                    val_outputs = [post_trans(i) for i in decollate_batch(val_logits)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                    val_loss += loss_function(val_logits, val_labels).item()
                    val_batches += 1

                    tp, fp, fn = _confusion_counts_by_channel(val_pred, val_labels)
                    tp_sum += tp
                    fp_sum += fp
                    fn_sum += fn

                    if args.max_val_steps > 0 and val_step >= args.max_val_steps:
                        break

                metric = dice_metric.aggregate().item()
                metric_batch = dice_metric_batch.aggregate()
                metric_batch = metric_batch.detach().cpu()
                dice_metric.reset()
                dice_metric_batch.reset()

                avg_val_loss = val_loss / max(val_batches, 1)
                precision = tp_sum / (tp_sum + fp_sum + 1e-8)
                recall = tp_sum / (tp_sum + fn_sum + 1e-8)
                precision = precision.detach().cpu()
                recall = recall.detach().cpu()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    ckpt_path = os.path.join(args.save_dir, "best_segmentation_model.pth")
                    save_checkpoint(
                        ckpt_path,
                        model=model,
                        metadata={
                            "task": "segmentation",
                            "dataset": args.dataset,
                            "model_backend": args.model_backend,
                            "in_channels": adapter.get_input_channels(),
                            "out_channels": adapter.get_output_channels(),
                            "best_metric": best_metric,
                            "best_metric_epoch": best_metric_epoch,
                            "args": vars(args),
                        },
                    )
                    print(f"Saved checkpoint: {ckpt_path}")

                summary = [f"val/loss={avg_val_loss:.4f}", f"val/dice_mean={metric:.4f}"]
                for idx, name in enumerate(channel_names):
                    summary.append(f"val/dice_{name}={metric_batch[idx].item():.4f}")
                    summary.append(f"val/precision_{name}={precision[idx].item():.4f}")
                    summary.append(f"val/recall_{name}={recall[idx].item():.4f}")
                print(" ".join(summary))

                if use_wandb:
                    log_payload = {
                        "val/loss": avg_val_loss,
                        "val/dice_mean": metric,
                        "val/best_dice": best_metric,
                        "val/time_sec": time.time() - val_start,
                        "epoch": epoch + 1,
                    }
                    for idx, name in enumerate(channel_names):
                        log_payload[f"val/dice_{name}"] = metric_batch[idx].item()
                        log_payload[f"val/precision_{name}"] = precision[idx].item()
                        log_payload[f"val/recall_{name}"] = recall[idx].item()
                    wandb_module.log(log_payload)

        print(f"epoch_time={time.time() - epoch_start:.1f}s")

    total_time = time.time() - total_start
    print("=" * 60)
    print("Training complete")
    print(f"Best mean Dice : {best_metric:.4f} @ epoch {best_metric_epoch}")
    print(f"Total time     : {total_time:.1f}s ({total_time / 3600:.2f}h)")
    print("=" * 60)

    if use_wandb:
        wandb_module.finish()


if __name__ == "__main__":
    main()
