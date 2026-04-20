"""Training loop and checkpoint helpers for the DynUNet trainer."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

from src.models.dynnet_config import (
    DEFAULT_GPU_PROFILE_NAME,
    GPU_PROFILE_CONFIGS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_run_name,
    parse_args,
    validate_args,
)
from src.models.dynnet_data import (
    build_dataset_splits,
    get_dataset_config,
    infer_cancervision_path_prefix_maps,
    resolve_cancervision_task_manifest_path,
)
from src.models.dynnet_runtime import (
    RuntimeContext,
    apply_gpu_profile_defaults,
    build_micro_batch_slices,
    cleanup_distributed,
    format_cuda_memory_summary,
    get_autocast_context,
    is_cuda_oom_error,
    is_main_process,
    rank0_print,
    reduce_mean,
    setup_device_and_distributed,
    should_use_amp,
    synchronize,
)


class NoOpWandbRun:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


DEFAULT_VALIDATION_THRESHOLD = 0.5


def _build_wandb_init_kwargs(config: dict[str, Any], mode: str) -> dict[str, Any]:
    init_kwargs: dict[str, Any] = {
        "project": WANDB_PROJECT,
        "config": config,
        "name": config["run_name"],
        "mode": mode,
    }
    if WANDB_ENTITY:
        init_kwargs["entity"] = WANDB_ENTITY
    return init_kwargs


def _format_wandb_target() -> str:
    if WANDB_ENTITY:
        return f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    return WANDB_PROJECT


def _format_wandb_init_error_hint(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "personal entities are disabled" in lowered or "permission_error" in lowered:
        if WANDB_ENTITY:
            return (
                f" WANDB_ENTITY={WANDB_ENTITY!r} looks wrong for this account. "
                "Use team slug or unset WANDB_ENTITY."
            )
        return " Set WANDB_ENTITY to team slug for shared workspace."
    return ""


def resolve_validation_thresholds(args: argparse.Namespace) -> tuple[float, ...]:
    if not args.val_thresholds:
        return (DEFAULT_VALIDATION_THRESHOLD,)

    seen: set[float] = set()
    thresholds: list[float] = []
    for threshold in args.val_thresholds:
        normalized = round(float(threshold), 6)
        if normalized in seen:
            continue
        seen.add(normalized)
        thresholds.append(normalized)
    return tuple(thresholds)


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


def build_loss_function(args: argparse.Namespace) -> torch.nn.Module:
    if args.loss == "dice":
        return DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )
    if args.loss == "dicece":
        return DiceCELoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
            lambda_dice=args.dicece_lambda_dice,
            lambda_ce=args.dicece_lambda_ce,
        )
    raise ValueError(f"Unsupported loss: {args.loss}")


def summarize_threshold_metrics(
    *,
    threshold_results: dict[float, dict[str, float]],
    metric_names: Sequence[str],
    include_per_threshold_metrics: bool,
) -> dict[str, float]:
    best_threshold = next(iter(threshold_results))
    best_metrics = threshold_results[best_threshold]
    for threshold, metrics in threshold_results.items():
        if metrics["dice_mean"] > best_metrics["dice_mean"]:
            best_threshold = threshold
            best_metrics = metrics

    summarized = {
        "dice_mean": best_metrics["dice_mean"],
        "selected_threshold": best_threshold,
    }
    for metric_name in metric_names:
        summarized[f"dice_{metric_name}"] = best_metrics[f"dice_{metric_name}"]

    if include_per_threshold_metrics:
        for threshold, metrics in threshold_results.items():
            suffix = _threshold_key(threshold)
            summarized[f"dice_mean_threshold_{suffix}"] = metrics["dice_mean"]
            for metric_name in metric_names:
                summarized[f"dice_{metric_name}_threshold_{suffix}"] = metrics[
                    f"dice_{metric_name}"
                ]

    return summarized


def build_model(*, in_channels: int = 4, out_channels: int = 3) -> DynUNet:
    return build_model_with_filters(
        in_channels=in_channels,
        out_channels=out_channels,
        filters=GPU_PROFILE_CONFIGS[DEFAULT_GPU_PROFILE_NAME].model_filters,
    )


def build_model_with_filters(
    *,
    in_channels: int = 4,
    out_channels: int = 3,
    filters: Sequence[int],
) -> DynUNet:
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=list(filters),
        dropout=0.2,
        res_block=True,
        deep_supervision=False,
    )


def maybe_init_wandb(
    args: argparse.Namespace,
    context: RuntimeContext,
    config: dict[str, Any],
) -> Any:
    if not is_main_process(context) or args.wandb_mode == "disabled":
        return NoOpWandbRun()

    try:
        import wandb
    except Exception:
        print("W&B unavailable in this environment; continuing without it.", flush=True)
        return NoOpWandbRun()

    mode = args.wandb_mode
    api_key = os.getenv("WANDB_API_KEY")
    if mode == "online" and not api_key:
        print("WANDB_API_KEY missing; falling back to offline mode.", flush=True)
        mode = "offline"

    print(f"W&B target: {_format_wandb_target()}  mode: {mode}", flush=True)

    try:
        if mode == "online" and api_key:
            wandb.login(key=api_key, relogin=True)
        run = wandb.init(**_build_wandb_init_kwargs(config, mode))
        return run if run is not None else NoOpWandbRun()
    except Exception as exc:
        hint = _format_wandb_init_error_hint(exc)
        if mode == "online":
            print(
                f"W&B online init failed ({exc}).{hint} Retrying offline in local wandb/ dir.",
                flush=True,
            )
            try:
                run = wandb.init(**_build_wandb_init_kwargs(config, "offline"))
                return run if run is not None else NoOpWandbRun()
            except Exception as offline_exc:
                print(
                    f"W&B offline init failed ({offline_exc}); continuing without it.",
                    flush=True,
                )
                return NoOpWandbRun()
        print(f"W&B init failed ({exc}).{hint} Continuing without it.", flush=True)
        return NoOpWandbRun()


def load_resume_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    resume_path: str | None,
    context: RuntimeContext,
) -> tuple[int, float, int, float, int]:
    if not resume_path:
        return 0, -1.0, -1, DEFAULT_VALIDATION_THRESHOLD, 0

    path = Path(resume_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(path, map_location=context.device)
    model_to_load = model

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_to_load.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
        best_metric_epoch = int(checkpoint.get("best_metric_epoch", -1))
        best_threshold = float(
            checkpoint.get("best_threshold", DEFAULT_VALIDATION_THRESHOLD)
        )
        epochs_since_improve = int(checkpoint.get("epochs_since_improve", 0))
        return (
            start_epoch,
            best_metric,
            best_metric_epoch,
            best_threshold,
            epochs_since_improve,
        )

    model_to_load.load_state_dict(checkpoint)
    return 0, -1.0, -1, DEFAULT_VALIDATION_THRESHOLD, 0


def save_last_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    best_metric_epoch: int,
    best_threshold: float,
    epochs_since_improve: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "best_threshold": best_threshold,
        "epochs_since_improve": epochs_since_improve,
    }
    torch.save(checkpoint, run_dir / "last_checkpoint.pt")


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    context: RuntimeContext,
) -> tuple[float, float]:
    del epoch
    model.train()
    epoch_loss = 0.0
    step_count = 0

    for step, batch_data in enumerate(train_loader, start=1):
        step_start = time.time()
        step_count += 1
        inputs = batch_data["image"]
        labels = batch_data["label"]
        micro_batch_slices = build_micro_batch_slices(
            total_batch_size=inputs.shape[0],
            requested_micro_batch_size=args.train_micro_batch_size,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = None
        step_loss = 0.0
        for batch_slice in micro_batch_slices:
            micro_inputs = inputs[batch_slice].to(context.device, non_blocking=True)
            micro_labels = labels[batch_slice].to(context.device, non_blocking=True)
            micro_weight = micro_inputs.shape[0] / inputs.shape[0]

            with get_autocast_context(context, should_use_amp(args, context)):
                outputs = model(micro_inputs)
                micro_loss = loss_function(outputs, micro_labels)
                scaled_loss = micro_loss * micro_weight

            step_loss += micro_loss.item() * micro_weight
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            loss = micro_loss

        if loss is None:
            raise RuntimeError("Training batch produced no micro-batches.")

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        epoch_loss += step_loss
        if is_main_process(context) and step % 10 == 0:
            print(
                f"  step {step}/{len(train_loader)}"
                f"  train_loss: {step_loss:.4f}"
                f"  step_time: {time.time() - step_start:.2f}s",
                flush=True,
            )

    scheduler.step()
    mean_epoch_loss = reduce_mean(epoch_loss, step_count, context)
    current_lr = scheduler.get_last_lr()[0]
    return mean_epoch_loss, current_lr


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader | None,
    roi_size: Sequence[int],
    args: argparse.Namespace,
    context: RuntimeContext,
    dataset_config: Any,
) -> dict[str, float] | None:
    if not is_main_process(context) or val_loader is None:
        return None

    thresholds = resolve_validation_thresholds(args)
    dice_metrics = {
        threshold: DiceMetric(include_background=True, reduction="mean")
        for threshold in thresholds
    }
    dice_metrics_batch = {
        threshold: DiceMetric(include_background=True, reduction="mean_batch")
        for threshold in thresholds
    }
    threshold_transforms = {
        threshold: AsDiscrete(threshold=threshold) for threshold in thresholds
    }
    model.eval()

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(context.device, non_blocking=True)
            val_labels = val_data["label"].to(context.device, non_blocking=True)
            with get_autocast_context(context, should_use_amp(args, context)):
                val_outputs = sliding_window_inference(
                    val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=args.val_sw_batch_size,
                    predictor=model,
                    overlap=0.5,
                )
            val_prob_outputs_list = [
                torch.sigmoid(item) for item in decollate_batch(val_outputs)
            ]
            val_labels_list = decollate_batch(val_labels)
            for threshold in thresholds:
                val_outputs_list = [
                    threshold_transforms[threshold](item)
                    for item in val_prob_outputs_list
                ]
                dice_metrics[threshold](y_pred=val_outputs_list, y=val_labels_list)
                dice_metrics_batch[threshold](y_pred=val_outputs_list, y=val_labels_list)

    threshold_results: dict[float, dict[str, float]] = {}
    for threshold in thresholds:
        metric = dice_metrics[threshold].aggregate().item()
        metric_batch = dice_metrics_batch[threshold].aggregate()
        dice_metrics[threshold].reset()
        dice_metrics_batch[threshold].reset()

        threshold_metrics = {"dice_mean": metric}
        for index, metric_name in enumerate(dataset_config.metric_names):
            threshold_metrics[f"dice_{metric_name}"] = metric_batch[index].item()
        threshold_results[threshold] = threshold_metrics

    return summarize_threshold_metrics(
        threshold_results=threshold_results,
        metric_names=dataset_config.metric_names,
        include_per_threshold_metrics=args.val_thresholds is not None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = setup_device_and_distributed()
    if context.device.type == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()
    gpu_profile = apply_gpu_profile_defaults(args, context)
    validate_args(args)

    run_name = build_run_name(args)
    run_dir = Path(args.save_dir).resolve() / run_name
    if is_main_process(context):
        run_dir.mkdir(parents=True, exist_ok=True)
    synchronize(context)

    wandb_run: Any = NoOpWandbRun()
    total_start = time.time()
    stop_reason = "max_epochs reached"
    try:
        if is_main_process(context):
            print_config()
        set_determinism(seed=args.seed)
        dataset_config = get_dataset_config(args.dataset_source)
        resolved_task_manifest = ""
        requested_task_manifest = ""
        path_prefix_maps: list[str] = []
        validation_thresholds = resolve_validation_thresholds(args)

        rank0_print(context, f"Using device: {context.device}")
        rank0_print(
            context,
            f"Distributed: {context.distributed} (world_size={context.world_size})",
        )
        rank0_print(context, f"GPU profile    : {gpu_profile.name}")
        rank0_print(context, f"Run directory: {run_dir}")
        rank0_print(context, f"Dataset source : {args.dataset_source}")
        if args.dataset_source == "brats":
            data_dir = os.path.normpath(args.data_dir)
            rank0_print(context, f"Data directory : {data_dir}")
            rank0_print(context, f"Exists         : {os.path.isdir(data_dir)}")
        else:
            requested_task_manifest = os.path.normpath(args.task_manifest)
            resolved_task_manifest = os.path.normpath(
                str(
                    resolve_cancervision_task_manifest_path(
                        args.task_manifest,
                        warn_on_fallback=False,
                    )
                )
            )
            path_prefix_maps = infer_cancervision_path_prefix_maps(args.path_prefix_map)
            rank0_print(context, f"Task manifest  : {resolved_task_manifest}")
            rank0_print(context, f"Exists         : {os.path.isfile(resolved_task_manifest)}")
            if resolved_task_manifest != requested_task_manifest:
                rank0_print(context, f"Requested task : {requested_task_manifest}")
            if path_prefix_maps:
                rank0_print(context, f"Path remaps    : {path_prefix_maps}")
        rank0_print(context, f"Train micro-batch size: {args.train_micro_batch_size}")
        rank0_print(context, f"Validation sw_batch_size: {args.val_sw_batch_size}")
        rank0_print(context, f"ROI size       : {tuple(args.roi_size)}")
        rank0_print(context, f"Num samples    : {args.num_samples}")
        rank0_print(context, f"Model filters  : {tuple(args.model_filters)}")
        rank0_print(
            context,
            f"Crop weights   : pos={args.crop_pos_weight:.2f} neg={args.crop_neg_weight:.2f}",
        )
        rank0_print(context, f"Loss           : {args.loss}")
        if args.loss == "dicece":
            rank0_print(
                context,
                "Loss lambdas   : "
                f"dice={args.dicece_lambda_dice:.2f} ce={args.dicece_lambda_ce:.2f}",
            )
        rank0_print(context, f"Val thresholds : {validation_thresholds}")
        rank0_print(context, f"Min epochs     : {args.min_epochs}")
        rank0_print(context, f"Early patience : {args.early_stop_patience}")

        train_dicts, val_dicts, test_dicts = build_dataset_splits(args)
        rank0_print(context, f"Train cases    : {len(train_dicts)}")
        rank0_print(context, f"Val cases      : {len(val_dicts)}")
        if args.dataset_source == "cancervision_binary_seg":
            rank0_print(context, f"Test cases     : {len(test_dicts)}")

        roi_size = tuple(args.roi_size)
        train_ds = Dataset(
            data=train_dicts,
            transform=dataset_config.train_transform_builder(
                roi_size,
                args.num_samples,
                args.crop_pos_weight,
                args.crop_neg_weight,
            ),
        )

        num_workers = min(args.num_workers, os.cpu_count() or 1)
        pin_memory = context.device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )

        val_ds = Dataset(
            data=val_dicts,
            transform=dataset_config.val_transform_builder(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )

        try:
            model = build_model_with_filters(
                in_channels=dataset_config.in_channels,
                out_channels=dataset_config.out_channels,
                filters=args.model_filters,
            ).to(context.device)
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            memory_summary = format_cuda_memory_summary(context)
            memory_hint = (
                f" Visible CUDA memory before model init: {memory_summary}."
                if memory_summary is not None
                else ""
            )
            raise RuntimeError(
                "CUDA OOM while initializing DynUNet. This happens before ROI crops "
                "or sliding-window inference, so lower --model-filters or choose a "
                "smaller --gpu-profile such as gpu32g/gpu16g, and confirm the Slurm "
                "job really has one mostly free GPU available."
                f"{memory_hint}"
            ) from exc

        loss_function = build_loss_function(args)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )
        scaler = (
            torch.cuda.amp.GradScaler(enabled=True)
            if should_use_amp(args, context)
            else None
        )

        (
            start_epoch,
            best_metric,
            best_metric_epoch,
            best_threshold,
            epochs_since_improve,
        ) = load_resume_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resume_path=args.resume,
            context=context,
        )
        if args.resume:
            rank0_print(
                context,
                f"Resumed from {args.resume} starting at epoch {start_epoch + 1}",
            )
            rank0_print(
                context,
                "Resume best    : "
                f"dice={best_metric:.4f} epoch={best_metric_epoch} threshold={best_threshold:.2f}",
            )

        wandb_config = {
            "run_name": run_name,
            "dataset_source": args.dataset_source,
            "data_dir": os.path.normpath(args.data_dir)
            if args.dataset_source == "brats"
            else "",
            "task_manifest": resolved_task_manifest
            if args.dataset_source == "cancervision_binary_seg"
            else "",
            "requested_task_manifest": requested_task_manifest,
            "path_prefix_maps": list(path_prefix_maps),
            "in_channels": dataset_config.in_channels,
            "out_channels": dataset_config.out_channels,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_interval": args.val_interval,
            "seed": args.seed,
            "gpu_profile": gpu_profile.name,
            "roi_size": list(roi_size),
            "num_samples": args.num_samples,
            "crop_pos_weight": args.crop_pos_weight,
            "crop_neg_weight": args.crop_neg_weight,
            "model_filters": list(args.model_filters),
            "train_micro_batch_size": args.train_micro_batch_size,
            "test_size": args.test_size,
            "amp": args.amp,
            "val_sw_batch_size": args.val_sw_batch_size,
            "loss": args.loss,
            "dicece_lambda_dice": args.dicece_lambda_dice,
            "dicece_lambda_ce": args.dicece_lambda_ce,
            "val_thresholds": list(validation_thresholds),
            "min_epochs": args.min_epochs,
            "early_stop_patience": args.early_stop_patience,
            "distributed": context.distributed,
            "world_size": context.world_size,
            "single_gpu_only": True,
        }
        wandb_run = maybe_init_wandb(args, context, wandb_config)

        for epoch in range(start_epoch, args.max_epochs):
            epoch_start = time.time()
            rank0_print(context, "-" * 40)
            rank0_print(context, f"Epoch {epoch + 1}/{args.max_epochs}")

            try:
                avg_loss, current_lr = train_one_epoch(
                    model=model,
                    train_loader=train_loader,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    epoch=epoch,
                    args=args,
                    context=context,
                )
            except Exception as exc:
                if not is_cuda_oom_error(exc):
                    raise
                raise RuntimeError(
                    "CUDA OOM during training step. Reduce --roi-size, keep "
                    "--num-samples 1, lower --gpu-profile if needed, and do not "
                    "launch more than one Slurm task."
                ) from exc

            if is_main_process(context):
                print(f"  avg loss: {avg_loss:.4f}  lr: {current_lr:.2e}", flush=True)
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "epoch": epoch + 1,
                    }
                )

            if (epoch + 1) % args.val_interval == 0:
                try:
                    metrics = validate(
                        model=model,
                        val_loader=val_loader,
                        roi_size=roi_size,
                        args=args,
                        context=context,
                        dataset_config=dataset_config,
                    )
                except Exception as exc:
                    if not is_cuda_oom_error(exc):
                        raise
                    raise RuntimeError(
                        "CUDA OOM during validation. Lower --roi-size first; "
                        "--val-sw-batch-size is already safest at 1, and a "
                        "smaller --gpu-profile can also help."
                    ) from exc
                if is_main_process(context) and metrics is not None:
                    if metrics["dice_mean"] > best_metric:
                        best_metric = metrics["dice_mean"]
                        best_metric_epoch = epoch + 1
                        best_threshold = metrics["selected_threshold"]
                        epochs_since_improve = 0
                        torch.save(model.state_dict(), run_dir / "best_metric_model.pth")
                        print(
                            f"  -> saved new best model to {run_dir / 'best_metric_model.pth'}",
                            flush=True,
                        )
                    else:
                        epochs_since_improve += 1

                    wandb_payload = {
                        "val/dice_mean": metrics["dice_mean"],
                        **{
                            f"val/dice_{metric_name}": metrics[f"dice_{metric_name}"]
                            for metric_name in dataset_config.metric_names
                        },
                        "val/selected_threshold": metrics["selected_threshold"],
                        "val/best_dice": best_metric,
                        "val/best_threshold": best_threshold,
                        "val/epochs_since_improve": epochs_since_improve,
                        "epoch": epoch + 1,
                    }
                    if args.val_thresholds is not None:
                        wandb_payload.update(
                            {
                                f"val/{key}": value
                                for key, value in metrics.items()
                                if key.startswith("dice_") and "_threshold_" in key
                            }
                        )
                    wandb_run.log(wandb_payload)
                    metric_summary = " ".join(
                        f"{metric_name.upper()}={metrics[f'dice_{metric_name}']:.4f}"
                        for metric_name in dataset_config.metric_names
                    )
                    print(
                        f"  val dice: {metrics['dice_mean']:.4f}"
                        f"  ({metric_summary})"
                        f"\n  selected threshold: {metrics['selected_threshold']:.2f}"
                        f"\n  best dice: {best_metric:.4f} @ epoch {best_metric_epoch}"
                        f"  (threshold={best_threshold:.2f})",
                        flush=True,
                    )
                    if args.val_thresholds is not None:
                        threshold_summary = " ".join(
                            f"{threshold:.2f}={metrics[f'dice_mean_threshold_{_threshold_key(threshold)}']:.4f}"
                            for threshold in validation_thresholds
                        )
                        print(f"  threshold sweep: {threshold_summary}", flush=True)
                    if (
                        args.early_stop_patience > 0
                        and (epoch + 1) >= args.min_epochs
                        and epochs_since_improve >= args.early_stop_patience
                    ):
                        stop_reason = (
                            "early stopping after "
                            f"{epochs_since_improve} validations without improvement"
                        )
                        print(f"  stopping: {stop_reason}", flush=True)
                        if is_main_process(context):
                            save_last_checkpoint(
                                run_dir=run_dir,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                epoch=epoch,
                                best_metric=best_metric,
                                best_metric_epoch=best_metric_epoch,
                                best_threshold=best_threshold,
                                epochs_since_improve=epochs_since_improve,
                            )
                            print(
                                f"  epoch time: {time.time() - epoch_start:.1f}s",
                                flush=True,
                            )
                        break

            if is_main_process(context):
                save_last_checkpoint(
                    run_dir=run_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_metric_epoch=best_metric_epoch,
                    best_threshold=best_threshold,
                    epochs_since_improve=epochs_since_improve,
                )
                print(f"  epoch time: {time.time() - epoch_start:.1f}s", flush=True)

        if is_main_process(context):
            total_time = time.time() - total_start
            print("=" * 60, flush=True)
            print("Training complete", flush=True)
            print(
                f"  Best mean Dice : {best_metric:.4f} @ epoch {best_metric_epoch}",
                flush=True,
            )
            print(f"  Best threshold : {best_threshold:.2f}", flush=True)
            print(f"  Stop reason    : {stop_reason}", flush=True)
            print(
                f"  Total time     : {total_time:.1f}s ({total_time / 3600:.2f}h)",
                flush=True,
            )
            print(f"  Checkpoint     : {run_dir / 'best_metric_model.pth'}", flush=True)
            print("=" * 60, flush=True)
    finally:
        wandb_run.finish()
        cleanup_distributed(context)
