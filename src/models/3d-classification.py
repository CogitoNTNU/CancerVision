import argparse
import glob
import logging
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from dotenv import load_dotenv
import wandb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "res", "output")


load_dotenv()  # loads .env from project root

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if not WANDB_API_KEY:
    raise RuntimeError("WANDB_API_KEY missing in .env")

wandb.login(key=WANDB_API_KEY)


def _default_batch_size(device: torch.device) -> int:
    if device.type == "mps":
        return 24
    if device.type == "cuda":
        return 32
    return 4


def _configure_max_compute(device: torch.device, max_compute: bool) -> None:
    if not max_compute:
        return

    cpu_threads = os.cpu_count() or 1
    torch.set_num_threads(cpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, cpu_threads // 2))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if device.type == "mps":
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def main(
    epochs: int = 100,
    batch_size: int = 0,
    num_workers: int = -2,
    prefetch_factor: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    max_compute: bool = True,
    use_compile: bool = True,
):
    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    _configure_max_compute(device, max_compute)

    if num_workers < 0:
        num_workers = max(1, (os.cpu_count() or 2))

    if batch_size <= 0:
        batch_size = _default_batch_size(device)

    pin_memory = torch.cuda.is_available()
    non_blocking_copy = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(
        "Using device=%s batch_size=%d num_workers=%d prefetch_factor=%d max_compute=%s compile=%s",
        device,
        batch_size,
        num_workers,
        prefetch_factor,
        max_compute,
        use_compile,
    )

    ixi_dir = os.path.join(PROJECT_ROOT, "res", "data", "IXI-T2")
    brats_dir = os.path.join(
        PROJECT_ROOT,
        "res",
        "data",
        "archive",
        "BraTS2020_TrainingData",
        "MICCAI_BraTS2020_TrainingData",
    )

    ixi_files = sorted(glob.glob(os.path.join(ixi_dir, "*.nii"))) + sorted(
        glob.glob(os.path.join(ixi_dir, "*.nii.gz"))
    )
    brats_tumor_files = sorted(
        glob.glob(os.path.join(brats_dir, "BraTS20_Training_*", "*_t2.nii"))
    ) + sorted(glob.glob(os.path.join(brats_dir, "BraTS20_Training_*", "*_t2.nii.gz")))

    if not ixi_files:
        raise FileNotFoundError(f"No IXI images found under '{ixi_dir}'")
    if not brats_tumor_files:
        raise FileNotFoundError(f"No BraTS T2 tumor images found under '{brats_dir}'")

    image_files = ixi_files + brats_tumor_files
    labels = np.array([0] * len(ixi_files) + [1] * len(brats_tumor_files), dtype=np.int64)
    logging.info("Loaded %d IXI images and %d BraTS tumor images.", len(ixi_files), len(brats_tumor_files))

    train_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96)), 
        RandRotate90()]
    )
    val_transforms = Compose(
        [ScaleIntensity(),
         EnsureChannelFirst(),
         Resize((96, 96, 96))])
    
    check_ds = ImageDataset(
        image_files=image_files,
        labels=labels,
        transform=train_transforms,
    )
    check_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        check_loader_kwargs["prefetch_factor"] = prefetch_factor
    check_loader = DataLoader(check_ds, **check_loader_kwargs)

    im, label = monai.utils.misc.first(check_loader)
    print(im.shape, label.shape)

    train_ds = ImageDataset(
        image_files=image_files,
        labels=labels,
        transform=train_transforms,
    )
    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_ds, **train_loader_kwargs)

    val_ds = ImageDataset(
        image_files=image_files,
        labels=labels,
        transform=val_transforms,
    )

    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
    val_loader = DataLoader(val_ds, **val_loader_kwargs)


    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logging.info("torch.compile enabled")
        except Exception as exc:
            logging.warning("torch.compile unavailable, continuing without compile: %s", exc)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epochs_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    wandb.init(
        project="cancervision",
        entity="cancervision",
        name="classification_1",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_compute": max_compute,
            "use_compile": use_compile,
            "device": str(device),
        },
    )

    try:
        for epoch in range(epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs = batch_data[0].to(device, non_blocking=non_blocking_copy)
                batch_labels = batch_data[1].to(device, non_blocking=non_blocking_copy)
                optimizer.zero_grad(set_to_none=True)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_ctx:
                    outputs = model(inputs)
                    loss = loss_function(outputs, batch_labels)

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                epoch_len = max(1, len(train_ds) // train_loader.batch_size)
                wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "step": step})
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

            epoch_loss /= max(1, step)
            epochs_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    num_correct = 0
                    metric_count = 0
                    for val_data in val_loader:
                        val_images = val_data[0].to(device, non_blocking=non_blocking_copy)
                        val_labels = val_data[1].to(device, non_blocking=non_blocking_copy)
                        val_outputs = model(val_images)
                        value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                        metric_count += len(value)
                        num_correct += value.sum().item()
                    metric = num_correct / max(1, metric_count)
                    metric_values.append(metric)

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, "best_metric_model_classification.pth"))
                        print("saved new best metric model")

                    print(f"epoch {epoch + 1} validation accuracy: {metric:.4f}")
                    writer.add_scalar("val_accuracy", metric, epoch + 1)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    finally:
        wandb.finish()
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D classification training")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=0, help="batch size (0 means auto for device)")
    parser.add_argument("--num-workers", type=int, default=-2, help="DataLoader workers (-2 means all CPU cores)")
    parser.add_argument("--prefetch-factor", type=int, default=8, help="DataLoader prefetch factor")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--disable-max-compute", action="store_true", help="disable aggressive compute tuning")
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_compute=not args.disable_max_compute,
        use_compile=not args.no_compile,
    )