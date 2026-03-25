"""Train a 3D binary classifier for tumor presence."""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.classifier.torch_classifier import SmallTumorClassifier3D
from src.core import resolve_device, save_checkpoint
from src.data.preprocess import preprocess_image_volume
from src.data.registry import get_dataset_adapter


class TumorPresenceDataset(Dataset):
    """Dataset producing full 3D volumes and binary tumor labels."""

    def __init__(self, records: list[dict[str, list[str] | str]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_paths = record["image"]
        label_path = record["label"]

        channels = [nib.load(path).get_fdata(dtype=np.float32) for path in image_paths]
        image = torch.from_numpy(np.stack(channels, axis=0)).float()
        image = preprocess_image_volume(image)

        label_volume = nib.load(label_path).get_fdata(dtype=np.float32)
        has_tumor = float(label_volume.max() > 0)
        target = torch.tensor(has_tumor, dtype=torch.float32)

        return image, target


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    default_dataset = "brats"
    default_data_dir = get_dataset_adapter(default_dataset).default_data_dir(project_root)

    parser = argparse.ArgumentParser(description="Train binary tumor classifier")
    parser.add_argument("--dataset", type=str, default=default_dataset)
    parser.add_argument("--data-dir", type=str, default=default_data_dir)
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.normpath(
            os.path.join(script_dir, "..", "..", "res", "models", "tumor_classifier.pth")
        ),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    adapter = get_dataset_adapter(args.dataset)

    records = adapter.build_training_records(args.data_dir)
    train_records, val_records = train_test_split(
        records,
        test_size=args.test_size,
        random_state=args.seed,
    )

    train_ds = TumorPresenceDataset(train_records)
    val_ds = TumorPresenceDataset(val_records)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    device = resolve_device(args.device)
    model = SmallTumorClassifier3D(in_channels=4).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for image, target in train_loader:
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for image, target in val_loader:
                image = image.to(device)
                target = target.to(device)
                probs = torch.sigmoid(model(image))
                preds = (probs >= 0.5).float()
                correct += int((preds == target).sum().item())
                total += target.numel()

        val_acc = correct / max(total, 1)
        avg_loss = train_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - train_loss={avg_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                args.save_path,
                model=model,
                metadata={
                    "task": "classification",
                    "dataset": args.dataset,
                    "model_backend": "small_classifier_3d",
                    "in_channels": 4,
                    "out_channels": 1,
                    "val_acc": best_val_acc,
                    "args": vars(args),
                },
            )
            print(f"Saved new best classifier checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
