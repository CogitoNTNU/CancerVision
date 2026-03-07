import logging
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from dotenv import load_dotenv
import wandb
path = os.path.join(os.path.dirname(__file__), "res/dataset/IXI-T1/")


load_dotenv()  # loads .env from project root

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if not WANDB_API_KEY:
    raise RuntimeError("WANDB_API_KEY missing in .env")

wandb.login(key=WANDB_API_KEY)


def main():
    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_path = os.path.join(os.path.dirname(__file__), "data")
    images = [
        "IXI314-IOP-0889-T1.nii.gz",
        "IXI249-Guys-1072-T1.nii.gz",
        "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz",
        "IXI020-Guys-0700-T1.nii.gz",
        "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz",
        "IXI577-HH-2661-T1.nii.gz",
        "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz",
        "IXI607-Guys-1097-T1.nii.gz",
        "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz",
        "IXI344-Guys-0905-T1.nii.gz",
        "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz",
        "IXI253-HH-1694-T1.nii.gz",
        "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz",
        "IXI585-Guys-1130-T1.nii.gz",
    ]
    images = os.sep.join([data_path, "images", "image_{}.nii.gz"])
    
    labels = labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

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
        images=images,
        labels=labels,
        transform=train_transforms,
    )
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=pin_memory)
    im, label = monai.utils.misc.first(check_loader)
    print(im.shape, label.shape)

    train_ds = ImageDataset(
        images=images,
        labels=labels,
        transform=train_transforms,
    )
    train_loader = DataLoader(train_ds, batch_size=2, num_workers=4, pin_memory=pin_memory)
    val_ds = ImageDataset(
        images=images,
        labels=labels,
        transform=val_transforms,
    )

    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=pin_memory)


    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    val_interval = 2
    best_metric = -1
    epochs_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    wandb.init(project="cancervision", entity="cancervision", name="classification_1")

    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "step": step})
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f

        epoch_loss /= step
        epochs_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(path, "best_metric_model_classification.pth"))
                    print("saved new best metric model")

                print(f"epoch {epoch + 1} validation accuracy: {metric:.4f}")
                writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == "__main__":
    main()