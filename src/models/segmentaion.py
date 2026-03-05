import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image


batch_size = 2
number_of_workers = 2
max_epochs = 5

def main(tempdir = None):
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
    for i in range(10):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        
        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, "im.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, "seg.nii.gz"))

    images = sorted(glob(os.path.join(tempfile.tempdir, "im.nii.gz")))

    segs = sorted(glob(os.path.join(tempfile.tempdir, "seg.nii.gz")))

    train_imtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            RandSpatialCrop((96, 96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        ]
    )
    train_segtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            RandSpatialCrop((96, 96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
        ]
    )

    val_imtrans = Compose([ScaleIntensity(), EnsureChannelFirst()])
    val_segtrans = Compose([ScaleIntensity(), EnsureChannelFirst()])

    check_ds = ImageDataset(
        images=images,
        segs=segs,
        im_transforms=train_imtrans,
        seg_transforms=train_segtrans,
    )
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=number_of_workers, pin_memory=pin_memory) 
    
    im, seg = monai.utils.misc.first(check_loader)
    print(im.shape, seg.shape)

    # create train data image loader
    train_ds = ImageDataset(
        images=images,
        segs=segs,
        im_transforms=train_imtrans,
        seg_transforms=train_segtrans,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=number_of_workers, pin_memory=pin_memory)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in train_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[0].to(device)
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in val_outputs]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation.pth")
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f} best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            plot_2d_or_3d_image(
                val_images, epoch + 1, writer, index=0, tag="val_image")
            plot_2d_or_3d_image(
                val_labels, epoch + 1, writer, index=0, tag="val_label")
            plot_2d_or_3d_image(
                val_outputs, epoch + 1, writer, index=0, tag="val_output")
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
        