# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
import pathlib
from glob import glob

from logger.train import TrainingLogger
from identifier import id

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.visualize import plot_2d_or_3d_image


def main(datadir: pathlib.Path):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # * Loading
    # Load the training data
    print(f"loading training data from {datadir} (this may take a while)")

    # Direcories for images and masks
    imageDirectory = datadir / "images"
    maskDirectory = datadir / "masks"
    
    # Get the list of image and mask files
    images = sorted(imageDirectory.glob("*.png"))
    segs = sorted(maskDirectory.glob("*.png"))

    # Total number of images in the dataset
    totalSamples = len(images)
    print(f"total number of images: {totalSamples}")

    # Split the dataset into training and validation sets (80% training, 20% validation)
    splitIndex = int(0.8 * totalSamples)

    train_files = [
        {"img": img, "seg": seg} 
        for img, seg in zip(
            images[:splitIndex], 
            segs[:splitIndex]
        )
    ]
    val_files = [
        {"img": img, "seg": seg} 
        for img, seg in zip(
            images[splitIndex:], 
            segs[splitIndex:]
        )
    ]


    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )


    # * Training
    # define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=TRAINING_BATCH_SIZE, num_workers=4, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=VALIDATION_BATCH_SIZE, num_workers=4, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logger.logEpochMetrics(epoch=(epoch + 1), average_loss=epoch_loss)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"models/{identifier}.pth")
                    print("saved new best metric model")
                logger.logValidationResults(
                    currentEpoch=epoch + 1,
                    current_mean_dice=metric,
                    best_mean_dice=best_metric,
                    best_mean_dice_epoch=best_metric_epoch
                )

                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    logger.logCompletion(best_metric, best_metric_epoch)
    writer.close()


if __name__ == "__main__":
    # Hyperparameters
    TRAINING_BATCH_SIZE     = 64
    VALIDATION_BATCH_SIZE   = 1
    EPOCHS                  = 10

    # Model type identifier (DO NOT CHANGE)
    MODEL_TYPE = "DICT"

    # Logging & saving
    identifier = id(MODEL_TYPE, TRAINING_BATCH_SIZE, VALIDATION_BATCH_SIZE, EPOCHS)
    logger = TrainingLogger(
        modelType=MODEL_TYPE,
        trainingBatchSize=TRAINING_BATCH_SIZE,
        validationBatchSize=VALIDATION_BATCH_SIZE,
        epochs=EPOCHS
    )

    # Dataset
    DATASET_NAME = "brain_tumor_dataset"
    datasetDirectory = os.path.join(pathlib.Path(__file__).parent.parent, "res/dataset")

    main(datadir=pathlib.Path(datasetDirectory, DATASET_NAME))
