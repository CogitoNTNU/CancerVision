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


def main(tempdir = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=True) 
    
    im, seg = monai.utils.misc.first(check_loader)
    print(im.shape, seg.shape)

    # create train data image loader
    train_ds = ImageDataset(
        images=images,
        segs=segs,
        im_transforms=train_imtrans,
        seg_transforms=train_segtrans,
    )
    train_loader = DataLoader(train_ds, batch_size=2, num_workers=2, pin_memory=True) 