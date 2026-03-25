"""MONAI 3D UNet backend implementation."""

from __future__ import annotations

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureTyped,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
)

from .base import SegmentationBackend


class MonaiUNetBackend(SegmentationBackend):
    """Default backend used for 3D tumor segmentation."""

    name = "monai_unet"
    description = "3D MONAI UNet with sliding-window inference"

    def build_model(self, in_channels: int, out_channels: int):
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.2,
        )

    def get_train_transforms(self, dataset_adapter, roi_size: tuple[int, int, int], num_samples: int):
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            EnsureTyped(keys=["image", "label"]),
        ]
        label_transform = dataset_adapter.get_segmentation_label_transform()
        if label_transform is not None:
            transforms.append(label_transform)

        transforms.extend(
            [
                ScaleIntensityRangePercentilesd(
                    keys="image",
                    lower=0.5,
                    upper=99.5,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                    channel_wise=True,
                ),
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

        return Compose(transforms)

    def get_val_transforms(self, dataset_adapter):
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            EnsureTyped(keys=["image", "label"]),
        ]
        label_transform = dataset_adapter.get_segmentation_label_transform()
        if label_transform is not None:
            transforms.append(label_transform)

        transforms.extend(
            [
                ScaleIntensityRangePercentilesd(
                    keys="image",
                    lower=0.5,
                    upper=99.5,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                    channel_wise=True,
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
        return Compose(transforms)
