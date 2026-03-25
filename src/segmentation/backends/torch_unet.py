"""Custom PyTorch 3D U-Net backend (non-MONAI model)."""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
)

from .base import SegmentationBackend


class _ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = _ConvBlock3D(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = _ConvBlock3D(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TorchUNet3D(nn.Module):
    """A performant 3D U-Net variant implemented purely in PyTorch."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 10

        self.stem = _ConvBlock3D(in_channels, c1)
        self.down1 = _DownBlock3D(c1, c2)
        self.down2 = _DownBlock3D(c2, c3)
        self.down3 = _DownBlock3D(c3, c4, dropout=0.1)
        self.bottleneck = _DownBlock3D(c4, c5, dropout=0.2)

        self.up3 = _UpBlock3D(c5, c4, c4, dropout=0.1)
        self.up2 = _UpBlock3D(c4, c3, c3)
        self.up1 = _UpBlock3D(c3, c2, c2)
        self.up0 = _UpBlock3D(c2, c1, c1)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bottleneck(s4)

        x = self.up3(b, s4)
        x = self.up2(x, s3)
        x = self.up1(x, s2)
        x = self.up0(x, s1)
        return self.head(x)


class TorchUNetBackend(SegmentationBackend):
    """Non-MONAI model backend with MONAI data transforms."""

    name = "torch_unet3d"
    description = "Custom PyTorch 3D U-Net with MONAI preprocessing/augmentation"

    def build_model(self, in_channels: int, out_channels: int):
        return TorchUNet3D(in_channels=in_channels, out_channels=out_channels, base_channels=32)

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
