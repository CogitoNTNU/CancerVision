"""DynUNet architecture builder tuned for BraTS (4 modalities -> TC/WT/ET)."""

from __future__ import annotations

from monai.networks.nets import DynUNet


def build_dynunet(
    in_channels: int = 4,
    out_channels: int = 3,
    dropout: float = 0.2,
) -> DynUNet:
    """Return a BraTS-shaped DynUNet.

    Defaults assume 4 MRI modalities in and 3 tumor-region channels out
    (TC, WT, ET). The kernel/stride/filter ladder matches the MONAI
    BraTS tutorial configuration that this project started from.
    """
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320],
        dropout=dropout,
        res_block=True,
        deep_supervision=False,
    )
