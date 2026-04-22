"""DynUNet architecture builder (MONAI's nnU-Net implementation) for BraTS.

When `deep_supervision=True` the network returns a stacked tensor with shape
(batch, num_heads, out_channels, H, W, D) during training, where num_heads =
1 + deep_supr_num. Deeper heads operate on downsampled feature maps; the
loss must downsample labels accordingly and combine the heads with
exponentially decaying weights. The training loop handles that.
"""

from __future__ import annotations

from monai.networks.nets import DynUNet


def build_dynunet(
    in_channels: int = 4,
    out_channels: int = 3,
    dropout: float = 0.2,
    deep_supervision: bool = False,
    deep_supr_num: int = 2,
) -> DynUNet:
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
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
    )
