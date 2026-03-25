"""Custom MONAI transforms for BraTS segmentation labels."""

from __future__ import annotations

import torch
from monai.transforms import MapTransform


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """Convert BraTS labels to multi-channel targets: TC, WT, ET."""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key][0]
            tc = torch.logical_or(label == 1, label == 4)
            wt = torch.logical_or(tc, label == 2)
            et = label == 4
            d[key] = torch.stack([tc, wt, et], dim=0).float()
        return d
