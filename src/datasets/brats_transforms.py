"""Custom MONAI transforms for BraTS brain tumor segmentation."""

import torch
from monai.transforms import MapTransform


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """Convert BraTS segmentation labels to multi-channel binary format.

    BraTS labels: 0=background, 1=necrotic/non-enhancing, 2=edema, 4=enhancing

    Output channels:
        - TC (Tumor Core): labels 1 + 4
        - WT (Whole Tumor): labels 1 + 2 + 4
        - ET (Enhancing Tumor): label 4
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key][0]  # remove channel dim: (1,H,W,D) -> (H,W,D)
            tc = torch.logical_or(label == 1, label == 4)
            wt = torch.logical_or(tc, label == 2)
            et = label == 4
            d[key] = torch.stack([tc, wt, et], dim=0).float()
        return d
