"""BraTS2020 pre-processed 2D HDF5 slice dataset."""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class BraTSH5Dataset(Dataset):
    """Dataset for BraTS2020 pre-processed 2D H5 slices.

    Each H5 file contains:
        - 'image': (240, 240, 4) float64 — 4 MRI modalities
        - 'mask':  (240, 240, 3) uint8   — 3 binary channels (TC, WT, ET)

    Returns dict with channel-first tensors:
        - 'image': (4, 240, 240) float32
        - 'label': (3, 240, 240) float32
    """

    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], "r") as f:
            image = torch.tensor(np.array(f["image"], dtype=np.float32)).permute(2, 0, 1)
            label = torch.tensor(np.array(f["mask"], dtype=np.float32)).permute(2, 0, 1)

        data = {"image": image, "label": label}
        if self.transform is not None:
            data = self.transform(data)
        return data
