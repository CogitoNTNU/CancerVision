import logging
import os
import sys

from nibabel.testing import data_path
import numpy as np
import torch

import monai
from monai.data import CSVSaver, ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, Resize, ScaleIntensity


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    images = [
        "IXI607-Guys-1097-T1.nii.gz",
        "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz",
        "IXI344-Guys-0905-T1.nii.gz",
        "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz",
        "IXI253-HH-1694-T1.nii.gz",
        "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz",
        "IXI585-Guys-1130-T1.nii.gz",
    ]
    images = [os.sep.join([data_path, f]) for f in images]

    labels = np.array([0,0,1,0,1,0,1,0,1,0], dtype=np.int64)
    

    val_tansforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96))
    ])

    val_ds = ImageDataset(
        images=images,
        labels=labels,
        transform=val_tansforms
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = monai.networks.nets.DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "res/model_3d_classification.pt")))
    model.eval()

    with torch.no_grad():
        num_correct = 0
        metric_count = 0
        saver = CSVSaver(output_dir=os.path.join(os.path.dirname(__file__), "res/3d_classification_results.csv"))
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images)
            value = torch.eq(val_outputs,val_labels)
            metric_count += len(value)
            num_correct += torch.sum(value).item()
            saver.save_batch(val_outputs, val_images.meta)
        metric = num_correct / metric_count
    print(f"Validation metric: {metric:.4f}")
    saver.finalize()

if __name__ == "__main__":
    main()