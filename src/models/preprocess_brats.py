#!/usr/bin/env python
"""Preprocess BraTS2020 NIfTI volumes to NumPy arrays for fast training.

Performs expensive one-time operations:
  - NIfTI decompression and loading
  - Intensity normalization (per-channel, nonzero voxels)
  - Segmentation label conversion to multi-channel (TC, WT, ET)

Output: one {patient}_image.npy and {patient}_label.npy per patient.

Usage:
    python preprocess_brats.py [--data-dir DIR] [--output-dir DIR] [--workers N]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_nifti(directory: str, pattern: str) -> str:
    """Find a NIfTI file matching *pattern* inside *directory*."""
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(directory, pattern + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {directory}"
    )


def preprocess_patient(patient_path: str, patient_name: str, output_dir: str) -> str:
    """Load, normalize, convert labels, and save one patient as .npy files.

    Replicates the deterministic MONAI transforms:
      LoadImaged -> EnsureChannelFirstd -> ConvertToMultiChannelBasedOnBratsClassesd
      -> NormalizeIntensityd(nonzero=True, channel_wise=True)
    """
    # Load 4 modalities and stack -> (4, H, W, D) float32
    modalities = []
    for suffix in ("_flair", "_t1", "_t1ce", "_t2"):
        nii = nib.load(find_nifti(patient_path, patient_name + suffix))
        modalities.append(nii.get_fdata(dtype=np.float32))
    image = np.stack(modalities, axis=0)  # (4, H, W, D)

    # Normalize: per-channel, nonzero voxels, zero-mean unit-variance
    # Matches MONAI NormalizeIntensityd(nonzero=True, channel_wise=True)
    for c in range(image.shape[0]):
        mask = image[c] != 0
        if mask.any():
            vals = image[c][mask]
            mean, std = float(vals.mean()), float(vals.std())
            if std > 0:
                image[c][mask] = (vals - mean) / std

    # Load segmentation and convert to multi-channel (TC, WT, ET)
    seg = nib.load(
        find_nifti(patient_path, patient_name + "_seg")
    ).get_fdata(dtype=np.float32)

    tc = np.logical_or(seg == 1, seg == 4).astype(np.float32)
    wt = np.logical_or(np.logical_or(seg == 1, seg == 2), seg == 4).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    label = np.stack([tc, wt, et], axis=0)  # (3, H, W, D)

    # Save as .npy (no compression = fast loading)
    np.save(os.path.join(output_dir, f"{patient_name}_image.npy"), image)
    np.save(os.path.join(output_dir, f"{patient_name}_label.npy"), label)

    return patient_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Preprocess BraTS2020 NIfTI volumes to .npy for fast training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.normpath(
            os.path.join(
                script_dir,
                "..",
                "..",
                "res",
                "data",
                "archive",
                "BraTS2020_TrainingData",
                "MICCAI_BraTS2020_TrainingData",
            )
        ),
        help="Path to MICCAI_BraTS2020_TrainingData folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.normpath(
            os.path.join(
                script_dir,
                "..",
                "..",
                "res",
                "data",
                "brats2020_preprocessed",
            )
        ),
        help="Directory to save preprocessed .npy files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers for preprocessing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    output_dir = os.path.normpath(args.output_dir)

    print(f"Data directory  : {data_dir}")
    print(f"Output directory: {output_dir}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Discover patients
    patient_names = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("BraTS20_Training_")
    )
    print(f"Found {len(patient_names)} patients")

    # Process in parallel
    t0 = time.time()
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                preprocess_patient,
                os.path.join(data_dir, name),
                name,
                output_dir,
            ): name
            for name in patient_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(patient_names):
                    print(f"  [{completed}/{len(patient_names)}] processed")
            except Exception as exc:
                errors += 1
                print(f"  ERROR: {name} -- {exc}")

    elapsed = time.time() - t0

    # Report output size
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".npy")
    )
    print(f"\nDone: {completed} patients, {errors} errors, {elapsed:.1f}s")
    print(f"Output size: {total_bytes / 1e9:.2f} GB in {output_dir}")


if __name__ == "__main__":
    main()
