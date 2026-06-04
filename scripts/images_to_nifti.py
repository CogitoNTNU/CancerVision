#!/usr/bin/env python3
"""
Convert a folder of 2D image slices (png/jpg) into a single NIfTI (.nii.gz) volume.

Usage:
  python scripts/images_to_nifti.py /path/to/slices output.nii.gz --spacing 1 1 1

The script sorts filenames using a natural sort (so slice_1.png, slice_2.png, ...).
"""
import os
import re
import argparse
from pathlib import Path

import numpy as np
try:
    import imageio
except Exception:
    raise RuntimeError('Please install imageio: pip install imageio')
try:
    import nibabel as nib
except Exception:
    raise RuntimeError('Please install nibabel: pip install nibabel')


def natural_key(s):
    # Split on digits so that numbers sort numerically
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s)]


def load_slices(folder, pattern=None):
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(folder)
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    if pattern:
        files = [f for f in files if pattern in f.name]
    files = sorted(files, key=lambda x: natural_key(x.name))
    if not files:
        raise RuntimeError('No image files found in folder: %s' % folder)
    imgs = []
    for f in files:
        img = imageio.imread(str(f))
        # If multi-channel, convert to grayscale by taking the first channel
        if img.ndim == 3:
            img = img[..., 0]
        imgs.append(img.astype(np.float32))
    return np.stack(imgs, axis=-1), [str(x) for x in files]


def main():
    ap = argparse.ArgumentParser(description='Stack 2D image slices to NIfTI (.nii.gz)')
    ap.add_argument('input_folder', help='Folder with image slices (png/jpg, sorted naturally)')
    ap.add_argument('output_nifti', help='Output filename (e.g. volume.nii.gz)')
    ap.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                    help='Voxel spacing for x y z (default: 1 1 1)')
    ap.add_argument('--pattern', default=None, help='Only include files whose name contains this')
    ap.add_argument('--transpose', action='store_true', help='Transpose axes to (z,y,x) before saving')
    args = ap.parse_args()

    data, files = load_slices(args.input_folder, args.pattern)

    # data shape is (H, W, Z) already because we stacked as axis=-1
    # ensure consistent dtype
    data = data.astype(np.float32)

    sx, sy, sz = args.spacing
    # Create a simple affine: diagonal of voxel sizes
    affine = np.diag([sx, sy, sz, 1.0])

    img = nib.Nifti1Image(data, affine)
    out_path = Path(args.output_nifti)
    nib.save(img, str(out_path))
    print(f'Saved {out_path} — shape={data.shape}, spacing=({sx},{sy},{sz})')
    print('Files used (first/last):', files[0], '...', files[-1])


if __name__ == '__main__':
    main()
