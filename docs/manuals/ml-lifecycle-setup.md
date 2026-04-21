# ML lifecycle setup

End-to-end glioma pipeline: segmentation training, inference, and rule-based
case classification.

## Layout

```text
src/
  datasets/
    brats.py              # NIfTI loader + MONAI transforms + BraTS label conversion
    standardize/          # multi-dataset ingestion (standalone subpackage)
  models/
    registry.py           # model-name -> builder registry ("dynunet", "unet")
    dynunet.py            # DynUNet architecture for BraTS
  training/
    train.py              # single training CLI (chooses model via --model)
    distributed.py        # device + DDP/Slurm setup
    checkpoint.py         # save/load helpers
  inference/
    inference.py          # segmentation inference CLI
    model_registry.py     # deployed-checkpoint registry (res/models/model_registry.json)
    architectures.py      # maps registry architecture ids onto models.registry
    pipeline.py           # shared inference utilities
  classification/
    rules.py              # feature extraction + binary + phenotype rules
    classify.py           # CLI: predicted mask dir -> CSV report
  web/                    # drag-and-drop browser inference UI
res/
  data/brats/             # BraTS NIfTI root (MICCAI_BraTS2020_TrainingData/...)
  models/                 # segmentation checkpoints + model_registry.json
  classification/reports/ # classification CSV outputs
  predictions/            # segmentation outputs from inference runs
```

## Segmentation training

Training is driven by a single CLI. The model is selected by name from the
registry in `src.models.registry`:

```bash
uv run python -m src.training.train --model dynunet \
    --data-dir res/data/brats/MICCAI_BraTS2020_TrainingData \
    --max-epochs 100
```

Currently registered models: `dynunet` (production path) and `unet` (reference).

See `python -m src.training.train --help` for the full flag list (batch size,
ROI, AMP, distributed launch, W&B mode, resume, etc.).

### Adding a new model

1. Implement a builder in `src/models/<your_model>.py` returning a
   `torch.nn.Module` with `in_channels=4, out_channels=3` by default.
2. Register it at the bottom of `src/models/registry.py`:

   ```python
   from .your_model import build_your_model
   register_model("your_model", build_your_model)
   ```

3. Run it: `python -m src.training.train --model your_model ...`.
4. To serve the trained checkpoint through the inference CLI, add an entry to
   `res/models/model_registry.json` and map its `architecture` id in
   `src.inference.architectures._ARCHITECTURE_TO_MODEL`.

## Inference

```bash
uv run python -m src.inference.inference \
    --model-id dynunet_latest \
    --input-root /path/to/cases \
    --output-root res/predictions/dynunet_latest
```

Model specs live in `res/models/model_registry.json`.

## Classification

Binary cancer-vs-non-cancer plus a coarse phenotype label, derived from a
directory of predicted BraTS segmentation masks:

```bash
uv run python -m src.classification.classify \
    --input-root res/predictions/dynunet_latest \
    --output-csv res/classification/reports/dynunet_latest.csv
```

The decision is purely volumetric: cases with fewer than `--min-tumor-voxels`
Whole-Tumor voxels are labelled `no_tumor`. Upstream segmentation quality
dominates the result.
