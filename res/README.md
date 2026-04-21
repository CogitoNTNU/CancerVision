# Dataset

The BraTS 2020 dataset is not included here (license). Download it and place
the training tree under `res/data/brats/MICCAI_BraTS2020_TrainingData/`
(that path is the default for `--data-dir` in the training CLI).

# Models and inference outputs

- Segmentation checkpoints live in `res/models/`.
- `res/models/model_registry.json` maps deployable model ids onto their
  architecture and checkpoint path for the inference CLI.
- Inference outputs default to `res/predictions/<model_id>/` (gitignored).

# Classification reports

Case-level classification CSVs produced by `src.classification.classify` go to
`res/classification/reports/`.
