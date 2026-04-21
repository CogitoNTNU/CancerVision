# Sustainable ML Lifecycle Setup

This manual covers a maintainable structure for the full brain tumor pipeline:

- Model training (segmentation and optional classifier training)
- Segmentation inference
- Case-level tumor classification

## Recommended Structure

```text
src/
  datasets/
  models/                 # model definitions + low-level training loops
  training/               # experiment registry + training dispatcher
  inference/              # model registry + inference runtime
  classification/         # classification rules, registry, and trainers
res/
  models/                 # segmentation checkpoints + model registry
  configs/                # training registries
  classification/
    classifier_registry.json
    models/               # trained classifier artifacts
    reports/              # prediction/classification reports
  predictions/            # segmentation outputs
docs/
  manuals/
tests/
```

## Training

Run registry-driven training:

uv run python -m src.training.train --experiment-id dynunet_brats_baseline

Dry-run a training spec:

uv run python -m src.training.train --experiment-id dynunet_brats_baseline --dry-run

Training configs live in res/configs/training_registry.json.

## Inference

Run segmentation inference with model id:

uv run python -m src.inference.inference --model-id dynunet_latest --input-root /path/to/cases --output-root res/predictions/dynunet_latest

Model specs live in res/models/model_registry.json.

## Classification

Classify predicted segmentations:

uv run python -m src.classification.classify --classifier-id brats_rule_based_v1 --input-root res/predictions/dynunet_latest

Classifier specs live in res/classification/classifier_registry.json.

## Optional Supervised Classifier Training

If you have case-level labels and feature CSVs, train a classifier model:

uv run python -m src.classification.train_classifier --features-csv /path/to/features.csv --target-column class_label --model-output res/classification/models/random_forest_v1.pkl --metrics-output res/classification/reports/random_forest_v1_metrics.json

## Sustainability Principles

- Registries decouple code from environment-specific paths and experiment names.
- Task modules are separated by responsibility, reducing cross-file coupling.
- All generated artifacts are written under res/ and can be cleanly ignored or archived.
- CLIs are stable entry points for local runs, Slurm jobs, and CI workflows.
