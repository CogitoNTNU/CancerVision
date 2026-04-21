# Sustainable Inference Setup

This project now supports registry-driven, multi-model inference through `src/inference`.

## Recommended Folder Structure

Use this layout to keep training code, inference runtime code, and artifacts separate:

```text
src/
  datasets/
  models/                 # model definitions + training
  inference/              # inference runtime (registry, pipeline, CLI)
res/
  models/                 # checkpoints + model_registry.json
  predictions/            # generated outputs (gitignored)
docs/
  manuals/
tests/
```

## Why This Scales Better

- `src/models` focuses on model/training concerns only.
- `src/inference` is runtime-focused and can evolve independently.
- `res/models/model_registry.json` creates a single source of truth for deployable model IDs.
- New model families can be added by registering architecture builders instead of rewriting CLI code.

## Multi-Model Inference Workflow

### 1. Add/Update checkpoint metadata

Edit `res/models/model_registry.json`:

```json
{
  "id": "my_new_model",
  "architecture": "dynunet_brats_v1",
  "checkpoint": "res/models/my_new_model/best_metric_model.pth",
  "roi_size": [128, 128, 128]
}
```

### 2. Run inference for a single case

```bash
uv run python -m src.inference.inference \
  --model-id dynunet_latest \
  --case-dir /path/to/BraTS20_Training_001
```

### 3. Run inference for multiple cases

```bash
uv run python -m src.inference.inference \
  --model-id dynunet_latest \
  --input-root /path/to/MICCAI_BraTS2020_TrainingData \
  --output-root res/predictions/dynunet_latest
```

## Adding new architectures

1. Implement and register the model in `src/models/registry.py` (see
   `docs/manuals/ml-lifecycle-setup.md` for the training-side recipe).
2. Map an inference `architecture` id to the registered model name in
   `src.inference.architectures._ARCHITECTURE_TO_MODEL`.
3. Reference that architecture id from entries in
   `res/models/model_registry.json`.

This keeps training and inference sharing one source of truth for architectures.
