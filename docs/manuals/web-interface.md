# Drag-and-Drop Web Inference

The `src/web` package serves a small FastAPI app that lets you run segmentation
inference directly from a browser. It accepts **arbitrary PyTorch weights** plus
four MRI modality NIfTI files via drag-and-drop, runs `sliding_window_inference`
through the registered architecture, and returns a downloadable prediction mask.

## Launching the server

Install dependencies (first time only):

```bash
uv sync
```

Then start the server:

```bash
uv run python -m src.web --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080> in a browser.

Flags:

- `--host` / `--port` — bind address (defaults: `127.0.0.1:8080`)
- `--reload` — enable uvicorn autoreload for local development

## What to drop

1. **Weights** — a PyTorch checkpoint (`.pth` / `.pt`). The file can be a raw
   `state_dict` or a training checkpoint containing a `model_state` key — the
   same formats the CLI inference pipeline accepts.
2. **FLAIR / T1 / T1CE / T2** — the four BraTS modalities as `.nii` or `.nii.gz`.
   Files whose names contain `flair`, `t1ce`, `t1`, or `t2` are auto-assigned
   to the correct slot. You can also drop all four files onto any single slot
   and they will be routed automatically.

## Configuration form

- **Architecture** — populated from `src/inference/architectures.py`.
  Register new ones with `register_architecture(name, builder_fn)` and they
  appear in the dropdown on the next page reload.
- **Device** — `auto`, `cpu`, `cuda`, or `mps`.
- **Threshold** — sigmoid probability threshold applied to the TC/WT/ET
  channels before label assembly.
- **ROI size** — sliding-window ROI, comma-separated (defaults to `128,128,128`).

## Output

On success the page shows a summary with per-label voxel counts and a download
link for the prediction NIfTI. Artifacts are written under
`res/web_uploads/<job_id>/` (gitignored). Delete that directory periodically to
reclaim disk space.

## Adding new architectures

Identical to the CLI path: implement a builder in
`src/inference/architectures.py` and call `register_architecture(...)`. The web
form will pick it up automatically via the `/api/architectures` endpoint.
