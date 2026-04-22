from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402  (must follow matplotlib.use)
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.inference.architectures import list_architectures  # noqa: E402
from src.inference.model_registry import resolve_repo_root  # noqa: E402

from .inference_runtime import (  # noqa: E402
    build_model_from_weights,
    infer_from_files,
    resolve_device,
)

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"

REPO_ROOT = resolve_repo_root(PACKAGE_DIR)
WORK_ROOT = REPO_ROOT / "res" / "web_uploads"

MODALITY_ORDER = ("flair", "t1", "t1ce", "t2")
VALID_PLANES = ("axial", "coronal", "sagittal")

# BraTS label -> (RGB in 0..1, alpha). Keep in sync with static/app.js legend.
_LABEL_OVERLAY = {
    1: ((1.0, 0.10, 0.10), 0.50),  # tumor core
    2: ((1.0, 0.82, 0.10), 0.35),  # edema / whole-tumor halo
    4: ((0.15, 0.55, 1.0), 0.60),  # enhancing tumor
}

# Per-job volume cache so slider scrubs don't re-read NIfTI from disk every frame.
# Keyed by job_id; simple FIFO eviction at _VOLUME_CACHE_MAX entries.
_VOLUME_CACHE: dict[str, dict] = {}
_VOLUME_CACHE_MAX = 4


def _load_job_volumes(job_id: str) -> dict:
    """Load modality + prediction volumes for a job; cache in memory."""
    cached = _VOLUME_CACHE.get(job_id)
    if cached is not None:
        return cached

    job_dir = WORK_ROOT / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    modalities: dict[str, np.ndarray] = {}
    for modality in MODALITY_ORDER:
        for ext in (".nii.gz", ".nii"):
            candidate = job_dir / f"case_{modality}{ext}"
            if candidate.exists():
                modalities[modality] = np.asarray(
                    nib.load(str(candidate)).get_fdata(dtype=np.float32)
                )
                break
    if not modalities:
        raise HTTPException(status_code=404, detail="No modality volumes on disk")

    predictions = sorted(job_dir.glob("prediction_*.nii.gz"))
    if not predictions:
        raise HTTPException(status_code=404, detail="Prediction not available")
    prediction = np.asarray(nib.load(str(predictions[0])).get_fdata()).astype(np.int16)

    # Fill any missing modalities from the first available so the UI can still render
    # a pane even if the user only uploaded one channel.
    fallback = next(iter(modalities.values()))
    for modality in MODALITY_ORDER:
        modalities.setdefault(modality, fallback)

    entry = {
        "modalities": modalities,
        "prediction": prediction,
        "shape": tuple(int(v) for v in prediction.shape),
    }
    _VOLUME_CACHE[job_id] = entry
    while len(_VOLUME_CACHE) > _VOLUME_CACHE_MAX:
        _VOLUME_CACHE.pop(next(iter(_VOLUME_CACHE)))
    return entry


def _extract_slice(volume: np.ndarray, plane: str, idx: int) -> np.ndarray:
    shape = volume.shape  # (X, Y, Z)
    if plane == "axial":
        idx = max(0, min(idx, shape[2] - 1))
        plane_slice = volume[:, :, idx]
    elif plane == "coronal":
        idx = max(0, min(idx, shape[1] - 1))
        plane_slice = volume[:, idx, :]
    elif plane == "sagittal":
        idx = max(0, min(idx, shape[0] - 1))
        plane_slice = volume[idx, :, :]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown plane: {plane}")
    # Rotate so anatomical superior is up in the rendered image.
    return np.rot90(plane_slice)


def _render_slice_png(
    modality_slice: np.ndarray,
    label_slice: np.ndarray | None,
) -> bytes:
    """Blend a grayscale MRI slice with an optional colored label overlay."""
    finite = modality_slice[np.isfinite(modality_slice)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    gray = np.clip(modality_slice / vmax, 0.0, 1.0)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    if label_slice is not None:
        for label_value, (color, alpha) in _LABEL_OVERLAY.items():
            mask = label_slice == label_value
            if not mask.any():
                continue
            color_arr = np.asarray(color, dtype=np.float32)
            rgb[mask] = rgb[mask] * (1.0 - alpha) + color_arr * alpha

    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    mpimg.imsave(buf, rgb_u8, format="png")
    return buf.getvalue()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CancerVision Web Inference",
        description="Drag-and-drop interface for BraTS segmentation inference.",
    )

    if not STATIC_DIR.is_dir():
        raise RuntimeError(f"Static directory missing: {STATIC_DIR}")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/architectures")
    def architectures_endpoint() -> JSONResponse:
        return JSONResponse({"architectures": list_architectures()})

    @app.post("/api/infer")
    async def run_inference(
        weights: UploadFile = File(...),
        flair: UploadFile | None = File(None),
        t1: UploadFile | None = File(None),
        t1ce: UploadFile | None = File(None),
        t2: UploadFile | None = File(None),
        architecture: str = Form("dynunet_brats_v1"),
        device: str = Form("auto"),
        threshold: float = Form(0.5),
        roi_size: str = Form("128,128,128"),
    ) -> JSONResponse:
        registered = list_architectures()
        if architecture not in registered:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown architecture '{architecture}'. "
                    f"Registered: {registered}"
                ),
            )

        try:
            parsed_roi = tuple(int(v.strip()) for v in roi_size.split(","))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid roi_size: {exc}")
        if len(parsed_roi) != 3:
            raise HTTPException(status_code=400, detail="roi_size must have 3 values")

        job_id = uuid.uuid4().hex[:12]
        job_dir = WORK_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_path = _save_upload(weights, job_dir / "weights.pth")

            uploads_by_modality = {
                "flair": flair,
                "t1": t1,
                "t1ce": t1ce,
                "t2": t2,
            }
            provided_uploads = {
                name: upload
                for name, upload in uploads_by_modality.items()
                if upload is not None and upload.filename
            }
            if not provided_uploads:
                raise HTTPException(
                    status_code=400,
                    detail="At least one modality file must be provided",
                )

            saved_modalities: dict[str, Path] = {}
            for modality_name, upload in provided_uploads.items():
                saved_modalities[modality_name] = _save_upload(
                    upload,
                    job_dir / f"case_{modality_name}{_extension(upload.filename)}",
                )

            fallback_path = next(iter(saved_modalities.values()))
            modality_paths = [saved_modalities.get(modality, fallback_path) for modality in MODALITY_ORDER]

            try:
                torch_device = resolve_device(device)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            try:
                model, spec = build_model_from_weights(
                    architecture=architecture,
                    checkpoint_path=weights_path,
                    device=torch_device,
                    roi_size=parsed_roi,
                    threshold=threshold,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load weights: {exc}",
                )

            output_path = job_dir / f"prediction_{job_id}.nii.gz"
            try:
                saved, counts = infer_from_files(
                    model=model,
                    spec=spec,
                    modality_files=modality_paths,
                    output_path=output_path,
                    device=torch_device,
                    threshold=threshold,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

            return JSONResponse(
                {
                    "job_id": job_id,
                    "architecture": spec.architecture,
                    "requested_architecture": architecture,
                    "device": str(torch_device),
                    "threshold": threshold,
                    "roi_size": list(parsed_roi),
                    "submitted_modalities": sorted(saved_modalities.keys()),
                    "filled_modalities": [
                        modality
                        for modality in MODALITY_ORDER
                        if modality not in saved_modalities
                    ],
                    "output_filename": saved.name,
                    "download_url": f"/api/download/{job_id}",
                    "label_counts": counts,
                }
            )
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    @app.get("/api/download/{job_id}")
    def download(job_id: str) -> FileResponse:
        job_dir = WORK_ROOT / job_id
        if not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="Job not found")

        predictions = sorted(job_dir.glob("prediction_*.nii.gz"))
        if not predictions:
            raise HTTPException(status_code=404, detail="Prediction not available")

        prediction = predictions[0]
        return FileResponse(
            prediction,
            media_type="application/gzip",
            filename=prediction.name,
        )

    @app.get("/api/render/{job_id}/meta")
    def render_meta(job_id: str) -> JSONResponse:
        entry = _load_job_volumes(job_id)
        x, y, z = entry["shape"]
        return JSONResponse(
            {
                "shape": [x, y, z],
                "modalities": list(entry["modalities"].keys()),
                "planes": {
                    "axial": {"size": z, "default": z // 2},
                    "coronal": {"size": y, "default": y // 2},
                    "sagittal": {"size": x, "default": x // 2},
                },
                "labels": [
                    {"value": 1, "name": "Tumor core", "color": "#ff1a1a"},
                    {"value": 2, "name": "Edema / WT halo", "color": "#ffd11a"},
                    {"value": 4, "name": "Enhancing tumor", "color": "#268cff"},
                ],
            }
        )

    @app.get("/api/render/{job_id}/{plane}/{idx}.png")
    def render_slice(
        job_id: str,
        plane: str,
        idx: int,
        modality: str = "flair",
        overlay: bool = True,
    ) -> Response:
        if plane not in VALID_PLANES:
            raise HTTPException(status_code=400, detail=f"Unknown plane: {plane}")
        entry = _load_job_volumes(job_id)
        modalities = entry["modalities"]
        chosen = modality if modality in modalities else next(iter(modalities))

        modality_slice = _extract_slice(modalities[chosen], plane, idx)
        label_slice = (
            _extract_slice(entry["prediction"], plane, idx) if overlay else None
        )
        png = _render_slice_png(modality_slice, label_slice)
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return app


def _extension(filename: str | None) -> str:
    if not filename:
        return ".nii.gz"
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    if lower.endswith(".nii"):
        return ".nii"
    return ".nii.gz"


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(destination.parent),
        delete=False,
        suffix=destination.suffix,
    ) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(upload.file, tmp)
    tmp_path.replace(destination)
    upload.file.close()
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the CancerVision drag-and-drop inference web interface."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload (development only)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    uvicorn.run(
        "src.web.server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
