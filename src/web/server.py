from __future__ import annotations

import argparse
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.inference.architectures import _ARCHITECTURE_BUILDERS
from src.inference.model_registry import resolve_repo_root

from .inference_runtime import (
    build_model_from_weights,
    infer_from_files,
    resolve_device,
)

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"

REPO_ROOT = resolve_repo_root(PACKAGE_DIR)
WORK_ROOT = REPO_ROOT / "res" / "web_uploads"

MODALITY_ORDER = ("flair", "t1", "t1ce", "t2")


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
    def list_architectures() -> JSONResponse:
        return JSONResponse({"architectures": sorted(_ARCHITECTURE_BUILDERS.keys())})

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
        if architecture not in _ARCHITECTURE_BUILDERS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown architecture '{architecture}'. "
                    f"Registered: {sorted(_ARCHITECTURE_BUILDERS.keys())}"
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
