"""Flask web app for running inference in browser."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.classifier import HeuristicTumorClassifier, load_torch_classifier
from src.data.registry import get_dataset_adapter, list_dataset_types
from src.inference import InferenceService, SegmentationInferer
from src.segmentation import list_segmentation_backends
from src.web.visualization import create_preview_png


def _project_root() -> Path:
    """Return repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[2]


def _web_output_dir() -> Path:
    """Directory used by the web app to store preview and prediction artifacts."""
    return _project_root() / "res" / "output" / "web"


def _is_nifti_filename(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _save_uploaded_nifti(sample_file, dataset: str) -> str:
    filename = (sample_file.filename or "").strip()
    if not filename or not _is_nifti_filename(filename):
        raise ValueError("Uploaded file must be a .nii or .nii.gz file.")

    safe_name = secure_filename(filename)
    if not safe_name:
        safe_name = f"upload_{uuid.uuid4().hex}.nii.gz"

    upload_dir = _web_output_dir() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{dataset}_{uuid.uuid4().hex[:10]}_{safe_name}"
    stored_path = upload_dir / stored_name
    sample_file.save(str(stored_path))
    return str(stored_path.resolve())


def _build_inference_service(
    classifier_checkpoint: str | None,
    segmentation_checkpoint: str,
    dataset: str,
    model_backend: str | None,
    classifier_threshold: float,
) -> tuple[InferenceService, str]:
    adapter = get_dataset_adapter(dataset)
    if classifier_checkpoint:
        classifier = load_torch_classifier(classifier_checkpoint)
        classifier_name = "torch"
    else:
        classifier = HeuristicTumorClassifier()
        classifier_name = "heuristic"

    segmenter = SegmentationInferer.from_checkpoint(
        segmentation_checkpoint,
        model_backend=model_backend,
        in_channels=adapter.get_input_channels(),
        out_channels=adapter.get_output_channels(),
    )
    service = InferenceService(
        classifier=classifier,
        segmenter=segmenter,
        classifier_threshold=classifier_threshold,
    )
    return service, classifier_name


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            datasets=list_dataset_types(),
            backends=list_segmentation_backends(),
            result=None,
            error=None,
        )

    @app.route("/infer", methods=["POST"])
    def infer():
        dataset = request.form.get("dataset", "brats")
        sample_path = request.form.get("sample_path", "").strip()
        segmentation_checkpoint = request.form.get("segmentation_checkpoint", "").strip()
        model_backend = request.form.get("model_backend", "").strip() or None
        classifier_checkpoint = request.form.get("classifier_checkpoint", "").strip() or None
        threshold_raw = request.form.get("classifier_threshold", "0.5").strip().replace(",", ".")
        sample_file = request.files.get("sample_file")
        has_uploaded_file = bool(sample_file and (sample_file.filename or "").strip())

        try:
            classifier_threshold = float(threshold_raw)
        except ValueError:
            return render_template(
                "index.html",
                datasets=list_dataset_types(),
                backends=list_segmentation_backends(),
                result=None,
                error="classifier_threshold must be a number between 0 and 1.",
            )

        if not 0.0 <= classifier_threshold <= 1.0:
            return render_template(
                "index.html",
                datasets=list_dataset_types(),
                backends=list_segmentation_backends(),
                result=None,
                error="classifier_threshold must be between 0 and 1.",
            )

        if not segmentation_checkpoint:
            return render_template(
                "index.html",
                datasets=list_dataset_types(),
                backends=list_segmentation_backends(),
                result=None,
                error="segmentation_checkpoint is required.",
            )

        if not sample_path and not has_uploaded_file:
            return render_template(
                "index.html",
                datasets=list_dataset_types(),
                backends=list_segmentation_backends(),
                result=None,
                error="Provide either sample_path or an uploaded NIfTI file.",
            )

        if has_uploaded_file and dataset != "ixi":
            return render_template(
                "index.html",
                datasets=list_dataset_types(),
                backends=list_segmentation_backends(),
                result=None,
                error=(
                    "File upload is currently supported for dataset 'ixi'. "
                    "For 'brats', provide sample_path as the patient directory path."
                ),
            )

        try:
            if has_uploaded_file and sample_file is not None:
                sample_path = _save_uploaded_nifti(sample_file, dataset=dataset)

            adapter = get_dataset_adapter(dataset)
            service, classifier_name = _build_inference_service(
                classifier_checkpoint=classifier_checkpoint,
                segmentation_checkpoint=segmentation_checkpoint,
                dataset=dataset,
                model_backend=model_backend,
                classifier_threshold=classifier_threshold,
            )

            image = adapter.load_inference_image(sample_path)
            result = service.run_tensor(image)

            output_dir = _web_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            run_id = uuid.uuid4().hex[:10]

            preview_filename = f"preview_{run_id}.png"
            preview_path = output_dir / preview_filename
            create_preview_png(
                image=image,
                mask=result.segmentation_mask,
                output_path=str(preview_path),
                title=f"Dataset={dataset} Tumor={result.has_tumor}",
            )

            prediction_filename = None
            if result.segmentation_mask is not None:
                prediction_filename = f"prediction_{run_id}.nii.gz"
                prediction_path = output_dir / prediction_filename
                service.save_prediction(
                    dataset_adapter=adapter,
                    sample_path=sample_path,
                    result=result,
                    output_path=str(prediction_path),
                )

            web_result = {
                "dataset": dataset,
                "classifier": classifier_name,
                "tumor_probability": f"{result.tumor_probability:.4f}",
                "has_tumor": result.has_tumor,
                "segmentation_ran": result.segmentation_mask is not None,
                "positive_voxels": int(result.segmentation_mask.sum().item()) if result.segmentation_mask is not None else 0,
                "output_dir": str(output_dir),
                "sample_path": sample_path,
                "uploaded_file": has_uploaded_file,
                "preview_path": f"/results/{preview_filename}",
                "prediction_path": f"/results/{prediction_filename}" if prediction_filename else None,
            }

            return render_template(
                "index.html",
                datasets=list_dataset_types(),
                backends=list_segmentation_backends(),
                result=web_result,
                error=None,
            )
        except Exception as exc:
            return render_template(
                "index.html",
                datasets=list_dataset_types(),
                backends=list_segmentation_backends(),
                result=None,
                error=str(exc),
            )

    @app.route("/results/<path:filename>")
    def result_file(filename: str):
        from flask import send_from_directory

        return send_from_directory(_web_output_dir(), filename)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CancerVision web interface")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
