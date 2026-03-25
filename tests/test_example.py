import torch

from src.inference.pipeline import TumorSegmentationPipeline


class _DummyClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, image: torch.Tensor) -> float:
        return self.probability


class _DummySegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def predict_mask(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        _, h, w, d = image.shape
        return torch.zeros((3, h, w, d), dtype=torch.float32)


def test_pipeline_skips_segmentation_when_probability_is_low():
    image = torch.randn(4, 8, 8, 8)
    classifier = _DummyClassifier(probability=0.2)
    segmenter = _DummySegmenter()

    pipeline = TumorSegmentationPipeline(
        classifier=classifier,
        segmenter=segmenter,
        classifier_threshold=0.5,
    )
    result = pipeline.run(image)

    assert result.has_tumor is False
    assert result.segmentation_mask is None
    assert segmenter.calls == 0


def test_pipeline_runs_segmentation_when_probability_is_high():
    image = torch.randn(4, 8, 8, 8)
    classifier = _DummyClassifier(probability=0.9)
    segmenter = _DummySegmenter()

    pipeline = TumorSegmentationPipeline(
        classifier=classifier,
        segmenter=segmenter,
        classifier_threshold=0.5,
    )
    result = pipeline.run(image)

    assert result.has_tumor is True
    assert result.segmentation_mask is not None
    assert tuple(result.segmentation_mask.shape) == (3, 8, 8, 8)
    assert segmenter.calls == 1
