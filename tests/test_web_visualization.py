from pathlib import Path

import torch

from src.web.visualization import create_preview_png


def test_create_preview_png_writes_file(tmp_path: Path):
    image = torch.randn(4, 16, 16, 16)
    mask = torch.zeros(3, 16, 16, 16)
    mask[1, 4:12, 4:12, 8] = 1

    output_path = tmp_path / "preview.png"
    create_preview_png(image=image, mask=mask, output_path=str(output_path), title="test")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
