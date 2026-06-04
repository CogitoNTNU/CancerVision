import subprocess
import unittest
from pathlib import Path


class TrainDynnetScriptTests(unittest.TestCase):
    def test_batch_script_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", "scripts/train_dynnet.sbatch"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_batch_script_runs_dynnet_through_uv(self) -> None:
        script = Path("scripts/train_dynnet.sbatch").read_text(encoding="utf-8")

        self.assertIn("command -v uv", script)
        self.assertIn("uv run --no-sync python -m src.models.dynnet", script)
        self.assertIn('GPU_PROFILE="${GPU_PROFILE:-gpu40g}"', script)
        self.assertIn('LOSS="${LOSS:-dicece}"', script)
        self.assertIn('CROP_POS_WEIGHT="${CROP_POS_WEIGHT:-3}"', script)
        self.assertIn('VAL_THRESHOLDS="${VAL_THRESHOLDS:-0.30 0.35 0.40 0.45 0.50 0.55 0.60}"', script)
        self.assertIn("--loss", script)
        self.assertIn("--crop-pos-weight", script)
        self.assertIn("--val-thresholds", script)
        self.assertIn("--early-stop-patience", script)
