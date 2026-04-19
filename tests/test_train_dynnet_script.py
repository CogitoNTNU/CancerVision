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
