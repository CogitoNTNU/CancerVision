import subprocess
import unittest


class TrainDynnetScriptTests(unittest.TestCase):
    def test_batch_script_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", "scripts/train_dynnet.sbatch"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
