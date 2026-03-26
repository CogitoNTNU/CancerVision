import subprocess
import sys
import unittest

import torch

from src.models.dynnet import build_model


class DynnetSmokeTests(unittest.TestCase):
    def test_build_model_smoke_forward(self) -> None:
        model = build_model()
        model.eval()
        inputs = torch.randn(1, 4, 32, 32, 32)

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (1, 3, 32, 32, 32))

    def test_dynnet_module_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.models.dynnet", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("DynUNet", result.stdout)


if __name__ == "__main__":
    unittest.main()
