import subprocess
import sys
import unittest
from collections import ChainMap

import torch

from src.models.dynnet import (
    build_micro_batch_slices,
    build_model,
    parse_args,
    resolve_distributed_launch_config,
)


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

    def test_parse_args_defaults_include_memory_safe_batching(self) -> None:
        args = parse_args([])

        self.assertEqual(args.train_micro_batch_size, 1)
        self.assertEqual(args.val_sw_batch_size, 1)

    def test_build_micro_batch_slices_splits_effective_batch(self) -> None:
        slices = build_micro_batch_slices(total_batch_size=4, requested_micro_batch_size=1)

        self.assertEqual(slices, [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4)])

    def test_build_micro_batch_slices_clamps_large_requests(self) -> None:
        slices = build_micro_batch_slices(total_batch_size=3, requested_micro_batch_size=8)

        self.assertEqual(slices, [slice(0, 3)])


class DistributedLaunchConfigTests(unittest.TestCase):
    def _env(self, **overrides: str) -> ChainMap[str, str]:
        return ChainMap(
            overrides,
            {
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
            },
        )

    def test_world_size_one_uses_non_distributed_path(self) -> None:
        config = resolve_distributed_launch_config(
            self._env(),
            cuda_available=True,
            visible_gpu_count=1,
        )

        self.assertFalse(config.distributed)
        self.assertEqual(config.world_size, 1)
        self.assertIsNone(config.device_index)

    def test_one_visible_gpu_per_process_maps_to_cuda_zero(self) -> None:
        config = resolve_distributed_launch_config(
            self._env(WORLD_SIZE="2", RANK="1", LOCAL_RANK="1", SLURM_GPUS_ON_NODE="2"),
            cuda_available=True,
            visible_gpu_count=1,
        )

        self.assertTrue(config.distributed)
        self.assertEqual(config.device_index, 0)

    def test_multi_visible_gpu_process_maps_to_local_rank(self) -> None:
        config = resolve_distributed_launch_config(
            self._env(WORLD_SIZE="2", RANK="1", LOCAL_RANK="1", SLURM_GPUS_ON_NODE="2"),
            cuda_available=True,
            visible_gpu_count=2,
        )

        self.assertEqual(config.device_index, 1)

    def test_allocated_gpu_count_smaller_than_world_size_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "allocated GPU count is smaller than world size"):
            resolve_distributed_launch_config(
                self._env(WORLD_SIZE="2", RANK="0", LOCAL_RANK="0", SLURM_GPUS_ON_NODE="1"),
                cuda_available=True,
                visible_gpu_count=1,
            )

    def test_zero_visible_gpus_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires at least one CUDA-visible GPU per process"):
            resolve_distributed_launch_config(
                self._env(WORLD_SIZE="2", RANK="0", LOCAL_RANK="0", SLURM_GPUS_ON_NODE="2"),
                cuda_available=True,
                visible_gpu_count=0,
            )


if __name__ == "__main__":
    unittest.main()
