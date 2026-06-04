import unittest

from src.models.dynnet_runtime import (
    bootstrap_distributed_env,
    get_distributed_env,
    resolve_cuda_device_index,
)


class DynnetRuntimeTests(unittest.TestCase):
    def test_get_distributed_env_prefers_torchrun_variables(self):
        env = {
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "SLURM_PROCID": "9",
            "SLURM_LOCALID": "4",
            "SLURM_NTASKS": "16",
        }

        self.assertEqual(get_distributed_env(env), (3, 1, 8))

    def test_get_distributed_env_falls_back_to_slurm_variables(self):
        env = {
            "SLURM_PROCID": "2",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "4",
        }

        self.assertEqual(get_distributed_env(env), (2, 1, 4))

    def test_bootstrap_distributed_env_sets_defaults(self):
        env = {
            "SLURM_PROCID": "2",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_JOB_ID": "12345",
        }

        bootstrap_distributed_env(env)

        self.assertEqual(env["RANK"], "2")
        self.assertEqual(env["LOCAL_RANK"], "1")
        self.assertEqual(env["WORLD_SIZE"], "4")
        self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(env["MASTER_PORT"], str(10000 + (12345 % 50000)))

    def test_resolve_cuda_device_index_uses_zero_when_one_gpu_is_visible(self):
        self.assertEqual(
            resolve_cuda_device_index(local_rank=3, visible_device_count=1),
            0,
        )

    def test_resolve_cuda_device_index_raises_for_invalid_rank_with_multiple_gpus(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid local rank"):
            resolve_cuda_device_index(local_rank=2, visible_device_count=2)
