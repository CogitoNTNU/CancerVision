from src.models.train_brats import (
    bootstrap_distributed_env,
    get_distributed_env,
    resolve_cuda_device_index,
)


def test_get_distributed_env_prefers_torchrun_variables():
    env = {
        "RANK": "3",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "8",
        "SLURM_PROCID": "9",
        "SLURM_LOCALID": "4",
        "SLURM_NTASKS": "16",
    }

    assert get_distributed_env(env) == (3, 1, 8)


def test_get_distributed_env_falls_back_to_slurm_variables():
    env = {
        "SLURM_PROCID": "2",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS": "4",
    }

    assert get_distributed_env(env) == (2, 1, 4)


def test_bootstrap_distributed_env_sets_torch_distributed_defaults():
    env = {
        "SLURM_PROCID": "2",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS": "4",
        "SLURM_JOB_ID": "12345",
    }

    bootstrap_distributed_env(env)

    assert env["RANK"] == "2"
    assert env["LOCAL_RANK"] == "1"
    assert env["WORLD_SIZE"] == "4"
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == str(10000 + (12345 % 50000))


def test_resolve_cuda_device_index_uses_zero_when_one_gpu_is_visible():
    assert resolve_cuda_device_index(local_rank=3, visible_device_count=1) == 0


def test_resolve_cuda_device_index_raises_for_invalid_rank_with_multiple_gpus():
    try:
        resolve_cuda_device_index(local_rank=2, visible_device_count=2)
    except RuntimeError as exc:
        assert "Invalid local rank" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for out-of-range local rank")
