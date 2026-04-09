import math
import subprocess
import sys
import unittest
from collections import ChainMap
from pathlib import Path

import torch

from src.datasets import ConvertToMultiChannelBasedOnBratsClassesd
from src.models.dynnet import (
    build_model,
    load_split_manifest,
    resolve_distributed_launch_config,
    resolve_num_workers,
    save_split_manifest,
    summarize_segmentation_metrics,
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
        self.assertIn("--label-schema", result.stdout)
        self.assertIn("--compute-hd95", result.stdout)


class LabelTransformTests(unittest.TestCase):
    def test_brats_label_conversion_maps_tc_wt_et(self) -> None:
        transform = ConvertToMultiChannelBasedOnBratsClassesd(keys="label")
        label = torch.tensor(
            [[[[0, 1], [2, 4]], [[1, 2], [4, 0]]]],
            dtype=torch.int64,
        )

        converted = transform({"label": label})["label"]

        expected_tc = torch.tensor(
            [[[0, 1], [0, 1]], [[1, 0], [1, 0]]],
            dtype=torch.float32,
        )
        expected_wt = torch.tensor(
            [[[0, 1], [1, 1]], [[1, 1], [1, 0]]],
            dtype=torch.float32,
        )
        expected_et = torch.tensor(
            [[[0, 0], [0, 1]], [[0, 0], [1, 0]]],
            dtype=torch.float32,
        )

        self.assertEqual(tuple(converted.shape), (3, 2, 2, 2))
        self.assertTrue(torch.equal(converted[0], expected_tc))
        self.assertTrue(torch.equal(converted[1], expected_wt))
        self.assertTrue(torch.equal(converted[2], expected_et))


class SplitManifestTests(unittest.TestCase):
    def test_split_manifest_round_trip(self) -> None:
        data_dicts = [
            {"patient_id": "p1", "image": ["a"], "label": "a_seg"},
            {"patient_id": "p2", "image": ["b"], "label": "b_seg"},
            {"patient_id": "p3", "image": ["c"], "label": "c_seg"},
        ]
        train_dicts = [data_dicts[0], data_dicts[2]]
        val_dicts = [data_dicts[1]]

        manifest_path = Path("split-manifest-test.json").resolve()
        try:
            save_split_manifest(
                manifest_path,
                train_dicts,
                val_dicts,
                data_dir="/tmp/data",
                seed=7,
                test_size=0.2,
                label_schema="brats3",
            )

            loaded_train, loaded_val = load_split_manifest(manifest_path, data_dicts)
        finally:
            if manifest_path.exists():
                manifest_path.unlink()

        self.assertEqual([item["patient_id"] for item in loaded_train], ["p1", "p3"])
        self.assertEqual([item["patient_id"] for item in loaded_val], ["p2"])


class MetricSummaryTests(unittest.TestCase):
    def test_hd95_uses_physical_spacing(self) -> None:
        prediction = torch.zeros((1, 6, 6, 6), dtype=torch.float32)
        label = torch.zeros((1, 6, 6, 6), dtype=torch.float32)
        prediction[0, 2:4, 1:3, 1:3] = 1
        label[0, 1:3, 1:3, 1:3] = 1

        metrics = summarize_segmentation_metrics(
            [prediction],
            [label],
            ["fg"],
            compute_hd95=True,
            spacings=[[2.0, 1.0, 1.0]],
        )

        self.assertAlmostEqual(metrics["hd95_fg"], 2.0, places=4)
        self.assertGreater(metrics["dice_fg"], 0.0)

    def test_empty_cases_are_excluded_and_surfaced(self) -> None:
        empty_prediction = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        empty_label = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        missed_prediction = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        missed_label = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        missed_label[0, 1:3, 1:3, 1:3] = 1

        metrics = summarize_segmentation_metrics(
            [empty_prediction, missed_prediction],
            [empty_label, missed_label],
            ["fg"],
            compute_hd95=True,
            spacings=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        )

        self.assertEqual(metrics["empty_both_case_count_fg"], 1.0)
        self.assertEqual(metrics["missed_case_count_fg"], 1.0)
        self.assertEqual(metrics["hd95_valid_case_count_fg"], 0.0)
        self.assertTrue(math.isnan(metrics["hd95_mean"]))


class WorkerResolutionTests(unittest.TestCase):
    def test_requested_workers_are_capped_by_slurm_cpu_budget(self) -> None:
        workers, reason = resolve_num_workers(
            4,
            env=ChainMap({"SLURM_CPUS_PER_TASK": "1"}, {}),
            cpu_count=8,
        )

        self.assertEqual(workers, 1)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("capped", reason)

    def test_default_workers_are_derived_from_cpu_budget(self) -> None:
        workers, reason = resolve_num_workers(
            None,
            env=ChainMap({"SLURM_CPUS_PER_TASK": "3"}, {}),
            cpu_count=16,
        )

        self.assertEqual(workers, 3)
        self.assertIsNotNone(reason)


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
