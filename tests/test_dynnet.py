import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import torch

import src.models.dynnet_data as dynnet_data_module
from src.models.dynnet import (
    DEFAULT_CANCERVISION_TASK_MANIFEST,
    DEFAULT_GPU_PROFILE_NAME,
    GPU_PROFILE_CONFIGS,
    GpuProfileConfig,
    apply_gpu_profile_defaults,
    apply_path_prefix_maps,
    build_cancervision_segmentation_splits,
    build_dataset_splits,
    build_micro_batch_slices,
    build_model,
    detect_gpu_profile_from_device,
    detect_gpu_profile_from_constraints,
    detect_requested_world_size,
    get_dataset_config,
    is_cuda_oom_error,
    load_resume_state,
    parse_args,
    parse_path_prefix_map,
    resolve_cancervision_task_manifest_path,
    resolve_gpu_profile,
    save_last_checkpoint,
    setup_device_and_distributed,
    validate_args,
)


class DynnetSmokeTests(unittest.TestCase):
    def _touch(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def _write_nifti(self, path: Path, shape: tuple[int, int, int]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine=np.eye(4)), path)
        return path

    def test_build_model_smoke_forward(self) -> None:
        model = build_model()
        model.eval()
        inputs = torch.randn(1, 4, 32, 32, 32)

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (1, 3, 32, 32, 32))

    def test_build_model_supports_cancervision_binary_shape(self) -> None:
        model = build_model(in_channels=1, out_channels=1)
        model.eval()
        inputs = torch.randn(1, 1, 32, 32, 32)

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (1, 1, 32, 32, 32))

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
        self.assertIsNone(args.val_sw_batch_size)
        self.assertIsNone(args.roi_size)
        self.assertIsNone(args.num_samples)
        self.assertEqual(args.gpu_profile, "auto")
        self.assertEqual(args.dataset_source, "brats")
        self.assertIn("cancervision-standardized", str(DEFAULT_CANCERVISION_TASK_MANIFEST))
        self.assertEqual(args.path_prefix_map, [])

    def test_parse_path_prefix_map_requires_from_to_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected FROM=TO"):
            parse_path_prefix_map("bad-mapping")

    def test_apply_path_prefix_maps_rewrites_matching_prefix(self) -> None:
        remapped = apply_path_prefix_maps(
            r"Z:\dataset\cancervision-standardized\segmentation_native\case\image.nii.gz",
            [
                r"Z:\dataset\cancervision-standardized=C:\Users\Polar\Documents\GitHub\CancerVision\res\dataset\cancervision-standardized"
            ],
        )

        self.assertEqual(
            remapped,
            os.path.normpath(
                r"C:\Users\Polar\Documents\GitHub\CancerVision\res\dataset\cancervision-standardized\segmentation_native\case\image.nii.gz"
            ),
        )

    def test_build_micro_batch_slices_splits_effective_batch(self) -> None:
        slices = build_micro_batch_slices(total_batch_size=4, requested_micro_batch_size=1)

        self.assertEqual(slices, [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4)])

    def test_build_micro_batch_slices_clamps_large_requests(self) -> None:
        slices = build_micro_batch_slices(total_batch_size=3, requested_micro_batch_size=8)

        self.assertEqual(slices, [slice(0, 3)])

    def test_get_dataset_config_supports_cancervision_binary_seg(self) -> None:
        config = get_dataset_config("cancervision_binary_seg")

        self.assertEqual(config.in_channels, 1)
        self.assertEqual(config.out_channels, 1)
        self.assertEqual(config.metric_names, ("lesion",))

    def test_build_cancervision_segmentation_splits_reads_task_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image1 = self._touch(root / "images" / "image1.nii.gz")
            mask1 = self._touch(root / "masks" / "mask1.nii.gz")
            image2 = self._touch(root / "images" / "image2.nii.gz")
            mask2 = self._touch(root / "masks" / "mask2.nii.gz")
            image3 = self._touch(root / "images" / "image3.nii.gz")
            mask3 = self._touch(root / "masks" / "mask3.nii.gz")
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,train,case-1,images/image1.nii.gz,masks/mask1.nii.gz,",
                        "segmentation_binary_curated,val,case-2,images/image2.nii.gz,masks/mask2.nii.gz,",
                        "segmentation_binary_curated,test,case-3,images/image3.nii.gz,masks/mask3.nii.gz,",
                        "segmentation_binary_curated,train,case-4,images/image4.nii.gz,,",
                    ]
                ),
                encoding="utf-8",
            )

            train_rows, val_rows, test_rows = build_cancervision_segmentation_splits(
                manifest
            )

            self.assertEqual(train_rows, [{"image": os.path.normpath(str(image1)), "label": os.path.normpath(str(mask1))}])
            self.assertEqual(val_rows, [{"image": os.path.normpath(str(image2)), "label": os.path.normpath(str(mask2))}])
            self.assertEqual(test_rows, [{"image": os.path.normpath(str(image3)), "label": os.path.normpath(str(mask3))}])

    def test_build_cancervision_segmentation_splits_ignores_extra_csv_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image1 = self._touch(root / "image1.nii.gz")
            mask1 = self._touch(root / "mask1.nii.gz")
            image2 = self._touch(root / "image2.nii.gz")
            mask2 = self._touch(root / "mask2.nii.gz")
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,train,case-1,image1.nii.gz,mask1.nii.gz,,extra-field",
                        "segmentation_binary_curated,val,case-2,image2.nii.gz,mask2.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            train_rows, val_rows, test_rows = build_cancervision_segmentation_splits(
                manifest
            )

            self.assertEqual(
                train_rows,
                [{"image": os.path.normpath(str(image1)), "label": os.path.normpath(str(mask1))}],
            )
            self.assertEqual(
                val_rows,
                [{"image": os.path.normpath(str(image2)), "label": os.path.normpath(str(mask2))}],
            )
            self.assertEqual(test_rows, [])

    def test_build_cancervision_segmentation_splits_rejects_duplicate_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._touch(root / "image1.nii.gz")
            self._touch(root / "mask1.nii.gz")
            self._touch(root / "image2.nii.gz")
            self._touch(root / "mask2.nii.gz")
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,train,dup-case,image1.nii.gz,mask1.nii.gz,",
                        "segmentation_binary_curated,val,dup-case,image2.nii.gz,mask2.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Duplicate case 'dup-case'"):
                build_cancervision_segmentation_splits(manifest)

    def test_build_cancervision_segmentation_splits_skips_shape_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            good_image = self._write_nifti(root / "good_image.nii.gz", (32, 32, 16))
            good_mask = self._write_nifti(root / "good_mask.nii.gz", (32, 32, 16))
            bad_image = self._write_nifti(root / "bad_image.nii.gz", (32, 32, 16))
            bad_mask = self._write_nifti(root / "bad_mask.nii.gz", (16, 16, 16))
            val_image = self._write_nifti(root / "val_image.nii.gz", (32, 32, 16))
            val_mask = self._write_nifti(root / "val_mask.nii.gz", (32, 32, 16))
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,dataset_key,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,train,good-case,brats2020,good_image.nii.gz,good_mask.nii.gz,",
                        "segmentation_binary_curated,train,bad-case,utsw_glioma,bad_image.nii.gz,bad_mask.nii.gz,",
                        "segmentation_binary_curated,val,val-case,brats2020,val_image.nii.gz,val_mask.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertWarnsRegex(UserWarning, "shape mismatches"):
                train_rows, val_rows, test_rows = build_cancervision_segmentation_splits(
                    manifest
                )

            self.assertEqual(
                train_rows,
                [{"image": os.path.normpath(str(good_image)), "label": os.path.normpath(str(good_mask))}],
            )
            self.assertEqual(
                val_rows,
                [{"image": os.path.normpath(str(val_image)), "label": os.path.normpath(str(val_mask))}],
            )
            self.assertEqual(test_rows, [])

    def test_build_dataset_splits_uses_cancervision_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image1 = self._touch(root / "image1.nii.gz")
            mask1 = self._touch(root / "mask1.nii.gz")
            image2 = self._touch(root / "image2.nii.gz")
            mask2 = self._touch(root / "mask2.nii.gz")
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,train,case-1,image1.nii.gz,mask1.nii.gz,",
                        "segmentation_binary_curated,val,case-2,image2.nii.gz,mask2.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--dataset-source",
                    "cancervision_binary_seg",
                    "--task-manifest",
                    str(manifest),
                ]
            )
            train_rows, val_rows, test_rows = build_dataset_splits(args)

            self.assertEqual(train_rows, [{"image": os.path.normpath(str(image1)), "label": os.path.normpath(str(mask1))}])
            self.assertEqual(val_rows, [{"image": os.path.normpath(str(image2)), "label": os.path.normpath(str(mask2))}])
            self.assertEqual(test_rows, [])

    def test_build_dataset_splits_applies_path_prefix_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image1 = self._touch(root / "segmentation_native" / "case-1" / "image.nii.gz")
            mask1 = self._touch(root / "segmentation_native" / "case-1" / "mask.nii.gz")
            image2 = self._touch(root / "segmentation_native" / "case-2" / "image.nii.gz")
            mask2 = self._touch(root / "segmentation_native" / "case-2" / "mask.nii.gz")
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        r"segmentation_binary_curated,train,case-1,Z:\dataset\cancervision-standardized\segmentation_native\case-1\image.nii.gz,Z:\dataset\cancervision-standardized\segmentation_native\case-1\mask.nii.gz,",
                        r"segmentation_binary_curated,val,case-2,Z:\dataset\cancervision-standardized\segmentation_native\case-2\image.nii.gz,Z:\dataset\cancervision-standardized\segmentation_native\case-2\mask.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--dataset-source",
                    "cancervision_binary_seg",
                    "--task-manifest",
                    str(manifest),
                    "--path-prefix-map",
                    rf"Z:\dataset\cancervision-standardized={root}",
                ]
            )
            train_rows, val_rows, test_rows = build_dataset_splits(args)

            self.assertEqual(
                train_rows,
                [{"image": os.path.normpath(str(image1)), "label": os.path.normpath(str(mask1))}],
            )
            self.assertEqual(
                val_rows,
                [{"image": os.path.normpath(str(image2)), "label": os.path.normpath(str(mask2))}],
            )
            self.assertEqual(test_rows, [])

    def test_resolve_cancervision_task_manifest_path_falls_back_to_materialized_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_dir = root / "task_manifests"
            seg_native_dir = root / "segmentation_native"
            task_dir.mkdir(parents=True, exist_ok=True)
            seg_native_dir.mkdir(parents=True, exist_ok=True)

            curated_manifest = task_dir / "segmentation_binary_curated.csv"
            curated_manifest.write_text(
                "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason\n",
                encoding="utf-8",
            )
            materialized_manifest = seg_native_dir / "segmentation_materialized_manifest.csv"
            materialized_manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,train,case-1,image1.nii.gz,mask1.nii.gz,",
                        "segmentation_binary_curated,val,case-2,image2.nii.gz,mask2.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertWarnsRegex(UserWarning, "falling back to materialized manifest"):
                resolved = resolve_cancervision_task_manifest_path(curated_manifest)

            self.assertEqual(resolved, materialized_manifest)

    def test_build_dataset_splits_falls_back_to_materialized_manifest_and_default_path_remap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_dir = root / "task_manifests"
            seg_native_dir = root / "segmentation_native"
            task_dir.mkdir(parents=True, exist_ok=True)
            seg_native_dir.mkdir(parents=True, exist_ok=True)

            image1 = self._touch(seg_native_dir / "case-1" / "image.nii.gz")
            mask1 = self._touch(seg_native_dir / "case-1" / "mask.nii.gz")
            image2 = self._touch(seg_native_dir / "case-2" / "image.nii.gz")
            mask2 = self._touch(seg_native_dir / "case-2" / "mask.nii.gz")

            curated_manifest = task_dir / "segmentation_binary_curated.csv"
            curated_manifest.write_text(
                "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason\n",
                encoding="utf-8",
            )
            materialized_manifest = seg_native_dir / "segmentation_materialized_manifest.csv"
            materialized_manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        r"segmentation_binary_curated,train,case-1,Z:\dataset\cancervision-standardized\segmentation_native\case-1\image.nii.gz,Z:\dataset\cancervision-standardized\segmentation_native\case-1\mask.nii.gz,",
                        r"segmentation_binary_curated,val,case-2,Z:\dataset\cancervision-standardized\segmentation_native\case-2\image.nii.gz,Z:\dataset\cancervision-standardized\segmentation_native\case-2\mask.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--dataset-source",
                    "cancervision_binary_seg",
                    "--task-manifest",
                    str(curated_manifest),
                ]
            )
            with mock.patch.object(
                dynnet_data_module,
                "DEFAULT_CANCERVISION_DATASET_ROOT",
                root,
            ):
                with self.assertWarnsRegex(
                    UserWarning, "falling back to materialized manifest"
                ):
                    train_rows, val_rows, test_rows = build_dataset_splits(args)

            self.assertEqual(
                train_rows,
                [{"image": os.path.normpath(str(image1)), "label": os.path.normpath(str(mask1))}],
            )
            self.assertEqual(
                val_rows,
                [{"image": os.path.normpath(str(image2)), "label": os.path.normpath(str(mask2))}],
            )
            self.assertEqual(test_rows, [])

    def test_build_dataset_splits_rejects_missing_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._touch(root / "image1.nii.gz")
            self._touch(root / "mask1.nii.gz")
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,val,case-1,image1.nii.gz,mask1.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--dataset-source",
                    "cancervision_binary_seg",
                    "--task-manifest",
                    str(manifest),
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "No train rows found"):
                build_dataset_splits(args)

    def test_build_dataset_splits_rejects_missing_val_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._touch(root / "image1.nii.gz")
            self._touch(root / "mask1.nii.gz")
            manifest = root / "segmentation_binary_curated.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "task_name,task_split,global_case_id,image_path,mask_path,exclude_reason",
                        "segmentation_binary_curated,train,case-1,image1.nii.gz,mask1.nii.gz,",
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--dataset-source",
                    "cancervision_binary_seg",
                    "--task-manifest",
                    str(manifest),
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "No val rows found"):
                build_dataset_splits(args)


class SingleGpuLaunchTests(unittest.TestCase):
    def _cpu_context(self) -> object:
        return type(
            "FakeContext",
            (),
            {"device": torch.device("cpu"), "distributed": False, "rank": 0, "local_rank": 0, "world_size": 1},
        )()

    def test_detect_requested_world_size_prefers_world_size(self) -> None:
        self.assertEqual(detect_requested_world_size({"WORLD_SIZE": "3", "SLURM_NTASKS": "1"}), 3)

    def test_detect_requested_world_size_falls_back_to_slurm_ntasks(self) -> None:
        self.assertEqual(detect_requested_world_size({"SLURM_NTASKS": "2"}), 2)

    def test_setup_device_and_distributed_rejects_multi_process_launch(self) -> None:
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "single-process single-GPU runs only"):
                setup_device_and_distributed()

    def test_detect_gpu_profile_from_constraints_prefers_memory_tier(self) -> None:
        self.assertEqual(
            detect_gpu_profile_from_constraints({"SLURM_JOB_CONSTRAINTS": "a100&gpu80g"}),
            "gpu80g",
        )

    def test_detect_gpu_profile_from_constraints_maps_architecture_alias(self) -> None:
        self.assertEqual(
            detect_gpu_profile_from_constraints({"SLURM_JOB_CONSTRAINTS": "v100"}),
            "gpu32g",
        )

    def test_detect_gpu_profile_from_device_prefers_free_memory_over_total_memory(self) -> None:
        context = type("FakeContext", (), {"device": torch.device("cuda:0")})()
        fake_properties = type(
            "FakeCudaProperties",
            (),
            {"total_memory": 80 * 1024**3},
        )()

        with mock.patch(
            "src.models.dynnet_runtime.torch.cuda.get_device_properties",
            return_value=fake_properties,
        ), mock.patch(
            "src.models.dynnet_runtime.torch.cuda.mem_get_info",
            return_value=(30 * 1024**3, 80 * 1024**3),
        ):
            self.assertEqual(detect_gpu_profile_from_device(context), "gpu32g")

    def test_is_cuda_oom_error_matches_runtime_error_text(self) -> None:
        exc = RuntimeError("CUDA error: out of memory")

        self.assertTrue(is_cuda_oom_error(exc))

    def test_resolve_gpu_profile_uses_default_when_no_constraints_or_cuda(self) -> None:
        profile = resolve_gpu_profile("auto", self._cpu_context(), {})

        self.assertIsInstance(profile, GpuProfileConfig)
        self.assertEqual(profile.name, DEFAULT_GPU_PROFILE_NAME)

    def test_apply_gpu_profile_defaults_uses_requested_constraint_profile(self) -> None:
        args = parse_args([])
        profile = apply_gpu_profile_defaults(
            args,
            self._cpu_context(),
            {"SLURM_JOB_CONSTRAINTS": "gpu16g"},
        )

        self.assertEqual(profile.name, "gpu16g")
        self.assertEqual(tuple(args.roi_size), GPU_PROFILE_CONFIGS["gpu16g"].roi_size)
        self.assertEqual(args.num_samples, GPU_PROFILE_CONFIGS["gpu16g"].num_samples)
        self.assertEqual(
            tuple(args.model_filters),
            GPU_PROFILE_CONFIGS["gpu16g"].model_filters,
        )
        self.assertEqual(
            args.val_sw_batch_size,
            GPU_PROFILE_CONFIGS["gpu16g"].val_sw_batch_size,
        )

    def test_apply_gpu_profile_defaults_keeps_explicit_overrides(self) -> None:
        args = parse_args(
            [
                "--gpu-profile",
                "gpu80g",
                "--roi-size",
                "72",
                "72",
                "72",
                "--num-samples",
                "3",
                "--model-filters",
                "8",
                "16",
                "32",
                "64",
                "96",
                "--val-sw-batch-size",
                "2",
            ]
        )
        profile = apply_gpu_profile_defaults(args, self._cpu_context(), {})

        self.assertEqual(profile.name, "gpu80g")
        self.assertEqual(tuple(args.roi_size), (72, 72, 72))
        self.assertEqual(args.num_samples, 3)
        self.assertEqual(tuple(args.model_filters), (8, 16, 32, 64, 96))
        self.assertEqual(args.val_sw_batch_size, 2)

    def test_validate_args_allows_thin_z_roi_for_segmentation_cases(self) -> None:
        args = parse_args(
            [
                "--roi-size",
                "48",
                "48",
                "16",
                "--num-samples",
                "1",
                "--model-filters",
                "8",
                "16",
                "32",
                "64",
                "96",
                "--val-sw-batch-size",
                "1",
            ]
        )

        validate_args(args)

    def test_save_and_load_resume_state_round_trip(self) -> None:
        model = build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_last_checkpoint(
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=3,
                best_metric=0.75,
                best_metric_epoch=2,
            )

            restored_model = build_model()
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-4)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer,
                T_max=5,
            )
            start_epoch, best_metric, best_metric_epoch = load_resume_state(
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                scaler=None,
                resume_path=str(run_dir / "last_checkpoint.pt"),
                context=self._cpu_context(),
            )

            self.assertEqual(start_epoch, 4)
            self.assertEqual(best_metric, 0.75)
            self.assertEqual(best_metric_epoch, 2)


if __name__ == "__main__":
    unittest.main()
