# Performance manual

Everything enabled in `scripts/run_2xa100.sh` is an independent flag on
`src.training.train`. This doc explains what each one actually does so you can
disable individual ones when diagnosing issues.

## Multi-GPU

### `torchrun --nproc_per_node=2`
One process per GPU. `src.training.distributed.setup_runtime()` reads
`WORLD_SIZE` / `RANK` / `LOCAL_RANK`, pins each process to its GPU, and calls
`init_process_group(backend="nccl", device_id=<cuda:N>)`. The `device_id`
argument silences the "barrier using current device" warning.

NCCL env tuning in the launcher:
- `NCCL_P2P_LEVEL=NVL` — prefer direct peer-to-peer over NVLink
- `TORCH_NCCL_TIMEOUT=1800` — val sliding-window at 192³ can run ~10 min per
  patient on startup before caches warm; default 600 s is too tight

### Sharded validation
`_build_loaders()` wraps the val set in a `DistributedSampler` so both GPUs
validate half the patients each. MONAI's `DiceMetric.aggregate()` all-gathers
results in float32. Only rank 0 logs / saves.

## Data path

### `--cache-rate 1.0 --cache-num-workers 8`
Wraps the train and val sets in `monai.data.CacheDataset`. The deterministic
prefix of the transform pipeline (LoadImaged → EnsureChannelFirstd →
ConvertToMultiChannelBasedOnBratsClassesd → NormalizeIntensityd) runs once up
front and results are kept in RAM; only the random crops, flips, and intensity
jitters rerun per step.

RAM cost for BraTS training: ~60 MB/patient × 295 ≈ 18 GB.
First-epoch warmup: ~1–3 min with 8 cache workers.

Set `--cache-rate 0.0` to fall back to lazy `Dataset` (disk on every step) if
you have a tight memory budget.

### `persistent_workers=True`, `pin_memory=True`
Standard. Avoids worker re-spawn cost every epoch and overlaps CPU→GPU copies.

### `torch.multiprocessing.set_sharing_strategy("file_system")`
Avoids the "received 0 items of ancdata" FD-exhaustion error with many
DataLoader workers feeding 3D volumes. Called at module import.

## Compute

### `--amp-dtype bf16`
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. BF16 has the same
8-bit exponent as FP32, so small Dice gradients don't underflow and a
`GradScaler` is unnecessary (set to `None` in `_scaler_for()`). FP16 is also
supported via `--amp-dtype fp16`; it enables `torch.amp.GradScaler("cuda")`.

### `--memory-format channels_last`
Converts the model and inputs to `torch.channels_last_3d` (NDHWC). On A100
this lets cuDNN pick Tensor-Core-friendly conv kernels for 3D convolutions.
Combined with AMP, typical throughput gain is 10–25% for DynUNet.

### `--compile`
`torch.compile(model, mode="default", fullgraph=False, dynamic=False)`
applied **after** DDP wrapping. The first epoch eats 20–60 s of JIT cost;
subsequent epochs run 10–20% faster. `dynamic=False` is safe because the
training ROI is fixed (`RandCropByPosNegLabeld(spatial_size=roi_size)` always
emits the same shape). Validation uses the same compiled forward via sliding
window — the window size is also fixed to `--roi-size`.

`mode="default"` is chosen over `"reduce-overhead"` because the latter's CUDA
graph capture conflicts with DDP's all-reduce bucket resync.

### `--fused-optimizer`
`torch.optim.Adam(fused=True)` replaces the stock per-parameter loop with one
fused CUDA kernel per step. On DynUNet (~16M params with deep supervision)
this is a ~5 % step-time win.

### `--fused-dice-loss`
Our custom CUDA kernel — see the next section.

## Custom CUDA kernel: `src/kernels/fused_dice.cu`

### Why

MONAI's `DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")` is
correct but bandwidth-bound. A straightforward implementation issues
roughly six distinct CUDA kernels:

1. `sigmoid(x)` — write `p`
2. `p * t` — write intersection tensor
3. `sum(...)` — intersection reduction
4. `p ** 2 + t ** 2` — write cardinality tensor
5. `sum(...)` — cardinality reduction
6. scalar epilogue + `.mean()`

Each op reads the full (B, C, D, H, W) activation tensor from HBM and writes
an intermediate back. At 192³ with deep supervision num_heads = 4 (so the
loss is called four times per step over tensors at 1× / 1/2× / 1/4× / 1/8×
scale), the loss alone moves hundreds of MB per step.

### What the kernel does

`fused_dice.cu` replaces those 6 launches with 2 (one forward, one backward)
and reads each logit/target pair *once* per direction.

**Forward.** Grid is `(blocks_x, B*C)`. Each block processes a slice of a
single (batch, channel) row. Threads stream through their slice, computing
`p = sigmoid(x)` inline and accumulating local partial sums for
intersection `p * t` and cardinality `p² + t²` in per-thread float32
registers. A standard shared-memory tree reduction collapses the block to
one (I, C) pair, which is `atomicAdd`-ed into per-row float32 accumulators.
The final scalar loss is computed in a cheap Python epilogue:
`(1 − (2·I + εₙ) / (C + ε_d)).mean()` over the B·C rows.

**Backward.** Uses saved `I_bc` and `C_bc` from the forward plus the original
logits/targets. The closed-form gradient is

```
dL/dx_i = scale · [ (−2 / (C_bc + ε_d)) · tᵢ · σ(xᵢ)(1−σ(xᵢ))
                  + ((2·I_bc + εₙ) / (C_bc + ε_d)²) · 2·σ(xᵢ)²·(1−σ(xᵢ)) ]
```

where `scale = grad_output / (B·C)` folds in the chain rule from the outer
`.mean()`. The backward kernel walks the same grid geometry as the forward and
writes the gradient in one pass.

Accumulation is always in float32 even for FP16/BF16 inputs; the template
parameter only controls loads and stores. `AT_DISPATCH_FLOATING_TYPES_AND2`
handles the three input dtypes (`float`, `at::Half`, `at::BFloat16`).

### How it's exposed to Python

`src/kernels/fused_dice.py`:
- `_get_module()` JIT compiles the `.cu` and a small pybind shim via
  `torch.utils.cpp_extension.load_inline` on first call, cached under
  `$TORCH_EXTENSIONS_DIR` (set to `<repo>/.torch_extensions` by the launcher).
- `_FusedDiceFn(torch.autograd.Function)` wires `forward` to the CUDA
  forward + a float32 epilogue, and `backward` to the CUDA backward using
  saved tensors.
- `FusedDiceLoss(nn.Module)` is the public surface. Drop-in replacement for
  MONAI's `DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")`.

### Correctness

`src/pipeline.preflight(..., fused_dice_loss=True)` compiles the kernel,
runs a forward+backward on random 2×3×16³ tensors, and compares both the
loss value (atol 1e-4) and the gradient (max-abs 1e-4) against MONAI. The
launcher calls this before any training starts, so a miscompiled kernel
fails fast with a clear error.

Unit tests live in `tests/test_fused_dice.py` and cover float32, float16,
and the backward pass. They skip gracefully when CUDA or `nvcc` is missing.

### Expected impact

- Loss forward+backward time: ≈ 2–3× faster than MONAI at 192³
- Full step wall-clock: ≈ 5–10 % faster at num_heads = 4 (four loss calls
  per step)
- Peak activation memory: a few hundred MB lower because the intermediate
  `p*t` and `p²+t²` tensors are never materialized

Small in absolute terms for DynUNet, but every kernel launch and every HBM
pass you remove compounds over 300 epochs × ~74 steps × 2 GPUs.

## Reproducibility

`--deterministic` switches on `set_determinism(seed=args.seed)` which forces
cuDNN to pick deterministic algorithms (slower). The default
`--no-deterministic` enables `cudnn.benchmark` so cuDNN picks the fastest
kernel for fixed shapes on the first iteration. Two runs with identical
hyperparameters will differ slightly at the last decimal of Dice; that's
expected and rarely affects final checkpoints.

## Memory budget (80 GB A100, 192×192×128 ROI, BF16, DS num=3)

| Component | Approx peak |
|---|---|
| Model params + grads + Adam state | ~1.3 GB |
| Activations (forward + DS) | ~30–40 GB |
| AMP temporaries | ~5 GB |
| MONAI CacheDataset in host RAM | ~18 GB (not GPU) |

On 40 GB A100s, drop `--roi-size` to `160 160 128` and `--num-samples` to 1–2.

## Diagnostic recipes

**Training slower than expected.** Confirm both GPUs are hot:
`watch -n 1 nvidia-smi`. If one is idle during training, DDP is broken.
If both are ~70 % and saturated on data, raise `--num-workers`.

**"OOM at epoch 0".** Disable flags in this order to localize: `--no-compile`
→ `--memory-format standard` → `--no-deep-supervision` → smaller ROI.

**"Fused Dice loss mismatch".** The preflight catches this. If it fires on
your hardware, rerun with `--no-fused-dice-loss` and open an issue with the
reported `|Δ|`; the most likely cause is a CUDA version mismatch between
what compiled the kernel and what's running it.
