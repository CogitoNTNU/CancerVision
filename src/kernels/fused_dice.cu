// Fused soft-Dice loss for BraTS-style segmentation outputs.
//
// Computes, in a single pair of kernels:
//   forward :   L = mean_{b,c} ( 1 - (2*I_bc + eps_n) / (C_bc + eps_d) )
//   backward:   dL/dx_i for each input logit x_i
//
// where for each (batch b, channel c) row:
//   p_i      = sigmoid(x_i)
//   I_bc     = sum_i p_i * t_i                 (intersection)
//   C_bc     = sum_i p_i^2 + t_i^2             (squared-pred cardinality)
//   t_i      = binary target in {0, 1}
//
// Each block processes a contiguous chunk of the (b, c) row; partial sums are
// reduced in shared memory and atomically added into per-row accumulators in
// float32. Kernels are templated on scalar_t to support {float16, bfloat16,
// float32} inputs while accumulating in float32 for numerical stability.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void fused_dice_forward_kernel(
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ targets,
    float* __restrict__ intersection,
    float* __restrict__ cardinality,
    const int64_t n_per_row
) {
    __shared__ float sh_i[BLOCK_SIZE];
    __shared__ float sh_c[BLOCK_SIZE];

    const int bc = blockIdx.y;
    const int64_t row_offset = static_cast<int64_t>(bc) * n_per_row;
    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    float local_i = 0.0f;
    float local_c = 0.0f;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_per_row; idx += stride) {
        const float x = static_cast<float>(logits[row_offset + idx]);
        const float t = static_cast<float>(targets[row_offset + idx]);
        const float p = 1.0f / (1.0f + __expf(-x));
        local_i += p * t;
        local_c += p * p + t * t;
    }
    sh_i[tid] = local_i;
    sh_c[tid] = local_c;
    __syncthreads();

    // Shared-memory tree reduction.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_i[tid] += sh_i[tid + s];
            sh_c[tid] += sh_c[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&intersection[bc], sh_i[0]);
        atomicAdd(&cardinality[bc], sh_c[0]);
    }
}

template <typename scalar_t>
__global__ void fused_dice_backward_kernel(
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ targets,
    const float* __restrict__ intersection,
    const float* __restrict__ cardinality,
    scalar_t* __restrict__ grad_logits,
    const int64_t n_per_row,
    const float eps_n,
    const float eps_d,
    const float scale
) {
    const int bc = blockIdx.y;
    const int64_t row_offset = static_cast<int64_t>(bc) * n_per_row;

    // Precompute the per-row Dice derivative factors.
    const float I = intersection[bc];
    const float C = cardinality[bc];
    const float denom = C + eps_d;
    const float inv_denom = 1.0f / denom;
    const float numer = 2.0f * I + eps_n;
    const float inv_denom2 = inv_denom * inv_denom;
    const float dL_dI = -2.0f * inv_denom;
    const float dL_dC = numer * inv_denom2;

    const int tid = threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_per_row; idx += stride) {
        const float x = static_cast<float>(logits[row_offset + idx]);
        const float t = static_cast<float>(targets[row_offset + idx]);
        const float p = 1.0f / (1.0f + __expf(-x));
        const float p_dp = p * (1.0f - p);          // sigmoid'(x)
        const float dI_dx = t * p_dp;               // d/dx [p * t]
        const float dC_dx = 2.0f * p * p_dp;        // d/dx [p^2]  (t^2 is const)
        const float g = scale * (dL_dI * dI_dx + dL_dC * dC_dx);
        grad_logits[row_offset + idx] = static_cast<scalar_t>(g);
    }
}

static void launch_dims(
    int64_t n_per_row, int64_t total_bc, dim3& grid, dim3& block
) {
    block = dim3(BLOCK_SIZE);
    const int64_t blocks_x = std::min<int64_t>(
        (n_per_row + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024
    );
    grid = dim3(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(total_bc));
}

void fused_dice_forward_cuda(
    at::Tensor logits, at::Tensor targets,
    at::Tensor intersection, at::Tensor cardinality
) {
    const int64_t B = logits.size(0);
    const int64_t C = logits.size(1);
    const int64_t total_bc = B * C;
    const int64_t n_per_row = logits.numel() / total_bc;

    dim3 grid, block;
    launch_dims(n_per_row, total_bc, grid, block);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(),
        "fused_dice_forward", [&] {
            fused_dice_forward_kernel<scalar_t><<<grid, block>>>(
                logits.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                intersection.data_ptr<float>(),
                cardinality.data_ptr<float>(),
                n_per_row
            );
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_dice_backward_cuda(
    at::Tensor logits, at::Tensor targets,
    at::Tensor intersection, at::Tensor cardinality,
    at::Tensor grad_logits,
    double eps_n, double eps_d, double grad_scale
) {
    const int64_t B = logits.size(0);
    const int64_t C = logits.size(1);
    const int64_t total_bc = B * C;
    const int64_t n_per_row = logits.numel() / total_bc;

    dim3 grid, block;
    launch_dims(n_per_row, total_bc, grid, block);

    // grad_scale already includes grad_output; divide by (B*C) because the
    // forward takes mean over (B, C).
    const float scale = static_cast<float>(grad_scale / static_cast<double>(total_bc));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(),
        "fused_dice_backward", [&] {
            fused_dice_backward_kernel<scalar_t><<<grid, block>>>(
                logits.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                intersection.data_ptr<float>(),
                cardinality.data_ptr<float>(),
                grad_logits.data_ptr<scalar_t>(),
                n_per_row,
                static_cast<float>(eps_n),
                static_cast<float>(eps_d),
                scale
            );
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
