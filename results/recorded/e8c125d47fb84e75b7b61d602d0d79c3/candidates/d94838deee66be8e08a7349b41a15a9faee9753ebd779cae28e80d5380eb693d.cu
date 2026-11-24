#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

namespace {
constexpr int kHiddenSize = 4096;

__device__ __forceinline__ float warp_sum(float sum) {
    for (int delta = 16; delta > 0; delta >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    return sum;
}

template<int Threads, bool Vectorized>
__global__ void rmsnorm_kernel(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                               __nv_bfloat16* output) {
    const auto offset = static_cast<int64_t>(blockIdx.x) * kHiddenSize;
    input += offset;
    output += offset;
    constexpr int Width = Vectorized ? 8 : 1;
    constexpr int Chunks = kHiddenSize / (Threads * Width);
    float values[Chunks][Width];
    float sum = 0.0f;
    #pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        const int col = (chunk * Threads + threadIdx.x) * Width;
        if constexpr (Vectorized) {
            union { uint4 vector; __nv_bfloat16 elements[8]; } data;
            data.vector = reinterpret_cast<const uint4*>(input + col)[0];
            #pragma unroll
            for (int i = 0; i < Width; ++i) {
                const float x = __bfloat162float(data.elements[i]);
                values[chunk][i] = x;
                sum = fmaf(x, x, sum);
            }
        } else {
            const float x = __bfloat162float(input[col]);
            values[chunk][0] = x;
            sum = fmaf(x, x, sum);
        }
    }
    __shared__ float partial[Threads / 32];
    __shared__ float scale;
    sum = warp_sum(sum);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < Threads / 32 ? partial[lane] : 0.0f);
        if (lane == 0) scale = rsqrtf(sum * (1.0f / kHiddenSize) + 1e-5f);
    }
    __syncthreads();
    const float inv_rms = scale;
    #pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        const int col = (chunk * Threads + threadIdx.x) * Width;
        if constexpr (Vectorized) {
            union { uint4 vector; __nv_bfloat16 elements[8]; } factors, result;
            factors.vector = reinterpret_cast<const uint4*>(weight + col)[0];
            #pragma unroll
            for (int i = 0; i < Width; ++i)
                result.elements[i] = __float2bfloat16_rn(
                    (values[chunk][i] * inv_rms) * __bfloat162float(factors.elements[i]));
            reinterpret_cast<uint4*>(output + col)[0] = result.vector;
        } else {
            output[col] = __float2bfloat16_rn(
                (values[chunk][0] * inv_rms) * __bfloat162float(weight[col]));
        }
    }
}
}  // namespace

void run(const torch::Tensor& input, const torch::Tensor& weight, torch::Tensor output) {
    const c10::cuda::CUDAGuard guard(input.device());
    const auto rows = input.size(0);
    if (rows == 0) return;
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
    auto* y = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto alignment = reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) |
                           reinterpret_cast<uintptr_t>(y);
    if (alignment & 15)
        rmsnorm_kernel<128, false><<<rows, 128, 0, stream>>>(x, w, y);
    else if (rows <= 170)
        rmsnorm_kernel<512, true><<<rows, 512, 0, stream>>>(x, w, y);
    else
        rmsnorm_kernel<128, true><<<rows, 128, 0, stream>>>(x, w, y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
