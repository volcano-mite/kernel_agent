#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace {
constexpr int kHiddenSize = 4096;
constexpr int kThreads = 256;
constexpr float kEpsilon = 1e-5f;

__global__ void rmsnorm_kernel(const at::BFloat16* input, const at::BFloat16* weight,
                              at::BFloat16* output) {
    __shared__ float sums[kThreads];
    const int lane = threadIdx.x;
    const auto row_offset = static_cast<int64_t>(blockIdx.x) * kHiddenSize;

    float squared_sum = 0.0f;
    for (int col = lane; col < kHiddenSize; col += kThreads) {
        const float value = static_cast<float>(input[row_offset + col]);
        squared_sum += value * value;
    }
    sums[lane] = squared_sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(sums[0] / kHiddenSize + kEpsilon);
    for (int col = lane; col < kHiddenSize; col += kThreads) {
        const float value = static_cast<float>(input[row_offset + col]);
        const float factor = static_cast<float>(weight[col]);
        output[row_offset + col] = at::BFloat16((value * scale) * factor);
    }
}
}  // namespace

void run(const torch::Tensor& input, const torch::Tensor& weight, torch::Tensor output) {
    const c10::cuda::CUDAGuard guard(input.device());
    const auto rows = input.size(0);
    if (rows == 0) {
        return;
    }
    rmsnorm_kernel<<<rows, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<at::BFloat16>(), weight.data_ptr<at::BFloat16>(),
        output.data_ptr<at::BFloat16>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
}
