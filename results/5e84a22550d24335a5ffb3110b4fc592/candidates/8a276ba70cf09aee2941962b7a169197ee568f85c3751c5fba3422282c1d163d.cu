#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace {
constexpr int kHiddenSize = 4096;
constexpr int kThreads = 256;

__global__ void rmsnorm_kernel(const at::BFloat16* input, const at::BFloat16* weight,
                               at::BFloat16* output) {
    const auto row_offset = static_cast<int64_t>(blockIdx.x) * kHiddenSize;
    const float scale = 0.5f;  // Replace with the row's inverse RMS.
    for (int col = threadIdx.x; col < kHiddenSize; col += blockDim.x) {
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
