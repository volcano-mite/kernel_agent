#include <cuda_runtime.h>

#include <cstddef>

namespace {
constexpr int kTileSize = 32;
constexpr int kBlockRows = 8;

// Pad each shared row so a warp reading one column avoids bank conflicts.
__global__ void transpose_kernel(const float* input, float* output, int rows, int columns) {
    __shared__ float tile[kTileSize][kTileSize + 1];
    const int tile_row = blockIdx.y * kTileSize;
    const int tile_column = blockIdx.x * kTileSize;
    const int local_row = threadIdx.y;
    const int local_column = threadIdx.x;

    for (int offset = 0; offset < kTileSize; offset += kBlockRows) {
        const int row = tile_row + local_row + offset;
        const int col = tile_column + local_column;
        if (row < rows && col < columns) {
            tile[local_row + offset][local_column] =
                input[static_cast<std::size_t>(row) * columns + col];
        }
    }
    __syncthreads();

    for (int offset = 0; offset < kTileSize; offset += kBlockRows) {
        const int row = tile_column + local_row + offset;
        const int col = tile_row + local_column;
        if (row < columns && col < rows) {
            output[static_cast<std::size_t>(row) * rows + col] =
                tile[local_column][local_row + offset];
        }
    }
}
}  // namespace

// Input and output are separate contiguous device buffers in row-major order.
cudaError_t transpose(const float* input, float* output, int rows, int columns,
                      cudaStream_t stream) {
    if (rows < 0 || columns < 0) {
        return cudaErrorInvalidValue;
    }
    if (rows == 0 || columns == 0) {
        return cudaSuccess;
    }
    const dim3 block(kTileSize, kBlockRows);
    const dim3 grid((static_cast<unsigned int>(columns) + kTileSize - 1) / kTileSize,
                    (static_cast<unsigned int>(rows) + kTileSize - 1) / kTileSize);
    transpose_kernel<<<grid, block, 0, stream>>>(input, output, rows, columns);
    return cudaGetLastError();
}
