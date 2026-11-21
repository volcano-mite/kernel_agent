#!/usr/bin/env bash
# Source this from Bash before running FlashInfer-Bench.
bench_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$bench_root/.venv/bin:$CUDA_HOME/bin:$PATH"
if [[ -x /opt/rh/gcc-toolset-14/root/usr/bin/g++ ]]; then
    export CC="${CC:-/opt/rh/gcc-toolset-14/root/usr/bin/gcc}"
    export CXX="${CXX:-/opt/rh/gcc-toolset-14/root/usr/bin/g++}"
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$bench_root}"
export FIB_CACHE_PATH="${FIB_CACHE_PATH:-$bench_root/.cache/flashinfer-bench}"
export FIB_DATASET_PATH="${FIB_DATASET_PATH:-$bench_root/.data/flashinfer-trace-rmsnorm}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$bench_root/.cache/uv}"
unset bench_root
