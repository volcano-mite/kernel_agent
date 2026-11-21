#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 || ! $1 =~ ^GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}$ ]]; then
    printf 'Usage: %s GPU-UUID\n' "$0" >&2
    printf 'Run nvidia-smi -L and pass the full UUID of an idle RTX 5090.\n' >&2
    exit 1
fi

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$experiment_dir/../gpu_env.sh"
export CUDA_VISIBLE_DEVICES="$1"
nvidia-smi -i "$1" --query-gpu=name,uuid --format=csv
artifact_dir="$experiment_dir/../../.artifacts/add-rmsnorm"
mkdir -p "$artifact_dir"
output="$(mktemp -d "$artifact_dir/profile.XXXXXX")"
printf 'Results: %s\n' "$output"

nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 -ccbin "${CXX:-g++}" \
    "$experiment_dir/bench.cu" -o "$output/bench"
"$output/bench" > "$output/timings.csv"

collect() {
    local mode="$2" report="$output/batch-$1"
    ncu --kernel-name 'regex:^(add|rmsnorm|add_rmsnorm)$' --launch-count 3 \
        --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
        -o "$report" "$output/bench" "$mode"
    ncu --import "$report.ncu-rep" --page details > "$report.txt"
}

collect 1 --check-only
collect 14509 --check-large
