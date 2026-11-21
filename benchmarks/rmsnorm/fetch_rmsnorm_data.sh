#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../gpu_env.sh"

revision=da915083d4c7c5e61aa3005e3d17ae488e0fc71c
base="https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/resolve/$revision"
for path in definitions/rmsnorm/rmsnorm_h4096.json workloads/rmsnorm/rmsnorm_h4096.jsonl; do
    target="$FIB_DATASET_PATH/$path"
    mkdir -p "$(dirname "$target")"
    download="$(mktemp "$target.XXXXXX")"
    trap 'rm -f -- "$download"' EXIT
    curl --fail --silent --show-error --location --retry 3 "$base/$path" --output "$download"
    mv -- "$download" "$target"
    trap - EXIT
done
printf 'FlashInfer-Trace %s RMSNorm files are in %s\n' "$revision" "$FIB_DATASET_PATH"
