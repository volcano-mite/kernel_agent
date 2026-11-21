# Kernel Agent

Kernel Agent evaluates CUDA kernel repairs on a Linux x86-64 NVIDIA GPU host. The RMSNorm smoke test uses FlashInfer-Bench's Python API to run `rmsnorm_h4096` on 14 official workloads. The repair evaluation compares one-shot answers with answers that can use benchmark feedback. The Add + RMSNorm experiment measures the effect of fusing a residual add with normalization.

| Path | Contents |
| --- | --- |
| [`scripts/`](scripts/) | Environment setup, data download, smoke test, and repair evaluation |
| [`examples/rmsnorm_smoke/`](examples/rmsnorm_smoke/) | Known-correct CUDA example used by the smoke test and task generator |
| [`benchmarks/rmsnorm/`](benchmarks/rmsnorm/) | Fixed 50-task suite with grading fixtures |
| [`experiments/add_rmsnorm/`](experiments/add_rmsnorm/README.md) | Standalone CUDA microbenchmark and Nsight Compute instructions |
| [`docs/`](docs/) | Evaluation instructions and project plan |

Commit source, docs, the fixed task suite, and the dependency lockfile. Keep virtual environments, downloaded traces, caches, compiled binaries, and per-run outputs local. The smoke test writes to `results/`, the repair evaluation to `eval_results/`, and the microbenchmark to `experiments/add_rmsnorm/profile.*/` when run through `profile.sh`.

See the [development plan](docs/kernel-agent-plan.md) for ownership, the evaluator interface, and the B/D/F experiment.

## Install

Use Python 3.11, `uv`, `git`, `curl`, an NVIDIA driver compatible with CUDA 13.0, and the CUDA 13.0 Toolkit with `nvcc`, `ncu`, and `compute-sanitizer`. The sourced script defaults to `/usr/local/cuda-13.0` and uses GCC 14 from the AlmaLinux toolset when it is available. You can set `CUDA_HOME`, `CC`, or `CXX` before sourcing the script to use a different installation.

```bash
source scripts/gpu_env.sh
uv sync --locked --python 3.11
bash scripts/fetch_rmsnorm_data.sh
```

If `uv` cannot discover a local Python 3.11 distribution, pass its executable path to `--python`. The dependencies and FlashInfer-Bench source revision are in `pyproject.toml` and `uv.lock`. The download script fetches the definition and workload files for this check into `.data/`.

## Verify on one GPU

```bash
source scripts/gpu_env.sh
nvidia-smi -L
export CUDA_VISIBLE_DEVICES=1  # RTX 5090 on Ryan's host. Use the index or UUID of your GPU.
flashinfer-bench run --help
python scripts/smoke_rmsnorm.py --diagnostics
```

This compiles the example with `nvcc` as a PyTorch extension and checks its output against the PyTorch reference. `results/rmsnorm-smoke.json` holds workload statuses, errors, candidate and reference latencies, and the benchmark configuration. `--diagnostics` checks one workload with memcheck, racecheck, and Nsight Compute, writing `results/rmsnorm-smoke.sanitizer.txt` and `results/rmsnorm-smoke.ncu.txt`. Running the script again overwrites those output files. Use `--output` to keep a separate result.

The benchmark uses short runs to verify setup. For the repair suite, see the [RMSNorm evaluation instructions](docs/rmsnorm-evaluation.md). For separate CUDA event timings and GPU profiles, see the [Add + RMSNorm experiment](experiments/add_rmsnorm/README.md).
