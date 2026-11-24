# Kernel Agent

Kernel Agent evaluates CUDA C++ candidates for FlashInfer-Bench `rmsnorm_h4096`. A run uses one Agent, one model and one GPU or CPU Mock backend. See [the design](docs/design.md) for the evaluation stages and B/D/F variants.

## Run the CPU workflow

```bash
uv venv --python 3.11 .venv-host
source .venv-host/bin/activate
uv pip install -r requirements-host.txt

RUN_ID=$(python -m harness init --backend mock | python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
python -m harness evaluate "$RUN_ID" "results/$RUN_ID/workspace/rmsnorm.cu"
python -m harness evaluate "$RUN_ID" tasks/rmsnorm_golden.cu
```

The starter has an intentional numerical error. Mock returns synthetic feedback and never assigns a GPU rank. To connect an MCP client manually, use the `run_id` and `token` printed by `init`, run `python -m harness serve "$RUN_ID"`, and connect to `http://127.0.0.1:8000/mcp` with `Authorization: Bearer <token>`.

Run the MCP regression test with `python -m unittest discover -s tests`. It uses a loopback HTTP server and the CPU Mock backend.

## Run an Agent on a GPU

Build the GPU-free Agent image on the host and set the API credential for the chosen Agent. Use the provider's model ID for `MODEL` and a full UUID from `nvidia-smi -L` for `GPU_UUID`.

```bash
podman build -f runners/Containerfile -t kernel-agent:local .
```

### Host GPU Worker

Use `--gpu-worker host` when the GPU is available only on the host. The host needs Linux, Bubblewrap with user namespaces enabled, the NVIDIA driver and CUDA toolkit, a C++ compiler supported by `nvcc`, Ninja and `nm`. Set `GPU_WORKER_PYTHON` to the Python executable in a trusted environment containing CUDA-enabled PyTorch and FlashInfer. `--gpu-worker-python` defaults to the Harness's Python. The launcher resolves and saves the CUDA toolkit and compiler paths at run creation using `CUDA_HOME`, `CC`, `CXX` and the host's executable search path.

```bash
python -m harness run --backend gpu --gpu-worker host --gpu-uuid "$GPU_UUID" \
  --gpu-worker-python "$GPU_WORKER_PYTHON" \
  --variant B --agent codex --model "$MODEL" --agent-image kernel-agent:local
```

Bubblewrap runs each job in a private PID, user and network namespace, with a fresh writable temporary filesystem and read-only runtime mounts. The process uses UID/GID 65534 inside its namespace, mapped to the non-root Harness account on the host. Only the selected GPU and its driver nodes are mounted. Run records and the Agent workspace stay outside this sandbox.

For a GPU smoke run without model calls, initialize a host run and submit the starter and golden candidate using the trusted CLI.

```bash
RUN_ID=$(python -m harness init --backend gpu --gpu-worker host --gpu-uuid "$GPU_UUID" \
  --gpu-worker-python "$GPU_WORKER_PYTHON" | python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
python -m harness evaluate "$RUN_ID" tasks/rmsnorm_starter.cu
python -m harness evaluate "$RUN_ID" tasks/rmsnorm_golden.cu
python -m harness accept "$RUN_ID"
```

`serve`, `evaluate` and `accept` read the Worker mode and settings from the saved run. Use `serve "$RUN_ID"` to expose the same evaluation path to an Agent container over MCP.

### Podman GPU Worker

The default mode uses a locally built GPU Worker image, rootless Podman and NVIDIA Container Toolkit CDI. The selected GPU must be present in `nvidia-ctk cdi list`. Runs created before `--gpu-worker` was added use Podman.

```bash
podman build -f containers/gpu.Containerfile -t kernel-worker:local .
python -m harness run --backend gpu --gpu-worker podman --variant B --agent claude --model "$MODEL" \
  --agent-image kernel-agent:local --gpu-image kernel-worker:local --gpu-uuid "$GPU_UUID"
```

For either Worker mode, use `--variant D` for the sanitizer and profiler, or `--variant F` for both tools and the read-only CUDA examples. D/F need `compute-sanitizer`, `ncu`, and access to GPU performance counters. An Agent can also run against `--backend mock`, with `--host-agent` available for local Mock development using a host-installed CLI.

## Read a run

Each `results/<run_id>/` directory contains the run configuration, reproducible source snapshots, public evaluation records, diagnostic logs, and any Agent logs. Independent acceptance seals the public shortlist and runs each attempted candidate in a fresh Worker. Its private records contain per-case candidate, PyTorch eager, and FlashInfer latency samples. Acceptance considers the geometric mean after subtracting two median absolute deviations from eager and adding two to the candidate. It selects the first shortlisted candidate above the configured speedup floor, which defaults to `1.03`, or falls back to eager.

Use identical evaluation and time budgets for a B/D/F comparison. Use the same Agent, model, task, starter, and GPU across variants. The optional `--results` flag precedes the subcommand and sets the location of the run records.
