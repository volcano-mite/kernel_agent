"""Run a fixed CUDA source snapshot on official FlashInfer-Bench workloads."""
import hashlib
import json
import os
import sys

import torch
from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import BuildSpec, Solution, SourceFile, TraceSet
from flashinfer_bench.env import get_fib_dataset_path

DEFINITION = "rmsnorm_h4096"


def main():
    request = json.load(sys.stdin)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one NVIDIA GPU with CUDA_VISIBLE_DEVICES")
    source = request["source"]
    digest = hashlib.sha256(source.encode()).hexdigest()
    torch.manual_seed(126)
    data = TraceSet.from_path(str(get_fib_dataset_path().resolve()))
    workloads = data.workloads[DEFINITION]
    if request["public"]:
        workloads = [trace for trace in workloads if trace.workload.axes["batch_size"] == 1]
    if not workloads:
        raise RuntimeError("No matching official workloads")
    data.workloads[DEFINITION] = workloads
    solution = Solution(
        name=f"rmsnorm_eval_{digest[:16]}",
        definition=DEFINITION,
        author="kernelAgent",
        spec=BuildSpec(language="cuda", binding="torch", target_hardware=["cuda"],
                       entry_point="rmsnorm.cu::run"),
        sources=[SourceFile(path="rmsnorm.cu", content=source)],
    )
    data.solutions[DEFINITION] = [solution]
    config = BenchmarkConfig.default(
        definitions=[DEFINITION], warmup_runs=1, iterations=3, num_trials=1,
    )
    benchmark = Benchmark(data, config)
    try:
        result = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()

    batches = {trace.workload.uuid: trace.workload.axes["batch_size"] for trace in workloads}
    checks = []
    for trace in result.traces.get(DEFINITION, []):
        record = trace.model_dump(mode="json")
        evaluation = record.get("evaluation") or {}
        checks.append({
            "batch": batches[trace.workload.uuid],
            "ok": trace.is_successful(),
            "status": evaluation.get("status", "unknown"),
            "correctness": evaluation.get("correctness"),
        })
    response = {
        "ok": {check["batch"] for check in checks} == set(batches.values())
              and len(checks) == len(batches)
              and all(check["ok"] for check in checks),
        "checks": checks,
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sha256": digest,
    }
    print("EVAL_RESULT " + json.dumps(response))


if __name__ == "__main__":
    main()
