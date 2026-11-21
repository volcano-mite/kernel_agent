"""Compile and evaluate a small CUDA extension on official RMSNorm workloads."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
from flashinfer_bench.agents import flashinfer_bench_run_ncu, flashinfer_bench_run_sanitizer
from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import BuildSpec, Solution, SourceFile, TraceSet
from flashinfer_bench.env import get_fib_dataset_path


ROOT = Path(__file__).resolve().parents[1]
DEFINITION = "rmsnorm_h4096"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/rmsnorm-smoke.json")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        parser.error("Make exactly one NVIDIA GPU visible with CUDA_VISIBLE_DEVICES.")

    dataset_path = str(get_fib_dataset_path().resolve())
    traces = TraceSet.from_path(dataset_path)
    workloads = traces.workloads.get(DEFINITION, [])
    if DEFINITION not in traces.definitions or not workloads:
        parser.error(f"Missing {DEFINITION} data in {dataset_path}. Run scripts/fetch_rmsnorm_data.sh.")

    source = ROOT / "examples/rmsnorm_smoke/rmsnorm.cu"
    solution = Solution(
        name="rmsnorm_cuda_smoke",
        definition=DEFINITION,
        author="kernelAgent",
        spec=BuildSpec(
            language="cuda",
            binding="torch",
            target_hardware=["cuda"],
            entry_point=f"{source.name}::run",
        ),
        sources=[SourceFile(path=source.name, content=source.read_text())],
    )
    traces.solutions[DEFINITION] = [solution]
    config = BenchmarkConfig.default(
        definitions=[DEFINITION], warmup_runs=3, iterations=20, num_trials=2
    )
    benchmark = Benchmark(traces, config)
    try:
        result = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()

    evaluations = result.traces.get(DEFINITION, [])
    passed = sum(trace.is_successful() for trace in evaluations)
    report = {
        "definition": DEFINITION,
        "solution": solution.name,
        "solution_hash": solution.hash(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_bench": version("flashinfer-bench"),
        "dataset_path": dataset_path,
        "benchmark_config": config.model_dump(exclude_none=True),
        "expected_workloads": len(workloads),
        "passed_workloads": passed,
        "traces": [trace.model_dump(mode="json") for trace in evaluations],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{passed}/{len(workloads)} {DEFINITION} workloads passed. Results: {args.output}")
    if passed != len(workloads):
        return 1
    if not args.diagnostics:
        return 0

    workload = workloads[0].workload
    sanitizer = flashinfer_bench_run_sanitizer(
        solution, workload, trace_set_path=dataset_path,
        sanitizer_types=["memcheck", "racecheck"], timeout=120,
    )
    profile = flashinfer_bench_run_ncu(
        solution, workload, trace_set_path=dataset_path,
        set="basic", kernel_name="rmsnorm_kernel", timeout=120,
    )
    sanitizer_path = args.output.with_suffix(".sanitizer.txt")
    profile_path = args.output.with_suffix(".ncu.txt")
    sanitizer_path.write_text(sanitizer.rstrip() + "\n")
    profile_path.write_text(profile.rstrip() + "\n")
    print(f"Diagnostics for {workload.uuid}: {sanitizer_path}, {profile_path}")
    sanitizer_passed = all(f"{tool} passed successfully" in sanitizer for tool in ("memcheck", "racecheck"))
    profile_passed = not profile.startswith("ERROR:") and "Section:" in profile
    return 0 if sanitizer_passed and profile_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
