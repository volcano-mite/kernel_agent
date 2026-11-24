"""Compile and evaluate one CUDA RMSNorm snapshot in a disposable process."""

import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

from tasks.rmsnorm_workload import EPS, correctness, generate_inputs, reference


BINDING = """#include <torch/extension.h>

void run(const torch::Tensor& input, const torch::Tensor& weight, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, bindings) {
    bindings.def("run", &run);
}
"""
MAX_BUILD_LOG = 64 * 1024
DEFAULT_WARMUPS = 3
DEFAULT_SAMPLES = 15


class InfrastructureError(RuntimeError):
    pass


class CompileError(RuntimeError):
    def __init__(self, detail, compiler):
        super().__init__(detail)
        self.compiler = compiler


class RejectedSymbols(CompileError):
    pass


def _compiler_report(log, truncated):
    kernels = []
    for line in log.splitlines():
        entry = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line)
        if entry:
            kernels.append({"symbol": entry[1], "architecture": entry[2]})
        if not kernels or not (line.startswith("ptxas info") or "bytes spill stores" in line):
            continue
        for key, pattern in (
            ("registers", r"Used (\d+) registers"),
            ("stack_bytes", r"(\d+) bytes stack frame"),
            ("spill_store_bytes", r"(\d+) bytes spill stores"),
            ("spill_load_bytes", r"(\d+) bytes spill loads"),
            ("static_shared_bytes", r"(\d+) bytes smem"),
        ):
            if match := re.search(pattern, line):
                kernels[-1][key] = int(match[1])
        if "Used" in line and "registers" in kernels[-1]:
            kernels[-1].setdefault("static_shared_bytes", 0)
    return {"log": log, "log_truncated": truncated, "kernels": kernels}


def _build_extension(source, directory):
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not CUDA_HOME or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        raise InfrastructureError("CUDA toolkit with nvcc is required")
    if not shutil.which("ninja"):
        raise InfrastructureError("Ninja is required to build the CUDA extension")

    os.environ["TORCH_NO_COMPILER_WRAPPER"] = "1"
    os.environ.setdefault("MAX_JOBS", "2")
    candidate = directory / "rmsnorm.cu"
    candidate.write_text(source, encoding="utf-8")
    sources = [str(candidate)]
    if not re.search(r"\bPYBIND11_MODULE\s*\(", source):
        binding = directory / "binding.cpp"
        binding.write_text(BINDING, encoding="utf-8")
        sources.append(str(binding))

    error = None
    with tempfile.TemporaryFile() as capture:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            os.dup2(capture.fileno(), 2)
            os.dup2(capture.fileno(), 1)
            try:
                module = load(
                    name="kernel_agent_candidate",
                    sources=sources,
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
                    build_directory=str(directory),
                    verbose=True,
                )
            except (RuntimeError, ImportError, OSError) as exc:
                error = exc
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)

        size = capture.seek(0, os.SEEK_END)
        capture.seek(max(size - MAX_BUILD_LOG, 0))
        log = capture.read().decode(errors="replace")

    if error is not None:
        detail = str(error).partition("\n")[0][:1500]
        # Ninja captures its output on a failed build and adds it to the error.
        if details := str(error).partition("\n")[2]:
            log = f"{log}\n{details}"
        compiler = _compiler_report(log[-MAX_BUILD_LOG:], size > MAX_BUILD_LOG or len(log) > MAX_BUILD_LOG)
        raise CompileError(detail, compiler) from error
    compiler = _compiler_report(log, size > MAX_BUILD_LOG)
    if not callable(getattr(module, "run", None)):
        raise CompileError("The compiled extension does not export callable run", compiler)
    try:
        symbols = subprocess.run(["nm", "-D", "--undefined-only", "--demangle", module.__file__],
                                 check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InfrastructureError("Unable to inspect the compiled extension symbols") from exc
    forbidden = re.search(r"\b(?:at::(?:_ops|native)::|flashinfer::|cu(?:blas|dnn)\w*\b)", symbols)
    if forbidden:
        raise RejectedSymbols("Compiled extension references an existing operator implementation", compiler)
    return module.run, compiler


class EagerRMSNorm:
    """Run the FP32 reference operations with preallocated intermediate buffers."""

    def __init__(self, torch, inputs, weight, output):
        options = {"dtype": torch.float32, "device": inputs.device}
        self.torch = torch
        self.inputs = inputs
        self.weight = weight
        self.output = output
        self.x = torch.empty_like(inputs, dtype=torch.float32)
        self.squared = torch.empty_like(self.x)
        self.inverse = torch.empty((inputs.shape[0], 1), **options)
        self.weight_fp32 = torch.empty_like(weight, dtype=torch.float32)

    def __call__(self):
        torch = self.torch
        self.x.copy_(self.inputs)
        self.weight_fp32.copy_(self.weight)
        torch.pow(self.x, 2, out=self.squared)
        torch.mean(self.squared, dim=-1, keepdim=True, out=self.inverse)
        self.inverse.add_(EPS).rsqrt_()
        self.x.mul_(self.inverse).mul_(self.weight_fp32)
        self.output.copy_(self.x)


def _latencies(torch, methods, flush_buffer, warmups, sample_count):
    for name, call in methods:
        try:
            for _ in range(warmups):
                call()
            torch.cuda.synchronize()
        except Exception as exc:
            if name:
                raise InfrastructureError(f"{name} baseline warmup failed: {exc}") from exc
            raise

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    stop.record()
    stop.synchronize()

    samples = {name: [] for name, _ in methods}
    for index in range(sample_count):
        # Rotate order to distribute timing drift across the implementations.
        shift = index % len(methods)
        for name, call in methods[shift:] + methods[:shift]:
            try:
                flush_buffer.add_(1)
                start.record()
                call()
                stop.record()
                stop.synchronize()
                latency = start.elapsed_time(stop) * 1000
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError("CUDA event returned an invalid latency")
            except Exception as exc:
                if name:
                    raise InfrastructureError(f"{name} baseline timing failed: {exc}") from exc
                raise
            samples[name].append(latency)
    result = {}
    for name, values in samples.items():
        prefix = f"{name}_" if name else ""
        median = statistics.median(values)
        result[f"{prefix}latency_us"] = median
        result[f"{prefix}samples_us"] = values
        result[f"{prefix}variation_us"] = statistics.median(abs(value - median) for value in values)
    return result


def _check_case(torch, candidate, case, device):
    inputs, weight = generate_inputs(case, device)
    expected = reference(inputs, weight)
    saved_inputs, saved_weight = inputs.clone(), weight.clone()
    # A sentinel turns partially unwritten output into a deterministic failure.
    output = torch.full_like(expected, torch.nan)
    candidate(inputs, weight, output)
    torch.cuda.synchronize()
    if not torch.equal(inputs, saved_inputs) or not torch.equal(weight, saved_weight):
        return {"ok": False, "reason": "input_modified"}
    return correctness(output, expected)


def _time_case(torch, candidate, case, device, flush_buffer, warmups, samples, flashinfer):
    inputs, weight = generate_inputs(case, device)
    output = torch.empty_like(inputs)
    eager_output = torch.empty_like(inputs)
    eager = EagerRMSNorm(torch, inputs, weight, eager_output)
    methods = [("", lambda: candidate(inputs, weight, output)), ("eager", eager)]
    if flashinfer is not None:
        flashinfer_output = torch.empty_like(inputs)
        methods.append((
            "flashinfer",
            lambda: flashinfer(inputs, weight, eps=EPS, out=flashinfer_output),
        ))
    return _latencies(torch, methods, flush_buffer, warmups, samples)


def _timing_options(timing):
    if not isinstance(timing, dict):
        raise ValueError("timing must be an object")
    warmups = timing.get("warmups", DEFAULT_WARMUPS)
    samples = timing.get("samples", DEFAULT_SAMPLES)
    if type(warmups) is not int or not 1 <= warmups <= 100:
        raise ValueError("timing.warmups must be an integer between 1 and 100")
    if type(samples) is not int or not 2 <= samples <= 1000:
        raise ValueError("timing.samples must be an integer between 2 and 1000")
    return warmups, samples


def _request_fields(request):
    source = request.get("source")
    if not isinstance(source, str) or not source.strip() or "\x00" in source:
        raise ValueError("source must contain nonempty CUDA source text")
    phase = request.get("phase", "public")
    if phase not in ("public", "holdout"):
        raise ValueError("phase must be public or holdout")
    operation = request.get("operation", "evaluate")
    if operation not in ("evaluate", "probe"):
        raise ValueError("Unsupported GPU worker operation")
    cases = request.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("case_id"), str)
            or not case["case_id"]
            or type(case.get("batch_size")) is not int
            or case["batch_size"] <= 0
            or type(case.get("seed")) is not int
        ):
            raise ValueError("Each case needs case_id, positive batch_size and an integer seed")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    return source, phase, operation, cases


def _response(status, stage, cases=(), **extra):
    return {"status": status, "stage": stage, "cases": list(cases), **extra}


def run(request):
    if not isinstance(request, dict):
        return _response("rejected", "static", detail="GPU worker request must be an object")
    try:
        source, phase, operation, cases = _request_fields(request)
        warmups, samples = _timing_options(request.get("timing", {}))
    except ValueError as exc:
        return _response("rejected", "static", detail=str(exc))

    try:
        import torch

        if torch.cuda.device_count() != 1:
            raise InfrastructureError("Expose exactly one CUDA device to the GPU worker")
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        environment = {
            "gpu": properties.name,
            "gpu_uuid": str(getattr(properties, "uuid", "")),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        }
    except Exception as exc:
        return _response("infrastructure_error", "initialize", detail=str(exc)[:1500])

    with tempfile.TemporaryDirectory(prefix="kernel-agent-build-") as temporary:
        try:
            candidate, compiler = _build_extension(source, Path(temporary))
        except RejectedSymbols as exc:
            return _response("rejected", "compile", detail=str(exc), compiler=exc.compiler)
        except CompileError as exc:
            return _response("compile_error", "compile", detail=str(exc), compiler=exc.compiler)
        except Exception as exc:
            return _response("infrastructure_error", "compile", detail=str(exc)[:1500])

        extra = {"compiler": compiler, "environment": environment}
        with torch.no_grad():
            if operation == "probe":
                try:
                    inputs, weight = generate_inputs(cases[0], device)
                    output = torch.empty_like(inputs)
                    torch.cuda.synchronize()
                    if request.get("profile") is True:
                        with torch.cuda.profiler.profile():
                            candidate(inputs, weight, output)
                            torch.cuda.synchronize()
                    else:
                        candidate(inputs, weight, output)
                        torch.cuda.synchronize()
                except Exception as exc:
                    return _response("runtime_error", "probe", detail=str(exc)[:1500], **extra)
                return _response("passed", "probe", **extra)

            checks = []
            for case in cases:
                check = {"case_id": case["case_id"], "batch_size": case["batch_size"]}
                checks.append(check)
                try:
                    error = _check_case(torch, candidate, case, device)
                except Exception as exc:
                    check.update(status="runtime_error")
                    return _response("runtime_error", "correctness", checks, detail=str(exc)[:1500], **extra)
                check.update(
                    status="passed" if error["ok"] else "wrong_answer",
                    max_absolute_error=error.get("max_abs_error"),
                    max_relative_error=error.get("max_rel_error"),
                )
                if not error["ok"]:
                    check["detail"] = error["reason"]
                    return _response("wrong_answer", "correctness", checks, detail=error["reason"], **extra)

            flashinfer = None
            try:
                cache_bytes = getattr(properties, "L2_cache_size", 0) or 32 * 1024 * 1024
                flush_buffer = torch.zeros(2 * cache_bytes // 4, dtype=torch.float32, device=device)
                if phase == "holdout":
                    from importlib.metadata import version

                    from flashinfer.norm import rmsnorm as flashinfer

                    environment["flashinfer"] = version("flashinfer-python")
            except Exception as exc:
                return _response("infrastructure_error", "timing", checks, detail=str(exc)[:1500], **extra)

            for case, check in zip(cases, checks, strict=True):
                try:
                    check.update(_time_case(torch, candidate, case, device, flush_buffer, warmups, samples, flashinfer))
                except InfrastructureError as exc:
                    return _response("infrastructure_error", "timing", checks, detail=str(exc)[:1500], **extra)
                except Exception as exc:
                    check.update(status="runtime_error")
                    return _response("runtime_error", "timing", checks, detail=str(exc)[:1500], **extra)
            return _response("passed", "timing", checks, **extra)


def main():
    # The caller receives exactly one JSON document. Native candidate output,
    # framework notices and diagnostic tool output use stderr.
    sys.stdout.flush()
    with os.fdopen(os.dup(1), "w", encoding="utf-8") as output:
        os.dup2(2, 1)
        try:
            response = run(json.load(sys.stdin))
        except ValueError as exc:
            response = _response("rejected", "static", detail=str(exc)[:1500])
        except Exception as exc:
            response = _response("infrastructure_error", "worker", detail=str(exc)[:1500])
        print(json.dumps(response, allow_nan=False), file=output, flush=True)


if __name__ == "__main__":
    main()
