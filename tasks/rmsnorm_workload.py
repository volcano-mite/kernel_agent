"""BF16 case generation and FP32 reference shared by the GPU workers."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch


HIDDEN_SIZE = 4096
EPS = 1e-5
ATOL = 1e-2
RTOL = 1e-2


class Case(TypedDict):
    case_id: str
    batch_size: int
    seed: int


class Correctness(TypedDict):
    ok: bool
    max_abs_error: float | None
    max_rel_error: float | None
    reason: str | None


def generate_inputs(case: Case, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Create contiguous BF16 inputs without advancing the global generator."""
    import torch

    generator = torch.Generator(device=device).manual_seed(case["seed"])
    shape = (case["batch_size"], HIDDEN_SIZE)
    input = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    weight = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device, generator=generator)
    return input, weight


def reference(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Official FP32 arithmetic with a single BF16 rounding at the output."""
    import torch

    x = input.float()
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
    return ((x * inv_rms) * weight.float()).to(input.dtype)


def correctness(output: torch.Tensor, expected: torch.Tensor) -> Correctness:
    """Apply the FlashInfer-Bench per-element tolerance and report finite stats."""
    import torch

    result: Correctness = {
        "ok": False,
        "max_abs_error": None,
        "max_rel_error": None,
        "reason": None,
    }
    if output.shape != expected.shape:
        result["reason"] = "shape"
        return result
    if output.dtype != expected.dtype:
        result["reason"] = "dtype"
        return result
    if not torch.isfinite(output).all().item():
        result["reason"] = "nonfinite"
        return result

    expected_fp32 = expected.float()
    absolute = (output.float() - expected_fp32).abs()
    relative = absolute / (expected_fp32.abs() + 1e-8)
    max_abs_error = float(absolute.amax().item())
    max_rel_error = float(relative.amax().item())
    result["max_abs_error"] = max_abs_error if isfinite(max_abs_error) else None
    result["max_rel_error"] = max_rel_error if isfinite(max_rel_error) else None
    result["ok"] = not ((absolute > ATOL) & (relative > RTOL)).any().item()
    if not result["ok"]:
        result["reason"] = "tolerance"
    return result
