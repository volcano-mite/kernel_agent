"""Fixed FlashInfer-Bench RMSNorm task and its standalone agent workspace."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from .rmsnorm_workload import ATOL, EPS, HIDDEN_SIZE, RTOL, Case


TASK_ID = "rmsnorm_h4096"


# The two manifests partition the 14 official batch sizes. The harness keeps
# the holdout manifest and its independently seeded inputs on the host.
PUBLIC_CASES: list[Case] = [
    {"case_id": "public-1", "batch_size": 1, "seed": 29583},
    {"case_id": "public-2", "batch_size": 34, "seed": 81064},
    {"case_id": "public-3", "batch_size": 170, "seed": 14067},
    {"case_id": "public-4", "batch_size": 8804, "seed": 65150},
    {"case_id": "public-5", "batch_size": 14509, "seed": 78051},
]

HOLDOUT_CASES: list[Case] = [
    {"case_id": "holdout-1", "batch_size": 7, "seed": 405132},
    {"case_id": "holdout-2", "batch_size": 15, "seed": 815026},
    {"case_id": "holdout-3", "batch_size": 16, "seed": 986178},
    {"case_id": "holdout-4", "batch_size": 63, "seed": 600385},
    {"case_id": "holdout-5", "batch_size": 64, "seed": 161237},
    {"case_id": "holdout-6", "batch_size": 79, "seed": 653026},
    {"case_id": "holdout-7", "batch_size": 10827, "seed": 257550},
    {"case_id": "holdout-8", "batch_size": 11832, "seed": 335178},
    {"case_id": "holdout-9", "batch_size": 14418, "seed": 890560},
]


def initialize_workspace(path: Path) -> None:
    """Write the public task and starting source to a fresh agent workspace."""
    path.mkdir(parents=True, exist_ok=True)
    batches = ", ".join(str(case["batch_size"]) for case in PUBLIC_CASES)
    specification = f"""# CUDA RMSNorm · `{TASK_ID}`

Implement and optimize `rmsnorm.cu` on a single NVIDIA GPU. The harness runs
the CUDA code and returns correctness and performance feedback.

- Input is a contiguous BF16 CUDA tensor of shape `[batch_size, {HIDDEN_SIZE}]`.
- Weight is a contiguous BF16 CUDA tensor of shape `[{HIDDEN_SIZE}]`.
- Output is a preallocated contiguous BF16 CUDA tensor with the input shape.
- Hidden size is fixed at {HIDDEN_SIZE}. Public batch sizes are {batches}.
  Your implementation must handle any positive batch size.
- Follow the FP32 reference below. For each output element, the harness checks
  finiteness and fails it if both absolute error exceeds {ATOL:g} and relative error
  exceeds {RTOL:g}. Relative error uses `abs(expected) + 1e-8` as the denominator.

```python
x = input.float()
inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + {EPS:g})
output = ((x * inv_rms) * weight.float()).to(torch.bfloat16)
```

Export the following function at global scope. The harness supplies the binding.

```cpp
void run(const torch::Tensor& input, const torch::Tensor& weight, torch::Tensor output);
```

Write to the supplied output tensor and launch on PyTorch's current CUDA stream.
Use CUDA C++ to implement the operator. PyTorch tensor metadata and data pointers,
CUDA runtime support, and CUDA C++ headers are available. Do not call existing
operator implementations or add dependencies or files.

Evaluate the starting kernel as written to check the feedback loop. Its scale
is a placeholder that needs to be replaced with the row's inverse RMS.
To evaluate a candidate, send the full contents of `rmsnorm.cu` to
`evaluate_kernel` with the run ID supplied at startup. Use the feedback to
iterate and leave your best evaluated source in `rmsnorm.cu`.
"""
    (path / "TASK.md").write_text(specification, encoding="utf-8")
    copyfile(Path(__file__).with_name("rmsnorm_starter.cu"), path / "rmsnorm.cu")
