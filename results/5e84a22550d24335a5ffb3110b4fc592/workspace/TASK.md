# CUDA RMSNorm · `rmsnorm_h4096`

Implement and optimize `rmsnorm.cu` on a single NVIDIA GPU. The harness runs
the CUDA code and returns correctness and performance feedback.

- Input is a contiguous BF16 CUDA tensor of shape `[batch_size, 4096]`.
- Weight is a contiguous BF16 CUDA tensor of shape `[4096]`.
- Output is a preallocated contiguous BF16 CUDA tensor with the input shape.
- Hidden size is fixed at 4096. Public batch sizes are 1, 34, 170, 8804, 14509.
  Your implementation must handle any positive batch size.
- Follow the FP32 reference below. For each output element, the harness checks
  finiteness and fails it if both absolute error exceeds 0.01 and relative error
  exceeds 0.01. Relative error uses `abs(expected) + 1e-8` as the denominator.

```python
x = input.float()
inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-05)
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
