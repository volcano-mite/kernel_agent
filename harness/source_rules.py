"""Screen candidate dependencies before allocating GPU time."""

import re


HEADERS = {
    "algorithm", "array", "cassert", "cmath", "cstddef", "cstdint", "limits", "type_traits", "utility",
    "cuda.h", "cuda_runtime.h", "cuda_runtime_api.h", "cuda_fp16.h", "cuda_bf16.h", "cooperative_groups.h",
    "torch/extension.h", "ATen/ATen.h", "ATen/Tensor.h",
}
HEADER_PREFIXES = ("ATen/cuda/", "c10/cuda/", "c10/util/", "cooperative_groups/", "cub/", "cuda/")
EXISTING_OPERATOR = re.compile(
    r"\b(?:flashinfer|at\s*::\s*(?:native|_ops)|(?:cu(?:blas|dnn)|nccl)\w*)\b|"
    r"\b(?:at|torch)\s*::\s*(?!(?:cuda|indexing)\s*::)[a-z_]\w*\s*\("
)


def violation(source):
    if not source.strip() or "\x00" in source:
        return "Source must contain nonempty CUDA source text"
    code = re.sub(r"\\\r?\n", "", source)
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", code, flags=re.DOTALL)
    if not re.search(r"\brun\s*\(", code):
        return "Expected a run entry point"
    if re.search(r"\bPYBIND11_MODULE\s*\(", code):
        return "The harness supplies the Python binding"
    for include in re.finditer(r"^\s*#\s*(include(?:_next)?|import)\s*([^\n]*)", code, flags=re.MULTILINE):
        match = re.fullmatch(r'<([\w./+-]+)>|"([\w./+-]+)"', include[2].strip())
        if include[1] != "include" or match is None:
            return "Use literal includes from the supported CUDA and PyTorch headers"
        header = match[1] or match[2]
        allowed = header in HEADERS or header.startswith(HEADER_PREFIXES)
        if not allowed or ".." in header.split("/"):
            return f"Unsupported dependency: {header}"
    if EXISTING_OPERATOR.search(code):
        return "Implement the operator in CUDA C++ without calling existing operator implementations"
    return None
