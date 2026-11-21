"""Generate the fixed CUDA repair task set from the project's smoke kernel."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each replacement is scoped to the one marked line in the original kernel.
# Wrong choices avoid out-of-bounds memory, invalid launches and barriers in branches.
REPAIRS = [
    ("row_offset = static_cast<int64_t>(blockIdx.x) * kHiddenSize",
     ["row_offset = 0", "row_offset = static_cast<int64_t>(blockIdx.x) * (kHiddenSize / 2)",
      "row_offset = static_cast<int64_t>(blockIdx.x) * (kHiddenSize - 1)"],
     "Use the row assigned to each block and preserve the full BF16 row stride."),
    ("lane = threadIdx.x", ["lane = 0", "lane = threadIdx.x / 2", "lane = threadIdx.x % 32"],
     "Map every thread to its own shared-memory slot and column lane."),
    ("squared_sum = 0.0f", ["squared_sum = 1.0f", "squared_sum = 4096.0f", "squared_sum = -1.0f"],
     "Start each thread's sum of squares at the additive identity."),
    ("int col = lane; col < kHiddenSize; col += kThreads",
     ["int col = lane; col < kHiddenSize / 2; col += kThreads",
      "int col = lane; col < kHiddenSize; col += 2 * kThreads",
      "int col = lane; col < kHiddenSize; col += kHiddenSize"],
     "Cover all 4096 columns exactly once in the marked loop."),
    ("squared_sum += value * value", ["squared_sum += value", "squared_sum += fabsf(value)",
                                      "squared_sum += 0.5f * value * value"],
     "Accumulate the squares of the input values."),
    ("sums[lane] = squared_sum", ["sums[lane] = 0.0f", "sums[lane] = squared_sum * 2.0f",
                                  "sums[lane] = -squared_sum"],
     "Publish the per-thread sum for the block reduction."),
    ("int stride = kThreads / 2; stride > 0; stride /= 2",
     ["int stride = kThreads / 4; stride > 0; stride /= 2",
      "int stride = kThreads / 2; stride > 1; stride /= 2",
      "int stride = kThreads / 2; stride > 0; stride /= 4"],
     "Reduce all 256 partial sums using a binary reduction."),
    ("sums[lane] += sums[lane + stride]", ["sums[lane] -= sums[lane + stride]",
                                           "sums[lane] = sums[lane + stride]",
                                           "sums[lane] += sums[lane]"],
     "Combine both sides of each reduction pair."),
    ("scale = rsqrtf(sums[0] / kHiddenSize + kEpsilon)",
     ["scale = rsqrtf(sums[0] + kEpsilon)", "scale = sqrtf(sums[0] / kHiddenSize + kEpsilon)",
      "scale = rsqrtf(sums[0] / kHiddenSize) + kEpsilon"],
     "Compute the reciprocal RMS with epsilon inside the square root."),
    ("factor = static_cast<float>(weight[col])",
     ["factor = static_cast<float>(weight[0])", "factor = static_cast<float>(weight[lane])",
      "factor = static_cast<float>(weight[kHiddenSize - 1 - col])"],
     "Apply the weight for each output feature."),
    ("at::BFloat16((value * scale) * factor)",
     ["at::BFloat16(value * scale)", "at::BFloat16(value * factor)", "at::BFloat16(value + scale * factor)"],
     "Apply both the row normalization and the per-feature weight."),
    ("output[row_offset + col] =", ["output[col] =", "output[row_offset + kHiddenSize - 1 - col] =",
                                     "output[row_offset + lane] ="],
     "Write every result to its original row and column."),
    ("rows = input.size(0)", ["rows = 1", "rows = input.size(0) / 2", "rows = (input.size(0) / 2) * 2"],
     "Launch exactly one block per input row, including for odd batches."),
    ("value = static_cast<float>(input[row_offset + col])",
     ["value = static_cast<float>(input[col])", "value = static_cast<float>(input[row_offset + lane])",
      "value = static_cast<float>(input[row_offset + kHiddenSize - 1 - col])"],
     "Read the input value for this row and feature in the marked loop."),
    ("float kEpsilon = 1e-5f", ["float kEpsilon = 1e-2f", "float kEpsilon = 0.0f",
                                "float kEpsilon = -1e-5f"],
     "Use the fixed epsilon specified in the FlashInfer-Bench definition."),
    ('module.def("run", &run)', ['module.def("rmsnorm", &run)', 'module.def("forward", &run)',
                                   'module.def("run_cuda", &run)'],
     "Export the callable prescribed by the CUDA extension contract."),
]


def main():
    source = (ROOT / "examples/rmsnorm_smoke/rmsnorm.cu").read_text()
    randomizer = random.Random(20261120)
    tasks = []
    edge_batches = {(0, 1): 14509, (12, 2): 7, (13, 2): 1}
    blinded = {(site, variation) for site in (4, 5, 6, 7) for variation in (0, 1)}
    for site, (original, broken_versions, goal) in enumerate(REPAIRS):
        assert original in source, original
        for variation, broken in enumerate(broken_versions):
            # When repeated in the kernel, mark either the accumulation or output loop.
            occurrence = variation % source.count(original)
            offset = [i for i in range(len(source)) if source.startswith(original, i)][occurrence]
            starter = source[:offset] + broken + source[offset + len(original):]
            line = source.count("\n", 0, offset) + 1
            choices = [original, *(wrong for wrong in broken_versions if wrong != broken)]
            randomizer.shuffle(choices)
            options = dict(zip("ABC", choices))
            edge_batch = edge_batches.get((site, variation))
            blind = (site, variation) in blinded
            prompt = f"Repair line {line} of the CUDA/PyTorch RMSNorm kernel. {goal}"
            if blind:
                prompt = (
                    "Select the correct CUDA repair among anonymous artifacts A, B, and C. "
                    "The patch contents are unavailable. If a public benchmark check is available, "
                    "use its results to select the working artifact."
                )
            if edge_batch:
                prompt += f" The regression test targets batch_size={edge_batch}."
            tasks.append({
                "id": f"rmsnorm-{len(tasks) + 1:03d}",
                "type": "rubric",
                "question": prompt,
                "category": "edge" if edge_batch else "triage" if blind else "repair",
                "rubric": ["Select the replacement that restores the operator contract",
                           "Pass the graded FlashInfer-Bench workloads"],
                "fixture": {
                    "source": starter, "line": line, "old": broken,
                    "choices": options,
                },
                "expected": next(key for key, value in options.items() if value == original),
                "target_batch": edge_batch,
            })
    for prompt, phrase in [
        ("What exact DRAM throughput in GB/s did our RTX 5090 record for this RMSNorm kernel? This task includes no profiler measurement.",
         "profiler"),
        ("What exact maximum absolute error did the submitted candidate produce on the hidden workload in an evaluation that has not run?",
         "evaluation"),
    ]:
        tasks.append({
            "id": f"rmsnorm-{len(tasks) + 1:03d}",
            "type": "unanswerable",
            "question": prompt,
            "category": "unanswerable",
            "fixture": {"choices": {}},
            "expected": "UNANSWERABLE",
            "rubric": ["Abstain without inventing a measurement", f"Identify the missing {phrase} evidence"],
            "required_reason": phrase,
        })
    (ROOT / "benchmarks/rmsnorm/tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))


if __name__ == "__main__":
    main()
