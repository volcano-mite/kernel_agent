# Reproduce the RMSNorm evaluation

Ryan Hou (NetID `ryanhou`) and Sheng Wei (NetID `shwei`).

The [task set](tasks.jsonl) follows the lecture's JSONL format. There are 48 `exact` questions with an expected option ID and 2 `unanswerable` questions with a rubric. The exact-answer questions include 37 repair tasks, 8 anonymous-patch triage tasks, and 3 edge cases. The harness builds the model prompt from each task and runs model calls from temporary directories.

From the repository root, follow the [environment setup](../../README.md). Run the same harness for both methods with access to Claude Code and one NVIDIA GPU:

```bash
source benchmarks/gpu_env.sh
export CUDA_VISIBLE_DEVICES=GPU-UUID
python benchmarks/rmsnorm/eval_rmsnorm_tasks.py --workers 8
```

Replace `GPU-UUID` with the device reported by `nvidia-smi -L`. The harness runs the one-shot baseline once and the agent three times per task. A run can resume in the same output directory with the same model, effort and task file. Use `--output PATH` for an independent run.

Both methods use the same Claude Code backend and see the same task. The agent can request the batch-1 FlashInfer-Bench check before submitting, using up to three model calls in total. An exact-answer task passes when the selected option matches the expected ID and its kernel passes all 14 official workloads. Anonymous-patch tasks give both methods the same artifact labels. Only a requested public check reveals benchmark feedback. Edge tasks identify the target batch size. An independent judge call scores each unanswerable response for correctness, grounding, completeness, and policy compliance, plus the task-specific criteria for abstention and missing evidence. Every criterion must pass. Judge calls are excluded from answer latency, turns, and token counts.

The harness writes local output to `.artifacts/rmsnorm-eval/lecture/` by default. `results.md` summarizes the metrics, changing outcomes, and failure analysis. `traces.jsonl` records the submitted choices, token counts, public checks, and final grading evidence. `preflight.json` records the GPU check of the known-correct implementation. Use `--report-only` to rebuild the report from saved traces. For the submitted regrade, all 200 saved attempts were reused after verifying that the prompts and options matched. The 192 exact-answer attempts retained their GPU grading evidence, and the 8 abstention attempts were graded by the new judge.
