# Reproduce the RMSNorm evaluation

Ryan Hou (NetID `ryanhou`) and Sheng Wei.

The [task set](../benchmarks/rmsnorm/tasks.jsonl) follows the lecture's JSONL format and includes grading fixtures and answer keys for the CUDA tasks. Each of the 50 lines has an ID, a type, a question, and grading criteria. The suite includes 37 repair tasks, 8 anonymous-patch triage tasks, 3 edge cases, and 2 unanswerable tasks. The harness builds the model prompt from each task and runs model calls from temporary directories.

From the repository root, follow the [environment setup](../README.md). Run the same harness for both methods with access to Claude Code and one NVIDIA GPU:

```bash
source scripts/gpu_env.sh
export CUDA_VISIBLE_DEVICES=GPU-UUID
python scripts/eval_rmsnorm_tasks.py --workers 8
```

Replace `GPU-UUID` with the device reported by `nvidia-smi -L`. The harness runs the one-shot baseline once and the agent three times per task. A run can resume in the same output directory with the same model, effort and task file. Use `--output PATH` for an independent run.

Both methods use the same Claude Code backend and see the same task. The agent can request the batch-1 FlashInfer-Bench check before submitting, using up to three model calls in total. Final grading uses the operator contract and all 14 official workloads. Anonymous-patch tasks give both methods the same artifact labels. Only a requested public check reveals benchmark feedback. Edge tasks identify the target batch size. The two unanswerable tasks require identifying the missing measurement or evaluation.

Each run writes local output to `eval_results/` by default. `results.md` summarizes the metrics, changing outcomes, and failure analysis. `traces.jsonl` records the submitted choices, token counts, public checks, and final grading evidence. `preflight.json` records the GPU check of the known-correct implementation. Use `--report-only` to rebuild the report from saved traces.
