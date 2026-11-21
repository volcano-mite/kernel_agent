"""Run the one-shot baseline and interactive CUDA repair agent, then report results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {
    "type": "object",
    "required": ["action", "answer", "reason"],
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["check", "final"]},
        "answer": {"type": "string", "enum": ["A", "B", "C", "UNANSWERABLE"]},
        "reason": {"type": "string"},
    },
}
SYSTEM = (
    "You solve CUDA RMSNorm repair tasks. Respond with a JSON object following the schema. "
    "For an unanswerable question submit UNANSWERABLE and identify the missing evidence "
    "without estimating the requested measurement. Keep reasons short."
)


def read_tasks(path):
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(tasks) != 50 or len({task["id"] for task in tasks}) != 50:
        raise ValueError("Expected 50 uniquely identified tasks")
    for category, minimum in (("edge", 3), ("unanswerable", 2)):
        if sum(task["category"] == category for task in tasks) < minimum:
            raise ValueError(f"Expected at least {minimum} {category} tasks")
    return tasks


def candidate(task, answer):
    problem = task["fixture"]
    lines = problem["source"].splitlines(keepends=True)
    index = problem["line"] - 1
    if lines[index].count(problem["old"]) != 1:
        raise ValueError(f"Ambiguous source replacement in {task['id']}")
    lines[index] = lines[index].replace(problem["old"], problem["choices"][answer], 1)
    return "".join(lines)


class BenchmarkTool:
    def __init__(self, timeout):
        self.timeout = timeout
        self.responses = {}
        self.lock = threading.Lock()

    def check(self, source, public):
        key = (hashlib.sha256(source.encode()).hexdigest(), public)
        with self.lock:
            if key in self.responses:
                return self.responses[key]
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/rmsnorm_eval_worker.py")],
                input=json.dumps({"source": source, "public": public}),
                capture_output=True, text=True, timeout=self.timeout, cwd=ROOT,
            )
            marker = "EVAL_RESULT "
            records = [line.removeprefix(marker) for line in result.stdout.splitlines()
                       if line.startswith(marker)]
            if result.returncode or len(records) != 1:
                raise RuntimeError(f"FlashInfer worker failed: {result.stderr[-1600:]}")
            self.responses[key] = json.loads(records[0])
            return self.responses[key]


def ask(prompt, args, cwd):
    result = subprocess.run(
        [args.claude, "-p", "--safe-mode", "--tools", "", "--strict-mcp-config",
         "--no-session-persistence", "--output-format", "json", "--model", args.model,
         "--effort", args.effort, "--system-prompt", SYSTEM,
         "--json-schema", json.dumps(SCHEMA)],
        input=prompt, text=True, capture_output=True, cwd=cwd, timeout=args.model_timeout,
    )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Claude Code produced no JSON: {result.stderr[-1000:]}") from exc
    if result.returncode or response.get("is_error"):
        raise RuntimeError(f"Claude Code call failed: {response.get('result', response.get('subtype'))}")
    usage = response["usage"]
    token_fields = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    answer = response.get("structured_output")
    if answer is None:
        try:
            answer = json.loads(response.get("result") or "")
        except json.JSONDecodeError:
            answer = {}
    if not isinstance(answer, dict):
        answer = {}
    return {
        "answer": answer,
        "raw_result": response.get("result"),
        "tokens": sum(usage.get(field, 0) for field in token_fields),
        "usage": usage,
        "api_duration_ms": response.get("duration_api_ms"),
        "models": sorted(response.get("modelUsage", {})),
    }


def score(task, answer, benchmark):
    if answer.get("action") != "final":
        return False, "No final submission within the turn budget", None
    selected = answer.get("answer")
    if task["category"] == "unanswerable":
        reason = str(answer.get("reason", "")).lower()
        if task["required_reason"] == "profiler":
            evidence = r"profil|nsight|ncu|measurement|benchmark|report"
        else:
            evidence = r"evaluat|benchmark|run|result|output|test"
        absence = r"\b(no|not|missing|without|unavailable|absent|cannot|unknown|need|lack)\b|can.t"
        invented = r"(?<![\w])[-+]?\d+(?:\.\d+)?\s*(?:GB/s|TB/s|[eE][+-]?\d+)"
        ok = (selected == task["expected"] and re.search(evidence, reason)
              and re.search(absence, reason) and not re.search(invented, reason, re.IGNORECASE))
        return bool(ok), "Missing evidence identified" if ok else "Abstention rubric failed", None
    if selected not in task["fixture"]["choices"]:
        return False, "No valid CUDA repair choice", None
    result = benchmark.check(candidate(task, selected), public=False)
    if selected != task["expected"]:
        return False, f"Selected {selected}; contract-preserving replacement is {task['expected']}", result
    if task["target_batch"] and not any(
        check["batch"] == task["target_batch"] and check["ok"] for check in result["checks"]
    ):
        return False, "Target edge workload failed", result
    return result["ok"], "All official workloads passed" if result["ok"] else "Official workloads failed", result


def attempt(task, trial, args, benchmark):
    mode = "baseline" if trial == 0 else "agent"
    problem = {"prompt": task["question"], **task["fixture"]}
    if task["category"] == "triage":
        problem = {"prompt": task["question"], "choices": dict.fromkeys(problem["choices"], "anonymous artifact")}
    shared = "Definition: FlashInfer-Bench rmsnorm_h4096, BF16, epsilon=1e-5.\n"
    if mode == "baseline" or task["category"] == "unanswerable":
        instructions = "Submit a final answer now with action=final."
    else:
        instructions = (
            "You can check a candidate with action=check and one choice. The harness then "
            "runs a public batch-1 FlashInfer-Bench check. You have at most three model turns; "
            "submit action=final by the last turn. You may submit without a check."
        )
    prompt = f"{shared}{instructions}\n\n{json.dumps(problem, ensure_ascii=False)}"
    history = []
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rmsnorm-model-") as cwd:
        for turn in range(1 if mode == "baseline" else 3):
            record = ask(prompt, args, cwd)
            history.append(record)
            answer = record["answer"]
            if answer.get("action") != "check" or mode == "baseline" or turn == 2:
                break
            choice = answer.get("answer")
            if choice in problem["choices"]:
                tested = benchmark.check(candidate(task, choice), public=True)
                feedback = {"ok": tested["ok"], "checks": tested["checks"]}
            else:
                feedback = {"error": "Choose a valid repair to check"}
            record["feedback"] = feedback
            prompt += (
                f"\n\nPrior response: {json.dumps(answer)}\n"
                f"Public test feedback: {json.dumps(feedback)}\n"
                f"Model turns remaining: {2 - turn}. "
                + ("Submit action=final now." if turn == 1 else "Submit or request another check.")
            )
    elapsed = time.monotonic() - start
    ok, reason, grading = score(task, answer, benchmark)
    return {
        "id": task["id"], "trial": trial, "passed": ok, "reason": reason,
        "turns": len(history), "tokens": sum(turn["tokens"] for turn in history),
        "latency_s": elapsed, "steps": history, "grading": grading,
    }


def metric_cells(rows):
    if not rows:
        return ["—"] * 5
    latency = sorted(row["latency_s"] for row in rows)
    return [
        fraction(sum(row['passed'] for row in rows), len(rows)),
        f"{statistics.mean(row['turns'] for row in rows):.2f}",
        f"{statistics.mean(latency):.2f}",
        f"{latency[math.ceil(0.95 * len(latency)) - 1]:.2f}",
        f"{statistics.mean(row['tokens'] for row in rows):.1f}",
    ]


def fraction(passed, count):
    return f"{passed / count:.0%} ({passed}/{count})"


def failure_paragraph(rows, tasks):
    failures = [row for row in rows if not row["passed"]]
    if not failures:
        return "No failed attempt is present in the recorded runs."
    failure = next((row for row in failures if row["trial"]), failures[0])
    task = tasks[failure["id"]]
    step = failure["steps"][-1]
    answer = step["answer"]
    choice = answer.get("answer")
    method = f"agent run {failure['trial']}" if failure["trial"] else "one-shot baseline"
    rationale = str(answer.get("reason", "")).replace("\n", " ")[:350]
    if choice in task["fixture"]["choices"]:
        chosen = task["fixture"]["choices"][choice].replace("`", "")
        expected = task["fixture"]["choices"][task["expected"]].replace("`", "")
        failed = [test["batch"] for test in (failure["grading"] or {}).get("checks", [])
                  if not test["ok"]]
        evidence = f" FlashInfer-Bench rejected batch sizes {failed}." if failed else ""
        diagnostic = ""
        if task["category"] == "triage":
            checks = [
                f"{entry['answer'].get('answer')} {'passed' if entry['feedback'].get('ok') else 'failed'}"
                for entry in failure["steps"] if "feedback" in entry
            ]
            diagnostic = (
                f" The public checks returned {', '.join(checks)}." if checks
                else " The task supplied opaque artifact labels, and no public check was run before submission."
            )
        return (
            f"On {failure['id']} ({method}), turn {failure['turns']} submitted `{chosen}`. "
            f"The required replacement was `{expected}`. {task['question']}"
            f"{diagnostic}{evidence} The model's stated reasoning was: “{rationale}”."
        )
    return (
        f"On {failure['id']} ({method}), the failure occurred at the final-submission step. "
        f"The grader recorded “{failure['reason']}” after turn {failure['turns']}. "
        f"The recorded answer was {choice!r} and the model's stated reasoning was “{rationale}”. "
        f"The expected answer was {task['expected']}. {' and '.join(task['rubric'])}."
    )


def write_report(path, rows, tasks, model, effort, sheng_netid="pending", trial_count=4):
    trials = {i: [row for row in rows if row["trial"] == i] for i in range(4)}
    task_map = {task["id"]: task for task in tasks}
    three = {task_id: {} for task_id in task_map}
    for row in rows:
        if row["trial"]:
            three[row["id"]][row["trial"]] = row["passed"]
    complete = all(len(results) == 3 for results in three.values())
    outcomes = {key: [result[index] for index in (1, 2, 3)]
                for key, result in three.items() if len(result) == 3}
    n = len(tasks)
    stability = [fraction(sum(any(flags) for flags in outcomes.values()), n),
                 fraction(sum(all(flags) for flags in outcomes.values()), n)] if complete else ["—", "—"]
    observed = sorted({m for row in rows for turn in row["steps"] for m in turn["models"]})
    definition = ("CUDA repair tasks for FlashInfer-Bench `rmsnorm_h4096`. "
                  "Some tasks present anonymous patch artifacts for benchmark-driven triage.")
    lines = [
        "# GPU Kernel Agent evaluation", "",
        f"Ryan Hou (NetID: ryanhou) · Sheng Wei (NetID: {sheng_netid})", "",
        f"{definition} Requested Claude Code model: `{model}` (effort: `{effort}`). "
        f"Observed backend model: {', '.join(observed) or 'pending'}.", "",
        "| Method | Accuracy | Average turns | Average latency (s) | p95 latency (s) | Average tokens | pass@3 | pass^3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, trial, extra in (("One-shot baseline", 0, ["—", "—"]), ("Agent, run 1", 1, stability)):
        lines.append("| " + " | ".join([label, *metric_cells(trials[trial]), *extra]) + " |")
    lines.extend(["", f"Recorded attempts: {len(rows)}/{trial_count * n}. "
                  "One turn is one model call. Latency includes model calls and any public checks through "
                  "submission. Hidden grading happens afterward. Identical source snapshots reuse benchmark "
                  "results in the same suite. Tokens include input, output, and cache tokens. "
                  "p95 uses the nearest-rank statistic. Both methods share the same task prompts. "
                  "For anonymous-patch triage, artifact contents are withheld and the agent can test "
                  "the artifacts through the public benchmark.", "", "## Accuracy by task category", "",
                  "| Category | One-shot baseline | Agent, run 1 |", "| --- | ---: | ---: |"])
    for category in ("repair", "triage", "edge", "unanswerable"):
        cells = []
        for trial in (0, 1):
            subset = [row for row in trials[trial] if task_map[row["id"]]["category"] == category]
            cells.append(fraction(sum(row["passed"] for row in subset), len(subset)) if subset else "—")
        lines.append(f"| {category.capitalize()} | {' | '.join(cells)} |")
    lines.extend(["", "## Outcomes that changed", ""])
    flips = [f"- `{task_id}` — " + " / ".join("pass" if value else "fail" for value in flags)
             for task_id, flags in outcomes.items() if len(set(flags)) > 1]
    lines.extend((flips or ["None."]) if complete else ["Pending completion of all three agent runs."])
    lines.extend(["", "## Failure trace", "", failure_paragraph(rows, task_map), ""])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "benchmarks/rmsnorm/tasks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval_results")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model-timeout", type=int, default=90)
    parser.add_argument("--benchmark-timeout", type=int, default=180)
    parser.add_argument("--sheng-netid", default="pending")
    parser.add_argument("--report-only", action="store_true", help="Regenerate results from saved traces.")
    parser.add_argument("--pilot", action="store_true", help="Run one task in each category for both methods.")
    args = parser.parse_args()
    tasks = read_tasks(args.tasks)
    if args.pilot:
        tasks = [next(task for task in tasks if task["category"] == category)
                 for category in ("repair", "edge", "unanswerable")]
    args.output.mkdir(parents=True, exist_ok=True)
    trace_path = args.output / "traces.jsonl"
    config_path = args.output / "config.json"
    config = {
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "model": args.model, "effort": args.effort, "pilot": args.pilot,
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Choose a new output directory for a different evaluation configuration")
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
    trial_count = 2 if args.pilot else 4

    def update_report():
        write_report(args.output / "results.md", rows, tasks, args.model, args.effort,
                     args.sheng_netid, trial_count)

    if args.report_only:
        if not rows:
            parser.error("No saved traces in this output directory")
        update_report()
        return
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = {(row["trial"], row["id"]) for row in rows}
    benchmark = BenchmarkTool(args.benchmark_timeout)
    reference = candidate(tasks[0], tasks[0]["expected"])
    warmup = benchmark.check(reference, public=False)
    if not warmup["ok"]:
        raise RuntimeError(f"Known-correct candidate failed the GPU preflight: {warmup}")
    (args.output / "preflight.json").write_text(json.dumps(warmup, indent=2) + "\n")
    jobs = [(task, trial) for trial in range(trial_count) for task in tasks]
    random.Random(20261120).shuffle(jobs)
    jobs = [(task, trial) for task, trial in jobs if (trial, task["id"]) not in done]
    with trace_path.open("a") as file, ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(lambda job: attempt(*job, args, benchmark), jobs):
            rows.append(row)
            file.write(json.dumps(row) + "\n")
            file.flush()
            update_report()
            print(f"{len(rows)}/{trial_count * len(tasks)} trial={row['trial']} {row['id']} passed={row['passed']}", flush=True)
    update_report()


if __name__ == "__main__":
    main()
