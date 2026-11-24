"""Run-scoped evaluation, ranking, and independent acceptance."""

import math
import time
import uuid

from . import mock_worker
from .feedback import public_feedback
from .results import exclusive_lock, read_json, timestamp, write_json
from .source_rules import violation


MAX_SOURCE_BYTES = 128 * 1024


class RunError(Exception):
    """An invalid operation requested for a run."""


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


class Evaluator:
    def __init__(self, store, gpu_worker=None):
        self.store = store
        self.gpu_worker = gpu_worker

    def evaluate(self, run_id, source):
        config = self.store.config(run_id)
        run = self.store.path(run_id)
        with exclusive_lock(run / ".lock"):
            self._require_open(run)
            evaluations = run / "evaluations"
            records = self.store.records(evaluations)
            used_budget = sum(record["status"] != "infrastructure_error" for record in records)
            if used_budget >= config["evaluation_budget"]:
                raise RunError("Evaluation budget exhausted")
            try:
                size = len(source.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise RunError("Source must be valid UTF-8") from error
            if not 0 < size <= MAX_SOURCE_BYTES:
                raise RunError(f"Source must contain 1 to {MAX_SOURCE_BYTES} bytes")
            candidate_id = self.store.snapshot(run, source)
            path = evaluations / f"{uuid.uuid4().hex}.json"
            record = self._record(path, config, candidate_id, "evaluate")
            request = self._request(config, source, "public")
            detail = violation(source)
            if detail:
                result = {"status": "rejected", "stage": "static", "cases": [], "detail": detail}
            else:
                result = self._invoke(config, request)
            score = self._score(result, request) if config["backend"] == "gpu" else None
            record.update(result)
            if score is not None:
                record["score"] = score
            self._finish(path, record)
            ranking = self._ranking([*records, record])
            if ranking:
                record["best_candidate_id"] = ranking[0]["candidate_id"]
            return public_feedback(record)

    def diagnostic(self, run_id, candidate_id, operation):
        config = self.store.config(run_id)
        if config["variant"] == "B" or operation not in {"sanitize", "profile"}:
            raise RunError("Diagnostic tool unavailable for this run")
        run = self.store.path(run_id)
        with exclusive_lock(run / ".lock"):
            self._require_open(run)
            directory = run / "diagnostics"
            if len(self.store.records(directory)) >= config["diagnostic_budget"]:
                raise RunError("Diagnostic budget exhausted")
            source = self.store.source(run, candidate_id)
            if violation(source):
                raise RunError("Candidate did not pass static checks")
            path = directory / f"{uuid.uuid4().hex}.json"
            record = self._record(path, config, candidate_id, operation)
            request = self._request(config, source, "public")
            record.update(self._invoke(config, request, operation))
            self._finish(path, record)
            return public_feedback(record)

    def accept(self, run_id):
        config = self.store.config(run_id)
        if config["backend"] != "gpu":
            raise RunError("Independent acceptance requires a GPU run")
        if self.gpu_worker is None:
            raise RunError("Configure the GPU worker before acceptance")
        run = self.store.path(run_id)
        directory = run / "acceptance"
        with exclusive_lock(run / ".lock"):
            final_path = directory / "decision.json"
            if final_path.exists():
                return read_json(final_path)
            shortlist_path = directory / "shortlist.json"
            if shortlist_path.exists():
                shortlist = read_json(shortlist_path)["candidates"]
            else:
                ranking = self._ranking(self.store.records(run / "evaluations"))
                shortlist = [{"candidate_id": item["candidate_id"], "public_score": item["score"]}
                             for item in ranking[:config["top_k"]]]
                write_json(shortlist_path, {"sealed_at": timestamp(), "candidates": shortlist})

            decision = {"status": "fallback", "implementation": "pytorch_eager", "candidate_id": None}
            for candidate in shortlist:
                candidate_id = candidate["candidate_id"]
                path = directory / f"{candidate_id}.json"
                result = read_json(path) if path.exists() else None
                if result is None or result["status"] in {"pending", "infrastructure_error"}:
                    result = self._holdout(config, run, candidate_id, path)
                if result["status"] == "infrastructure_error":
                    raise RunError("Independent acceptance interrupted by a GPU infrastructure error; retry accept")
                if result.get("conservative_speedup", 0) > config["minimum_speedup"]:
                    decision = {"status": "accepted", "implementation": "cuda", "candidate_id": candidate_id,
                                "speedup": result["score"], "conservative_speedup": result["conservative_speedup"]}
                    break
            decision["run_id"] = run_id
            decision["decided_at"] = timestamp()
            write_json(final_path, decision)
            return decision

    def _holdout(self, config, run, candidate_id, path):
        source = self.store.source(run, candidate_id)
        request = self._request(config, source, "holdout")
        record = self._record(path, config, candidate_id, "accept")
        result = self._invoke(config, request)
        record.update(result)
        score = self._score(result, request)
        if score is not None:
            record["score"] = score
            record["conservative_speedup"] = self._conservative_speedup(result["cases"])
        self._finish(path, record)
        return record

    @staticmethod
    def _request(config, source, phase):
        return {"source": source, "phase": phase,
                "cases": config["public_cases"] if phase == "public" else config["holdout_cases"],
                "timing": {"warmups": config["warmups"], "samples": config["samples"]}}

    def _invoke(self, config, request, operation=None):
        if config["backend"] == "mock":
            return mock_worker.diagnose(request, operation) if operation else mock_worker.evaluate(request)
        if self.gpu_worker is None:
            return {"status": "infrastructure_error", "stage": "worker", "cases": [],
                    "detail": "No GPU worker configured"}
        return self.gpu_worker.run(request, operation=operation)

    @staticmethod
    def _require_open(run):
        if (run / "acceptance" / "shortlist.json").exists():
            raise RunError("Search has ended for this run")

    @staticmethod
    def _record(path, config, candidate_id, operation):
        record = {"candidate_id": candidate_id, "backend": config["backend"], "operation": operation,
                  "status": "pending", "started_at": timestamp(), "started_monotonic": time.monotonic()}
        write_json(path, record)
        return record

    @staticmethod
    def _finish(path, record):
        record["duration_seconds"] = time.monotonic() - record.pop("started_monotonic")
        record["finished_at"] = timestamp()
        write_json(path, record)

    @staticmethod
    def _ranking(records):
        eligible = [item for item in records if item["backend"] == "gpu"
                    and item["status"] == "passed" and "score" in item]
        first_by_candidate = {}
        for item in sorted(eligible, key=lambda item: item["started_at"]):
            first_by_candidate.setdefault(item["candidate_id"], item)
        return sorted(first_by_candidate.values(), key=lambda item: (-item["score"], item["candidate_id"]))

    @staticmethod
    def _score(result, request):
        if result["status"] != "passed":
            return None
        cases = result["cases"]
        expected = {(case["case_id"], case["batch_size"]) for case in request["cases"]}
        actual = {(case["case_id"], case["batch_size"]) for case in cases}
        if actual != expected or len(cases) != len(expected):
            return None
        ratios = []
        for case in cases:
            eager = case.get("eager_latency_us", 0)
            candidate = case.get("latency_us", 0)
            if case["status"] != "passed" or not (math.isfinite(eager) and math.isfinite(candidate)):
                return None
            if min(eager, candidate) <= 0:
                return None
            ratios.append(eager / candidate)
        return geometric_mean(ratios)

    @staticmethod
    def _conservative_speedup(cases):
        # Compare medians after allowing two MADs in each implementation's unfavorable direction.
        ratios = []
        for case in cases:
            eager_mad = case.get("eager_variation_us")
            candidate_mad = case.get("variation_us")
            if eager_mad is None or candidate_mad is None:
                return 0.0
            eager_lower = case["eager_latency_us"] - 2 * eager_mad
            candidate_upper = case["latency_us"] + 2 * candidate_mad
            if eager_lower <= 0:
                return 0.0
            ratios.append(eager_lower / candidate_upper)
        return geometric_mean(ratios)
