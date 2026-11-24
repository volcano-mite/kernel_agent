"""A deterministic backend for developing the agent feedback loop on CPU."""


def evaluate(request):
    source = request["source"]
    if "// mock: compile_error" in source or "run(" not in source.replace(" ", ""):
        return {"status": "compile_error", "stage": "compile", "cases": [],
                "detail": "Mock compiler expected a run entry point"}

    if "// mock: timeout" in source:
        return {"status": "timeout", "stage": "correctness", "cases": [],
                "detail": "Mock evaluation timed out"}

    wrong = "// mock: wrong_answer" in source or "const float scale = 0.5f;" in source
    cases = [{"case_id": case["case_id"], "batch_size": case["batch_size"],
              "status": "wrong_answer" if wrong else "passed"} for case in request["cases"]]
    return {"status": "wrong_answer" if wrong else "passed", "stage": "correctness", "cases": cases,
            "detail": "Simulated result. No CUDA compilation or measurement was performed."}


def diagnose(_request, operation):
    return {"status": "passed", "stage": operation, "cases": [],
            "detail": "Simulated diagnostic result. No GPU tool was run."}
