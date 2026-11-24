"""Project internal evaluation records onto the public MCP response."""


def public_feedback(record):
    fields = ("candidate_id", "backend", "status", "stage", "score", "best_candidate_id", "operation")
    response = {name: record[name] for name in fields if name in record}
    case_fields = ("case_id", "batch_size", "status", "latency_us", "eager_latency_us", "variation_us",
                   "eager_variation_us", "max_absolute_error", "max_relative_error", "matched_ratio", "detail")
    response["cases"] = [
        {name: case[name] for name in case_fields if name in case}
        for case in record.get("cases", [])
    ]
    if record.get("detail"):
        response["detail"] = record["detail"][:4000]
    if record.get("compiler"):
        compiler = record["compiler"]
        response["compiler"] = {"kernels": compiler.get("kernels", [])[:32]}
        if record["status"] == "compile_error" and compiler.get("log"):
            response["compiler"]["log_excerpt"] = compiler["log"][-6000:]
    if record.get("diagnostic"):
        response["diagnostic"] = record["diagnostic"]
    return response
