"""Commands and public summaries for the optional GPU diagnostics."""

import csv
import re


METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


def commands(operation):
    if operation == "sanitize":
        return [(tool, ["compute-sanitizer", "--tool", tool, "--error-exitcode", "86",
                        "--log-file", "/dev/stderr"]) for tool in ("memcheck", "racecheck")]
    return [("ncu", ["ncu", "--profile-from-start", "off", "--csv", "--page", "raw",
                     "--log-file", "/dev/stderr", "--metrics", ",".join(METRICS)])]


def summarize(tool, log, exit_code):
    if tool == "ncu":
        reader = csv.DictReader(line for line in log.splitlines() if line.startswith('"'))
        counters = []
        for row in reader:
            if row.get("Metric Name") in METRICS:
                counters.append({"kernel": row.get("Kernel Name", ""), "metric": row["Metric Name"],
                                 "unit": row.get("Metric Unit", ""), "value": row.get("Metric Value", "")})
                if len(counters) == 24:
                    break
        return {"tool": tool, "status": "passed" if exit_code == 0 and counters else "diagnostic_error",
                "counters": counters}

    counts = [int(value) for value in re.findall(r"(?:ERROR|RACECHECK) SUMMARY:\s*(\d+)", log)]
    errors = [line.lstrip("= ")[:300] for line in log.splitlines()
              if re.match(r"^=+\s*(?:Error|Warning|Invalid|Potential)", line)]
    if exit_code == 86 or any(counts):
        status = "issues_found"
    else:
        status = "passed" if exit_code == 0 and counts else "diagnostic_error"
    return {"tool": tool, "status": status, "errors": max(counts, default=0), "messages": errors[:8]}


def error_detail(log):
    for line in log.splitlines():
        if line.startswith("==ERROR==") or "ERR_NVGPUCTRPERM" in line:
            return line[:1000]
    return "The diagnostic tool failed"
