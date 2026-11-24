"""Attach an agent to an existing Harness run."""

import argparse
import json
import os
from pathlib import Path

from .agent import MCP_TOKEN_ENV, run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("claude", "codex"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="B", choices=("B", "D", "F"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--image", help="local rootless Podman image with the agent CLI")
    parser.add_argument("--executable", help="override the Claude Code or Codex executable")
    parser.add_argument("--examples", type=Path, help="read-only examples for variant F")
    args = parser.parse_args()
    token = os.environ.get(MCP_TOKEN_ENV)
    if not token:
        parser.error(f"set {MCP_TOKEN_ENV} to the bearer token for this run")
    try:
        result = run_agent(**vars(args), token=token)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result.to_dict()))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
