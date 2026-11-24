"""Create and operate a kernel-agent run from the host."""

import argparse
import json
import math
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from tasks.rmsnorm import HOLDOUT_CASES, PUBLIC_CASES, TASK_ID, initialize_workspace

from .evaluator import Evaluator, RunError
from .host_worker import host_settings
from .results import RunStore, write_json
from .worker import GPUWorker


REPOSITORY = Path(__file__).resolve().parents[1]


def _configure(parser):
    parser.add_argument("--backend", choices=("mock", "gpu"), default="mock")
    parser.add_argument("--variant", choices=("B", "D", "F"), default="B")
    parser.add_argument("--agent", choices=("claude", "codex"), default="claude")
    parser.add_argument("--model")
    parser.add_argument("--evaluation-budget", type=int, default=10)
    parser.add_argument("--diagnostic-budget", type=int, default=2)
    parser.add_argument("--agent-timeout", type=float, default=900)
    parser.add_argument("--worker-timeout", type=float, default=300)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--gpu-worker", choices=("podman", "host"), default="podman",
                        help="GPU Worker execution mode (default: podman)")
    parser.add_argument("--gpu-worker-python", type=Path,
                        help="CUDA-enabled Python executable for host mode (default: current Python)")
    parser.add_argument("--gpu-image", help="Local GPU Worker image for Podman mode")
    parser.add_argument("--gpu-uuid")


def _create(store, args):
    if min(args.evaluation_budget, args.top_k) < 1 or args.diagnostic_budget < 0:
        raise RunError("Evaluation budget and top K must be positive; diagnostics budget may be zero")
    if not 1 <= args.warmups <= 100 or not 3 <= args.samples <= 1000:
        raise RunError("Use 1 to 100 warmups and 3 to 1000 samples")
    if not all(math.isfinite(value) and value > 0 for value in (args.agent_timeout, args.worker_timeout)):
        raise RunError("Agent and worker timeouts must be positive numbers")
    if not math.isfinite(args.minimum_speedup) or args.minimum_speedup < 1.0:
        raise RunError("Minimum speedup must be at least 1.0")
    use_host = args.backend == "gpu" and args.gpu_worker == "host"
    if args.backend != "gpu" and args.gpu_worker == "host":
        raise RunError("--gpu-worker host requires --backend gpu")
    if args.gpu_worker_python and not use_host:
        raise RunError("--gpu-worker-python requires --backend gpu --gpu-worker host")
    if args.backend == "gpu" and not args.gpu_uuid:
        raise RunError("GPU runs require --gpu-uuid")
    if args.backend == "gpu" and not use_host and not args.gpu_image:
        raise RunError("Podman GPU Workers require --gpu-image")
    if use_host and args.gpu_image:
        raise RunError("--gpu-image is available for Podman GPU Workers")

    config = {
        "task": TASK_ID,
        "task_revision": "flashinfer-trace@da915083d4c7c5e61aa3005e3d17ae488e0fc71c",
        "public_cases": PUBLIC_CASES,
        "holdout_cases": HOLDOUT_CASES,
        "backend": args.backend,
        "variant": args.variant,
        "agent": args.agent,
        "model": args.model,
        "evaluation_budget": args.evaluation_budget,
        "diagnostic_budget": args.diagnostic_budget,
        "agent_timeout": args.agent_timeout,
        "worker_timeout": args.worker_timeout,
        "warmups": args.warmups,
        "samples": args.samples,
        "top_k": args.top_k,
        "minimum_speedup": args.minimum_speedup,
        "gpu_worker": args.gpu_worker,
        "gpu_image": args.gpu_image,
        "gpu_uuid": args.gpu_uuid,
    }
    if use_host:
        config["host_worker"] = host_settings(args.gpu_worker_python)
    if args.backend == "gpu":
        _gpu_worker(config)
    run_id, token = store.create(config)
    workspace = store.path(run_id) / "workspace"
    initialize_workspace(workspace)
    return run_id, token, workspace


def _evaluator(store, run_id):
    config = store.config(run_id)
    worker = _gpu_worker(config) if config["backend"] == "gpu" else None
    return Evaluator(store, gpu_worker=worker)


def _gpu_worker(config):
    mode = config.get("gpu_worker", "podman")
    if mode == "host":
        return GPUWorker(None, config["gpu_uuid"], config["worker_timeout"], host=config["host_worker"])
    if mode == "podman":
        return GPUWorker(config["gpu_image"], config["gpu_uuid"], config["worker_timeout"])
    raise RunError(f"Unknown GPU Worker mode: {mode}")


def _server(evaluator, run_id, port):
    import uvicorn

    from .mcp_server import create_app

    return uvicorn.Server(uvicorn.Config(create_app(evaluator, run_id), host="127.0.0.1", port=port,
                                         log_level="warning", timeout_graceful_shutdown=5))


@contextmanager
def _local_mcp(evaluator, run_id):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = _server(evaluator, run_id, port)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RunError("MCP server failed to start")
                time.sleep(0.02)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def _run_agent(store, args):
    from runners import run_agent

    if not args.model:
        raise RunError("Provide --model for a noninteractive Agent run")
    if args.host_agent and args.backend != "mock":
        raise RunError("Host mode is available for Mock development runs")
    run_id, token, workspace = _create(store, args)
    run = store.path(run_id)
    print(f"Started run {run_id} in {workspace}", file=sys.stderr, flush=True)
    evaluator = _evaluator(store, run_id)
    log_dir = run / "agent"
    with _local_mcp(evaluator, run_id) as mcp_url:
        process = run_agent(backend=args.agent, model=args.model, variant=args.variant, run_id=run_id,
                            mcp_url=mcp_url, token=token, workspace=workspace, log_dir=log_dir,
                            timeout=args.agent_timeout, image=args.agent_image,
                            executable=args.agent_executable,
                            examples=REPOSITORY / "examples" if args.variant == "F" else None)
    outcome = process.to_dict()
    write_json(log_dir / "result.json", outcome)
    result = {"run_id": run_id, "workspace": str(workspace), "agent": outcome}
    if args.backend == "gpu" and process.status in {"completed", "timeout"}:
        result["acceptance"] = evaluator.accept(run_id)
    return result, int(process.status != "completed")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Prepare a task workspace and issue a run token")
    _configure(init)
    serve = commands.add_parser("serve", help="Expose the run's MCP tools on loopback")
    serve.add_argument("run_id")
    serve.add_argument("--port", type=int, default=8000)
    evaluate = commands.add_parser("evaluate", help="Submit a source file as the trusted host operator")
    evaluate.add_argument("run_id")
    evaluate.add_argument("source", type=Path)
    accept = commands.add_parser("accept", help="Seal a GPU run and perform independent acceptance")
    accept.add_argument("run_id")
    run = commands.add_parser("run", help="Launch the MCP service and one noninteractive Agent")
    _configure(run)
    location = run.add_mutually_exclusive_group(required=True)
    location.add_argument("--agent-image", help="Locally built Podman image with Claude Code and Codex")
    location.add_argument("--host-agent", action="store_true", help="Run the Agent on the host for Mock development")
    run.add_argument("--agent-executable", help="Override the Agent executable for development")
    return parser.parse_args()


def main():
    args = _arguments()
    store = RunStore(args.results)
    try:
        if args.command == "init":
            run_id, token, workspace = _create(store, args)
            result = {"run_id": run_id, "token": token, "workspace": str(workspace)}
        elif args.command == "run":
            result, code = _run_agent(store, args)
            print(json.dumps(result, indent=2))
            return code
        else:
            evaluator = _evaluator(store, args.run_id)
            if args.command == "serve":
                _server(evaluator, args.run_id, args.port).run()
                return 0
            if args.command == "evaluate":
                result = evaluator.evaluate(args.run_id, args.source.read_bytes().decode("utf-8"))
            else:
                result = evaluator.accept(args.run_id)
        print(json.dumps(result, indent=2))
    except (OSError, RunError, ValueError, ModuleNotFoundError) as error:
        print(f"Kernel Agent: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
