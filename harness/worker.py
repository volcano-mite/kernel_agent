"""Launch isolated GPU jobs and serialize access to each device."""

import hashlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import time
import uuid
from pathlib import Path
from tempfile import TemporaryFile

from . import diagnostics
from .host_worker import LAUNCH_ENV, SANDBOX_ROOT, WORKER_FILES, HostWorkerLaunch
from .results import exclusive_lock


ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT = 2 * 1024 * 1024
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


class WorkerInterrupted(Exception):
    pass


def _capture(command, value, timeout, *, env=None):
    with TemporaryFile() as input_file:
        input_file.write(json.dumps(value).encode())
        input_file.seek(0)
        with subprocess.Popen(command, stdin=input_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=env, start_new_session=True) as process, selectors.DefaultSelector() as selector:
            output = {process.stdout: bytearray(), process.stderr: bytearray()}
            for stream in output:
                selector.register(stream, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            try:
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        buffer = output[key.fileobj]
                        buffer.extend(chunk)
                        if len(buffer) > MAX_OUTPUT:
                            raise WorkerInterrupted("Worker exceeded the output limit")
                remaining = max(0, deadline - time.monotonic())
                code = process.wait(timeout=remaining)
                return code, *(output[stream].decode(errors="replace") for stream in (process.stdout, process.stderr))
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise


class GPUWorker:
    def __init__(self, image, gpu_uuid, timeout, *, host=None):
        if host is None and (not image or image.startswith("-") or any(c.isspace() for c in image)):
            raise ValueError("Supply a valid local GPU Worker image name")
        if not gpu_uuid or not GPU_UUID.fullmatch(gpu_uuid):
            raise ValueError("The GPU must be identified by its NVIDIA GPU UUID")
        if os.geteuid() == 0:
            raise ValueError("Run the GPU Worker from an unprivileged account")
        self.image = image
        self.gpu_uuid = gpu_uuid
        self.timeout = timeout
        self.host = HostWorkerLaunch(host, gpu_uuid, ROOT) if host is not None else None
        lock_root = Path("/tmp") / f"kernel-agent-gpus-{os.getuid()}"
        lock_root.mkdir(mode=0o700, exist_ok=True)
        if lock_root.is_symlink() or lock_root.stat().st_uid != os.getuid():
            raise ValueError("GPU lock directory must be owned by the operator")
        device = hashlib.sha256(gpu_uuid.lower().encode()).hexdigest()
        self.lock = lock_root / f"{device}.lock"
        self.unhealthy = lock_root / f"{device}.unhealthy"

    def run(self, request, operation=None):
        with exclusive_lock(self.lock):
            if self.unhealthy.exists() and not self._verify_device():
                return self._failure("infrastructure_error", "GPU health check failed")
            if operation is None:
                return self._launch(request)

            probe = {**request, "operation": "probe", "profile": operation == "profile",
                     "cases": [request["cases"][len(request["cases"]) // 2]]}
            summaries = []
            logs = []
            for tool, command in diagnostics.commands(operation):
                result = self._launch(probe, command)
                if result["status"] not in {"passed", "diagnostic_exit"}:
                    return result
                log = result.get("worker_log", "")
                logs.append(log)
                summaries.append(diagnostics.summarize(tool, log, result["worker_exit_code"]))
            statuses = {item["status"] for item in summaries}
            status = "diagnostic_error" if "diagnostic_error" in statuses else (
                "issues_found" if "issues_found" in statuses else "passed")
            return {"status": status, "stage": operation, "cases": [], "diagnostic": summaries,
                    "worker_log": "\n".join(logs)}

    def _base(self, name):
        return ["podman", "run", "--rm", "--pull=never", f"--name={name}", "--interactive",
                "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--security-opt=label=disable", "--pids-limit=256", "--memory=8g", "--cpus=4",
                "--tmpfs=/tmp:rw,exec,mode=1777,size=4g", "--http-proxy=false", "--log-driver=none",
                f"--timeout={math.ceil(self.timeout) + 2}",
                "--env=HOME=/tmp", "--env=XDG_CACHE_HOME=/tmp/cache", "--env=MAX_JOBS=2",
                f"--device=nvidia.com/gpu={self.gpu_uuid}"]

    def _execute(self, name, value, python_args, timeout, wrapper=()):
        if self.host is not None:
            command = self.host.command(python_args, wrapper)
            return _capture(command, value, timeout, env=LAUNCH_ENV)
        command = self._base(name)
        for filename in WORKER_FILES:
            command.extend(("--volume", f"{ROOT / filename}:{SANDBOX_ROOT / filename}:ro"))
        command.extend((f"--workdir={SANDBOX_ROOT}", self.image, *wrapper, "python", *python_args))
        return _capture(command, value, timeout)

    def _launch(self, request, wrapper=()):
        name = f"kernel-agent-worker-{uuid.uuid4().hex}"
        try:
            exit_code, stdout, stderr = self._execute(name, request, ("-m", "harness.gpu_worker"),
                                                      self.timeout, wrapper)
        except (subprocess.TimeoutExpired, WorkerInterrupted) as error:
            self._recover(name)
            detail = "Worker deadline exceeded" if isinstance(error, subprocess.TimeoutExpired) else str(error)
            return self._failure("infrastructure_error", detail)
        except (OSError, ValueError) as error:
            return self._failure("infrastructure_error", "GPU Worker setup failed. Check the worker log.", str(error))
        except BaseException:
            self._recover(name)
            raise

        if exit_code == 0:
            try:
                result = json.loads(stdout)
                if (not isinstance(result, dict) or not isinstance(result.get("status"), str)
                        or not isinstance(result.get("stage"), str) or not isinstance(result.get("cases"), list)):
                    raise ValueError("Incomplete worker reply")
            except ValueError:
                self._recover(name)
                return self._failure("infrastructure_error", "Worker did not return a valid result", stderr)
            result.update(worker_log=stderr, worker_exit_code=exit_code)
            if result["status"] == "runtime_error":
                self._recover(name)
            return result
        if self.host is not None and stderr.startswith("bwrap:"):
            return self._failure("infrastructure_error", "Host GPU Worker isolation failed. Check the worker log.", stderr)
        if self.host is None and exit_code in {125, 126, 127}:
            return self._failure("infrastructure_error", "Podman could not launch the GPU worker", stderr)
        if wrapper and exit_code == 86:
            return {"status": "diagnostic_exit", "stage": "diagnostic", "cases": [],
                    "worker_log": stderr, "worker_exit_code": exit_code}
        if wrapper and exit_code > 0 and exit_code < 128:
            return self._failure("diagnostic_error", diagnostics.error_detail(stderr), stderr)
        self._recover(name)
        return self._failure("infrastructure_error", f"Worker exited without a result (status {exit_code})", stderr)

    def _remove(self, name):
        if self.host is not None:
            # Bubblewrap's PID namespace is torn down when the launcher exits.
            return True
        try:
            subprocess.run(["podman", "rm", "--ignore", "--force", name], timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (OSError, subprocess.SubprocessError):
            return False
        return True

    def _recover(self, name):
        self.unhealthy.touch()
        if self._remove(name):
            self._verify_device()

    def _verify_device(self):
        name = f"kernel-agent-health-{uuid.uuid4().hex}"
        probe = ("import sys, torch; assert torch.cuda.device_count() == 1; "
                 "uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-'); "
                 "assert uuid == sys.argv[1].lower().removeprefix('gpu-'); "
                 "torch.zeros(1, device='cuda'); torch.cuda.synchronize()")
        try:
            code, _, _ = self._execute(name, {}, ("-c", probe, self.gpu_uuid), 30)
        except (OSError, ValueError, subprocess.SubprocessError, WorkerInterrupted):
            self._remove(name)
            return False
        if code == 0:
            self.unhealthy.unlink(missing_ok=True)
            return True
        return False

    @staticmethod
    def _failure(status, detail, log=""):
        return {"status": status, "stage": "worker", "cases": [], "detail": detail, "worker_log": log}
