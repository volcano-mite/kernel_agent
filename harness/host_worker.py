"""Build the sandbox for one GPU Worker process on the Linux host."""

import os
import re
import shutil
import sys
from pathlib import Path


SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
LAUNCH_ENV = {"PATH": SYSTEM_PATH, "LC_ALL": "C"}
NVIDIA_GPUS = Path("/proc/driver/nvidia/gpus")
WORKER_FILES = ("harness/gpu_worker.py", "tasks/rmsnorm_workload.py")
SANDBOX_ROOT = Path("/opt/kernel-agent")


def _absolute(path):
    # Preserve the final symlink so Python recognizes a virtual environment.
    return Path(os.path.abspath(Path(path).expanduser()))


def _compiler(variable, default):
    value = os.environ.get(variable)
    executable = shutil.which(value or default)
    if value and not executable:
        raise ValueError(f"{variable} must name an executable compiler")
    return str(_absolute(executable)) if executable else None


def host_settings(python=None):
    """Record the operator-selected runtime paths in the run manifest."""
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = _absolute(nvcc).resolve().parent.parent if nvcc else "/usr/local/cuda"
    interpreter = _absolute(python or sys.executable)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"GPU Worker Python is not executable: {interpreter}")
    return {
        "python": str(interpreter),
        "cuda_home": str(_absolute(cuda_home).resolve()),
        "cc": _compiler("CC", "cc"),
        "cxx": _compiler("CXX", "c++"),
    }


def _gpu_device(gpu_uuid):
    # CUDA ordinals can differ from Linux device minors.
    for information in NVIDIA_GPUS.glob("*/information"):
        fields = dict(line.split(":", 1) for line in information.read_text().splitlines() if ":" in line)
        if fields.get("GPU UUID", "").strip().lower() != gpu_uuid.lower():
            continue
        minor = fields.get("Device Minor", "").strip()
        if re.fullmatch(r"[0-9]+", minor):
            return Path(f"/dev/nvidia{int(minor)}")
        break
    raise FileNotFoundError(f"No NVIDIA device minor found for {gpu_uuid}")


def _runtime_prefix(executable):
    if executable.parent in (Path("/bin"), Path("/sbin")):
        return Path("/usr")
    prefix = executable.parent.parent
    # Software Collections compilers also use their own etc/alternatives.
    if prefix.name == "usr" and prefix != Path("/usr") and (prefix.parent / "etc/alternatives").is_dir():
        return prefix.parent
    return prefix


def _bind_runtime(command, paths, repository):
    mounted = [Path("/usr")]
    for path in sorted(set(paths), key=lambda item: (len(item.parts), str(item))):
        if any(path.is_relative_to(root) for root in mounted):
            continue
        if any(private.is_relative_to(path) for private in (Path.home(), repository, Path("/tmp"))):
            raise ValueError(f"Use a dedicated installation directory for GPU Worker runtime: {path}")
        command.extend(("--ro-bind", str(path), str(path)))
        mounted.append(path)


class HostWorkerLaunch:
    def __init__(self, settings, gpu_uuid, repository):
        self.settings = settings
        self.gpu_uuid = gpu_uuid
        self.repository = repository

    def command(self, python_args, wrapper=()):
        bwrap = shutil.which("bwrap", path=SYSTEM_PATH)
        if not bwrap:
            raise FileNotFoundError("Host GPU Worker requires Bubblewrap (bwrap)")
        python = Path(self.settings["python"])
        cuda_home = Path(self.settings["cuda_home"])
        if not (cuda_home / "bin/nvcc").is_file():
            raise FileNotFoundError(f"No nvcc under the configured CUDA toolkit: {cuda_home}")

        command = [bwrap, "--unshare-all", "--uid", "65534", "--gid", "65534", "--cap-drop", "ALL",
                   "--die-with-parent", "--new-session", "--proc", "/proc", "--dev", "/dev",
                   "--tmpfs", "/tmp", "--ro-bind", "/usr", "/usr", "--ro-bind", "/sys", "/sys"]
        for path in map(Path, ("/bin", "/sbin", "/lib", "/lib64")):
            if path.is_symlink():
                command.extend(("--symlink", os.readlink(path), str(path)))
            elif path.is_dir():
                command.extend(("--ro-bind", str(path), str(path)))
        for path in ("/etc/alternatives", "/etc/ld.so.cache"):
            command.extend(("--ro-bind-try", path, path))

        compiler_paths = {name: Path(self.settings[key]) for name, key in (("CC", "cc"), ("CXX", "cxx"))
                          if self.settings[key]}
        tools = [python, *compiler_paths.values()]
        prefixes = [_runtime_prefix(binary) for tool in tools for binary in (tool, tool.resolve())]
        _bind_runtime(command, [cuda_home, *prefixes], self.repository)
        for filename in WORKER_FILES:
            command.extend(("--ro-bind", str(self.repository / filename), str(SANDBOX_ROOT / filename)))

        for device in (_gpu_device(self.gpu_uuid), Path("/dev/nvidiactl"), Path("/dev/nvidia-uvm")):
            if not device.is_char_device():
                raise FileNotFoundError(f"Host GPU Worker requires NVIDIA device node {device}")
            command.extend(("--dev-bind", str(device), str(device)))
        command.extend(("--dev-bind-try", "/dev/nvidia-uvm-tools", "/dev/nvidia-uvm-tools"))

        path = os.pathsep.join(dict.fromkeys(map(str, [*(tool.parent for tool in tools), cuda_home / "bin"])))
        env = {
            "PATH": f"{path}:{SYSTEM_PATH}",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "XDG_CACHE_HOME": "/tmp/cache",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_HOME": str(cuda_home),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid,
            "FLASHINFER_WORKSPACE_BASE": "/tmp",
            "MAX_JOBS": "2",
            **{name: str(value) for name, value in compiler_paths.items()},
        }
        for key, value in env.items():
            command.extend(("--setenv", key, value))
        return [*command, "--remount-ro", "/", "--chdir", str(SANDBOX_ROOT), "--", *wrapper, str(python), *python_args]
