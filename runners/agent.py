"""Launch one Claude Code or Codex session for a prepared run workspace."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4


MCP_TOKEN_ENV = "KERNEL_AGENT_MCP_TOKEN"
MCP_SERVER = "kernel_harness"
BASE_TOOLS = ("evaluate_kernel",)
DIAGNOSTIC_TOOLS = ("sanitize_kernel", "profile_kernel")
MODEL_ENV = {
    "claude": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ),
    "codex": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY"),
}
HOST_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TMPDIR",
    "CODEX_HOME", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "SSL_CERT_FILE",
    "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
)
PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_EXAMPLES = "/examples"
CONTAINER_HOME = "/run/agent-home"


@dataclass(frozen=True)
class AgentResult:
    status: str
    returncode: int | None
    duration_seconds: float
    usage: dict[str, int | float]
    stdout_path: Path
    stderr_path: Path
    error: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["stdout_path"] = str(self.stdout_path)
        result["stderr_path"] = str(self.stderr_path)
        return result


def _prompt(run_id: str, variant: str, examples_dir: str | None) -> str:
    prompt = (
        "Read TASK.md and the starting rmsnorm.cu. Optimize the CUDA C++ candidate "
        "by editing rmsnorm.cu. To evaluate a revision, call the kernel_harness MCP "
        f"tool evaluate_kernel with run_id {json.dumps(run_id)} and the entire current "
        "contents of rmsnorm.cu as source. Use the public feedback to improve subsequent "
        "submissions. Finish with a short summary and the candidate ID for your best "
        "publicly evaluated revision."
    )
    if variant != "B":
        prompt += (
            " For additional evidence, call sanitize_kernel or profile_kernel with the same "
            "run_id and a candidate_id returned by evaluate_kernel."
        )
    if examples_dir:
        prompt += f" Read-only examples for other CUDA operators are available at {json.dumps(examples_dir)}."
    return prompt


def _tool_names(variant: str) -> tuple[str, ...]:
    if variant == "B":
        return BASE_TOOLS
    return BASE_TOOLS + DIAGNOSTIC_TOOLS


def _command(
    backend: str,
    executable: str,
    model: str,
    variant: str,
    mcp_url: str,
    timeout: float,
    container: bool,
    examples_dir: str | None,
) -> list[str]:
    tools = _tool_names(variant)
    if backend == "claude":
        mcp_config = {"mcpServers": {MCP_SERVER: {
            "type": "http",
            "url": mcp_url,
            "headers": {"Authorization": f"Bearer ${{{MCP_TOKEN_ENV}}}"},
            "timeout": math.ceil(timeout * 1000),
        }}}
        allowed_tools = ",".join(("Edit(rmsnorm.cu)", *(f"mcp__{MCP_SERVER}__{tool}" for tool in tools)))
        command = [
            executable, "-p", "--model", model, "--output-format", "stream-json",
            "--verbose", "--no-session-persistence", "--permission-mode", "dontAsk",
            "--setting-sources", "", "--disable-slash-commands",
            "--mcp-config", json.dumps(mcp_config), "--strict-mcp-config",
            "--tools", "Read,Edit,Write,Glob,Grep", "--allowed-tools", allowed_tools,
        ]
        if examples_dir:
            command.extend(("--add-dir", examples_dir))
        return command

    namespace = f"mcp_servers.{MCP_SERVER}"
    command = [
        executable, "exec", "--model", model, "--json", "--ephemeral",
        "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
        "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
        "-c", f"{namespace}.url={json.dumps(mcp_url)}",
        "-c", f'{namespace}.bearer_token_env_var="{MCP_TOKEN_ENV}"',
        "-c", f"{namespace}.required=true",
        "-c", f"{namespace}.tool_timeout_sec={math.ceil(timeout)}",
        "-c", f"{namespace}.enabled_tools={json.dumps(tools)}",
        "-c", f'{namespace}.default_tools_approval_mode="approve"',
    ]
    # Podman already provides filesystem and device isolation without nested namespaces.
    sandbox = "danger-full-access" if container else "workspace-write"
    command.extend(("--sandbox", sandbox, "-"))
    return command


def _agent_url(mcp_url: str, container: bool) -> tuple[str, bool]:
    url = urlsplit(mcp_url)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("MCP URL must be an HTTP(S) endpoint without credentials, query or fragment")
    port = url.port
    if not container or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return mcp_url, False
    if url.scheme == "https" or url.hostname == "::1":
        raise ValueError("Use an MCP URL reachable from Podman for HTTPS or IPv6 loopback")
    authority = "host.containers.internal"
    if port is not None:
        authority += f":{port}"
    return urlunsplit((url.scheme, authority, url.path, "", "")), True


def _environment(backend: str, token: str, container: bool) -> dict[str, str]:
    names = MODEL_ENV[backend] + PROXY_ENV
    if not container:
        names += HOST_ENV
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update({
        MCP_TOKEN_ENV: token,
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    if container:
        environment.update({
            "HOME": CONTAINER_HOME,
            # An explicit CODEX_HOME must already exist when Codex starts.
            "CODEX_HOME": CONTAINER_HOME,
            "CLAUDE_CONFIG_DIR": f"{CONTAINER_HOME}/.claude",
        })
    return environment


def _podman_command(
    command: list[str],
    image: str,
    name: str,
    workspace: Path,
    examples: Path | None,
    environment: dict[str, str],
    loopback: bool,
    timeout: float,
) -> list[str]:
    podman = [
        "podman", "run", "--rm", "--interactive", "--name", name,
        "--timeout", str(math.ceil(timeout) + 1),
        "--pull=never", "--userns=keep-id", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--http-proxy=false",
        "--tmpfs", f"{CONTAINER_HOME}:rw,mode=1777,size=128m",
        "--tmpfs", "/tmp:rw,exec,mode=1777,size=256m",
        "--volume", f"{workspace}:{CONTAINER_WORKSPACE}:rw,Z",
        "--workdir", CONTAINER_WORKSPACE,
    ]
    if loopback:
        podman.extend((
            "--network", "slirp4netns:allow_host_loopback=true",
            "--add-host", "host.containers.internal:10.0.2.2",
        ))
    if examples is not None:
        podman.extend(("--volume", f"{examples}:{CONTAINER_EXAMPLES}:ro,z"))
    for variable, value in environment.items():
        assignment = f"{variable}={value}" if variable == "HOME" else variable
        podman.extend(("--env", assignment))
    podman.extend(("--entrypoint", command[0], image, *command[1:]))
    return podman


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _remove_container(name: str, environment: dict[str, str], stderr: BinaryIO) -> None:
    try:
        subprocess.run(
            ["podman", "rm", "--force", "--ignore", name], env=environment,
            stdout=subprocess.DEVNULL, stderr=stderr, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        stderr.write(b"Podman cleanup failed\n")


def _copy_log(source: BinaryIO, path: Path, secrets: tuple[bytes, ...]) -> None:
    source.seek(0)
    output_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(output_fd, "wb") as output:
        for line in source:
            for secret in secrets:
                line = line.replace(secret, b"[redacted]")
            output.write(line)


def _read_usage(backend: str, stdout_path: Path) -> tuple[dict[str, int | float], bool]:
    usage: dict[str, int | float] = {}
    failed = False
    with stdout_path.open(encoding="utf-8", errors="replace") as stdout:
        for line in stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if backend == "codex":
                if event.get("type") == "turn.failed":
                    failed = True
                if event.get("type") != "turn.completed":
                    continue
            else:
                if event.get("type") != "result":
                    continue
                failed = bool(event.get("is_error"))
                usage = {
                    key: event[key] for key in ("total_cost_usd", "num_turns")
                    if type(event.get(key)) in (int, float)
                }
            tokens = event.get("usage")
            if isinstance(tokens, dict):
                for key, count in tokens.items():
                    if type(count) is int and count >= 0:
                        usage[key] = usage.get(key, 0) + count
    return usage, failed


def run_agent(
    *,
    backend: str,
    model: str,
    variant: str,
    run_id: str,
    mcp_url: str,
    token: str,
    workspace: Path,
    log_dir: Path,
    timeout: float,
    image: str | None = None,
    executable: str | None = None,
    examples: Path | None = None,
) -> AgentResult:
    """Run one session. Supply a local Podman image for workspace isolation."""
    backend = {"claude-code": "claude"}.get(backend, backend)
    if backend not in MODEL_ENV:
        raise ValueError("Agent backend must be claude or codex")
    if variant not in {"B", "D", "F"}:
        raise ValueError("Agent variant must be B, D or F")
    if not (math.isfinite(timeout) and timeout > 0):
        raise ValueError("Agent timeout must be a positive number of seconds")
    if not model or not run_id or not token or any(c.isspace() for c in token):
        raise ValueError("Model, run ID and a bearer token are required")
    if (variant == "F") != (examples is not None):
        raise ValueError("Provide an examples directory for variant F only")
    workspace = Path(workspace).resolve()
    log_dir = Path(log_dir).resolve()
    if not workspace.is_dir():
        raise ValueError("Agent workspace must be a directory")
    if log_dir.is_relative_to(workspace):
        raise ValueError("Agent logs must be outside the workspace")
    if examples is not None:
        examples = Path(examples).resolve()
        if (not examples.is_dir() or examples.is_relative_to(workspace)
                or workspace.is_relative_to(examples) or log_dir.is_relative_to(examples)):
            raise ValueError("Agent examples must be in a separate directory")
    if image is not None and (not image.strip() or image.startswith("-")):
        raise ValueError("Supply a valid local Podman image name")

    container = image is not None
    agent_url, use_loopback = _agent_url(mcp_url, container)
    examples_dir = None
    if examples is not None:
        examples_dir = CONTAINER_EXAMPLES if container else str(examples)
    environment = _environment(backend, token, container)
    command = _command(backend, executable or backend, model, variant, agent_url,
                       timeout, container, examples_dir)
    container_name = None
    if image is not None:
        if ":" in str(workspace) or (examples is not None and ":" in str(examples)):
            raise ValueError("Podman bind mount paths must not contain a colon")
        if use_loopback:
            for variable in ("NO_PROXY", "no_proxy"):
                addresses = environment.get(variable, "")
                environment[variable] = ",".join(filter(None, (addresses, "host.containers.internal")))
        container_name = f"kernel-agent-{uuid4().hex}"
        command = _podman_command(command, image, container_name, workspace, examples,
                                  environment, use_loopback, timeout)
        # The Podman client uses host identity and storage. Only --env names enter the container.
        environment = {**os.environ, **environment, "HOME": str(Path.home())}

    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    stdout_path = log_dir / "agent.stdout.jsonl"
    stderr_path = log_dir / "agent.stderr.log"
    credential_names = (MCP_TOKEN_ENV,) + tuple(
        name for name in MODEL_ENV[backend] if name.endswith(("_KEY", "_TOKEN"))
    )
    secrets = tuple(sorted(
        {environment[name].encode() for name in credential_names if environment.get(name)},
        key=len, reverse=True,
    ))
    prompt = _prompt(run_id, variant, examples_dir).encode()
    start = time.monotonic()
    status = "launch_error"
    returncode = None
    error = None
    with tempfile.TemporaryFile(dir=log_dir) as stdout, tempfile.TemporaryFile(dir=log_dir) as stderr:
        try:
            try:
                process = subprocess.Popen(
                    command, cwd=workspace, env=environment, start_new_session=True,
                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                )
            except OSError as exc:
                error = exc.strerror
                stderr.write(f"Could not start {command[0]}: {error}\n".encode())
            else:
                try:
                    process.communicate(prompt, timeout=timeout)
                    status = "completed" if process.returncode == 0 else "failed"
                    if container and process.returncode in (125, 126, 127):
                        status = "launch_error"
                except BaseException as exc:
                    _kill_group(process)
                    if container_name is not None:
                        _remove_container(container_name, environment, stderr)
                    if not isinstance(exc, subprocess.TimeoutExpired):
                        raise
                    status = "timeout"
                    error = f"Agent exceeded its {timeout:g} second deadline"
                returncode = process.returncode
        finally:
            duration = time.monotonic() - start
            _copy_log(stdout, stdout_path, secrets)
            _copy_log(stderr, stderr_path, secrets)

    usage, reported_failure = _read_usage(backend, stdout_path)
    if status == "completed" and reported_failure:
        status = "failed"
    return AgentResult(status, returncode, duration, usage, stdout_path, stderr_path, error)
