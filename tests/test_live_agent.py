import json
import os
import select
import signal
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runners.agent import run_agent


MONITOR = Path(__file__).resolve().parents[1] / "scripts/watch_run.sh"


class LiveAgentTest(unittest.TestCase):
    def test_monitor_receives_redacted_event_before_exit(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as pool:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (root / "run.json").write_text('{"backend":"mock"}')
            exe = root / "fake-codex"
            exe.write_text("""#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
sys.stdin.read()
event = {"type":"item.completed", "item":{"type":"agent_message", "text": os.environ["KERNEL_AGENT_MCP_TOKEN"]}}
prefix, suffix = json.dumps(event).split("private-")
sys.stdout.write(prefix + "private-")
sys.stdout.flush()
time.sleep(.5)
print(suffix, flush=True)
while not Path("release").exists():
    time.sleep(.05)
print(json.dumps({"type":"turn.completed", "usage":{"output_tokens":3}}), flush=True)
""")
            exe.chmod(0o700)
            log = root / "agent/agent.stdout.jsonl"
            result = pool.submit(run_agent, backend="codex", model="test", variant="B", run_id="test",
                                 mcp_url="http://localhost:8000/mcp", token="private-credential",
                                 workspace=workspace, log_dir=log.parent, timeout=15, executable=str(exe))
            # Start following before the Agent is allowed to exit.
            follower = subprocess.Popen(["sh", str(MONITOR), "--agent", str(root)],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        start_new_session=True)
            try:
                self.assertTrue(select.select([follower.stdout], [], [], 8)[0])
                self.assertEqual(follower.stdout.readline(), b"Codex | [redacted]\n")
                self.assertFalse(result.done())
                self.assertEqual(log.stat().st_mode & 0o777, 0o600)
            finally:
                (workspace / "release").touch()
                os.killpg(follower.pid, signal.SIGTERM)
                follower.wait(timeout=3)
                follower.stdout.close()
            outcome = result.result(timeout=5)
            self.assertEqual((outcome.status, outcome.usage), ("completed", {"output_tokens": 3}))
            (log.parent / "result.json").write_text(json.dumps(outcome.to_dict()))
            replay = subprocess.check_output(["sh", str(MONITOR), "--agent", str(root)])
            self.assertEqual(replay, b"Codex | [redacted]\nSession | turn.completed\n")
