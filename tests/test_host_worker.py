"""Regression tests for host launch settings and the existing Podman path."""

import argparse
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness.__main__ import _configure, _create, _evaluator
from harness.evaluator import RunError
from harness.host_worker import LAUNCH_ENV, HostWorkerLaunch, _gpu_device
from harness.results import RunStore


GPU_UUID = "GPU-12345678-1234-1234-1234-123456789012"


class HostWorkerTest(unittest.TestCase):
    def test_gpu_uuid_resolves_to_device_minor(self):
        with TemporaryDirectory() as directory:
            devices = Path(directory)
            first = devices / "0000:01:00.0"
            first.mkdir()
            (first / "information").write_text("GPU UUID: GPU-other\nDevice Minor: 1\n")
            second = devices / "0000:80:00.0"
            second.mkdir()
            (second / "information").write_text(f"GPU UUID: {GPU_UUID}\nDevice Minor: 5\n")
            with patch("harness.host_worker.NVIDIA_GPUS", devices):
                self.assertEqual(_gpu_device(GPU_UUID), Path("/dev/nvidia5"))
                with self.assertRaises(FileNotFoundError):
                    _gpu_device("GPU-missing")

    def test_sandbox_mounts_selected_device_and_worker_files(self):
        settings = {"python": "/opt/worker/bin/python3", "cuda_home": "/usr/local/cuda",
                    "cc": None, "cxx": None}
        repo = Path("/home/test/kernel-agent")
        launch = HostWorkerLaunch(settings, GPU_UUID, repo)
        with (patch("harness.host_worker.shutil.which", return_value="/usr/bin/bwrap"),
              patch("harness.host_worker._gpu_device", return_value=Path("/dev/nvidia5")),
              patch.object(Path, "is_file", return_value=True),
              patch.object(Path, "is_char_device", return_value=True),
              patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"})):
            command = launch.command(("-m", "harness.gpu_worker"), ("ncu",))

        args = command.index("--")
        self.assertEqual(command[args + 1:], ["ncu", settings["python"], "-m", "harness.gpu_worker"])
        mounts = {command[index + 1] for index, item in enumerate(command) if item == "--dev-bind"}
        self.assertEqual(mounts, {"/dev/nvidia5", "/dev/nvidiactl", "/dev/nvidia-uvm"})
        env = {command[index + 1]: command[index + 2] for index, item in enumerate(command) if item == "--setenv"}
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPU_UUID)
        self.assertNotIn("OPENAI_API_KEY", env | LAUNCH_ENV)
        mounted = {command[index + 1] for index, item in enumerate(command) if item == "--ro-bind"}
        self.assertIn(str(repo / "harness/gpu_worker.py"), mounted)
        self.assertNotIn(str(repo), mounted)

    def test_new_host_run_and_legacy_podman_run_use_saved_configuration(self):
        parser = argparse.ArgumentParser()
        _configure(parser)
        args = parser.parse_args(["--backend", "gpu", "--gpu-worker", "host", "--gpu-uuid", GPU_UUID,
                                  "--gpu-worker-python", sys.executable, "--evaluation-budget", "1"])
        with (TemporaryDirectory() as directory,
              patch("harness.__main__.GPUWorker") as worker,
              patch.dict(os.environ, {"CUDA_HOME": "/opt/cuda-test", "CC": "", "CXX": ""})):
            store = RunStore(directory)
            run_id, _, _ = _create(store, args)
            config = store.config(run_id)
            self.assertEqual(config["gpu_worker"], "host")
            self.assertIsNone(config["gpu_image"])
            self.assertEqual(config["host_worker"]["python"], sys.executable)
            self.assertEqual(config["host_worker"]["cuda_home"], "/opt/cuda-test")

            evaluator = _evaluator(store, run_id)
            worker.assert_called_with(None, GPU_UUID, args.worker_timeout, host=config["host_worker"])
            evaluator.gpu_worker.run.side_effect = [
                {"status": "infrastructure_error", "stage": "worker", "cases": []},
                {"status": "wrong_answer", "stage": "correctness", "cases": []},
            ]
            self.assertEqual(evaluator.evaluate(run_id, "void run() {}")["status"], "infrastructure_error")
            self.assertEqual(evaluator.evaluate(run_id, "void run() {}")["status"], "wrong_answer")
            with self.assertRaisesRegex(RunError, "budget exhausted"):
                evaluator.evaluate(run_id, "void run() {}")

            legacy, _ = store.create({"backend": "gpu", "gpu_image": "worker:local",
                                      "gpu_uuid": GPU_UUID, "worker_timeout": 42})
            _evaluator(store, legacy)
            worker.assert_called_with("worker:local", GPU_UUID, 42)
