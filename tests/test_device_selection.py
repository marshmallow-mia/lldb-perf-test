"""Tests for device selection, --list-devices parsing, and single-log features."""
from __future__ import annotations

import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from llama_bench.config import BenchConfig, configs_from_args
from llama_bench.gpu import list_devices_from_server
from llama_bench.runner import _build_server_cmd, start_server


# ---------------------------------------------------------------------------
# list_devices_from_server — parsing
# ---------------------------------------------------------------------------

class TestListDevicesFromServer:
    """Unit tests for list_devices_from_server()."""

    def _make_run_result(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        r = MagicMock()
        r.stdout = stdout
        r.stderr = stderr
        r.returncode = returncode
        return r

    def test_parses_two_vulkan_devices(self):
        output = textwrap.dedent("""\
            Available devices:
              Vulkan0: NVIDIA GeForce RTX 2080 Ti (driver 560.35.03) | VRAM: 11264 MiB
              Vulkan1: AMD Radeon RX 7800 XT (driver 2.0.303) | VRAM: 16384 MiB
        """)
        result = self._make_run_result(stdout=output)
        with patch("subprocess.run", return_value=result):
            devices = list_devices_from_server("/fake/llama-server")
        assert devices == {0: "Vulkan0", 1: "Vulkan1"}

    def test_parses_single_device(self):
        output = textwrap.dedent("""\
            Available devices:
              Vulkan0: NVIDIA GeForce RTX 4090 (driver 550.54) | VRAM: 24576 MiB
        """)
        result = self._make_run_result(stdout=output)
        with patch("subprocess.run", return_value=result):
            devices = list_devices_from_server("/fake/llama-server")
        assert devices == {0: "Vulkan0"}

    def test_returns_empty_on_no_available_devices_section(self):
        output = "llama-server: invalid option --list-devices\n"
        result = self._make_run_result(stdout=output)
        with patch("subprocess.run", return_value=result):
            devices = list_devices_from_server("/fake/llama-server")
        assert devices == {}

    def test_returns_empty_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            devices = list_devices_from_server("/nonexistent/llama-server")
        assert devices == {}

    def test_returns_empty_on_timeout(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=15)):
            devices = list_devices_from_server("/fake/llama-server")
        assert devices == {}

    def test_parses_from_stderr(self):
        """Device listing may appear in stderr output."""
        output = textwrap.dedent("""\
            Available devices:
              Vulkan0: Intel Arc A770
        """)
        result = self._make_run_result(stderr=output)
        with patch("subprocess.run", return_value=result):
            devices = list_devices_from_server("/fake/llama-server")
        assert devices == {0: "Vulkan0"}

    def test_ignores_lines_before_section(self):
        output = textwrap.dedent("""\
            Log: loading model...
            Available devices:
              Vulkan0: NVIDIA RTX 3080
        """)
        result = self._make_run_result(stdout=output)
        with patch("subprocess.run", return_value=result):
            devices = list_devices_from_server("/fake/llama-server")
        assert devices == {0: "Vulkan0"}


# ---------------------------------------------------------------------------
# _build_server_cmd — --device flag
# ---------------------------------------------------------------------------

class TestBuildServerCmdDevice:
    """Tests that _build_server_cmd includes --device correctly."""

    def _base_cfg(self, **kwargs) -> BenchConfig:
        return BenchConfig(
            model_path="/models/test.gguf",
            server_binary="/usr/bin/llama-server",
            host="127.0.0.1",
            port=5001,
            use_sudo=False,
            **kwargs,
        )

    def test_no_device_flag_when_device_is_none(self):
        cfg = self._base_cfg(device=None)
        cmd = _build_server_cmd(cfg)
        assert "--device" not in cmd

    def test_device_none_for_cpu_engine(self):
        cfg = self._base_cfg(device="none", engine="cpu")
        cmd = _build_server_cmd(cfg)
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "none"

    def test_device_vulkan0_vulkan1(self):
        cfg = self._base_cfg(device="Vulkan0,Vulkan1", engine="vulkan")
        cmd = _build_server_cmd(cfg)
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "Vulkan0,Vulkan1"

    def test_device_single_vulkan(self):
        cfg = self._base_cfg(device="Vulkan0", engine="vulkan")
        cmd = _build_server_cmd(cfg)
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "Vulkan0"

    def test_vulkan_no_vk_devices_no_device_flag(self):
        """--engine vulkan with no --vk-devices: no --device flag."""
        cfg = self._base_cfg(device=None, engine="vulkan")
        cmd = _build_server_cmd(cfg)
        assert "--device" not in cmd


# ---------------------------------------------------------------------------
# BenchConfig — device field
# ---------------------------------------------------------------------------

class TestBenchConfigDevice:
    def test_default_device_is_none(self):
        cfg = BenchConfig(model_path="/tmp/model.gguf")
        assert cfg.device is None

    def test_device_set_to_none_string(self):
        cfg = BenchConfig(model_path="/tmp/model.gguf", device="none")
        assert cfg.device == "none"

    def test_device_set_to_vulkan(self):
        cfg = BenchConfig(model_path="/tmp/model.gguf", device="Vulkan0")
        assert cfg.device == "Vulkan0"

    def test_configs_from_args_device_none_default(self):
        cfg = configs_from_args(model="/tmp/model.gguf")
        assert cfg.device is None

    def test_configs_from_args_device_explicit(self):
        cfg = configs_from_args(model="/tmp/model.gguf", device="Vulkan0,Vulkan1")
        assert cfg.device == "Vulkan0,Vulkan1"

    def test_configs_from_args_cpu_device(self):
        cfg = configs_from_args(model="/tmp/model.gguf", engine="cpu", device="none")
        assert cfg.device == "none"
        assert cfg.engine == "cpu"


# ---------------------------------------------------------------------------
# start_server — single log file with separator headers
# ---------------------------------------------------------------------------

class TestStartServerSingleLog:
    """Tests that start_server appends separator headers when fixed paths are given."""

    def test_single_log_creates_separator(self, tmp_path):
        """When stdout_path/stderr_path provided, a separator header is written."""
        stdout_log = str(tmp_path / "server_stdout.log")
        stderr_log = str(tmp_path / "server_stderr.log")

        cfg = BenchConfig(
            model_path="/tmp/model.gguf",
            server_binary="/fake/llama-server",
            use_sudo=False,
        )

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("subprocess.Popen", return_value=mock_proc):
            handle = start_server(
                cfg,
                artifacts_dir=str(tmp_path),
                stdout_path=stdout_log,
                stderr_path=stderr_log,
                attempt_header="ctx=49152 ngl=45 batch=1536 engine=vulkan",
            )

        # Separator must be in the log files
        with open(stdout_log) as fh:
            stdout_content = fh.read()
        with open(stderr_log) as fh:
            stderr_content = fh.read()
        assert "ctx=49152 ngl=45 batch=1536 engine=vulkan" in stdout_content
        assert "ctx=49152 ngl=45 batch=1536 engine=vulkan" in stderr_content
        # Handle must reference the provided paths
        assert handle.stdout_path == stdout_log
        assert handle.stderr_path == stderr_log

    def test_per_attempt_log_created_when_no_paths(self, tmp_path):
        """When no fixed paths, timestamp-named per-attempt files are created."""
        cfg = BenchConfig(
            model_path="/tmp/model.gguf",
            server_binary="/fake/llama-server",
            use_sudo=False,
        )

        mock_proc = MagicMock()
        mock_proc.pid = 99
        with patch("subprocess.Popen", return_value=mock_proc):
            handle = start_server(cfg, artifacts_dir=str(tmp_path))

        assert "server_stdout_" in handle.stdout_path
        assert "server_stderr_" in handle.stderr_path
        assert os.path.exists(handle.stdout_path)
        assert os.path.exists(handle.stderr_path)

    def test_multiple_attempts_append_to_same_log(self, tmp_path):
        """Two calls with the same paths append both separators."""
        stdout_log = str(tmp_path / "server_stdout.log")
        stderr_log = str(tmp_path / "server_stderr.log")

        cfg = BenchConfig(
            model_path="/tmp/model.gguf",
            server_binary="/fake/llama-server",
            use_sudo=False,
        )
        mock_proc = MagicMock()
        mock_proc.pid = 1

        with patch("subprocess.Popen", return_value=mock_proc):
            start_server(cfg, artifacts_dir=str(tmp_path),
                         stdout_path=stdout_log, stderr_path=stderr_log,
                         attempt_header="attempt 1")
            start_server(cfg, artifacts_dir=str(tmp_path),
                         stdout_path=stdout_log, stderr_path=stderr_log,
                         attempt_header="attempt 2")

        with open(stdout_log) as fh:
            content = fh.read()
        assert "attempt 1" in content
        assert "attempt 2" in content
