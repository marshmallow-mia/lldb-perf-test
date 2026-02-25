"""Tests for llama_bench.runner version parsing and server path resolution."""
from __future__ import annotations

import os
import stat
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_bench.runner import get_server_version, _build_server_cmd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (_FIXTURES_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_version_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


# ---------------------------------------------------------------------------
# get_server_version — version parsing
# ---------------------------------------------------------------------------

class TestGetServerVersionParsing:
    """Unit tests for version string parsing in get_server_version."""

    def test_parses_version_with_hash_from_fixture(self):
        """Parses numeric version '8133' from multiline fixture output."""
        output = _fixture("llama_server_version.txt")
        result = _make_version_result(stdout=output)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version == "8133"

    def test_parses_version_stdout_only(self):
        stdout = "version: 8133 (2b6dfe824)\n"
        result = _make_version_result(stdout=stdout)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version == "8133"

    def test_parses_version_stderr_only(self):
        stderr = "version: 4200 (abc1234)\n"
        result = _make_version_result(stderr=stderr)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version == "4200"

    def test_parses_version_without_hash(self):
        """Version line without a hash component is returned as plain number."""
        stdout = "version: 8133\n"
        result = _make_version_result(stdout=stdout)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version == "8133"

    def test_returns_none_on_nonzero_exit(self):
        """Does not parse error output when command exits with non-zero code."""
        stderr = "sudo: ./llama-server: command not found\n"
        result = _make_version_result(stderr=stderr, returncode=1)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version is None

    def test_returns_none_on_nonzero_exit_even_if_version_line_present(self):
        """A version line in error output is not parsed on non-zero exit."""
        stderr = "version: 8133 (2b6dfe824)\nsudo: command not found\n"
        result = _make_version_result(stderr=stderr, returncode=127)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version is None

    def test_returns_none_when_no_version_line(self):
        stdout = "usage: llama-server [options]\n"
        result = _make_version_result(stdout=stdout)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version is None

    def test_returns_none_on_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            version = get_server_version("/nonexistent/llama-server")
        assert version is None

    def test_returns_none_on_timeout(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=10)):
            version = get_server_version("/fake/llama-server")
        assert version is None

    def test_version_line_with_extra_whitespace(self):
        stdout = "  version:   9000  (deadbeef)  \n"
        result = _make_version_result(stdout=stdout)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version == "9000"

    def test_invoked_without_sudo(self):
        """get_server_version must never prepend 'sudo' to the command."""
        result = _make_version_result(stdout="version: 1 (a)\n")
        with patch("subprocess.run", return_value=result) as mock_run:
            get_server_version("/my/llama-server", use_sudo=True)  # use_sudo ignored
        cmd = mock_run.call_args[0][0]
        assert cmd[0] != "sudo", "sudo must NOT be prepended to version check"
        assert cmd[0] == "/my/llama-server"

    def test_multiline_backend_logs_before_version(self):
        """Finds 'version:' even when backend log lines precede it."""
        stdout = textwrap.dedent("""\
            ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
            ggml_cuda_init: found 1 CUDA devices:
            build: 8133 (2b6dfe824)
            version: 8133 (2b6dfe824)
            system info: n_threads = 8
        """)
        result = _make_version_result(stdout=stdout)
        with patch("subprocess.run", return_value=result):
            version = get_server_version("/fake/llama-server")
        assert version == "8133"


# ---------------------------------------------------------------------------
# _resolve_server_path — path resolution
# ---------------------------------------------------------------------------

class TestResolveServerPath:
    """Unit tests for server path resolution in cli._resolve_server_path."""

    def _get_fn(self):
        from llama_bench.cli import _resolve_server_path
        return _resolve_server_path

    def test_absolute_path_returned_unchanged(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        fn = self._get_fn()
        assert fn(str(binary)) == str(binary)

    def test_tilde_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        fn = self._get_fn()
        assert fn("~/llama-server") == str(binary)

    def test_relative_path_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        fn = self._get_fn()
        result = fn("./llama-server")
        assert os.path.isabs(result)
        assert result == str(binary)

    def test_bare_name_resolved_via_which(self, tmp_path, monkeypatch):
        import shutil
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        monkeypatch.chdir(tmp_path)
        # Patch shutil.which to return the binary
        with patch("llama_bench.cli.shutil.which", return_value=str(binary)):
            fn = self._get_fn()
            result = fn("llama-server")
        assert result == str(binary)

    def test_missing_binary_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fn = self._get_fn()
        with pytest.raises(SystemExit):
            fn("./llama-server-does-not-exist")

    def test_non_executable_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"")
        # Do NOT set executable bit
        binary.chmod(0o644)
        fn = self._get_fn()
        with pytest.raises(SystemExit):
            fn(str(binary))


# ---------------------------------------------------------------------------
# wait_for_server_ready — liveness checks
# ---------------------------------------------------------------------------

class TestWaitForServerReady:
    """Unit tests for wait_for_server_ready with process liveness checking."""

    def _fn(self):
        from llama_bench.runner import wait_for_server_ready
        return wait_for_server_ready

    def test_returns_none_when_health_check_succeeds(self):
        """Returns None (success) when /health responds 200 immediately."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            result = self._fn()("localhost", 5001, timeout=5.0)
        assert result is None

    def test_returns_server_startup_timeout_on_timeout(self):
        """Returns 'server_startup_timeout' if server never responds within timeout."""
        import requests as req
        with patch("requests.get", side_effect=req.ConnectionError("refused")):
            with patch("time.sleep"):  # skip actual sleeps
                result = self._fn()("localhost", 5001, timeout=0.01)
        assert result == "server_startup_timeout"

    def test_returns_server_exited_when_process_exits_early(self):
        """Returns 'server_exited' immediately if proc.poll() returns non-None."""
        import requests as req
        proc = MagicMock()
        proc.poll.return_value = 1  # process already exited
        proc.returncode = 1
        with patch("requests.get", side_effect=req.ConnectionError("refused")):
            result = self._fn()("localhost", 5001, timeout=30.0, proc=proc)
        assert result == "server_exited"

    def test_returns_none_when_process_alive_and_health_ok(self):
        """Returns None when proc is alive and /health eventually returns 200."""
        proc = MagicMock()
        proc.poll.return_value = None  # still running

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            result = self._fn()("localhost", 5001, timeout=30.0, proc=proc)
        assert result is None

    def test_no_proc_argument_still_works(self):
        """Backward-compatible: omitting proc returns None on successful health check."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            result = self._fn()("localhost", 5001, timeout=5.0)
        assert result is None


# ---------------------------------------------------------------------------
# _build_server_cmd — max_predict_tokens in -n flag
# ---------------------------------------------------------------------------

class TestBuildServerCmd:
    """Unit tests for _build_server_cmd with max_predict_tokens."""

    def _make_cfg(self, **kwargs):
        from llama_bench.config import BenchConfig
        defaults = dict(model_path="/tmp/m.gguf", server_binary="/bin/srv",
                        host="0.0.0.0", port=5001, use_sudo=False,
                        vk_visible_devices="0", np=1, ctx=4096,
                        n_gpu_layers=10, flash_attn=True, batch_size=512,
                        ubatch_size=128, cache_type_k="q8_0",
                        cache_type_v="q8_0", kv_unified=True,
                        cache_reuse=512, cont_batching=True,
                        threads=4, threads_batch=4, split_mode="none",
                        max_predict_tokens=256)
        defaults.update(kwargs)
        return BenchConfig(**defaults)

    def test_n_flag_uses_max_predict_tokens(self):
        """Server -n flag must equal cfg.max_predict_tokens, not hardcoded 4096."""
        cfg = self._make_cfg(max_predict_tokens=128)
        cmd = _build_server_cmd(cfg)
        assert "-n" in cmd
        n_idx = cmd.index("-n")
        assert cmd[n_idx + 1] == "128"

    def test_n_flag_default_is_512(self):
        cfg = self._make_cfg(max_predict_tokens=512)
        cmd = _build_server_cmd(cfg)
        n_idx = cmd.index("-n")
        assert cmd[n_idx + 1] == "512"

    def test_no_sudo_when_disabled(self):
        cfg = self._make_cfg(use_sudo=False)
        cmd = _build_server_cmd(cfg)
        assert cmd[0] != "sudo"

    def test_sudo_prepended_when_enabled(self):
        cfg = self._make_cfg(use_sudo=True)
        cmd = _build_server_cmd(cfg)
        assert cmd[0] == "sudo"
