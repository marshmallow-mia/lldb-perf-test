"""Server lifecycle management and benchmark runner for llama-bench."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from llama_bench import EXPECTED_LLAMA_SERVER_VERSION
from llama_bench.config import BenchConfig
from llama_bench.gpu import build_env
from llama_bench.metrics import (
    ClientMetrics,
    RunMetrics,
    classify_failure,
    metrics_to_dict,
    parse_server_log,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ServerHandle
# ---------------------------------------------------------------------------

@dataclass
class ServerHandle:
    process: subprocess.Popen
    stdout_path: str
    stderr_path: str
    pid: int


# ---------------------------------------------------------------------------
# start_server
# ---------------------------------------------------------------------------

def _build_server_cmd(cfg: BenchConfig) -> list[str]:
    """Build the command-line list for launching llama-server."""
    cmd: list[str] = []
    if cfg.use_sudo:
        cmd.append("sudo")

    cmd += [
        cfg.server_binary,
        "-m", cfg.model_path,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "-np", str(cfg.np),
        "-c", str(cfg.ctx),
        "-n", "4096",
        "--batch-size", str(cfg.batch_size),
        "--ubatch-size", str(cfg.ubatch_size),
        "--n-gpu-layers", str(cfg.n_gpu_layers),
        "--cache-type-k", cfg.cache_type_k,
        "--cache-type-v", cfg.cache_type_v,
        "--cache-reuse", str(cfg.cache_reuse),
        "--threads", str(cfg.threads),
        "--threads-batch", str(cfg.threads_batch),
        "--split-mode", cfg.split_mode,
    ]

    if cfg.flash_attn:
        cmd += ["--flash-attn", "on"]
    else:
        cmd += ["--flash-attn", "off"]

    if cfg.kv_unified:
        cmd.append("--kv-unified")

    if cfg.cont_batching:
        cmd.append("--cont-batching")

    return cmd


def start_server(cfg: BenchConfig, artifacts_dir: str = "results") -> ServerHandle:
    """Launch llama-server as a subprocess, redirecting I/O to log files.

    Returns a :class:`ServerHandle` describing the running process.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    stdout_path = os.path.join(artifacts_dir, f"server_stdout_{timestamp}.log")
    stderr_path = os.path.join(artifacts_dir, f"server_stderr_{timestamp}.log")

    cmd = _build_server_cmd(cfg)
    env = build_env(cfg.vk_visible_devices)

    logger.info("Server binary: %s", cfg.server_binary)
    logger.info("Server command: %s", " ".join(cmd))
    logger.info("Server stdout log: %s", stdout_path)
    logger.info("Server stderr log: %s", stderr_path)
    logger.debug("Starting server: %s", " ".join(cmd))

    stdout_fh = open(stdout_path, "w")  # noqa: WPS515 – kept open for the lifetime of the server process
    stderr_fh = open(stderr_path, "w")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_fh,
            stderr=stderr_fh,
            env=env,
            close_fds=True,
        )
    except Exception:
        stdout_fh.close()
        stderr_fh.close()
        raise

    logger.info("Server started with PID %d", proc.pid)
    return ServerHandle(
        process=proc,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        pid=proc.pid,
    )


# ---------------------------------------------------------------------------
# wait_for_server_ready
# ---------------------------------------------------------------------------

def wait_for_server_ready(host: str, port: int, timeout: float = 60.0) -> bool:
    """Poll ``GET /health`` until 200 OK or *timeout* seconds elapse."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = requests.get(url, timeout=2.0)
            if resp.status_code == 200:
                logger.info("Server ready after %d health-check attempt(s)", attempt)
                return True
            logger.debug("Health check attempt %d: HTTP %d", attempt, resp.status_code)
        except requests.RequestException as exc:
            logger.debug("Health check attempt %d: %s", attempt, exc)
        time.sleep(1.0)

    logger.warning("Server did not become ready within %.0fs (%d attempts)", timeout, attempt)
    return False


# ---------------------------------------------------------------------------
# stop_server
# ---------------------------------------------------------------------------

def stop_server(handle: ServerHandle) -> None:
    """Terminate the server process gracefully, force-killing if needed."""
    proc = handle.process
    if proc.poll() is not None:
        return  # already exited

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server process %d did not die after SIGKILL", handle.pid)


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def get_server_version(binary: str, use_sudo: bool = False) -> Optional[str]:
    """Run ``llama-server --version`` and return the numeric build version, or None.

    The binary is always invoked directly (never with sudo) so that the version
    check does not fail with ``sudo: ./llama-server: command not found`` when
    sudo has a restricted PATH.

    The version is extracted from a line of the form::

        version: 8133 (2b6dfe824)

    and returned as the numeric string ``"8133"``.  If the command exits with a
    non-zero code the output is not parsed and ``None`` is returned.

    The *use_sudo* parameter is accepted for backward compatibility but is
    ignored; version detection never uses sudo.
    """
    import re
    cmd = [binary, "--version"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.debug(
                "get_server_version: non-zero exit %d for %r; stderr=%r",
                result.returncode, binary, result.stderr[:200],
            )
            return None
        output = result.stdout + result.stderr
        m = re.search(r"^\s*version:\s*(\d+)\b", output, re.MULTILINE)
        if m:
            return m.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as exc:
        logger.debug("get_server_version: failed to run %r: %s", binary, exc)
    return None


def check_version_mismatch(binary: str, use_sudo: bool = False) -> Optional[str]:
    """Return a warning message if the binary version differs from the expected version."""
    version = get_server_version(binary)
    if version is None:
        return f"Could not determine llama-server version (binary: {binary!r})."
    if version != EXPECTED_LLAMA_SERVER_VERSION:
        return (
            f"llama-server version mismatch: expected {EXPECTED_LLAMA_SERVER_VERSION!r}, "
            f"got {version!r}. Results may differ from baseline."
        )
    return None


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

def _config_hash(cfg: BenchConfig) -> str:
    import dataclasses
    cfg_dict = dataclasses.asdict(cfg)
    return hashlib.md5(
        json.dumps(cfg_dict, sort_keys=True).encode()
    ).hexdigest()[:8]


def _make_run_id(cfg: BenchConfig) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    ch = _config_hash(cfg)
    return f"run_{ts}_{ch}"


class BenchmarkRunner:
    """Orchestrate start/stop of llama-server and execution of prompts."""

    def __init__(self, cfg: BenchConfig, artifacts_dir: str = "results",
                 log_file: Optional[str] = None) -> None:
        self.cfg = cfg
        self.artifacts_dir = artifacts_dir
        self.log_file = log_file
        os.makedirs(artifacts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        prompt_sequence: list[dict],
        n_followups: int = 4,
    ) -> list[RunMetrics]:
        """Start server, run *prompt_sequence*, stop server, return metrics."""
        handle: Optional[ServerHandle] = None
        results: list[RunMetrics] = []

        try:
            logger.info("Starting server for run (config_hash will be assigned per-prompt)")
            handle = start_server(self.cfg, self.artifacts_dir)
            logger.info("Waiting for server to become ready (timeout=90s)")
            ready = wait_for_server_ready(self.cfg.host, self.cfg.port, timeout=90.0)

            if not ready:
                run_id = _make_run_id(self.cfg)
                logger.warning("Server not ready; recording failure run_id=%s", run_id)
                results.append(
                    RunMetrics(
                        success=False,
                        failure_reason="server_crash",
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        config_hash=_config_hash(self.cfg),
                        log_file=self.log_file,
                    )
                )
                return results

            for idx, item in enumerate(prompt_sequence):
                messages = item.get("messages", [])
                run_id = _make_run_id(self.cfg)
                ts = datetime.now(timezone.utc).isoformat()
                ch = _config_hash(self.cfg)
                logger.info("Benchmark request %d/%d run_id=%s", idx + 1, len(prompt_sequence), run_id)

                try:
                    # Run streaming request
                    logger.debug("Starting streaming request (prompt messages=%d)", len(messages))
                    client_metrics = self._run_streaming(messages)
                    logger.info(
                        "Streaming done: ttft=%.1fms e2e=%.1fms tok/s=%.1f tokens=%d",
                        client_metrics.ttft_ms,
                        client_metrics.end_to_end_latency_ms,
                        client_metrics.streaming_tok_per_s,
                        client_metrics.total_tokens,
                    )
                    server_metrics = self._read_server_metrics(handle)
                    if server_metrics is None:
                        logger.debug("No server metrics parsed from log (run_id=%s)", run_id)
                    else:
                        logger.debug(
                            "Server metrics: prompt_tok/s=%.1f decode_tok/s=%.1f",
                            server_metrics.prompt_tok_per_s,
                            server_metrics.decode_tok_per_s,
                        )
                    results.append(
                        RunMetrics(
                            server=server_metrics,
                            client=client_metrics,
                            success=True,
                            run_id=run_id,
                            timestamp=ts,
                            config_hash=ch,
                            log_file=self.log_file,
                        )
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    reason = classify_failure(0, "", exc)
                    logger.warning("Request failed run_id=%s reason=%s exc=%s", run_id, reason, exc)
                    results.append(
                        RunMetrics(
                            success=False,
                            failure_reason=reason,
                            run_id=run_id,
                            timestamp=ts,
                            config_hash=ch,
                            log_file=self.log_file,
                        )
                    )
        finally:
            if handle is not None:
                logger.info("Stopping server PID %d", handle.pid)
                stop_server(handle)
                logger.info("Server stopped; stdout=%s stderr=%s", handle.stdout_path, handle.stderr_path)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _endpoint(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}/v1/chat/completions"

    def _run_streaming(
        self,
        messages: list[dict],
        timeout: float = 120.0,
    ) -> ClientMetrics:
        """POST to /v1/chat/completions with stream=True and measure TTFT + tok/s."""
        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": 4096,
        }
        start = time.monotonic()
        ttft_ms = 0.0
        total_tokens = 0
        first_token = True

        with requests.post(
            self._endpoint(),
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token:
                            ttft_ms = (time.monotonic() - start) * 1000
                            first_token = False
                        total_tokens += len(content.split())  # rough token count

        end_to_end_ms = (time.monotonic() - start) * 1000
        duration_s = end_to_end_ms / 1000.0
        tok_per_s = total_tokens / duration_s if duration_s > 0 else 0.0

        return ClientMetrics(
            ttft_ms=ttft_ms,
            end_to_end_latency_ms=end_to_end_ms,
            streaming_tok_per_s=tok_per_s,
            total_tokens=total_tokens,
            is_streaming=True,
        )

    def _run_nonstreaming(
        self,
        messages: list[dict],
        timeout: float = 120.0,
    ) -> ClientMetrics:
        """POST to /v1/chat/completions with stream=False and measure e2e latency."""
        payload = {
            "messages": messages,
            "stream": False,
            "max_tokens": 4096,
        }
        start = time.monotonic()
        resp = requests.post(self._endpoint(), json=payload, timeout=timeout)
        resp.raise_for_status()
        end_to_end_ms = (time.monotonic() - start) * 1000

        data = resp.json()
        total_tokens = 0
        try:
            usage = data.get("usage", {})
            total_tokens = usage.get("completion_tokens", 0)
        except (AttributeError, KeyError):
            pass

        return ClientMetrics(
            ttft_ms=0.0,  # not measurable without streaming
            end_to_end_latency_ms=end_to_end_ms,
            streaming_tok_per_s=0.0,
            total_tokens=total_tokens,
            is_streaming=False,
        )

    def _read_server_metrics(self, handle: ServerHandle) -> Optional[object]:
        """Attempt to parse server metrics from the stderr log."""
        try:
            with open(handle.stderr_path, "r", encoding="utf-8", errors="replace") as fh:
                log_text = fh.read()
            metrics = parse_server_log(log_text)
            if metrics is None:
                logger.debug("Server log parse: no metrics found in %s", handle.stderr_path)
            else:
                logger.debug(
                    "Server log parse: found metrics (prompt_ms=%.1f eval_ms=%.1f) in %s",
                    metrics.prompt_eval_time_ms,
                    metrics.eval_time_ms,
                    handle.stderr_path,
                )
            return metrics
        except OSError as exc:
            logger.debug("Server log parse: could not read %s: %s", handle.stderr_path, exc)
            return None
