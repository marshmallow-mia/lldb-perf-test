"""Metrics data structures and parsing for llama-bench."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServerMetrics:
    prompt_eval_time_ms: float = 0.0
    prompt_eval_count: int = 0
    eval_time_ms: float = 0.0
    eval_count: int = 0
    prompt_tok_per_s: float = 0.0
    decode_tok_per_s: float = 0.0
    total_time_ms: float = 0.0
    has_memory_info: bool = False
    memory_used_mb: Optional[float] = None


@dataclass
class ClientMetrics:
    ttft_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    streaming_tok_per_s: float = 0.0
    total_tokens: int = 0
    is_streaming: bool = False


@dataclass
class RunMetrics:
    server: Optional[ServerMetrics] = None
    client: ClientMetrics = field(default_factory=ClientMetrics)
    success: bool = False
    failure_reason: Optional[str] = None
    run_id: str = ""
    timestamp: str = ""
    config_hash: str = ""
    log_file: Optional[str] = None
    stderr_path: Optional[str] = None
    stdout_path: Optional[str] = None
    server_error_excerpt: Optional[str] = None


# ---------------------------------------------------------------------------
# parse_server_log
# ---------------------------------------------------------------------------

# Text patterns emitted by llama-server / llama.cpp
_RE_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r"(?:.*?([\d.]+)\s*tokens per second)?",
    re.IGNORECASE,
)
_RE_EVAL = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r"(?:.*?([\d.]+)\s*tokens per second)?",
    re.IGNORECASE,
)
_RE_TOTAL = re.compile(
    r"total time\s*=\s*([\d.]+)\s*ms",
    re.IGNORECASE,
)
_RE_MEMORY = re.compile(
    r"ggml_vulkan:\s*([\d.]+)\s*MiB",
    re.IGNORECASE,
)


def parse_server_log(log_text: str) -> Optional[ServerMetrics]:
    """Parse llama-server log text and extract timing/memory metrics.

    Handles both the human-readable log format and the JSON ``timings`` format.
    Returns ``None`` if no recognisable metrics are found.
    """
    metrics = ServerMetrics()
    found_any = False

    # --- JSON timings format ---
    # {"prompt_n":X,"prompt_ms":Y,"predicted_n":Z,"predicted_ms":W}
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "prompt_n" in obj or "predicted_n" in obj:
            prompt_n = obj.get("prompt_n", 0)
            prompt_ms = obj.get("prompt_ms", 0.0)
            predicted_n = obj.get("predicted_n", 0)
            predicted_ms = obj.get("predicted_ms", 0.0)
            metrics.prompt_eval_count = int(prompt_n)
            metrics.prompt_eval_time_ms = float(prompt_ms)
            metrics.eval_count = int(predicted_n)
            metrics.eval_time_ms = float(predicted_ms)
            if prompt_ms > 0:
                metrics.prompt_tok_per_s = prompt_n / (prompt_ms / 1000.0) if prompt_ms else 0.0
            if predicted_ms > 0:
                metrics.decode_tok_per_s = predicted_n / (predicted_ms / 1000.0) if predicted_ms else 0.0
            metrics.total_time_ms = float(obj.get("total_ms", prompt_ms + predicted_ms))
            found_any = True

    # --- Text log format ---
    m = _RE_PROMPT_EVAL.search(log_text)
    if m:
        metrics.prompt_eval_time_ms = float(m.group(1))
        metrics.prompt_eval_count = int(m.group(2))
        if m.group(3):
            metrics.prompt_tok_per_s = float(m.group(3))
        elif metrics.prompt_eval_time_ms > 0:
            metrics.prompt_tok_per_s = metrics.prompt_eval_count / (metrics.prompt_eval_time_ms / 1000.0)
        found_any = True

    m = _RE_EVAL.search(log_text)
    if m:
        metrics.eval_time_ms = float(m.group(1))
        metrics.eval_count = int(m.group(2))
        if m.group(3):
            metrics.decode_tok_per_s = float(m.group(3))
        elif metrics.eval_time_ms > 0:
            metrics.decode_tok_per_s = metrics.eval_count / (metrics.eval_time_ms / 1000.0)
        found_any = True

    m = _RE_TOTAL.search(log_text)
    if m:
        metrics.total_time_ms = float(m.group(1))
        found_any = True

    # Memory info
    m = _RE_MEMORY.search(log_text)
    if m:
        metrics.memory_used_mb = float(m.group(1))
        metrics.has_memory_info = True

    return metrics if found_any else None


# ---------------------------------------------------------------------------
# classify_server_stderr / extract_server_error_excerpt
# ---------------------------------------------------------------------------

# Ordered list of (reason, pattern) pairs; first match wins.
# Priority: out_of_vram before model_not_found (per spec).
_STARTUP_FAILURE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("out_of_vram", re.compile(
        r"ErrorOutOfDeviceMemory"
        r"|Device memory allocation .* failed"
        r"|(?:failed|unable) to allocate Vulkan\d+ buffer"
        r"|failed to fit params to free device memory",
        re.IGNORECASE,
    )),
    ("model_not_found", re.compile(
        r"failed to open GGUF file"
        r"|No such file or directory",
        re.IGNORECASE,
    )),
]


def classify_server_stderr(stderr_text: str) -> str:
    """Classify a server startup failure from stderr text.

    Returns one of ``"model_not_found"``, ``"out_of_vram"``, or
    ``"server_exited"`` (fallback when no specific pattern matches).
    """
    for reason, pattern in _STARTUP_FAILURE_PATTERNS:
        if pattern.search(stderr_text):
            return reason
    return "server_exited"


def extract_server_error_excerpt(stderr_text: str, max_lines: int = 5) -> str:
    """Return a short excerpt of error/fatal lines from server stderr.

    Falls back to the last *max_lines* lines if no error lines are found.
    """
    _ERROR_KEYWORDS = ("error", "failed", "fatal", "abort", "oom", "vulkan")
    lines = stderr_text.splitlines()
    error_lines = [ln for ln in lines if any(kw in ln.lower() for kw in _ERROR_KEYWORDS)]
    excerpt = error_lines[-max_lines:] if error_lines else lines[-max_lines:]
    return "\n".join(excerpt)


# Pattern for llama.cpp fit-failure memory lines:
# "llama_params_fit_impl: projected to use 12345 MiB, 9876 MiB free"
_RE_FIT_MEMORY = re.compile(
    r"llama_params_fit_impl:.*?projected to use\s+([\d.]+)\s*MiB.*?(\d[\d.]*)\s*MiB\s*free",
    re.IGNORECASE,
)


def parse_memory_fit_heuristic(stderr_text: str) -> "Optional[tuple[float, float]]":
    """Parse projected vs free memory from a llama.cpp fit-failure line.

    Looks for lines of the form::

        llama_params_fit_impl: projected to use X MiB ... Y MiB free

    Returns ``(projected_mib, free_mib)`` when a match is found, otherwise ``None``.
    """
    m = _RE_FIT_MEMORY.search(stderr_text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

def classify_failure(
    status_code: int,
    response_text: str,
    exception: Optional[Exception],
) -> str:
    """Map an HTTP response or exception to a short failure-reason string."""
    if exception is not None:
        exc_name = type(exception).__name__.lower()
        if "timeout" in exc_name or isinstance(exception, TimeoutError):
            return "timeout"
        if "connectionerror" in exc_name or "connection" in str(exception).lower():
            return "server_crash"
        return "unknown"

    if status_code == 200:
        return ""  # not a failure

    text_lower = response_text.lower()

    if "out of memory" in text_lower or "oom" in text_lower or status_code == 507:
        return "oom"

    if "context" in text_lower and ("overflow" in text_lower or "full" in text_lower or "exceed" in text_lower):
        return "context_overflow"

    if "truncat" in text_lower:
        return "truncation"

    if status_code in (502, 503, 504):
        return "server_crash"

    if status_code != 200:
        return "non_200"

    return "unknown"


# ---------------------------------------------------------------------------
# score_run / metrics_to_dict
# ---------------------------------------------------------------------------

def score_run(metrics: RunMetrics) -> float:
    """Return a scalar score (lower is better) for a run.

    Failed runs score ``float('inf')``.
    """
    if not metrics.success:
        return float("inf")
    return metrics.client.end_to_end_latency_ms


def metrics_to_dict(m: RunMetrics) -> dict:
    """Serialise a :class:`RunMetrics` to a JSON-serialisable dict."""
    server_dict = None
    if m.server is not None:
        server_dict = {
            "prompt_eval_time_ms": m.server.prompt_eval_time_ms,
            "prompt_eval_count": m.server.prompt_eval_count,
            "eval_time_ms": m.server.eval_time_ms,
            "eval_count": m.server.eval_count,
            "prompt_tok_per_s": m.server.prompt_tok_per_s,
            "decode_tok_per_s": m.server.decode_tok_per_s,
            "total_time_ms": m.server.total_time_ms,
            "has_memory_info": m.server.has_memory_info,
            "memory_used_mb": m.server.memory_used_mb,
        }

    return {
        "run_id": m.run_id,
        "timestamp": m.timestamp,
        "config_hash": m.config_hash,
        "success": m.success,
        "failure_reason": m.failure_reason,
        "log_file": m.log_file,
        "stderr_path": m.stderr_path,
        "stdout_path": m.stdout_path,
        "server_error_excerpt": m.server_error_excerpt,
        "client": {
            "ttft_ms": m.client.ttft_ms,
            "end_to_end_latency_ms": m.client.end_to_end_latency_ms,
            "streaming_tok_per_s": m.client.streaming_tok_per_s,
            "total_tokens": m.client.total_tokens,
            "is_streaming": m.client.is_streaming,
        },
        "server": server_dict,
    }


# ---------------------------------------------------------------------------
# detect_vulkan_used
# ---------------------------------------------------------------------------

# Lines emitted by llama.cpp when the Vulkan backend is initialised.
_RE_VULKAN_BACKEND = re.compile(
    r"ggml_vulkan",
    re.IGNORECASE,
)


def detect_vulkan_used(stderr_text: str) -> bool:
    """Return ``True`` if the server stderr contains Vulkan backend log lines.

    Used to detect *engine mismatch*: when ``--engine vulkan`` was requested
    but the server binary was not built with Vulkan support (or defaulted to
    a different backend).
    """
    return bool(_RE_VULKAN_BACKEND.search(stderr_text))
