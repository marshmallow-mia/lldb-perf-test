"""Tests for llama_bench.metrics."""
import pytest

from llama_bench.metrics import (
    ClientMetrics,
    RunMetrics,
    classify_failure,
    classify_server_stderr,
    extract_server_error_excerpt,
    parse_server_log,
    score_run,
)


# ---------------------------------------------------------------------------
# parse_server_log
# ---------------------------------------------------------------------------

SAMPLE_LOG = """\
llama server listening at http://0.0.0.0:5001
ggml_vulkan: 12345.6 MiB
llama_print_timings:        load time =   450.12 ms
llama_print_timings: prompt eval time =  1234.56 ms /   512 tokens (   2.41 ms per token, 414.79 tokens per second)
llama_print_timings:        eval time =  5678.90 ms /   256 tokens (  22.18 ms per token,  45.08 tokens per second)
llama_print_timings:       total time =  6913.46 ms /   768 tokens
"""


def test_parse_server_log_prompt_eval():
    m = parse_server_log(SAMPLE_LOG)
    assert m is not None
    assert abs(m.prompt_eval_time_ms - 1234.56) < 0.01
    assert m.prompt_eval_count == 512


def test_parse_server_log_eval():
    m = parse_server_log(SAMPLE_LOG)
    assert m is not None
    assert abs(m.eval_time_ms - 5678.90) < 0.01
    assert m.eval_count == 256


def test_parse_server_log_total():
    m = parse_server_log(SAMPLE_LOG)
    assert m is not None
    assert abs(m.total_time_ms - 6913.46) < 0.01


def test_parse_server_log_memory():
    m = parse_server_log(SAMPLE_LOG)
    assert m is not None
    assert m.has_memory_info is True
    assert abs(m.memory_used_mb - 12345.6) < 0.1


def test_parse_server_log_tok_per_s():
    m = parse_server_log(SAMPLE_LOG)
    assert m is not None
    # Should have extracted or computed tok/s
    assert m.prompt_tok_per_s > 0
    assert m.decode_tok_per_s > 0


def test_parse_server_log_json_format():
    log = '{"prompt_n":128,"prompt_ms":600.0,"predicted_n":64,"predicted_ms":1200.0}'
    m = parse_server_log(log)
    assert m is not None
    assert m.prompt_eval_count == 128
    assert abs(m.prompt_eval_time_ms - 600.0) < 0.01
    assert m.eval_count == 64
    assert abs(m.eval_time_ms - 1200.0) < 0.01


def test_parse_server_log_empty():
    m = parse_server_log("")
    assert m is None


def test_parse_server_log_unrelated():
    m = parse_server_log("INFO: server started\nDEBUG: connected\n")
    assert m is None


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

def test_classify_failure_success():
    result = classify_failure(200, '{"choices":[]}', None)
    assert result == ""


def test_classify_failure_timeout():
    result = classify_failure(0, "", TimeoutError("timed out"))
    assert result == "timeout"


def test_classify_failure_oom():
    result = classify_failure(503, "out of memory", None)
    assert result == "oom"


def test_classify_failure_oom_507():
    result = classify_failure(507, "", None)
    assert result == "oom"


def test_classify_failure_non_200():
    result = classify_failure(400, "bad request", None)
    assert result == "non_200"


def test_classify_failure_server_crash():
    result = classify_failure(502, "bad gateway", None)
    assert result == "server_crash"


def test_classify_failure_context_overflow():
    result = classify_failure(400, "context overflow exceeded", None)
    assert result == "context_overflow"


def test_classify_failure_exception_connection():
    import requests
    exc = requests.exceptions.ConnectionError("refused")
    result = classify_failure(0, "", exc)
    assert result == "server_crash"


# ---------------------------------------------------------------------------
# score_run
# ---------------------------------------------------------------------------

def test_score_run_success():
    m = RunMetrics(
        success=True,
        client=ClientMetrics(end_to_end_latency_ms=1234.5),
    )
    assert abs(score_run(m) - 1234.5) < 0.01


def test_score_run_failure():
    m = RunMetrics(success=False, failure_reason="timeout")
    assert score_run(m) == float("inf")


# ---------------------------------------------------------------------------
# classify_server_stderr
# ---------------------------------------------------------------------------

# Fixture lines from problem statement
_MODEL_NOT_FOUND_LINES = [
    "gguf_init_from_file: failed to open GGUF file /path/to/model.gguf (No such file or directory)",
    "srv    load_model: failed to load model, '/path/to/model.gguf'",
    "common_init_from_params: failed to load model '/path/to/model.gguf'",
    "llama_model_load: error loading model:",
]

_OUT_OF_VRAM_LINES = [
    "ggml_vulkan: Device memory allocation of size 1234567890 failed.",
    "vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
    "failed to allocate Vulkan0 buffer",
    "unable to allocate Vulkan0 buffer",
    "main: exiting due to model loading error",
]


class TestClassifyServerStderr:
    """Unit tests for classify_server_stderr using problem-statement fixtures."""

    @pytest.mark.parametrize("line", _MODEL_NOT_FOUND_LINES)
    def test_model_not_found(self, line):
        assert classify_server_stderr(line) == "model_not_found"

    @pytest.mark.parametrize("line", _OUT_OF_VRAM_LINES)
    def test_out_of_vram(self, line):
        assert classify_server_stderr(line) == "out_of_vram"

    def test_unknown_stderr_returns_server_exited(self):
        assert classify_server_stderr("some unrecognised log line") == "server_exited"

    def test_empty_stderr_returns_server_exited(self):
        assert classify_server_stderr("") == "server_exited"

    def test_model_not_found_takes_priority_over_model_loading_error(self):
        # Both patterns present; model_not_found should win (listed first)
        text = (
            "gguf_init_from_file: failed to open GGUF file x.gguf (No such file or directory)\n"
            "main: exiting due to model loading error\n"
        )
        assert classify_server_stderr(text) == "model_not_found"

    def test_multiline_oom_log(self):
        log = (
            "ggml_vulkan: initialising...\n"
            "ggml_vulkan: Device memory allocation of size 4294967296 failed.\n"
            "vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n"
        )
        assert classify_server_stderr(log) == "out_of_vram"


# ---------------------------------------------------------------------------
# extract_server_error_excerpt
# ---------------------------------------------------------------------------

class TestExtractServerErrorExcerpt:
    def test_picks_error_lines(self):
        log = "INFO: starting\nERROR: something failed\nDEBUG: done\n"
        excerpt = extract_server_error_excerpt(log)
        assert "ERROR: something failed" in excerpt

    def test_falls_back_to_last_lines_when_no_error(self):
        log = "line1\nline2\nline3\nline4\nline5\nline6\n"
        excerpt = extract_server_error_excerpt(log, max_lines=3)
        lines = excerpt.splitlines()
        assert len(lines) <= 3
        assert "line6" in excerpt

    def test_empty_text_returns_empty(self):
        assert extract_server_error_excerpt("") == ""
