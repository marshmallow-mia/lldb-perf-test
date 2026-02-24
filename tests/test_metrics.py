"""Tests for llama_bench.metrics."""
import pytest

from llama_bench.metrics import (
    ClientMetrics,
    RunMetrics,
    classify_failure,
    classify_server_stderr,
    extract_server_error_excerpt,
    parse_memory_fit_heuristic,
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
# classify_server_stderr — updated patterns per spec
# ---------------------------------------------------------------------------

# Lines that must classify as out_of_vram (checked before model_not_found)
_OUT_OF_VRAM_LINES = [
    "vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
    "ErrorOutOfDeviceMemory",
    "ggml_vulkan: Device memory allocation of size 1234567890 failed.",
    "Device memory allocation of size 9999 failed",
    "failed to allocate Vulkan0 buffer",
    "unable to allocate Vulkan0 buffer",
    "failed to fit params to free device memory",
]

# Lines that must classify as model_not_found (missing-file patterns only)
_MODEL_NOT_FOUND_LINES = [
    "gguf_init_from_file: failed to open GGUF file /path/to/model.gguf (No such file or directory)",
    "error: failed to open GGUF file model.gguf",
    "fatal: No such file or directory: /tmp/model.gguf",
]

# Lines that should fall through to server_exited (no specific match)
_SERVER_EXITED_LINES = [
    "some unrecognised log line",
    "",
    "srv    load_model: failed to load model, '/path/to/model.gguf'",
    "main: exiting due to model loading error",
]


class TestClassifyServerStderr:
    """Unit tests for classify_server_stderr using problem-statement fixtures."""

    @pytest.mark.parametrize("line", _OUT_OF_VRAM_LINES)
    def test_out_of_vram(self, line):
        assert classify_server_stderr(line) == "out_of_vram"

    @pytest.mark.parametrize("line", _MODEL_NOT_FOUND_LINES)
    def test_model_not_found(self, line):
        assert classify_server_stderr(line) == "model_not_found"

    @pytest.mark.parametrize("line", _SERVER_EXITED_LINES)
    def test_server_exited_fallback(self, line):
        assert classify_server_stderr(line) == "server_exited"

    def test_out_of_vram_takes_priority_over_model_not_found(self):
        """out_of_vram is checked first (higher priority)."""
        text = (
            "gguf_init_from_file: failed to open GGUF file x.gguf (No such file or directory)\n"
            "failed to fit params to free device memory\n"
        )
        assert classify_server_stderr(text) == "out_of_vram"

    def test_model_not_found_without_oom(self):
        """model_not_found returned when only missing-file patterns present."""
        text = "gguf_init_from_file: failed to open GGUF file x.gguf (No such file or directory)\n"
        assert classify_server_stderr(text) == "model_not_found"

    def test_multiline_oom_log(self):
        log = (
            "ggml_vulkan: initialising...\n"
            "ggml_vulkan: Device memory allocation of size 4294967296 failed.\n"
            "vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n"
        )
        assert classify_server_stderr(log) == "out_of_vram"

    def test_fit_failure_classified_as_out_of_vram(self):
        log = (
            "llama_params_fit_impl: projected to use 12345 MiB, 9876 MiB free\n"
            "failed to fit params to free device memory\n"
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


# ---------------------------------------------------------------------------
# parse_memory_fit_heuristic
# ---------------------------------------------------------------------------

class TestParseMemoryFitHeuristic:
    """Unit tests for the ctx adjustment heuristic parser."""

    def test_parses_standard_fit_line(self):
        log = "llama_params_fit_impl: projected to use 12345 MiB, 9876 MiB free\n"
        result = parse_memory_fit_heuristic(log)
        assert result is not None
        projected, free = result
        assert abs(projected - 12345.0) < 0.1
        assert abs(free - 9876.0) < 0.1

    def test_parses_with_extra_text_on_line(self):
        log = (
            "llama_params_fit_impl: model requires projected to use 8192.5 MiB "
            "of device memory, but only 4096.0 MiB free\n"
        )
        result = parse_memory_fit_heuristic(log)
        assert result is not None
        projected, free = result
        assert abs(projected - 8192.5) < 0.1
        assert abs(free - 4096.0) < 0.1

    def test_returns_none_when_no_fit_line(self):
        log = "Some other server log without fit info\n"
        assert parse_memory_fit_heuristic(log) is None

    def test_returns_none_on_empty_string(self):
        assert parse_memory_fit_heuristic("") is None

    def test_picks_first_match_in_multiline(self):
        log = (
            "llama_params_fit_impl: projected to use 10000 MiB, 5000 MiB free\n"
            "llama_params_fit_impl: projected to use 20000 MiB, 3000 MiB free\n"
        )
        result = parse_memory_fit_heuristic(log)
        assert result is not None
        projected, _ = result
        assert abs(projected - 10000.0) < 0.1
