"""Tests for llama_bench.metrics."""
import pytest

from llama_bench.metrics import (
    ClientMetrics,
    RunMetrics,
    classify_failure,
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
