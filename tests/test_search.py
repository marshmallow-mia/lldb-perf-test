"""Tests for llama_bench.search."""
import json
import os
import tempfile

import pytest

from llama_bench.config import BenchConfig, SearchSpace, default_search_space, parse_range
from llama_bench.metrics import ClientMetrics, RunMetrics
from llama_bench.search import EarlyStopReason, SearchResult, StagedSearcher, save_results
from llama_bench.report import load_results


# ---------------------------------------------------------------------------
# parse_range integration (re-test via config import)
# ---------------------------------------------------------------------------

def test_parse_range_integration():
    assert parse_range("1,2") == [1, 2]
    assert parse_range("10-12") == [10, 11, 12]
    assert parse_range("99") == [99]


# ---------------------------------------------------------------------------
# EarlyStopReason
# ---------------------------------------------------------------------------

def test_early_stop_reason_members():
    assert hasattr(EarlyStopReason, "FAILED")
    assert hasattr(EarlyStopReason, "TIMEOUT")
    assert hasattr(EarlyStopReason, "MUCH_WORSE_THAN_BEST")
    assert hasattr(EarlyStopReason, "OOM")


def test_early_stop_reason_is_enum():
    from enum import Enum
    assert issubclass(EarlyStopReason, Enum)


# ---------------------------------------------------------------------------
# StagedSearcher instantiation
# ---------------------------------------------------------------------------

def test_staged_searcher_instantiation():
    space = default_search_space()
    base = BenchConfig(model_path="/tmp/model.gguf")
    searcher = StagedSearcher(space=space, base_cfg=base, max_configs=5)
    assert searcher is not None
    assert searcher.max_configs == 5


def test_staged_searcher_best_config_initially_none():
    space = default_search_space()
    base = BenchConfig(model_path="/tmp/model.gguf")
    searcher = StagedSearcher(space=space, base_cfg=base, max_configs=5)
    # Before run(), best_config returns None (no results)
    assert searcher.best_config() is None


# ---------------------------------------------------------------------------
# save_results / load_results roundtrip
# ---------------------------------------------------------------------------

def _make_search_result(run_id: str = "run_001", success: bool = True) -> SearchResult:
    m = RunMetrics(
        success=success,
        run_id=run_id,
        timestamp="2024-01-01T00:00:00+00:00",
        config_hash="abcd1234",
        client=ClientMetrics(
            ttft_ms=100.0,
            end_to_end_latency_ms=2000.0,
            streaming_tok_per_s=25.0,
            total_tokens=50,
            is_streaming=True,
        ),
    )
    cfg = BenchConfig(model_path="/tmp/model.gguf")
    return SearchResult(
        config=cfg,
        metrics=[m],
        best_score=2000.0 if success else float("inf"),
        phase=1,
        run_id=run_id,
    )


def test_save_results_creates_file():
    results = [_make_search_result("r1"), _make_search_result("r2")]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.jsonl")
        save_results(results, path)
        assert os.path.exists(path)


def test_save_results_line_count():
    results = [_make_search_result("r1"), _make_search_result("r2")]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.jsonl")
        save_results(results, path)
        with open(path) as fh:
            lines = [l for l in fh.readlines() if l.strip()]
        assert len(lines) == 2


def test_save_load_roundtrip():
    results = [_make_search_result("r1", success=True), _make_search_result("r2", success=False)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.jsonl")
        save_results(results, path)
        loaded = load_results(path)
    assert len(loaded) == 2
    # Check run_ids
    run_ids = {r["run_id"] for r in loaded}
    assert "r1" in run_ids
    assert "r2" in run_ids


def test_save_load_scores():
    results = [_make_search_result("r1", success=True)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.jsonl")
        save_results(results, path)
        loaded = load_results(path)
    assert loaded[0]["best_score"] == pytest.approx(2000.0)


def test_save_load_failed_score_is_null():
    """Failed runs should save best_score as null (not infinity which is not JSON-serialisable)."""
    results = [_make_search_result("r_fail", success=False)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.jsonl")
        save_results(results, path)
        loaded = load_results(path)
    assert loaded[0]["best_score"] is None
