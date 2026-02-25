"""Tests for llama_bench.explorer — HallOfFame and ContinuousExplorer."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from llama_bench.explorer import HallOfFame, HallOfFameEntry, ContinuousExplorer, _bal_score
from llama_bench.tuner import TuneAttempt, TunerBounds, TunerThresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attempt(
    ctx: int = 65536,
    ngl: int = 45,
    batch: int = 1536,
    success: bool = True,
    cold_ttft_s: float = 2.5,
    warm_ttft_s: float = 0.5,
    tokens_per_sec: float = 20.0,
    failure_reason: str | None = None,
) -> TuneAttempt:
    from llama_bench.config import BenchConfig
    cfg = BenchConfig(
        model_path="/fake/model.gguf",
        server_binary="/fake/llama-server",
        ctx=ctx,
        n_gpu_layers=ngl,
        batch_size=batch,
        ubatch_size=512,
    )
    return TuneAttempt(
        config=cfg,
        ctx=ctx,
        n_gpu_layers=ngl,
        batch_size=batch,
        success=success,
        cold_ttft_s=cold_ttft_s,
        warm_ttft_s=warm_ttft_s,
        tokens_per_sec=tokens_per_sec,
        failure_reason=failure_reason,
        ttft_s=cold_ttft_s,
        run_id="test-run",
    )


def _make_base_cfg():
    from llama_bench.config import BenchConfig
    return BenchConfig(
        model_path="/fake/model.gguf",
        server_binary="/fake/llama-server",
        ctx=65536,
        n_gpu_layers=45,
        batch_size=1536,
        ubatch_size=512,
    )


def _make_bounds():
    return TunerBounds(
        ctx_min=8192,
        ctx_max=65536,
        ctx_step=8192,
        ngl_min=0,
        ngl_step=4,
        batch_min=256,
        batch_step=256,
        max_retries=3,
    )


def _make_thresholds():
    return TunerThresholds(max_ttft_s=60.0, min_tokens_per_sec=1.0)


# ---------------------------------------------------------------------------
# HallOfFame.update — basic successful update
# ---------------------------------------------------------------------------

def test_hof_update_success():
    hof = HallOfFame()
    a = _attempt()
    changed = hof.update(a)
    assert changed is True
    assert hof.max_ctx is not None
    assert hof.max_ctx.ctx == 65536
    assert hof.fastest_ttft is not None
    assert hof.best_warm is not None
    assert hof.best_throughput is not None
    assert hof.best_balanced is not None


# ---------------------------------------------------------------------------
# HallOfFame.update — failed attempt does NOT update
# ---------------------------------------------------------------------------

def test_hof_update_failure_ignored():
    hof = HallOfFame()
    a = _attempt(success=False, failure_reason="out_of_vram")
    changed = hof.update(a)
    assert changed is False
    assert hof.max_ctx is None
    assert hof.fastest_ttft is None
    assert hof.best_throughput is None
    assert hof.total_tested == 0


# ---------------------------------------------------------------------------
# HallOfFame — max_ctx tracking
# ---------------------------------------------------------------------------

def test_hof_max_ctx_tracking():
    hof = HallOfFame()
    a1 = _attempt(ctx=32768)
    a2 = _attempt(ctx=65536)
    a3 = _attempt(ctx=16384)
    hof.update(a1)
    hof.update(a2)
    hof.update(a3)
    assert hof.max_ctx is not None
    assert hof.max_ctx.ctx == 65536
    assert hof.max_ctx.label == "Max Context"


# ---------------------------------------------------------------------------
# HallOfFame — fastest_ttft tracking (lower is better)
# ---------------------------------------------------------------------------

def test_hof_fastest_ttft_tracking():
    hof = HallOfFame()
    hof.update(_attempt(cold_ttft_s=5.0))
    hof.update(_attempt(cold_ttft_s=1.5))  # faster
    hof.update(_attempt(cold_ttft_s=3.0))  # slower
    assert hof.fastest_ttft is not None
    assert hof.fastest_ttft.cold_ttft_s == pytest.approx(1.5)
    assert hof.fastest_ttft.label == "Fastest TTFT"


# ---------------------------------------------------------------------------
# HallOfFame — best_throughput tracking (higher is better)
# ---------------------------------------------------------------------------

def test_hof_best_throughput_tracking():
    hof = HallOfFame()
    hof.update(_attempt(tokens_per_sec=15.0))
    hof.update(_attempt(tokens_per_sec=30.0))  # better
    hof.update(_attempt(tokens_per_sec=10.0))  # worse
    assert hof.best_throughput is not None
    assert hof.best_throughput.tokens_per_sec == pytest.approx(30.0)
    assert hof.best_throughput.label == "Best Throughput"


# ---------------------------------------------------------------------------
# HallOfFame — best_balanced tracking
# ---------------------------------------------------------------------------

def test_hof_best_balanced_tracking():
    hof = HallOfFame()
    # low ctx, high tps, low ttft
    a1 = _attempt(ctx=8192, tokens_per_sec=50.0, cold_ttft_s=0.5)
    # high ctx, low tps, moderate ttft
    a2 = _attempt(ctx=131072, tokens_per_sec=5.0, cold_ttft_s=10.0)

    score1 = _bal_score(8192, 50.0, 0.5)
    score2 = _bal_score(131072, 5.0, 10.0)

    hof.update(a1)
    hof.update(a2)

    assert hof.best_balanced is not None
    expected_ctx = a1.ctx if score1 > score2 else a2.ctx
    assert hof.best_balanced.ctx == expected_ctx
    assert hof.best_balanced.label == "Best Overall"


# ---------------------------------------------------------------------------
# HallOfFame.to_dict — serialization
# ---------------------------------------------------------------------------

def test_hof_to_dict_empty():
    hof = HallOfFame()
    d = hof.to_dict()
    assert d["max_ctx"] is None
    assert d["fastest_ttft"] is None
    assert d["best_warm"] is None
    assert d["best_throughput"] is None
    assert d["best_balanced"] is None
    assert d["total_tested"] == 0
    assert d["round_num"] == 1


def test_hof_to_dict_with_entry():
    hof = HallOfFame()
    hof.update(_attempt(ctx=65536, ngl=45, batch=1536, cold_ttft_s=2.5, warm_ttft_s=0.5, tokens_per_sec=20.0))
    d = hof.to_dict()
    entry = d["max_ctx"]
    assert entry is not None
    assert entry["label"] == "Max Context"
    assert entry["ctx"] == 65536
    assert entry["ngl"] == 45
    assert entry["batch"] == 1536
    assert entry["cold_ttft_s"] == pytest.approx(2.5)
    assert entry["warm_ttft_s"] == pytest.approx(0.5)
    assert entry["tokens_per_sec"] == pytest.approx(20.0)


def test_hof_to_dict_keys():
    hof = HallOfFame()
    hof.update(_attempt())
    d = hof.to_dict()
    expected_keys = {"max_ctx", "fastest_ttft", "best_warm", "best_throughput", "best_balanced", "total_tested", "round_num"}
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# HallOfFame — best_warm tracking
# ---------------------------------------------------------------------------

def test_hof_best_warm_tracking():
    hof = HallOfFame()
    hof.update(_attempt(warm_ttft_s=1.0))
    hof.update(_attempt(warm_ttft_s=0.2))  # better (lower)
    hof.update(_attempt(warm_ttft_s=0.8))  # worse
    assert hof.best_warm is not None
    assert hof.best_warm.warm_ttft_s == pytest.approx(0.2)
    assert hof.best_warm.label == "Best Warm TTFT"


# ---------------------------------------------------------------------------
# HallOfFame — total_tested and round_num are not auto-updated by update()
# ---------------------------------------------------------------------------

def test_hof_total_tested_managed_externally():
    """HallOfFame.update() does not increment total_tested — caller does it."""
    hof = HallOfFame()
    hof.update(_attempt())
    assert hof.total_tested == 0  # caller must increment manually


# ---------------------------------------------------------------------------
# ContinuousExplorer._generate_round — correct Cartesian product
# ---------------------------------------------------------------------------

def test_generate_round_product():
    base_cfg = _make_base_cfg()
    bounds = _make_bounds()
    thresholds = _make_thresholds()

    ngl_values = [37, 41, 45]
    batch_values = [1024, 1536]

    explorer = ContinuousExplorer(
        base_cfg=base_cfg,
        bounds=bounds,
        thresholds=thresholds,
        ngl_values=ngl_values,
        batch_values=batch_values,
        output_path=None,
    )

    candidates = explorer._generate_round(1)

    # ctx_values: 65536 down to 8192 by 8192 → 65536, 57344, ..., 8192
    ctx_vals = list(range(65536, 8192 - 1, -8192))
    expected_count = len(ctx_vals) * len(ngl_values) * len(batch_values)
    assert len(candidates) == expected_count

    # All generated ctx values should be in the expected set
    generated_ctxs = {c.ctx for c in candidates}
    assert generated_ctxs == set(ctx_vals)

    # All ngl values should match
    generated_ngls = {c.n_gpu_layers for c in candidates}
    assert generated_ngls == set(ngl_values)

    # All batch values should match
    generated_batches = {c.batch_size for c in candidates}
    assert generated_batches == set(batch_values)


# ---------------------------------------------------------------------------
# ContinuousExplorer._generate_round — round 1 sorted, round 2+ shuffled
# ---------------------------------------------------------------------------

def test_generate_round_1_sorted_descending():
    """Round 1 should be sorted high-to-low (ctx first, then ngl, then batch)."""
    base_cfg = _make_base_cfg()
    bounds = _make_bounds()
    thresholds = _make_thresholds()

    explorer = ContinuousExplorer(
        base_cfg=base_cfg,
        bounds=bounds,
        thresholds=thresholds,
        ngl_values=[37, 45],
        batch_values=[1024, 1536],
        output_path=None,
    )

    candidates = explorer._generate_round(1)

    # The first candidate should have the highest ctx (ctx_max = 65536)
    assert candidates[0].ctx == 65536

    # Verify ctxs are non-increasing throughout (they iterate outer loop)
    # ctx changes every len(ngl)*len(batch) entries
    block = len([37, 45]) * len([1024, 1536])
    for i in range(0, len(candidates) - block, block):
        assert candidates[i].ctx >= candidates[i + block].ctx


def test_generate_round_2_not_necessarily_sorted():
    """Round 2+ is shuffled; over many attempts it won't match round 1 order."""
    base_cfg = _make_base_cfg()
    bounds = TunerBounds(
        ctx_min=8192, ctx_max=65536, ctx_step=8192,
        ngl_min=0, ngl_step=4, batch_min=256, batch_step=256, max_retries=3,
    )
    thresholds = _make_thresholds()

    explorer = ContinuousExplorer(
        base_cfg=base_cfg,
        bounds=bounds,
        thresholds=thresholds,
        ngl_values=[37, 41, 45],
        batch_values=[1024, 1536],
        output_path=None,
    )

    round1 = [(c.ctx, c.n_gpu_layers, c.batch_size) for c in explorer._generate_round(1)]

    # Run many round-2 generations; at least one should differ from round 1
    differs = False
    for _ in range(20):
        round2 = [(c.ctx, c.n_gpu_layers, c.batch_size) for c in explorer._generate_round(2)]
        if round2 != round1:
            differs = True
            break
    assert differs, "Round 2 should be shuffled and differ from round 1 order"
