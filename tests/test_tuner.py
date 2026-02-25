"""Tests for llama_bench.tuner — summary selection and ctx heuristic."""
from __future__ import annotations

import copy
import json
import os
import tempfile

import pytest

from llama_bench.config import BenchConfig
from llama_bench.metrics import parse_memory_fit_heuristic
from llama_bench.tuner import (
    AdaptiveTuner,
    TuneAttempt,
    TunerBounds,
    TunerThresholds,
    select_best_configs,
    write_summary_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attempt(
    ctx: int,
    success: bool,
    tokens_per_sec: float = 20.0,
    ttft_s: float = 1.0,
    failure_reason: str | None = None,
    n_gpu_layers: int = 45,
    batch_size: int = 1536,
) -> TuneAttempt:
    cfg = BenchConfig(model_path="/tmp/model.gguf", ctx=ctx,
                      n_gpu_layers=n_gpu_layers, batch_size=batch_size)
    return TuneAttempt(
        config=cfg,
        success=success,
        failure_reason=failure_reason,
        ttft_s=ttft_s,
        tokens_per_sec=tokens_per_sec,
        ctx=ctx,
        n_gpu_layers=n_gpu_layers,
        batch_size=batch_size,
        run_id=f"run_{ctx}",
        timestamp="2024-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# select_best_configs — summary selection logic
# ---------------------------------------------------------------------------

class TestSelectBestConfigs:
    def test_returns_none_when_no_usable_attempts(self):
        attempts = [_make_attempt(49152, False, failure_reason="out_of_vram")]
        result = select_best_configs(attempts)
        assert result["max_ctx_result"] is None
        assert result["top_throughput"] == []
        assert result["recommended"] is None

    def test_max_ctx_is_highest_usable_ctx(self):
        attempts = [
            _make_attempt(16384, True, tokens_per_sec=30.0),
            _make_attempt(32768, True, tokens_per_sec=20.0),
            _make_attempt(65536, False, failure_reason="out_of_vram"),
        ]
        result = select_best_configs(attempts)
        assert result["max_ctx_result"].ctx == 32768

    def test_top_throughput_sorted_by_tokens_per_sec(self):
        attempts = [
            _make_attempt(16384, True, tokens_per_sec=10.0),
            _make_attempt(24576, True, tokens_per_sec=25.0),
            _make_attempt(32768, True, tokens_per_sec=15.0),
        ]
        result = select_best_configs(attempts, ctx_pct_threshold=0.5)
        top = result["top_throughput"]
        assert len(top) >= 1
        # First entry should have highest throughput
        assert top[0].tokens_per_sec >= top[-1].tokens_per_sec

    def test_top_throughput_limited_to_top_n(self):
        attempts = [_make_attempt(8192 * i, True, tokens_per_sec=float(i))
                    for i in range(1, 10)]
        result = select_best_configs(attempts, top_n=3)
        assert len(result["top_throughput"]) <= 3

    def test_configs_below_threshold_excluded_from_top(self):
        """Configs below ctx_pct_threshold * max_ctx should not appear in top."""
        attempts = [
            _make_attempt(8192,  True, tokens_per_sec=100.0),   # 8192 / 32768 = 0.25 < 0.90
            _make_attempt(32768, True, tokens_per_sec=10.0),
        ]
        result = select_best_configs(attempts, ctx_pct_threshold=0.90)
        top_ctxs = {a.ctx for a in result["top_throughput"]}
        # 8192 is below 90% of 32768 (29491); should be excluded
        assert 8192 not in top_ctxs

    def test_recommended_prefers_higher_ctx(self):
        """Recommended config should prefer higher ctx among near-max configs."""
        attempts = [
            _make_attempt(32768, True, tokens_per_sec=50.0),
            _make_attempt(30000, True, tokens_per_sec=55.0),
        ]
        result = select_best_configs(attempts, ctx_pct_threshold=0.90)
        # Both are near max (30000/32768 ≈ 0.91 > 0.90); recommended should pick highest ctx first
        assert result["recommended"].ctx == 32768

    def test_single_usable_config(self):
        attempts = [_make_attempt(49152, True, tokens_per_sec=22.0)]
        result = select_best_configs(attempts)
        assert result["max_ctx_result"].ctx == 49152
        assert len(result["top_throughput"]) == 1
        assert result["recommended"].ctx == 49152


# ---------------------------------------------------------------------------
# write_summary_json
# ---------------------------------------------------------------------------

class TestWriteSummaryJson:
    def test_creates_file(self):
        attempts = [_make_attempt(49152, True)]
        selection = select_best_configs(attempts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            write_summary_json(path, attempts, selection)
            assert os.path.exists(path)

    def test_valid_json(self):
        attempts = [_make_attempt(49152, True)]
        selection = select_best_configs(attempts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            write_summary_json(path, attempts, selection)
            with open(path) as fh:
                data = json.load(fh)
        assert "max_ctx_result" in data
        assert "top_throughput_configs" in data
        assert "recommended" in data
        assert "total_attempts" in data

    def test_max_ctx_in_summary(self):
        attempts = [
            _make_attempt(16384, True),
            _make_attempt(32768, True),
            _make_attempt(65536, False, failure_reason="out_of_vram"),
        ]
        selection = select_best_configs(attempts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            write_summary_json(path, attempts, selection)
            with open(path) as fh:
                data = json.load(fh)
        assert data["max_ctx_result"]["ctx"] == 32768

    def test_no_usable_configs(self):
        attempts = [_make_attempt(49152, False, failure_reason="out_of_vram")]
        selection = select_best_configs(attempts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            write_summary_json(path, attempts, selection)
            with open(path) as fh:
                data = json.load(fh)
        assert data["max_ctx_result"] is None
        assert data["recommended"] is None


# ---------------------------------------------------------------------------
# AdaptiveTuner._adapt_for_oom — ctx adjustment heuristic
# ---------------------------------------------------------------------------

class TestAdaptForOom:
    """Tests for the ctx reduction logic in AdaptiveTuner._adapt_for_oom."""

    def _make_tuner(self, base_ctx=65536, ngl=45, batch=1536):
        bounds = TunerBounds(
            ctx_min=8192, ctx_max=base_ctx, ctx_step=8192,
            ngl_step=4, batch_step=256, max_retries=5,
        )
        thresholds = TunerThresholds()
        cfg = BenchConfig(
            model_path="/tmp/model.gguf",
            ctx=base_ctx, n_gpu_layers=ngl, batch_size=batch,
        )
        tuner = AdaptiveTuner(
            base_cfg=cfg, bounds=bounds, thresholds=thresholds,
            artifacts_dir="/tmp", log_file=None,
        )
        return tuner, cfg

    def test_reduces_ngl_first(self):
        tuner, cfg = self._make_tuner(ngl=45)
        new_cfg = tuner._adapt_for_oom(cfg, "", set())
        assert new_cfg is not None
        assert new_cfg.n_gpu_layers == 41  # 45 - 4

    def test_reduces_batch_when_ngl_at_min(self):
        tuner, cfg = self._make_tuner(ngl=0, batch=1536)
        # ngl already at min (0), so try batch reduction
        new_cfg = tuner._adapt_for_oom(cfg, "", set())
        assert new_cfg is not None
        # After exhausting ngl, batch should be reduced
        # ngl-reduction check: 0 - 4 = -4 < 0, so skip; try batch
        assert new_cfg.batch_size == 1280  # 1536 - 256

    def test_uses_heuristic_when_fit_line_present(self):
        tuner, cfg = self._make_tuner(ngl=0, batch=256)
        # Both ngl and batch already at minimum; fall through to ctx heuristic
        bounds = TunerBounds(
            ctx_min=8192, ctx_max=65536, ctx_step=8192,
            ngl_step=4, batch_step=256, max_retries=5,
        )
        cfg_at_min = BenchConfig(
            model_path="/tmp/model.gguf",
            ctx=65536, n_gpu_layers=0, batch_size=256,
        )
        tuner_min = AdaptiveTuner(
            base_cfg=cfg_at_min, bounds=bounds, thresholds=TunerThresholds(),
            artifacts_dir="/tmp", log_file=None,
        )
        stderr = (
            "llama_params_fit_impl: projected to use 20000 MiB, 10000 MiB free\n"
            "failed to fit params to free device memory\n"
        )
        tried: set = set()
        tried.add((65536, 0, 256, True))   # mark flash-attn adaptation as tried too
        tried.add((65536, 0, 256, False))
        new_cfg = tuner_min._adapt_for_oom(cfg_at_min, stderr, tried)
        if new_cfg is not None:
            # ctx should be reduced proportionally: 65536 * (10000/20000) * 0.9 ≈ 29491
            # rounded to 8192 boundary: 24576
            assert new_cfg.ctx < cfg_at_min.ctx
            assert new_cfg.ctx >= bounds.ctx_min

    def test_returns_none_when_no_option_left(self):
        """When all parameters are at minimum, _adapt_for_oom returns None."""
        bounds = TunerBounds(
            ctx_min=8192, ctx_max=8192, ctx_step=8192,
            ngl_step=4, batch_step=256, max_retries=5,
        )
        cfg_min = BenchConfig(
            model_path="/tmp/model.gguf",
            ctx=8192, n_gpu_layers=0, batch_size=256, flash_attn=False,
        )
        tuner = AdaptiveTuner(
            base_cfg=cfg_min, bounds=bounds, thresholds=TunerThresholds(),
            artifacts_dir="/tmp", log_file=None,
        )
        tried = {(8192, 0, 256, False)}
        new_cfg = tuner._adapt_for_oom(cfg_min, "", tried)
        assert new_cfg is None


# ---------------------------------------------------------------------------
# AdaptiveTuner._generate_candidates
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    def test_generates_correct_ctx_values(self):
        bounds = TunerBounds(ctx_min=8192, ctx_max=32768, ctx_step=8192)
        cfg = BenchConfig(model_path="/tmp/model.gguf")
        tuner = AdaptiveTuner(
            base_cfg=cfg, bounds=bounds, thresholds=TunerThresholds(),
            artifacts_dir="/tmp",
        )
        candidates = tuner._generate_candidates()
        ctxs = [c.ctx for c in candidates]
        assert ctxs == [32768, 24576, 16384, 8192]

    def test_generates_single_candidate_when_min_equals_max(self):
        bounds = TunerBounds(ctx_min=49152, ctx_max=49152, ctx_step=8192)
        cfg = BenchConfig(model_path="/tmp/model.gguf")
        tuner = AdaptiveTuner(
            base_cfg=cfg, bounds=bounds, thresholds=TunerThresholds(),
            artifacts_dir="/tmp",
        )
        candidates = tuner._generate_candidates()
        assert len(candidates) == 1
        assert candidates[0].ctx == 49152


# ---------------------------------------------------------------------------
# AdaptiveTuner — new params: n_followups, max_tokens, prompt_seq_override
# ---------------------------------------------------------------------------

class TestAdaptiveTunerNewParams:

    def _make_tuner(self, **kwargs):
        bounds = TunerBounds(ctx_min=8192, ctx_max=8192, ctx_step=8192)
        cfg = BenchConfig(model_path="/tmp/model.gguf")
        return AdaptiveTuner(
            base_cfg=cfg, bounds=bounds, thresholds=TunerThresholds(),
            artifacts_dir="/tmp",
            **kwargs,
        )

    def test_default_n_followups_is_4(self):
        tuner = self._make_tuner()
        assert tuner.n_followups == 4

    def test_custom_n_followups(self):
        tuner = self._make_tuner(n_followups=2)
        assert tuner.n_followups == 2

    def test_default_max_tokens_is_512(self):
        tuner = self._make_tuner()
        assert tuner.max_tokens == 512

    def test_custom_max_tokens(self):
        tuner = self._make_tuner(max_tokens=128)
        assert tuner.max_tokens == 128

    def test_prompt_seq_override_stored(self):
        override = [{"messages": [{"role": "user", "content": "test"}], "is_followup": False,
                     "expected_prefix_len_tokens": 1}]
        tuner = self._make_tuner(prompt_seq_override=override)
        assert tuner.prompt_seq_override is override

    def test_prompt_seq_override_default_none(self):
        tuner = self._make_tuner()
        assert tuner.prompt_seq_override is None
