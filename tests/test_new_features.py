"""Tests for newly-introduced features:
  - gpu.build_env with Optional[str]
  - metrics.detect_vulkan_used
  - TunerThresholds updated defaults
  - TTFT gating disabled by default
  - Big-step ctx ladder in _adapt_for_oom
  - BenchConfig.engine field
  - TuneAttempt.engine_mismatch field
"""
from __future__ import annotations

import pytest

from llama_bench.config import BenchConfig, configs_from_args
from llama_bench.gpu import build_env
from llama_bench.metrics import detect_vulkan_used
from llama_bench.tuner import (
    AdaptiveTuner,
    TuneAttempt,
    TunerBounds,
    TunerThresholds,
)


# ---------------------------------------------------------------------------
# build_env — Optional[str] support
# ---------------------------------------------------------------------------

class TestBuildEnv:
    def test_sets_vk_visible_devices_when_provided(self):
        env = build_env("0,1")
        assert env.get("GGML_VK_VISIBLE_DEVICES") == "0,1"

    def test_does_not_set_env_var_when_none(self):
        import os
        # Remove env var from os.environ so the test is clean
        original = os.environ.pop("GGML_VK_VISIBLE_DEVICES", None)
        try:
            env = build_env(None)
            assert "GGML_VK_VISIBLE_DEVICES" not in env
        finally:
            if original is not None:
                os.environ["GGML_VK_VISIBLE_DEVICES"] = original

    def test_single_device_index(self):
        env = build_env("2")
        assert env["GGML_VK_VISIBLE_DEVICES"] == "2"

    def test_empty_string_still_sets_var(self):
        env = build_env("")
        assert "GGML_VK_VISIBLE_DEVICES" in env
        assert env["GGML_VK_VISIBLE_DEVICES"] == ""


# ---------------------------------------------------------------------------
# detect_vulkan_used
# ---------------------------------------------------------------------------

class TestDetectVulkanUsed:
    def test_detects_vulkan_init_line(self):
        log = "ggml_vulkan: Using Vulkan device: AMD Radeon RX 7900 XTX\n"
        assert detect_vulkan_used(log) is True

    def test_detects_lowercase_variant(self):
        log = "ggml_vulkan: found 1 device(s)\n"
        assert detect_vulkan_used(log) is True

    def test_returns_false_when_no_vulkan_lines(self):
        log = (
            "llama_model_loader: loaded model /path/to/model.gguf\n"
            "llm_load_tensors: using CPU backend\n"
            "llama server ready\n"
        )
        assert detect_vulkan_used(log) is False

    def test_returns_false_on_empty_string(self):
        assert detect_vulkan_used("") is False

    def test_detects_with_mixed_log(self):
        log = (
            "ggml_cuda_init: no CUDA devices found\n"
            "ggml_vulkan: 12345.0 MiB\n"
        )
        assert detect_vulkan_used(log) is True


# ---------------------------------------------------------------------------
# TunerThresholds — updated defaults
# ---------------------------------------------------------------------------

class TestTunerThresholdsDefaults:
    def test_min_tokens_per_sec_default_is_4(self):
        t = TunerThresholds()
        assert t.min_tokens_per_sec == 4.0

    def test_max_ttft_s_default_is_none(self):
        t = TunerThresholds()
        assert t.max_ttft_s is None

    def test_explicit_max_ttft_s_accepted(self):
        t = TunerThresholds(max_ttft_s=30.0)
        assert t.max_ttft_s == 30.0


# ---------------------------------------------------------------------------
# TuneAttempt — engine_mismatch field
# ---------------------------------------------------------------------------

class TestTuneAttemptEngineMismatch:
    def test_default_engine_mismatch_false(self):
        cfg = BenchConfig(model_path="/tmp/model.gguf")
        attempt = TuneAttempt(config=cfg, success=True)
        assert attempt.engine_mismatch is False

    def test_engine_mismatch_set_true(self):
        cfg = BenchConfig(model_path="/tmp/model.gguf")
        attempt = TuneAttempt(config=cfg, success=True, engine_mismatch=True)
        assert attempt.engine_mismatch is True


# ---------------------------------------------------------------------------
# BenchConfig — engine field
# ---------------------------------------------------------------------------

class TestBenchConfigEngine:
    def test_default_engine_is_vulkan(self):
        cfg = BenchConfig(model_path="/tmp/model.gguf")
        assert cfg.engine == "vulkan"

    def test_engine_set_to_cpu(self):
        cfg = BenchConfig(model_path="/tmp/model.gguf", engine="cpu")
        assert cfg.engine == "cpu"

    def test_configs_from_args_engine(self):
        cfg = configs_from_args(model="/tmp/model.gguf", engine="cpu")
        assert cfg.engine == "cpu"

    def test_configs_from_args_default_engine_vulkan(self):
        cfg = configs_from_args(model="/tmp/model.gguf")
        assert cfg.engine == "vulkan"

    def test_configs_from_args_vk_devices_none_default(self):
        cfg = configs_from_args(model="/tmp/model.gguf")
        assert cfg.vk_visible_devices is None


# ---------------------------------------------------------------------------
# TTFT gating disabled by default (max_ttft_s=None)
# ---------------------------------------------------------------------------

class TestTtftGatingDisabled:
    """When max_ttft_s is None, TTFT alone must never reject a config."""

    def _make_tuner(self, max_ttft_s=None, min_tps=4.0):
        bounds = TunerBounds(ctx_min=8192, ctx_max=8192, ctx_step=8192)
        thresholds = TunerThresholds(max_ttft_s=max_ttft_s, min_tokens_per_sec=min_tps)
        cfg = BenchConfig(model_path="/tmp/model.gguf", ctx=8192)
        return AdaptiveTuner(base_cfg=cfg, bounds=bounds, thresholds=thresholds,
                             artifacts_dir="/tmp")

    def test_none_ttft_does_not_reject_high_ttft(self):
        tuner = self._make_tuner(max_ttft_s=None, min_tps=1.0)
        # Simulate metrics with very high TTFT
        from llama_bench.metrics import ClientMetrics, RunMetrics
        from unittest.mock import patch

        good_metrics = RunMetrics(
            client=ClientMetrics(ttft_ms=300_000, streaming_tok_per_s=5.0),
            success=True,
            run_id="r1",
            timestamp="2024-01-01T00:00:00+00:00",
        )
        prompt_seq = [{"messages": [{"role": "user", "content": "hi"}]}]
        with patch.object(tuner, "_single_run") as mock_run:
            mock_run.return_value = TuneAttempt(
                config=tuner.base_cfg,
                success=True,
                ttft_s=300.0,
                tokens_per_sec=5.0,
                ctx=8192,
                n_gpu_layers=45,
                batch_size=1536,
                run_id="r1",
                timestamp="2024-01-01T00:00:00+00:00",
            )
            attempts = tuner.run()

        assert len(attempts) == 1
        assert attempts[0].success is True

    def test_explicit_max_ttft_rejects_high_ttft(self):
        tuner = self._make_tuner(max_ttft_s=10.0, min_tps=1.0)
        prompt_seq = [{"messages": [{"role": "user", "content": "hi"}]}]
        with patch.object(tuner, "_single_run") as mock_run:
            from unittest.mock import patch as _patch
            mock_run.return_value = TuneAttempt(
                config=tuner.base_cfg,
                success=False,
                failure_reason="ttft_exceeded",
                ttft_s=300.0,
                tokens_per_sec=5.0,
                ctx=8192,
                n_gpu_layers=45,
                batch_size=1536,
                run_id="r1",
                timestamp="2024-01-01T00:00:00+00:00",
            )
            attempts = tuner.run()

        assert attempts[0].success is False
        assert attempts[0].failure_reason == "ttft_exceeded"


from unittest.mock import patch


# ---------------------------------------------------------------------------
# Big-step ctx ladder in _adapt_for_oom
# ---------------------------------------------------------------------------

class TestBigStepCtxLadder:
    """When free_mib / projected_mib < 0.5 (memory way too low), ctx should halve."""

    def _make_tuner_at_min_params(self, ctx=65536):
        """Create a tuner where ngl and batch are already at minimum."""
        bounds = TunerBounds(
            ctx_min=8192, ctx_max=ctx, ctx_step=8192,
            ngl_step=4, batch_step=256, max_retries=5,
        )
        cfg = BenchConfig(
            model_path="/tmp/model.gguf",
            ctx=ctx, n_gpu_layers=0, batch_size=256, flash_attn=False,
        )
        tuner = AdaptiveTuner(
            base_cfg=cfg, bounds=bounds, thresholds=TunerThresholds(),
            artifacts_dir="/tmp",
        )
        return tuner, cfg

    def test_big_step_used_when_ratio_below_half(self):
        tuner, cfg = self._make_tuner_at_min_params(ctx=65536)
        # projected=20000, free=5000 → ratio=0.25 < 0.5 → big step (halve)
        stderr = (
            "llama_params_fit_impl: projected to use 20000 MiB, 5000 MiB free\n"
            "failed to fit params to free device memory\n"
        )
        tried = {(65536, 0, 256, True), (65536, 0, 256, False)}
        new_cfg = tuner._adapt_for_oom(cfg, stderr, tried)
        assert new_cfg is not None
        # Expected: 65536 // 2 = 32768, then rounded to 8192 boundary = 32768
        assert new_cfg.ctx == 32768

    def test_heuristic_step_used_when_ratio_above_half(self):
        tuner, cfg = self._make_tuner_at_min_params(ctx=65536)
        # projected=10000, free=6000 → ratio=0.60 > 0.5 → heuristic
        # new_ctx = 65536 * 0.60 * 0.90 ≈ 35389 → rounded to 8192 boundary = 32768
        stderr = (
            "llama_params_fit_impl: projected to use 10000 MiB, 6000 MiB free\n"
            "failed to fit params to free device memory\n"
        )
        tried = {(65536, 0, 256, True), (65536, 0, 256, False)}
        new_cfg = tuner._adapt_for_oom(cfg, stderr, tried)
        assert new_cfg is not None
        assert new_cfg.ctx < cfg.ctx
        assert new_cfg.ctx >= 8192  # ctx_min
