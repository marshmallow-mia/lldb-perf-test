"""Tests for llama_bench.orchestrator — PresetBenchOrchestrator with mocked runners."""
from __future__ import annotations

import threading
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from llama_bench.config import BenchConfig
from llama_bench.tuner import TuneAttempt
from llama_bench.presets import (
    ALL_PRESET_NAMES,
    PRESET_FASTEST_RESPONSE,
    PRESET_LONG_CONTEXT_RAG,
    PRESET_MAX_CONTEXT,
    PRESET_REGISTRY,
    PRESET_THROUGHPUT_KING,
    GoalPreset,
    PresetResult,
    PresetRunner,
)
from llama_bench.orchestrator import OrchestratorResult, PresetBenchOrchestrator
from llama_bench.scoring import ScoringResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attempt(
    ctx: int = 4096,
    ngl: int = 45,
    batch: int = 1536,
    success: bool = True,
    cold_ttft_s: float = 1.0,
    tokens_per_sec: float = 20.0,
) -> TuneAttempt:
    cfg = BenchConfig(model_path="/tmp/model.gguf", ctx=ctx,
                      n_gpu_layers=ngl, batch_size=batch)
    return TuneAttempt(
        config=cfg,
        success=success,
        failure_reason=None if success else "out_of_vram",
        ttft_s=cold_ttft_s,
        cold_ttft_s=cold_ttft_s,
        tokens_per_sec=tokens_per_sec,
        ctx=ctx,
        n_gpu_layers=ngl,
        batch_size=batch,
        run_id=f"run_{ctx}_{ngl}_{batch}",
        timestamp="2024-01-01T00:00:00+00:00",
    )


def _make_preset_result(
    preset_name: str,
    success: bool = True,
    skipped: bool = False,
    ctx: int = 4096,
    ngl: int = 45,
    batch: int = 1536,
    tokens_per_sec: float = 20.0,
    cold_ttft_s: float = 1.0,
) -> PresetResult:
    if skipped:
        return PresetResult(
            preset_name=preset_name,
            primary_metric=PRESET_REGISTRY[preset_name].primary_metric,
            skipped=True,
        )
    attempt = _make_attempt(ctx=ctx, ngl=ngl, batch=batch, success=success,
                            tokens_per_sec=tokens_per_sec, cold_ttft_s=cold_ttft_s)
    pr = PresetResult(
        preset_name=preset_name,
        primary_metric=PRESET_REGISTRY[preset_name].primary_metric,
        attempts=[attempt],
        best_attempt=attempt if success else None,
        best_value=float(getattr(attempt, PRESET_REGISTRY[preset_name].primary_metric, 0.0)) if success else None,
        skipped=False,
    )
    return pr


def _base_cfg(ngl: int = 45, batch: int = 1536, ctx: int = 65536) -> BenchConfig:
    return BenchConfig(
        model_path="/tmp/model.gguf",
        ctx=ctx,
        n_gpu_layers=ngl,
        batch_size=batch,
    )


def _all_active_goal() -> GoalPreset:
    return GoalPreset(name="general", max_context=5, fastest_response=5,
                      throughput=5, long_context_rag=5)


def _only_throughput_goal() -> GoalPreset:
    return GoalPreset(name="tput_only", max_context=0, fastest_response=0,
                      throughput=20, long_context_rag=0)


# ---------------------------------------------------------------------------
# OrchestratorResult defaults
# ---------------------------------------------------------------------------

class TestOrchestratorResultDefaults:
    def test_defaults(self):
        result = OrchestratorResult()
        assert result.preset_results == {}
        assert result.scoring_result is None
        assert result.goal is None
        assert result.completed_presets == []
        assert result.interrupted is False


# ---------------------------------------------------------------------------
# Weight=0 presets are skipped
# ---------------------------------------------------------------------------

class TestWeightZeroSkip:
    def _run_with_goal(self, goal: GoalPreset, preset_results: dict) -> OrchestratorResult:
        """Run orchestrator with mocked PresetRunner.run() returning preset_results."""
        def fake_run(self_runner):
            return preset_results.get(
                self_runner.preset_cfg.name,
                _make_preset_result(self_runner.preset_cfg.name),
            )

        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                artifacts_dir="/tmp",
            )
            return orch.run()

    def test_zero_weight_preset_is_skipped_in_result(self):
        goal = _only_throughput_goal()
        preset_results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING),
        }
        result = self._run_with_goal(goal, preset_results)
        # max_context, fastest_response, long_context_rag have weight=0 → skipped=True
        assert result.preset_results[PRESET_MAX_CONTEXT].skipped is True
        assert result.preset_results[PRESET_FASTEST_RESPONSE].skipped is True
        assert result.preset_results[PRESET_LONG_CONTEXT_RAG].skipped is True

    def test_zero_weight_preset_not_in_completed(self):
        goal = _only_throughput_goal()
        preset_results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING),
        }
        result = self._run_with_goal(goal, preset_results)
        assert PRESET_MAX_CONTEXT not in result.completed_presets
        assert PRESET_FASTEST_RESPONSE not in result.completed_presets
        assert PRESET_LONG_CONTEXT_RAG not in result.completed_presets

    def test_active_preset_in_completed(self):
        goal = _only_throughput_goal()
        preset_results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING),
        }
        result = self._run_with_goal(goal, preset_results)
        assert PRESET_THROUGHPUT_KING in result.completed_presets

    def test_scoring_result_produced_for_active_presets(self):
        goal = _only_throughput_goal()
        preset_results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING),
        }
        result = self._run_with_goal(goal, preset_results)
        assert result.scoring_result is not None
        assert isinstance(result.scoring_result, ScoringResult)


# ---------------------------------------------------------------------------
# stop_event halts between presets
# ---------------------------------------------------------------------------

class TestStopEvent:
    def test_stop_event_before_second_preset_halts(self):
        """stop_event set immediately → only first active preset may run."""
        stop_event = threading.Event()

        run_count = {"n": 0}

        def fake_run(self_runner):
            run_count["n"] += 1
            # Set stop after first run so the orchestrator stops between presets
            stop_event.set()
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _all_active_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                stop_event=stop_event,
                artifacts_dir="/tmp",
            )
            result = orch.run()

        # Only max_context (first active preset) should have run
        assert run_count["n"] == 1
        assert result.interrupted is True

    def test_stop_event_already_set_skips_all_active(self):
        """If stop_event is already set before run(), no preset should execute."""
        stop_event = threading.Event()
        stop_event.set()

        run_count = {"n": 0}

        def fake_run(self_runner):
            run_count["n"] += 1
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _all_active_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                stop_event=stop_event,
                artifacts_dir="/tmp",
            )
            result = orch.run()

        assert run_count["n"] == 0
        assert result.interrupted is True

    def test_interrupted_flag_false_when_no_stop(self):
        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _only_throughput_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                artifacts_dir="/tmp",
            )
            result = orch.run()

        assert result.interrupted is False


# ---------------------------------------------------------------------------
# accumulated_results
# ---------------------------------------------------------------------------

class TestAccumulatedResults:
    def test_empty_before_run(self):
        orch = PresetBenchOrchestrator(
            base_cfg=_base_cfg(),
            goal=_only_throughput_goal(),
        )
        assert orch.accumulated_results == {}

    def test_returns_copy(self):
        """accumulated_results should return a copy, not the internal dict."""
        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _only_throughput_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                artifacts_dir="/tmp",
            )
            orch.run()
            acc = orch.accumulated_results
            acc["injected_key"] = None
            # Internal dict should not be modified
            assert "injected_key" not in orch._preset_results


# ---------------------------------------------------------------------------
# _ctx_ceiling_from_max_context
# ---------------------------------------------------------------------------

class TestCtxCeilingFromMaxContext:
    def _make_orch(self) -> PresetBenchOrchestrator:
        return PresetBenchOrchestrator(
            base_cfg=_base_cfg(),
            goal=_all_active_goal(),
        )

    def test_returns_none_when_no_max_context_result(self):
        orch = self._make_orch()
        assert orch._ctx_ceiling_from_max_context() is None

    def test_returns_none_when_max_context_skipped(self):
        orch = self._make_orch()
        orch._preset_results[PRESET_MAX_CONTEXT] = _make_preset_result(
            PRESET_MAX_CONTEXT, skipped=True
        )
        assert orch._ctx_ceiling_from_max_context() is None

    def test_returns_none_when_no_best_attempt(self):
        orch = self._make_orch()
        orch._preset_results[PRESET_MAX_CONTEXT] = PresetResult(
            preset_name=PRESET_MAX_CONTEXT,
            primary_metric="ctx",
            best_attempt=None,
        )
        assert orch._ctx_ceiling_from_max_context() is None

    def test_returns_ctx_from_best_attempt(self):
        orch = self._make_orch()
        attempt = _make_attempt(ctx=65536, ngl=45, batch=1536)
        pr = PresetResult(
            preset_name=PRESET_MAX_CONTEXT,
            primary_metric="ctx",
            attempts=[attempt],
            best_attempt=attempt,
            best_value=65536.0,
        )
        orch._preset_results[PRESET_MAX_CONTEXT] = pr
        assert orch._ctx_ceiling_from_max_context() == 65536


# ---------------------------------------------------------------------------
# Events emitted
# ---------------------------------------------------------------------------

class TestEventsEmitted:
    def test_orchestrator_preset_start_and_done_emitted(self):
        emitted = []

        def cb(event, data):
            emitted.append((event, data))

        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _only_throughput_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                event_cb=cb,
                artifacts_dir="/tmp",
            )
            orch.run()

        event_names = [e for e, _ in emitted]
        assert "orchestrator_preset_start" in event_names
        assert "orchestrator_preset_done" in event_names

    def test_scoring_complete_emitted_after_run(self):
        emitted = []

        def cb(event, data):
            emitted.append((event, data))

        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _only_throughput_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                event_cb=cb,
                artifacts_dir="/tmp",
            )
            orch.run()

        event_names = [e for e, _ in emitted]
        assert "scoring_complete" in event_names

    def test_event_data_contains_preset_name(self):
        emitted = []

        def cb(event, data):
            emitted.append((event, data))

        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _only_throughput_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                event_cb=cb,
                artifacts_dir="/tmp",
            )
            orch.run()

        start_events = [(e, d) for e, d in emitted if e == "orchestrator_preset_start"]
        assert any(d["preset"] == PRESET_THROUGHPUT_KING for _, d in start_events)

    def test_broken_event_cb_does_not_crash(self):
        """If the event callback raises, the orchestrator should continue."""
        def bad_cb(event, data):
            raise RuntimeError("callback exploded")

        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _only_throughput_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                event_cb=bad_cb,
                artifacts_dir="/tmp",
            )
            # Should not raise
            result = orch.run()

        assert PRESET_THROUGHPUT_KING in result.completed_presets


# ---------------------------------------------------------------------------
# Full run with all presets active
# ---------------------------------------------------------------------------

class TestFullRun:
    def test_all_presets_complete(self):
        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _all_active_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                artifacts_dir="/tmp",
            )
            result = orch.run()

        assert set(result.completed_presets) == set(ALL_PRESET_NAMES)
        assert result.interrupted is False

    def test_result_contains_all_preset_results(self):
        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _all_active_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                artifacts_dir="/tmp",
            )
            result = orch.run()

        for name in ALL_PRESET_NAMES:
            assert name in result.preset_results

    def test_goal_stored_in_result(self):
        def fake_run(self_runner):
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _all_active_goal()
        with patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                artifacts_dir="/tmp",
            )
            result = orch.run()

        assert result.goal is goal

    def test_long_context_rag_receives_ctx_ceiling(self):
        """long_context_rag runner should receive ctx_ceiling from max_context result."""
        ctx_ceiling_received = {}

        original_init = PresetRunner.__init__

        def capturing_init(self_runner, preset_cfg, base_cfg, **kwargs):
            original_init(self_runner, preset_cfg=preset_cfg, base_cfg=base_cfg, **kwargs)
            if preset_cfg.name == PRESET_LONG_CONTEXT_RAG:
                ctx_ceiling_received["value"] = kwargs.get("ctx_ceiling")

        def fake_run(self_runner):
            if self_runner.preset_cfg.name == PRESET_MAX_CONTEXT:
                attempt = _make_attempt(ctx=65536, ngl=45, batch=1536)
                return PresetResult(
                    preset_name=PRESET_MAX_CONTEXT,
                    primary_metric="ctx",
                    attempts=[attempt],
                    best_attempt=attempt,
                    best_value=65536.0,
                )
            return _make_preset_result(self_runner.preset_cfg.name)

        goal = _all_active_goal()
        with patch("llama_bench.orchestrator.PresetRunner.__init__", capturing_init), \
             patch("llama_bench.orchestrator.PresetRunner.run", fake_run):
            orch = PresetBenchOrchestrator(
                base_cfg=_base_cfg(),
                goal=goal,
                artifacts_dir="/tmp",
            )
            orch.run()

        assert ctx_ceiling_received.get("value") == 65536
