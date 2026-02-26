"""Tests for llama_bench.presets — GoalPreset, PresetResult, PresetRunner helpers."""
from __future__ import annotations

import itertools

import pytest

from llama_bench.config import BenchConfig
from llama_bench.tuner import TuneAttempt
from llama_bench.presets import (
    ALL_PRESET_NAMES,
    BUILTIN_GOAL_NAMES,
    BUILTIN_GOALS,
    GOAL_TOTAL_POINTS,
    PRESET_FASTEST_RESPONSE,
    PRESET_LONG_CONTEXT_RAG,
    PRESET_MAX_CONTEXT,
    PRESET_REGISTRY,
    PRESET_THROUGHPUT_KING,
    GoalPreset,
    PresetConfig,
    PresetResult,
    PresetRunner,
    get_goal_preset,
)


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


def _base_cfg(ngl: int = 45, batch: int = 1536, ctx: int = 8192) -> BenchConfig:
    return BenchConfig(
        model_path="/tmp/model.gguf",
        ctx=ctx,
        n_gpu_layers=ngl,
        batch_size=batch,
    )


# ---------------------------------------------------------------------------
# PRESET_REGISTRY sanity
# ---------------------------------------------------------------------------

class TestPresetRegistry:
    def test_all_preset_names_in_registry(self):
        for name in ALL_PRESET_NAMES:
            assert name in PRESET_REGISTRY

    def test_preset_configs_have_required_fields(self):
        for name, pc in PRESET_REGISTRY.items():
            assert isinstance(pc.primary_metric, str)
            assert isinstance(pc.lower_is_better, bool)
            assert isinstance(pc.max_tokens, int)
            assert isinstance(pc.n_followups, int)

    def test_max_context_not_sweep_ngl_batch(self):
        pc = PRESET_REGISTRY[PRESET_MAX_CONTEXT]
        assert pc.sweep_ngl_batch is False

    def test_fastest_response_sweep_ngl_batch(self):
        pc = PRESET_REGISTRY[PRESET_FASTEST_RESPONSE]
        assert pc.sweep_ngl_batch is True

    def test_throughput_king_sweep_ngl_batch(self):
        pc = PRESET_REGISTRY[PRESET_THROUGHPUT_KING]
        assert pc.sweep_ngl_batch is True

    def test_long_context_rag_not_sweep_ngl_batch(self):
        pc = PRESET_REGISTRY[PRESET_LONG_CONTEXT_RAG]
        assert pc.sweep_ngl_batch is False

    def test_fastest_response_lower_is_better(self):
        assert PRESET_REGISTRY[PRESET_FASTEST_RESPONSE].lower_is_better is True

    def test_throughput_king_higher_is_better(self):
        assert PRESET_REGISTRY[PRESET_THROUGHPUT_KING].lower_is_better is False

    def test_max_context_higher_is_better(self):
        assert PRESET_REGISTRY[PRESET_MAX_CONTEXT].lower_is_better is False

    def test_long_context_rag_lower_is_better(self):
        assert PRESET_REGISTRY[PRESET_LONG_CONTEXT_RAG].lower_is_better is True


# ---------------------------------------------------------------------------
# ALL_PRESET_NAMES
# ---------------------------------------------------------------------------

class TestAllPresetNames:
    def test_contains_exactly_four_entries(self):
        assert len(ALL_PRESET_NAMES) == 4

    def test_canonical_order(self):
        assert ALL_PRESET_NAMES == (
            PRESET_MAX_CONTEXT,
            PRESET_FASTEST_RESPONSE,
            PRESET_THROUGHPUT_KING,
            PRESET_LONG_CONTEXT_RAG,
        )


# ---------------------------------------------------------------------------
# GoalPreset
# ---------------------------------------------------------------------------

class TestGoalPreset:
    def test_weight_for_max_context(self):
        gp = GoalPreset(name="test", max_context=12, fastest_response=1,
                        throughput=3, long_context_rag=4)
        assert gp.weight_for(PRESET_MAX_CONTEXT) == 12

    def test_weight_for_fastest_response(self):
        gp = GoalPreset(name="test", max_context=12, fastest_response=1,
                        throughput=3, long_context_rag=4)
        assert gp.weight_for(PRESET_FASTEST_RESPONSE) == 1

    def test_weight_for_throughput_king(self):
        gp = GoalPreset(name="test", max_context=12, fastest_response=1,
                        throughput=3, long_context_rag=4)
        assert gp.weight_for(PRESET_THROUGHPUT_KING) == 3

    def test_weight_for_long_context_rag(self):
        gp = GoalPreset(name="test", max_context=12, fastest_response=1,
                        throughput=3, long_context_rag=4)
        assert gp.weight_for(PRESET_LONG_CONTEXT_RAG) == 4

    def test_weight_for_unknown_raises(self):
        gp = GoalPreset(name="test", max_context=5, fastest_response=5,
                        throughput=5, long_context_rag=5)
        with pytest.raises(KeyError):
            gp.weight_for("nonexistent_preset")

    def test_active_presets_excludes_zero_weight(self):
        gp = GoalPreset(name="test", max_context=20, fastest_response=0,
                        throughput=0, long_context_rag=0)
        assert gp.active_presets() == [PRESET_MAX_CONTEXT]

    def test_active_presets_all_active(self):
        gp = GoalPreset(name="test", max_context=5, fastest_response=5,
                        throughput=5, long_context_rag=5)
        assert set(gp.active_presets()) == set(ALL_PRESET_NAMES)

    def test_as_dict_returns_all_four_keys(self):
        gp = GoalPreset(name="test", max_context=5, fastest_response=5,
                        throughput=5, long_context_rag=5)
        d = gp.as_dict()
        assert set(d.keys()) == set(ALL_PRESET_NAMES)

    def test_as_dict_values_match(self):
        gp = GoalPreset(name="test", max_context=12, fastest_response=1,
                        throughput=3, long_context_rag=4)
        d = gp.as_dict()
        assert d[PRESET_MAX_CONTEXT] == 12
        assert d[PRESET_FASTEST_RESPONSE] == 1
        assert d[PRESET_THROUGHPUT_KING] == 3
        assert d[PRESET_LONG_CONTEXT_RAG] == 4


# ---------------------------------------------------------------------------
# BUILTIN_GOALS
# ---------------------------------------------------------------------------

class TestBuiltinGoals:
    def test_all_five_goals_present(self):
        expected = {"reverse_engineering", "coding", "chatting", "rag_research", "general"}
        assert set(BUILTIN_GOALS.keys()) == expected

    def test_all_builtin_goals_sum_to_20(self):
        for name, gp in BUILTIN_GOALS.items():
            total = (gp.max_context + gp.fastest_response +
                     gp.throughput + gp.long_context_rag)
            assert total == GOAL_TOTAL_POINTS, (
                f"BUILTIN_GOALS[{name!r}] sums to {total}, expected {GOAL_TOTAL_POINTS}"
            )

    def test_reverse_engineering_weights(self):
        gp = BUILTIN_GOALS["reverse_engineering"]
        assert gp.weight_for(PRESET_MAX_CONTEXT) == 12

    def test_coding_fastest_response_high(self):
        gp = BUILTIN_GOALS["coding"]
        assert gp.weight_for(PRESET_FASTEST_RESPONSE) == 10

    def test_rag_research_long_context_rag_high(self):
        gp = BUILTIN_GOALS["rag_research"]
        assert gp.weight_for(PRESET_LONG_CONTEXT_RAG) == 12

    def test_general_all_equal(self):
        gp = BUILTIN_GOALS["general"]
        weights = [gp.weight_for(n) for n in ALL_PRESET_NAMES]
        assert all(w == 5 for w in weights)

    def test_builtin_goal_names_list(self):
        assert set(BUILTIN_GOAL_NAMES) == set(BUILTIN_GOALS.keys())


# ---------------------------------------------------------------------------
# get_goal_preset
# ---------------------------------------------------------------------------

class TestGetGoalPreset:
    def test_returns_builtin_by_name(self):
        gp = get_goal_preset("coding")
        assert gp.name == "coding"
        assert gp.weight_for(PRESET_FASTEST_RESPONSE) == 10

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_goal_preset("does_not_exist")

    def test_custom_with_all_weights(self):
        gp = get_goal_preset(
            "custom",
            w_max_context=8,
            w_fastest_response=6,
            w_throughput=4,
            w_long_context_rag=2,
        )
        assert gp.name == "custom"
        assert gp.weight_for(PRESET_MAX_CONTEXT) == 8

    def test_custom_missing_weights_raises(self):
        with pytest.raises(TypeError):
            get_goal_preset("custom")

    def test_custom_bad_sum_raises(self):
        with pytest.raises(ValueError):
            get_goal_preset(
                "custom",
                w_max_context=5,
                w_fastest_response=5,
                w_throughput=5,
                w_long_context_rag=4,  # sums to 19
            )


# ---------------------------------------------------------------------------
# PresetResult defaults
# ---------------------------------------------------------------------------

class TestPresetResult:
    def test_default_skipped_false(self):
        pr = PresetResult(preset_name=PRESET_THROUGHPUT_KING,
                          primary_metric="tokens_per_sec")
        assert pr.skipped is False

    def test_default_attempts_empty(self):
        pr = PresetResult(preset_name=PRESET_THROUGHPUT_KING,
                          primary_metric="tokens_per_sec")
        assert pr.attempts == []

    def test_default_best_attempt_none(self):
        pr = PresetResult(preset_name=PRESET_THROUGHPUT_KING,
                          primary_metric="tokens_per_sec")
        assert pr.best_attempt is None

    def test_default_best_value_none(self):
        pr = PresetResult(preset_name=PRESET_THROUGHPUT_KING,
                          primary_metric="tokens_per_sec")
        assert pr.best_value is None

    def test_successful_attempts_filters(self):
        a1 = _make_attempt(success=True)
        a2 = _make_attempt(success=False)
        pr = PresetResult(preset_name=PRESET_THROUGHPUT_KING,
                          primary_metric="tokens_per_sec",
                          attempts=[a1, a2])
        assert pr.successful_attempts() == [a1]


# ---------------------------------------------------------------------------
# PresetRunner helpers
# ---------------------------------------------------------------------------

class TestPresetRunnerNglBatchCombos:
    def _make_runner(self, preset_name=PRESET_FASTEST_RESPONSE,
                     ngl_values=None, batch_values=None,
                     ngl=45, batch=1536) -> PresetRunner:
        return PresetRunner(
            preset_cfg=PRESET_REGISTRY[preset_name],
            base_cfg=_base_cfg(ngl=ngl, batch=batch),
            ngl_values=ngl_values,
            batch_values=batch_values,
        )

    def test_no_override_uses_base_cfg(self):
        runner = self._make_runner(ngl=45, batch=1536)
        combos = runner._ngl_batch_combos()
        assert combos == [(45, 1536)]

    def test_ngl_override_only(self):
        runner = self._make_runner(ngl_values=[40, 45], batch_values=None, batch=1536)
        combos = runner._ngl_batch_combos()
        assert set(combos) == {(40, 1536), (45, 1536)}

    def test_batch_override_only(self):
        runner = self._make_runner(ngl_values=None, batch_values=[1024, 1536], ngl=45)
        combos = runner._ngl_batch_combos()
        assert set(combos) == {(45, 1024), (45, 1536)}

    def test_both_overrides_cartesian_product(self):
        runner = self._make_runner(ngl_values=[40, 45], batch_values=[1024, 1536])
        combos = runner._ngl_batch_combos()
        expected = list(itertools.product([40, 45], [1024, 1536]))
        assert set(combos) == set(expected)
        assert len(combos) == 4

    def test_single_ngl_single_batch(self):
        runner = self._make_runner(ngl_values=[45], batch_values=[1536])
        combos = runner._ngl_batch_combos()
        assert combos == [(45, 1536)]


class TestPresetRunnerEffectiveCtxForRag:
    def _make_rag_runner(self, ctx_ceiling=None) -> PresetRunner:
        return PresetRunner(
            preset_cfg=PRESET_REGISTRY[PRESET_LONG_CONTEXT_RAG],
            base_cfg=_base_cfg(),
            ctx_ceiling=ctx_ceiling,
        )

    def test_no_ceiling_returns_max_candidate(self):
        runner = self._make_rag_runner(ctx_ceiling=None)
        # ctx_candidates = [32768, 65536, 131072]
        assert runner._effective_ctx_for_rag() == 131072

    def test_ceiling_filters_candidates(self):
        runner = self._make_rag_runner(ctx_ceiling=65536)
        # Only 32768 and 65536 are <= 65536
        assert runner._effective_ctx_for_rag() == 65536

    def test_ceiling_below_all_candidates_uses_max_candidate(self):
        runner = self._make_rag_runner(ctx_ceiling=16384)
        # No candidates <= 16384 → falls back to max(candidates) = 131072
        assert runner._effective_ctx_for_rag() == 131072

    def test_ceiling_exactly_equal_to_candidate(self):
        runner = self._make_rag_runner(ctx_ceiling=32768)
        assert runner._effective_ctx_for_rag() == 32768
