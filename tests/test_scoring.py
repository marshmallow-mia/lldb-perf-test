"""Tests for llama_bench.scoring — normalization, weighted scoring, and ranking."""
from __future__ import annotations

import pytest

from llama_bench.config import BenchConfig
from llama_bench.tuner import TuneAttempt
from llama_bench.presets import (
    GoalPreset,
    PresetResult,
    BUILTIN_GOALS,
    PRESET_REGISTRY,
    ALL_PRESET_NAMES,
    PRESET_MAX_CONTEXT,
    PRESET_FASTEST_RESPONSE,
    PRESET_THROUGHPUT_KING,
    PRESET_LONG_CONTEXT_RAG,
)
from llama_bench.scoring import (
    ConfigScore,
    ScoringResult,
    score_presets,
    _normalize,
    _extract_per_config_values,
    find_best_overall_attempt,
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
    failure_reason: str | None = None,
) -> TuneAttempt:
    cfg = BenchConfig(model_path="/tmp/model.gguf", ctx=ctx,
                      n_gpu_layers=ngl, batch_size=batch)
    return TuneAttempt(
        config=cfg,
        success=success,
        failure_reason=failure_reason,
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
    attempts: list | None = None,
    skipped: bool = False,
) -> PresetResult:
    pr = PresetResult(
        preset_name=preset_name,
        primary_metric=PRESET_REGISTRY[preset_name].primary_metric,
        skipped=skipped,
    )
    if attempts:
        pr.attempts = attempts
        successful = [a for a in attempts if a.success]
        if successful:
            metric = pr.primary_metric
            lower_is_better = PRESET_REGISTRY[preset_name].lower_is_better
            if lower_is_better:
                pr.best_attempt = min(successful, key=lambda a: getattr(a, metric, float("inf")))
            else:
                pr.best_attempt = max(successful, key=lambda a: getattr(a, metric, 0.0))
            pr.best_value = float(getattr(pr.best_attempt, metric, 0.0))
    return pr


def _goal(max_context=5, fastest_response=5, throughput=5, long_context_rag=5):
    return GoalPreset(
        name="test",
        max_context=max_context,
        fastest_response=fastest_response,
        throughput=throughput,
        long_context_rag=long_context_rag,
    )


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_higher_is_better_basic(self):
        values = {(45, 1536): 10.0, (41, 1024): 20.0}
        normalized, raw_min, raw_max = _normalize(values, lower_is_better=False)
        assert raw_min == pytest.approx(10.0)
        assert raw_max == pytest.approx(20.0)
        # (10 - 10) / (20 - 10) = 0.0; (20 - 10) / (20 - 10) = 1.0
        assert normalized[(45, 1536)] == pytest.approx(0.0)
        assert normalized[(41, 1024)] == pytest.approx(1.0)

    def test_lower_is_better_inverts(self):
        values = {(45, 1536): 1.0, (41, 1024): 2.0}
        normalized, _, _ = _normalize(values, lower_is_better=True)
        # lower raw → higher normalized
        assert normalized[(45, 1536)] == pytest.approx(1.0)
        assert normalized[(41, 1024)] == pytest.approx(0.0)

    def test_single_value_returns_full_credit(self):
        values = {(45, 1536): 5.0}
        normalized, raw_min, raw_max = _normalize(values, lower_is_better=False)
        assert normalized[(45, 1536)] == pytest.approx(1.0)
        assert raw_min == pytest.approx(5.0)
        assert raw_max == pytest.approx(5.0)

    def test_all_same_values_all_get_full_credit(self):
        values = {(45, 1536): 10.0, (41, 1024): 10.0, (37, 512): 10.0}
        normalized, _, _ = _normalize(values, lower_is_better=False)
        for v in normalized.values():
            assert v == pytest.approx(1.0)

    def test_empty_dict_returns_empty(self):
        normalized, raw_min, raw_max = _normalize({}, lower_is_better=False)
        assert normalized == {}
        assert raw_min == pytest.approx(0.0)
        assert raw_max == pytest.approx(0.0)

    def test_normalized_values_in_zero_to_one_range(self):
        values = {(i, 1536): float(i * 10) for i in range(1, 11)}
        normalized, _, _ = _normalize(values, lower_is_better=False)
        for v in normalized.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# score_presets — basic functionality
# ---------------------------------------------------------------------------

class TestScorePresetsBasic:
    def test_returns_scoring_result(self):
        attempts = [_make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0)]
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING, attempts),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        assert isinstance(sr, ScoringResult)

    def test_single_config_gets_full_score(self):
        """One config → normalized=1.0, weighted = weight/20."""
        attempts = [_make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0)]
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING, attempts),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        assert len(sr.config_scores) == 1
        assert sr.config_scores[0].weighted_score == pytest.approx(1.0)

    def test_weight_zero_skips_preset(self):
        """Preset with weight=0 should not contribute to any score."""
        attempts_tput = [_make_attempt(ngl=45, batch=1536, tokens_per_sec=50.0)]
        attempts_ttft = [_make_attempt(ngl=41, batch=1024, cold_ttft_s=0.5)]
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING, attempts_tput),
            PRESET_FASTEST_RESPONSE: _make_preset_result(PRESET_FASTEST_RESPONSE, attempts_ttft),
        }
        # throughput=0 → only fastest_response contributes
        goal = GoalPreset(name="test", max_context=0, fastest_response=20,
                         throughput=0, long_context_rag=0)
        sr = score_presets(results, goal)
        # Only fastest_response config (41, 1024) should appear
        keys = {(cs.ngl, cs.batch_size) for cs in sr.config_scores}
        assert (41, 1024) in keys
        # (45, 1536) appeared only in throughput (weight=0) → should not score
        assert (45, 1536) not in keys

    def test_no_successful_attempts_empty_result(self):
        attempts = [_make_attempt(success=False, failure_reason="out_of_vram")]
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING, attempts),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        assert sr.config_scores == []
        assert sr.optimal is None

    def test_empty_results_dict(self):
        goal = _goal()
        sr = score_presets({}, goal)
        assert sr.config_scores == []
        assert sr.optimal is None

    def test_skipped_preset_excluded(self):
        attempts = [_make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0)]
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING, attempts),
            PRESET_FASTEST_RESPONSE: PresetResult(
                preset_name=PRESET_FASTEST_RESPONSE,
                primary_metric=PRESET_REGISTRY[PRESET_FASTEST_RESPONSE].primary_metric,
                skipped=True,
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=5,
                         throughput=15, long_context_rag=0)
        sr = score_presets(results, goal)
        # skipped preset contributes nothing — still produces a result from throughput
        assert len(sr.config_scores) >= 1


# ---------------------------------------------------------------------------
# score_presets — normalization & ranking
# ---------------------------------------------------------------------------

class TestScorePresetsRanking:
    def test_higher_tps_scores_higher(self):
        """For throughput_king (higher=better), the config with more tok/s wins."""
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [
                    _make_attempt(ngl=45, batch=1536, tokens_per_sec=50.0),
                    _make_attempt(ngl=41, batch=1024, tokens_per_sec=10.0),
                ],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        assert sr.config_scores[0].ngl == 45
        assert sr.config_scores[0].batch_size == 1536
        assert sr.config_scores[0].weighted_score > sr.config_scores[1].weighted_score

    def test_lower_ttft_scores_higher(self):
        """For fastest_response (lower=better), the config with lower TTFT wins."""
        results = {
            PRESET_FASTEST_RESPONSE: _make_preset_result(
                PRESET_FASTEST_RESPONSE,
                [
                    _make_attempt(ngl=45, batch=1536, cold_ttft_s=0.5),
                    _make_attempt(ngl=41, batch=1024, cold_ttft_s=2.0),
                ],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=20,
                         throughput=0, long_context_rag=0)
        sr = score_presets(results, goal)
        # ngl=45 has lower TTFT → should rank first
        assert sr.config_scores[0].ngl == 45

    def test_optimal_is_first_config_score(self):
        """ScoringResult.optimal must equal config_scores[0]."""
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [
                    _make_attempt(ngl=45, batch=1536, tokens_per_sec=30.0),
                    _make_attempt(ngl=41, batch=1024, tokens_per_sec=10.0),
                ],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        assert sr.optimal is sr.config_scores[0]

    def test_all_same_value_all_get_equal_score(self):
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [
                    _make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0),
                    _make_attempt(ngl=41, batch=1024, tokens_per_sec=20.0),
                ],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        scores = [cs.weighted_score for cs in sr.config_scores]
        assert all(s == pytest.approx(scores[0]) for s in scores)

    def test_weighted_score_in_zero_to_one_range(self):
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [_make_attempt(ngl=i * 4, batch=1536, tokens_per_sec=float(i * 5))
                 for i in range(1, 6)],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        for cs in sr.config_scores:
            assert 0.0 <= cs.weighted_score <= 1.0

    def test_multi_preset_weighted_combination(self):
        """Two presets, equal weight, same config — should get 1.0."""
        results = {
            PRESET_FASTEST_RESPONSE: _make_preset_result(
                PRESET_FASTEST_RESPONSE,
                [_make_attempt(ngl=45, batch=1536, cold_ttft_s=1.0)],
            ),
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [_make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0)],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=10,
                         throughput=10, long_context_rag=0)
        sr = score_presets(results, goal)
        # single config → normalized = 1.0 for each preset → weighted = 1.0
        assert len(sr.config_scores) == 1
        assert sr.config_scores[0].weighted_score == pytest.approx(1.0)

    def test_config_missing_from_one_preset_penalized(self):
        """A config missing from an active preset gets 0 for that preset's contribution.

        Config A (45, 1536) appears in both presets.
        Config B (41, 1024) appears only in throughput_king.
        With fastest_response weighted heavily, A wins because B contributes nothing there.
        """
        results = {
            PRESET_FASTEST_RESPONSE: _make_preset_result(
                PRESET_FASTEST_RESPONSE,
                [_make_attempt(ngl=45, batch=1536, cold_ttft_s=0.5)],
            ),
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [
                    _make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0),
                    _make_attempt(ngl=41, batch=1024, tokens_per_sec=100.0),
                ],
            ),
        }
        # fastest_response weight=16, throughput weight=4
        # Config (45,1536): 1.0*16 + 0.0*4 = 16/20 = 0.80
        # Config (41,1024): 0.0*16 + 1.0*4 =  4/20 = 0.20
        goal = GoalPreset(name="test", max_context=0, fastest_response=16,
                         throughput=4, long_context_rag=0)
        sr = score_presets(results, goal)
        score_45 = next(cs for cs in sr.config_scores if cs.ngl == 45)
        score_41 = next(cs for cs in sr.config_scores if cs.ngl == 41)
        assert score_45.weighted_score > score_41.weighted_score
        # Verify (41,1024) has 0 on fastest_response because it was never tested
        assert score_41.normalized_scores[PRESET_FASTEST_RESPONSE] == pytest.approx(0.0)

# ---------------------------------------------------------------------------
# GoalPreset validation
# ---------------------------------------------------------------------------

class TestGoalPresetValidation:
    def test_valid_preset_does_not_raise(self):
        gp = GoalPreset(name="ok", max_context=5, fastest_response=5,
                        throughput=5, long_context_rag=5)
        assert gp.max_context == 5

    def test_weights_not_summing_to_20_raises(self):
        with pytest.raises(ValueError, match="20"):
            GoalPreset(name="bad", max_context=5, fastest_response=5,
                       throughput=5, long_context_rag=4)  # sums to 19

    def test_weights_summing_to_more_than_20_raises(self):
        with pytest.raises(ValueError):
            GoalPreset(name="bad", max_context=6, fastest_response=6,
                       throughput=6, long_context_rag=6)  # sums to 24

    def test_zero_weight_presets_valid(self):
        gp = GoalPreset(name="zero", max_context=20, fastest_response=0,
                        throughput=0, long_context_rag=0)
        assert gp.weight_for(PRESET_MAX_CONTEXT) == 20

    def test_builtin_goals_all_sum_to_20(self):
        for name, gp in BUILTIN_GOALS.items():
            total = (gp.max_context + gp.fastest_response +
                     gp.throughput + gp.long_context_rag)
            assert total == 20, f"BUILTIN_GOALS[{name!r}] sums to {total}"


# ---------------------------------------------------------------------------
# find_best_overall_attempt
# ---------------------------------------------------------------------------

class TestFindBestOverallAttempt:
    def test_returns_highest_tps(self):
        a_low = _make_attempt(ngl=45, batch=1536, tokens_per_sec=10.0)
        a_high = _make_attempt(ngl=45, batch=1536, tokens_per_sec=50.0)
        results = {
            PRESET_FASTEST_RESPONSE: _make_preset_result(
                PRESET_FASTEST_RESPONSE, [a_low]
            ),
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING, [a_high]
            ),
        }
        best = find_best_overall_attempt(results, config_key=(45, 1536))
        assert best is not None
        assert best.tokens_per_sec == pytest.approx(50.0)

    def test_returns_none_for_missing_config(self):
        attempts = [_make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0)]
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(PRESET_THROUGHPUT_KING, attempts),
        }
        best = find_best_overall_attempt(results, config_key=(41, 1024))
        assert best is None

    def test_skips_failed_attempts(self):
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [_make_attempt(ngl=45, batch=1536, success=False,
                               failure_reason="out_of_vram")],
            ),
        }
        best = find_best_overall_attempt(results, config_key=(45, 1536))
        assert best is None

    def test_empty_results_returns_none(self):
        best = find_best_overall_attempt({}, config_key=(45, 1536))
        assert best is None


# ---------------------------------------------------------------------------
# per_preset_ranges populated correctly
# ---------------------------------------------------------------------------

class TestPerPresetRanges:
    def test_ranges_populated_for_active_presets(self):
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [
                    _make_attempt(ngl=45, batch=1536, tokens_per_sec=10.0),
                    _make_attempt(ngl=41, batch=1024, tokens_per_sec=20.0),
                ],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        assert PRESET_THROUGHPUT_KING in sr.per_preset_ranges
        r = sr.per_preset_ranges[PRESET_THROUGHPUT_KING]
        assert r["raw_min"] == pytest.approx(10.0)
        assert r["raw_max"] == pytest.approx(20.0)
        assert r["lower_is_better"] is False
        assert r["n_configs"] == 2

    def test_weight_zero_preset_not_in_ranges(self):
        results = {
            PRESET_THROUGHPUT_KING: _make_preset_result(
                PRESET_THROUGHPUT_KING,
                [_make_attempt(ngl=45, batch=1536, tokens_per_sec=20.0)],
            ),
        }
        goal = GoalPreset(name="test", max_context=0, fastest_response=0,
                         throughput=20, long_context_rag=0)
        sr = score_presets(results, goal)
        assert PRESET_FASTEST_RESPONSE not in sr.per_preset_ranges
