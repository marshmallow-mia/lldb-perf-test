"""Scoring module for the 4-preset bench characterization system.

Given results from up to 4 preset runs and a GoalPreset (weight distribution),
this module computes a normalized weighted score per (ngl, batch) configuration
pair and identifies the optimal configuration for the user's use case.

Normalization:
    For each preset, scores are min-max normalized across all (ngl, batch)
    configs that appeared in that preset's results:

        normalized = (raw - min) / (max - min)

    For lower-is-better metrics (e.g. TTFT), the normalized value is inverted:

        normalized = 1 - (raw - min) / (max - min)

    When only one config succeeded in a preset (min == max), the normalized
    score is 1.0 (full credit for being the only successful config).

Weighted score:
    final_score(config) = Σ(weight_i × normalized_i) / GOAL_TOTAL_POINTS

    The result is in [0, 1].  A higher score is always better.

Optimal config:
    The (ngl, batch) pair with the highest final_score.  Ties are broken by
    preferring the config with the highest tokens_per_sec across all presets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llama_bench.presets import (
    ALL_PRESET_NAMES,
    GOAL_TOTAL_POINTS,
    PRESET_MAX_CONTEXT,
    PRESET_FASTEST_RESPONSE,
    PRESET_THROUGHPUT_KING,
    PRESET_LONG_CONTEXT_RAG,
    PRESET_REGISTRY,
    GoalPreset,
    PresetResult,
)


# ---------------------------------------------------------------------------
# Config key
# ---------------------------------------------------------------------------

ConfigKey = tuple[int, int]  # (n_gpu_layers, batch_size)


# ---------------------------------------------------------------------------
# Per-config score breakdown
# ---------------------------------------------------------------------------

@dataclass
class ConfigScore:
    """Scoring details for a single (ngl, batch) configuration.

    Attributes:
        ngl:              Number of GPU layers.
        batch_size:       Batch size.
        preset_scores:    Raw primary metric values per preset (None if the
                          preset was skipped or the config never succeeded in it).
        normalized_scores: Min-max normalized score per preset in [0, 1].
        weighted_score:   Final weighted score in [0, 1] (higher = better).
        contributing_attempts: Number of presets that contributed a score.
    """

    ngl: int
    batch_size: int
    preset_scores: dict[str, Optional[float]] = field(default_factory=dict)
    normalized_scores: dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    contributing_attempts: int = 0

    @property
    def config_key(self) -> ConfigKey:
        return (self.ngl, self.batch_size)


# ---------------------------------------------------------------------------
# Scoring result
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Output of :func:`score_presets`.

    Attributes:
        config_scores:     All config scores, sorted by weighted_score descending.
        optimal:           The best :class:`ConfigScore`, or None if no configs scored.
        goal:              The GoalPreset used for scoring.
        recommendation:    Human-readable recommendation string.
        per_preset_ranges: Raw value ranges used for normalization, keyed by
                           preset name.  Useful for debugging / reports.
    """

    config_scores: list[ConfigScore] = field(default_factory=list)
    optimal: Optional[ConfigScore] = None
    goal: Optional[GoalPreset] = None
    recommendation: str = ""
    per_preset_ranges: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

def _extract_per_config_values(
    results: dict[str, PresetResult],
    goal: GoalPreset,
) -> dict[str, dict[ConfigKey, float]]:
    """Extract the primary metric value per (ngl, batch) key per preset.

    For presets that sweep ngl×batch, each attempt already has distinct
    (n_gpu_layers, batch_size).  For presets that pin ngl/batch (max_context,
    long_context_rag), all attempts share the same key — the *best* value
    (per the preset's lower_is_better flag) is used.

    Returns:
        A dict:  preset_name → {ConfigKey → metric_value}
        Only presets with weight > 0 and at least one successful attempt
        are included.
    """
    out: dict[str, dict[ConfigKey, float]] = {}

    for preset_name in ALL_PRESET_NAMES:
        if goal.weight_for(preset_name) == 0:
            continue

        result = results.get(preset_name)
        if result is None or result.skipped:
            continue

        preset_def = PRESET_REGISTRY[preset_name]
        metric = preset_def.primary_metric
        lower_is_better = preset_def.lower_is_better

        per_config: dict[ConfigKey, float] = {}
        for attempt in result.attempts:
            if not attempt.success:
                continue
            key: ConfigKey = (attempt.n_gpu_layers, attempt.batch_size)
            value = float(getattr(attempt, metric, 0.0))
            if key not in per_config:
                per_config[key] = value
            else:
                # Keep best: lower for lower_is_better, higher otherwise
                if lower_is_better:
                    per_config[key] = min(per_config[key], value)
                else:
                    per_config[key] = max(per_config[key], value)

        if per_config:
            out[preset_name] = per_config

    return out


def _normalize(
    values: dict[ConfigKey, float],
    lower_is_better: bool,
) -> tuple[dict[ConfigKey, float], float, float]:
    """Min-max normalize a dict of raw values.

    Args:
        values:          Raw metric values keyed by ConfigKey.
        lower_is_better: Whether lower values are preferable.

    Returns:
        A 3-tuple of:
        - normalized dict (same keys, values in [0, 1] where 1 = best)
        - raw minimum value
        - raw maximum value
    """
    if not values:
        return {}, 0.0, 0.0

    raw_min = min(values.values())
    raw_max = max(values.values())
    spread = raw_max - raw_min

    normalized: dict[ConfigKey, float] = {}
    for key, v in values.items():
        if spread == 0.0:
            normalized[key] = 1.0  # only one distinct value → full credit
        else:
            norm = (v - raw_min) / spread
            normalized[key] = (1.0 - norm) if lower_is_better else norm

    return normalized, raw_min, raw_max


def score_presets(
    results: dict[str, PresetResult],
    goal: GoalPreset,
) -> ScoringResult:
    """Compute weighted scores for all (ngl, batch) configs.

    Args:
        results: Mapping of preset_name → PresetResult from an orchestrator run.
                 Missing entries are treated as "no data".
        goal:    The GoalPreset that defines the weight distribution.

    Returns:
        A :class:`ScoringResult` with all config scores sorted best-first.
    """
    # Step 1: extract raw per-config values for each active preset
    per_config_values = _extract_per_config_values(results, goal)

    # Step 2: normalize each preset independently
    per_preset_normalized: dict[str, dict[ConfigKey, float]] = {}
    per_preset_ranges: dict[str, dict] = {}

    for preset_name, cfg_values in per_config_values.items():
        preset_def = PRESET_REGISTRY[preset_name]
        normalized, raw_min, raw_max = _normalize(cfg_values, preset_def.lower_is_better)
        per_preset_normalized[preset_name] = normalized
        per_preset_ranges[preset_name] = {
            "raw_min": raw_min,
            "raw_max": raw_max,
            "lower_is_better": preset_def.lower_is_better,
            "metric": preset_def.primary_metric,
            "n_configs": len(cfg_values),
        }

    # Step 3: collect all config keys that appeared in any preset
    all_keys: set[ConfigKey] = set()
    for cfg_values in per_config_values.values():
        all_keys.update(cfg_values.keys())

    # Step 4: compute weighted score per config
    config_scores: list[ConfigScore] = []
    for key in sorted(all_keys):  # deterministic order
        ngl, batch = key
        cs = ConfigScore(ngl=ngl, batch_size=batch)

        weighted_sum = 0.0
        contributing = 0

        for preset_name in ALL_PRESET_NAMES:
            weight = goal.weight_for(preset_name)
            if weight == 0:
                cs.preset_scores[preset_name] = None
                cs.normalized_scores[preset_name] = 0.0
                continue

            raw_vals = per_config_values.get(preset_name, {})
            norm_vals = per_preset_normalized.get(preset_name, {})

            if key in raw_vals:
                cs.preset_scores[preset_name] = raw_vals[key]
                norm_val = norm_vals.get(key, 0.0)
                cs.normalized_scores[preset_name] = norm_val
                weighted_sum += weight * norm_val
                contributing += 1
            else:
                # This config wasn't tested in this preset — zero contribution
                cs.preset_scores[preset_name] = None
                cs.normalized_scores[preset_name] = 0.0

        cs.weighted_score = weighted_sum / GOAL_TOTAL_POINTS
        cs.contributing_attempts = contributing
        config_scores.append(cs)

    # Step 5: sort by weighted score descending, tie-break by n_gpu_layers desc
    config_scores.sort(key=lambda s: (-s.weighted_score, -s.ngl, -s.batch_size))

    optimal = config_scores[0] if config_scores else None

    return ScoringResult(
        config_scores=config_scores,
        optimal=optimal,
        goal=goal,
        recommendation=_generate_recommendation(optimal, results, goal) if optimal else "",
        per_preset_ranges=per_preset_ranges,
    )


# ---------------------------------------------------------------------------
# Recommendation text
# ---------------------------------------------------------------------------

def _generate_recommendation(
    optimal: ConfigScore,
    results: dict[str, PresetResult],
    goal: GoalPreset,
) -> str:
    """Generate a human-readable recommendation string."""
    lines: list[str] = []
    lines.append(
        f"Optimal config: ngl={optimal.ngl}, batch={optimal.batch_size} "
        f"(score={optimal.weighted_score:.3f}/1.000)"
    )

    # Summarize what each active preset contributed
    detail_parts: list[str] = []
    for preset_name in ALL_PRESET_NAMES:
        weight = goal.weight_for(preset_name)
        if weight == 0:
            continue
        raw = optimal.preset_scores.get(preset_name)
        norm = optimal.normalized_scores.get(preset_name, 0.0)
        if raw is None:
            detail_parts.append(f"  {preset_name}: not tested")
        else:
            metric = PRESET_REGISTRY[preset_name].primary_metric
            unit = "ctx" if metric == "ctx" else ("s" if "ttft" in metric else "tok/s")
            detail_parts.append(
                f"  {preset_name} (weight={weight}/20): "
                f"{metric}={raw:.1f}{unit}, normalized={norm:.3f}"
            )
    if detail_parts:
        lines.append("Goal breakdown:")
        lines.extend(detail_parts)

    # Dominant dimension warning
    dominant = max(ALL_PRESET_NAMES, key=lambda p: goal.weight_for(p))
    dominant_weight = goal.weight_for(dominant)
    if dominant_weight >= 12:
        lines.append(
            f"Note: {dominant!r} carries {dominant_weight}/20 points "
            f"and dominates this recommendation."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: find best attempt for a given config key across all presets
# ---------------------------------------------------------------------------

def find_best_overall_attempt(
    results: dict[str, PresetResult],
    config_key: ConfigKey,
) -> Optional[object]:  # Optional[TuneAttempt]
    """Return the TuneAttempt for *config_key* with the highest tokens/s.

    Searches all presets.  Used as a tie-breaker when multiple configs have
    equal weighted scores.

    Returns None if no successful attempt for this config was found.
    """
    best = None
    best_tps: float = -1.0

    for result in results.values():
        for attempt in result.attempts:
            if not attempt.success:
                continue
            key: ConfigKey = (attempt.n_gpu_layers, attempt.batch_size)
            if key != config_key:
                continue
            tps = float(getattr(attempt, "tokens_per_sec", 0.0))
            if tps > best_tps:
                best_tps = tps
                best = attempt

    return best
