"""Orchestrator for the 4-preset bench characterization system.

:class:`PresetBenchOrchestrator` runs the four presets in the canonical order:

    1. max_context        (determines ctx ceiling for long_context_rag)
    2. fastest_response   (ngl×batch grid, TTFT-optimised)
    3. throughput_king    (ngl×batch grid, tok/s-optimised)
    4. long_context_rag   (large-ctx TTFT, uses max_context ceiling)

Presets with ``GoalPreset.weight == 0`` are silently skipped.

After all presets complete (or after a graceful stop), the orchestrator passes
the collected :class:`~llama_bench.presets.PresetResult` objects to
:func:`~llama_bench.scoring.score_presets` and returns an
:class:`OrchestratorResult`.
"""
from __future__ import annotations

import logging
import os
import threading
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from llama_bench.presets import (
    ALL_PRESET_NAMES,
    PRESET_LONG_CONTEXT_RAG,
    PRESET_MAX_CONTEXT,
    PRESET_REGISTRY,
    GoalPreset,
    PresetResult,
    PresetRunner,
)
from llama_bench.scoring import ScoringResult, score_presets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OrchestratorResult
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    """Aggregated output of a full (or partial) preset bench run.

    Attributes:
        preset_results:     Mapping from preset name → :class:`PresetResult`.
                            Only contains presets that were actually executed.
        scoring_result:     Weighted scoring output; ``None`` if scoring could
                            not be computed (e.g. all presets were skipped or
                            interrupted before any results were collected).
        goal:               The :class:`GoalPreset` used to weight the run.
        completed_presets:  Names of presets that finished successfully.
        interrupted:        ``True`` if the run was stopped early (stop_event
                            was set before all active presets completed).
    """

    preset_results: dict[str, PresetResult] = field(default_factory=dict)
    scoring_result: Optional[ScoringResult] = None
    goal: Optional[GoalPreset] = None
    completed_presets: list[str] = field(default_factory=list)
    interrupted: bool = False


# ---------------------------------------------------------------------------
# PresetBenchOrchestrator
# ---------------------------------------------------------------------------


class PresetBenchOrchestrator:
    """Run all four benchmark presets and produce a weighted scoring result.

    Args:
        base_cfg:       Base :class:`~llama_bench.config.BenchConfig`.  Each
                        preset runner receives a deep copy of this config and
                        overrides the parameters it sweeps.
        goal:           :class:`GoalPreset` specifying how to weight each
                        preset dimension.  Presets with weight 0 are skipped.
        event_cb:       Optional callback ``(event: str, data: dict) -> None``
                        forwarded to every :class:`~llama_bench.presets.PresetRunner`
                        and used to emit orchestrator-level events
                        (``preset_phase_start``, ``preset_phase_done``,
                        ``scoring_complete``).
        stop_event:     A :class:`threading.Event`-compatible object.  When
                        set, the orchestrator finishes the current preset and
                        then stops, marking the result as ``interrupted=True``.
        artifacts_dir:  Directory for server logs / JSONL files.
        log_file:       Path to the tool's own log file (passed through to
                        each :class:`~llama_bench.tuner.AdaptiveTuner`).
        ngl_values:     ngl values swept by the ngl×batch-grid presets
                        (``fastest_response``, ``throughput_king``).
                        ``None`` → use ``base_cfg.n_gpu_layers`` only.
        batch_values:   batch_size values swept by the ngl×batch-grid presets.
                        ``None`` → use ``base_cfg.batch_size`` only.
    """

    def __init__(
        self,
        base_cfg: "BenchConfig",  # type: ignore[name-defined]  # noqa: F821
        goal: GoalPreset,
        event_cb: Optional[Callable[[str, dict], None]] = None,
        stop_event: Optional[threading.Event] = None,
        artifacts_dir: str = "results",
        log_file: Optional[str] = None,
        ngl_values: Optional[list[int]] = None,
        batch_values: Optional[list[int]] = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.goal = goal
        self.event_cb = event_cb
        self.stop_event = stop_event
        self.artifacts_dir = artifacts_dir
        self.log_file = log_file
        self.ngl_values = ngl_values
        self.batch_values = batch_values

        # Partial results — accumulated even on interrupt
        self._preset_results: dict[str, PresetResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> OrchestratorResult:
        """Execute all active presets in order and return the final result.

        Execution order:
            1. ``max_context``      (provides ctx_ceiling for step 4)
            2. ``fastest_response``
            3. ``throughput_king``
            4. ``long_context_rag`` (uses ctx_ceiling from step 1)

        Returns:
            :class:`OrchestratorResult` with scoring results and per-preset
            details.  If the stop_event fires mid-run, ``interrupted=True``
            and ``scoring_result`` reflects only the completed presets.
        """
        interrupted = False
        # Emit base settings so the TUI Settings panel populates immediately.
        self._emit(
            "bench_settings",
            model=os.path.basename(self.base_cfg.model_path),
            model_path=self.base_cfg.model_path,
            ctx_max=self.base_cfg.ctx,
            ctx_min=self.base_cfg.ctx,
            ngl_initial=self.base_cfg.n_gpu_layers,
            batch=self.base_cfg.batch_size,
            ubatch=self.base_cfg.ubatch_size,
            flash_attn=self.base_cfg.flash_attn,
            kv_unified=self.base_cfg.kv_unified,
            cache_reuse=self.base_cfg.cache_reuse,
            cache_type_k=self.base_cfg.cache_type_k,
            cache_type_v=self.base_cfg.cache_type_v,
            split_mode=self.base_cfg.split_mode,
            vk_devices=self.base_cfg.vk_visible_devices,
            np=self.base_cfg.np,
            threads=self.base_cfg.threads,
            threads_batch=self.base_cfg.threads_batch,
            cont_batching=self.base_cfg.cont_batching,
            goal=self.goal.name,
            weights=self.goal.as_dict(),
        )
        ctx_ceiling: Optional[int] = None
        completed: list[str] = []

        total_active = sum(
            1 for name in ALL_PRESET_NAMES if self.goal.weight_for(name) > 0
        )
        active_index = 0

        for preset_name in ALL_PRESET_NAMES:
            weight = self.goal.weight_for(preset_name)

            if weight == 0:
                logger.info("Skipping preset %r (weight=0)", preset_name)
                self._preset_results[preset_name] = PresetResult(
                    preset_name=preset_name,
                    primary_metric=PRESET_REGISTRY[preset_name].primary_metric,
                    skipped=True,
                )
                continue

            # Check stop_event between presets (not mid-preset)
            if self.stop_event is not None and self.stop_event.is_set():
                logger.info(
                    "Orchestrator: stop_event set before preset %r, halting.",
                    preset_name,
                )
                interrupted = True
                break

            active_index += 1
            self._emit(
                "orchestrator_preset_start",
                preset=preset_name,
                phase_index=active_index,
                total_phases=total_active,
            )

            logger.info(
                "Running preset %r (%d/%d)", preset_name, active_index, total_active
            )

            # Build ctx_ceiling for long_context_rag from max_context results
            if preset_name == PRESET_LONG_CONTEXT_RAG:
                ctx_ceiling = self._ctx_ceiling_from_max_context()

            runner = PresetRunner(
                preset_cfg=PRESET_REGISTRY[preset_name],
                base_cfg=self.base_cfg,
                artifacts_dir=self.artifacts_dir,
                log_file=self.log_file,
                event_cb=self.event_cb,
                stop_event=self.stop_event,
                ngl_values=(
                    self.ngl_values
                    if PRESET_REGISTRY[preset_name].sweep_ngl_batch
                    else None
                ),
                batch_values=(
                    self.batch_values
                    if PRESET_REGISTRY[preset_name].sweep_ngl_batch
                    else None
                ),
                ctx_ceiling=ctx_ceiling if preset_name == PRESET_LONG_CONTEXT_RAG else None,
            )

            result = runner.run()
            self._preset_results[preset_name] = result
            completed.append(preset_name)

            logger.info(
                "Preset %r done. best_value=%s skipped=%s",
                preset_name,
                result.best_value,
                result.skipped,
            )

            self._emit(
                "orchestrator_preset_done",
                preset=preset_name,
                phase_index=active_index,
                total_phases=total_active,
                best_value=result.best_value,
                primary_metric=result.primary_metric,
                skipped=result.skipped,
            )

        # ---- Scoring --------------------------------------------------------
        scoring_result: Optional[ScoringResult] = None
        if self._preset_results:
            try:
                scoring_result = score_presets(
                    results=self._preset_results,
                    goal=self.goal,
                )
                opt = scoring_result.optimal
                best_config_dict = (
                    {
                        "ngl": opt.ngl,
                        "batch": opt.batch_size,
                        "final_score": opt.weighted_score,
                    }
                    if opt is not None
                    else None
                )
                self._emit(
                    "scoring_complete",
                    best_config=best_config_dict,
                    recommendation=scoring_result.recommendation,
                    scores=[
                        {
                            "ngl": cs.ngl,
                            "batch": cs.batch_size,
                            "final_score": cs.weighted_score,
                            "preset_scores": cs.preset_scores,
                            "normalized_scores": cs.normalized_scores,
                        }
                        for cs in scoring_result.config_scores[:5]
                    ],
                    goal=self.goal.name,
                    per_preset_ranges=scoring_result.per_preset_ranges,
                )
                logger.info(
                    "Scoring complete. Best config: ngl=%s batch=%s score=%.3f — %s",
                    scoring_result.optimal.ngl if scoring_result.optimal else None,
                    scoring_result.optimal.batch_size if scoring_result.optimal else None,
                    scoring_result.optimal.weighted_score if scoring_result.optimal else 0.0,
                    scoring_result.recommendation,
                )
            except Exception as exc:
                logger.warning("Scoring failed: %s", exc, exc_info=True)

        return OrchestratorResult(
            preset_results=self._preset_results,
            scoring_result=scoring_result,
            goal=self.goal,
            completed_presets=completed,
            interrupted=interrupted,
        )

    # ------------------------------------------------------------------
    # Partial results (for interrupt recovery)
    # ------------------------------------------------------------------

    @property
    def accumulated_results(self) -> dict[str, PresetResult]:
        """Return whatever preset results have been collected so far.

        Safe to call from a different thread while ``run()`` is active.
        """
        return dict(self._preset_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(self, event: str, **data: object) -> None:
        if self.event_cb is not None:
            try:
                self.event_cb(event, dict(data))
            except Exception:  # noqa: BLE001
                pass

    def _ctx_ceiling_from_max_context(self) -> Optional[int]:
        """Extract the highest successful ctx from the max_context preset result.

        Returns ``None`` if the max_context preset was skipped or produced no
        successful attempts.
        """
        mc_result = self._preset_results.get(PRESET_MAX_CONTEXT)
        if mc_result is None or mc_result.skipped:
            return None
        if mc_result.best_attempt is None:
            return None
        try:
            return int(getattr(mc_result.best_attempt, "ctx", 0)) or None
        except (TypeError, ValueError):
            return None
