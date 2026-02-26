"""Preset configurations for the 4-dimension bench characterization system.

Each preset sweeps its own parameter space independently to characterize
hardware+model performance across four distinct dimensions:

  1. max_context        — highest usable context window
  2. fastest_response   — lowest cold TTFT for interactive use
  3. throughput_king    — highest decode tokens/s for batch workloads
  4. long_context_rag   — TTFT at very large context (RAG / doc analysis)

A GoalPreset assigns weights (summing to exactly 20) across these four
dimensions. The scoring module uses these weights to compute a final
weighted score per (ngl, batch) configuration pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Preset name constants
# ---------------------------------------------------------------------------

PRESET_MAX_CONTEXT = "max_context"
PRESET_FASTEST_RESPONSE = "fastest_response"
PRESET_THROUGHPUT_KING = "throughput_king"
PRESET_LONG_CONTEXT_RAG = "long_context_rag"

ALL_PRESET_NAMES = (
    PRESET_MAX_CONTEXT,
    PRESET_FASTEST_RESPONSE,
    PRESET_THROUGHPUT_KING,
    PRESET_LONG_CONTEXT_RAG,
)


# ---------------------------------------------------------------------------
# PresetConfig — describes how a single preset should be run
# ---------------------------------------------------------------------------

@dataclass
class PresetConfig:
    """Static configuration for one benchmark preset.

    Attributes:
        name:              Unique preset identifier (one of ALL_PRESET_NAMES).
        description:       Human-readable description of what this preset measures.
        primary_metric:    Name of the TuneAttempt field used as the primary
                           scoring metric for this preset.
        lower_is_better:   True if a *lower* primary_metric value is better
                           (e.g. TTFT), False if higher is better (e.g. tok/s).
        fixed_ctx:         If not None, the context size is fixed (no ctx sweep).
        ctx_candidates:    Explicit list of ctx values to try (used by
                           max_context and long_context_rag presets).
        ctx_min:           Minimum ctx for OOM-adaptive sweep (max_context only).
        ctx_step:          Step size when sweeping ctx downward (max_context only).
        max_tokens:        Max tokens to generate per request.
        n_followups:       Number of follow-up turns (0 = single-turn only).
        prompt_builder:    Name of the prompt builder function to call (resolved
                           at runtime by PresetRunner from prompts.py).
        sweep_ngl_batch:   True if the preset sweeps the ngl×batch grid.
        ngl_values:        ngl values to sweep (None → use base_cfg value only).
        batch_values:      batch values to sweep (None → use base_cfg value only).
    """

    name: str
    description: str
    primary_metric: str
    lower_is_better: bool

    # Context handling
    fixed_ctx: Optional[int] = None
    ctx_candidates: Optional[list[int]] = None
    ctx_min: int = 8192
    ctx_step: int = 8192

    # Request shape
    max_tokens: int = 128
    n_followups: int = 0

    # Prompt builder (function name in prompts.py)
    prompt_builder: str = "build_fastest_response_prompt"

    # Grid sweep
    sweep_ngl_batch: bool = True
    ngl_values: Optional[list[int]] = None   # None → single value from base_cfg
    batch_values: Optional[list[int]] = None  # None → single value from base_cfg


# ---------------------------------------------------------------------------
# Built-in preset definitions
# ---------------------------------------------------------------------------

PRESET_REGISTRY: dict[str, PresetConfig] = {
    PRESET_MAX_CONTEXT: PresetConfig(
        name=PRESET_MAX_CONTEXT,
        description=(
            "Finds the highest usable context window for this model+hardware. "
            "Sweeps ctx from high to low, using OOM-adaptive retry logic. "
            "Primary metric: highest successful ctx."
        ),
        primary_metric="ctx",
        lower_is_better=False,
        # ctx_candidates=None → will be generated from ctx_max down to ctx_min
        ctx_min=8192,
        ctx_step=8192,
        max_tokens=32,
        n_followups=0,
        prompt_builder="build_max_context_prompt",
        sweep_ngl_batch=False,  # OOM retry handles ngl/batch adaptation internally
    ),
    PRESET_FASTEST_RESPONSE: PresetConfig(
        name=PRESET_FASTEST_RESPONSE,
        description=(
            "Finds the ngl×batch config that minimises cold TTFT. "
            "Fixed ctx=4096, short prompt (~300 tok), single response. "
            "Primary metric: cold TTFT (lower=better)."
        ),
        primary_metric="cold_ttft_s",
        lower_is_better=True,
        fixed_ctx=4096,
        max_tokens=128,
        n_followups=0,
        prompt_builder="build_fastest_response_prompt",
        sweep_ngl_batch=True,
    ),
    PRESET_THROUGHPUT_KING: PresetConfig(
        name=PRESET_THROUGHPUT_KING,
        description=(
            "Finds the ngl×batch config that maximises decode throughput. "
            "Fixed ctx=8192, long generation, single turn. "
            "Primary metric: tokens/s (higher=better)."
        ),
        primary_metric="tokens_per_sec",
        lower_is_better=False,
        fixed_ctx=8192,
        max_tokens=2048,
        n_followups=0,
        prompt_builder="build_throughput_prompt",
        sweep_ngl_batch=True,
    ),
    PRESET_LONG_CONTEXT_RAG: PresetConfig(
        name=PRESET_LONG_CONTEXT_RAG,
        description=(
            "Measures TTFT at large context sizes (32k / 64k / 128k) with a "
            "synthetically padded prompt that fills the context window. "
            "Uses max_context results to skip known-OOM sizes. "
            "Primary metric: TTFT at largest successful ctx (lower=better)."
        ),
        primary_metric="cold_ttft_s",
        lower_is_better=True,
        ctx_candidates=[32768, 65536, 131072],
        max_tokens=256,
        n_followups=1,
        prompt_builder="build_long_context_rag_prompt",
        sweep_ngl_batch=False,  # uses the best config found by fastest_response/throughput
    ),
}


# ---------------------------------------------------------------------------
# PresetResult — result from running a single preset
# ---------------------------------------------------------------------------

@dataclass
class PresetResult:
    """Aggregated result from one completed preset run.

    Attributes:
        preset_name:    Which preset produced this result.
        attempts:       All TuneAttempt records collected during the preset run.
        best_attempt:   The attempt selected as "best" for this preset's primary
                        metric, or None if no successful attempt exists.
        primary_metric: Name of the metric that was optimised.
        best_value:     Value of the primary metric for best_attempt (None if
                        no success).
        skipped:        True if the preset was skipped (weight=0).
    """

    preset_name: str
    attempts: list = field(default_factory=list)  # list[TuneAttempt]
    best_attempt: Optional[object] = None          # TuneAttempt | None
    primary_metric: str = ""
    best_value: Optional[float] = None
    skipped: bool = False

    def successful_attempts(self) -> list:
        """Return only the attempts where success=True."""
        return [a for a in self.attempts if a.success]


# ---------------------------------------------------------------------------
# GoalPreset — user-facing weight distribution
# ---------------------------------------------------------------------------

GOAL_TOTAL_POINTS = 20


@dataclass
class GoalPreset:
    """Weight distribution across the 4 presets.

    All weights must be non-negative integers that sum to exactly
    GOAL_TOTAL_POINTS (20). A weight of 0 means the preset is skipped.

    Attributes:
        name:              Identifier for this goal preset.
        max_context:       Points allocated to max_context dimension.
        fastest_response:  Points allocated to fastest_response dimension.
        throughput:        Points allocated to throughput_king dimension.
        long_context_rag:  Points allocated to long_context_rag dimension.
    """

    name: str
    max_context: int
    fastest_response: int
    throughput: int
    long_context_rag: int

    def __post_init__(self) -> None:
        total = self.max_context + self.fastest_response + self.throughput + self.long_context_rag
        if total != GOAL_TOTAL_POINTS:
            raise ValueError(
                f"GoalPreset '{self.name}': weights must sum to {GOAL_TOTAL_POINTS}, "
                f"got {total} (max_context={self.max_context}, "
                f"fastest_response={self.fastest_response}, "
                f"throughput={self.throughput}, "
                f"long_context_rag={self.long_context_rag})"
            )

    def weight_for(self, preset_name: str) -> int:
        """Return the weight for the given preset name."""
        mapping = {
            PRESET_MAX_CONTEXT: self.max_context,
            PRESET_FASTEST_RESPONSE: self.fastest_response,
            PRESET_THROUGHPUT_KING: self.throughput,
            PRESET_LONG_CONTEXT_RAG: self.long_context_rag,
        }
        if preset_name not in mapping:
            raise KeyError(f"Unknown preset name: {preset_name!r}")
        return mapping[preset_name]

    def active_presets(self) -> list[str]:
        """Return preset names with weight > 0, in execution order."""
        return [p for p in ALL_PRESET_NAMES if self.weight_for(p) > 0]

    def as_dict(self) -> dict[str, int]:
        """Return weights as a plain dict keyed by preset name."""
        return {
            PRESET_MAX_CONTEXT: self.max_context,
            PRESET_FASTEST_RESPONSE: self.fastest_response,
            PRESET_THROUGHPUT_KING: self.throughput,
            PRESET_LONG_CONTEXT_RAG: self.long_context_rag,
        }


# ---------------------------------------------------------------------------
# Built-in goal presets
# ---------------------------------------------------------------------------

BUILTIN_GOALS: dict[str, GoalPreset] = {
    "reverse_engineering": GoalPreset(
        name="reverse_engineering",
        max_context=12,
        fastest_response=1,
        throughput=3,
        long_context_rag=4,
    ),
    "coding": GoalPreset(
        name="coding",
        max_context=3,
        fastest_response=10,
        throughput=5,
        long_context_rag=2,
    ),
    "chatting": GoalPreset(
        name="chatting",
        max_context=3,
        fastest_response=8,
        throughput=5,
        long_context_rag=4,
    ),
    "rag_research": GoalPreset(
        name="rag_research",
        max_context=4,
        fastest_response=2,
        throughput=2,
        long_context_rag=12,
    ),
    "general": GoalPreset(
        name="general",
        max_context=5,
        fastest_response=5,
        throughput=5,
        long_context_rag=5,
    ),
}

BUILTIN_GOAL_NAMES = list(BUILTIN_GOALS.keys())


def get_goal_preset(
    name: str,
    *,
    w_max_context: Optional[int] = None,
    w_fastest_response: Optional[int] = None,
    w_throughput: Optional[int] = None,
    w_long_context_rag: Optional[int] = None,
) -> GoalPreset:
    """Resolve a GoalPreset by name, or build a custom one from weights.

    If name == "custom", all four weight keyword arguments must be provided
    and must sum to GOAL_TOTAL_POINTS.

    Args:
        name:               Built-in goal name or "custom".
        w_max_context:      Weight for max_context (required when name="custom").
        w_fastest_response: Weight for fastest_response (required when name="custom").
        w_throughput:       Weight for throughput_king (required when name="custom").
        w_long_context_rag: Weight for long_context_rag (required when name="custom").

    Returns:
        A GoalPreset instance (validated: weights sum to 20).

    Raises:
        ValueError: If name is unknown, or if "custom" weights don't sum to 20.
        TypeError:  If name="custom" but weights are missing.
    """
    if name == "custom":
        missing = [
            k for k, v in {
                "w_max_context": w_max_context,
                "w_fastest_response": w_fastest_response,
                "w_throughput": w_throughput,
                "w_long_context_rag": w_long_context_rag,
            }.items()
            if v is None
        ]
        if missing:
            raise TypeError(
                f"Custom goal requires all weight arguments. Missing: {missing}"
            )
        return GoalPreset(
            name="custom",
            max_context=w_max_context,       # type: ignore[arg-type]
            fastest_response=w_fastest_response,  # type: ignore[arg-type]
            throughput=w_throughput,         # type: ignore[arg-type]
            long_context_rag=w_long_context_rag,  # type: ignore[arg-type]
        )

    if name not in BUILTIN_GOALS:
        valid = ", ".join(sorted(BUILTIN_GOALS)) + ", custom"
        raise ValueError(f"Unknown goal preset {name!r}. Valid options: {valid}")

    return BUILTIN_GOALS[name]


# ---------------------------------------------------------------------------
# PresetRunner — executes a single preset and returns a PresetResult
# ---------------------------------------------------------------------------

import copy
import itertools
import logging
from typing import Callable


logger = logging.getLogger(__name__)


class PresetRunner:
    """Execute one benchmark preset and collect results.

    Delegates all server management, OOM-adaptive retry logic, and actual
    benchmarking to :class:`~llama_bench.tuner.AdaptiveTuner`.  The runner
    is responsible for:

    * Building the appropriate parameter sweep (ngl×batch grid or fixed).
    * Selecting the correct prompt builder for the preset.
    * Passing optional context-ceiling information (from a previous
      max_context run) to the long_context_rag preset.
    * Collecting all :class:`~llama_bench.tuner.TuneAttempt` records and
      packaging them into a :class:`PresetResult`.

    Args:
        preset_cfg:       Which preset to run.
        base_cfg:         Base :class:`~llama_bench.config.BenchConfig`.  The
                          runner will vary *n_gpu_layers*, *batch_size*, and
                          *ctx* around this baseline.
        artifacts_dir:    Directory for server logs / artifacts.
        log_file:         Path to the tool's own log file (passed through to
                          ``AdaptiveTuner`` for JSONL records).
        event_cb:         Optional callback ``(event: str, data: dict) -> None``
                          fired for TUI events.
        stop_event:       A :class:`threading.Event`-like object; when set,
                          stops the run after the current attempt.
        ngl_values:       ngl values to sweep.  If None, uses
                          ``base_cfg.n_gpu_layers`` only.
        batch_values:     batch_size values to sweep.  If None, uses
                          ``base_cfg.batch_size`` only.
        ctx_ceiling:      For the long_context_rag preset: skip ctx values
                          above this ceiling (learned from a prior
                          max_context run).  Ignored for other presets.
    """

    def __init__(
        self,
        preset_cfg: PresetConfig,
        base_cfg: "BenchConfig",
        artifacts_dir: str = "results",
        log_file: "Optional[str]" = None,
        event_cb: "Optional[Callable[[str, dict], None]]" = None,
        stop_event: "Optional[object]" = None,
        ngl_values: "Optional[list[int]]" = None,
        batch_values: "Optional[list[int]]" = None,
        ctx_ceiling: "Optional[int]" = None,
    ) -> None:
        self.preset_cfg = preset_cfg
        self.base_cfg = base_cfg
        self.artifacts_dir = artifacts_dir
        self.log_file = log_file
        self.event_cb = event_cb
        self.stop_event = stop_event
        self._ngl_values = ngl_values
        self._batch_values = batch_values
        self.ctx_ceiling = ctx_ceiling

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PresetResult:
        """Execute the preset and return the aggregated :class:`PresetResult`.

        Dispatches to the appropriate internal strategy based on
        ``self.preset_cfg.sweep_ngl_batch``.
        """
        self._emit(
            "preset_phase_start",
            preset=self.preset_cfg.name,
            description=self.preset_cfg.description,
        )

        if self.preset_cfg.sweep_ngl_batch:
            result = self._run_ngl_batch_grid()
        else:
            result = self._run_fixed_or_ctx_sweep()

        self._emit(
            "preset_phase_done",
            preset=self.preset_cfg.name,
            success=result.best_attempt is not None,
            best_value=result.best_value,
            primary_metric=result.primary_metric,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(self, event: str, **data: object) -> None:
        if self.event_cb is not None:
            try:
                self.event_cb(event, dict(data))
            except Exception:  # noqa: BLE001
                pass

    def _prompt_sequence(self) -> list[dict]:
        """Build the prompt sequence for this preset.

        Resolves the prompt builder name to the actual function in
        ``llama_bench.prompts`` and calls it.
        """
        import llama_bench.prompts as _prompts

        builder_name = self.preset_cfg.prompt_builder
        builder = getattr(_prompts, builder_name, None)
        if builder is None:
            raise ValueError(
                f"Preset {self.preset_cfg.name!r}: prompt builder \'s {builder_name!r} not found ", "in llama_bench.prompts"
            )

        if builder_name == "build_long_context_rag_prompt":
            # Calculate approximate padding needed to fill the context.
            # Subtract overhead for system prompt + question + response.
            ctx = self._effective_ctx_for_rag()
            overhead_tokens = 300  # system + question + max_tokens buffer
            padding_tokens = max(1000, ctx - overhead_tokens)
            return builder(padding_tokens=padding_tokens)

        return builder()

    def _effective_ctx_for_rag(self) -> int:
        """Return the context size to use for padding in the RAG preset."""
        # Use the largest ctx_candidate that is <= ctx_ceiling (if provided).
        candidates = self.preset_cfg.ctx_candidates or [32768]
        if self.ctx_ceiling is not None:
            valid = [c for c in candidates if c <= self.ctx_ceiling]
            if valid:
                return max(valid)
        return max(candidates)

    def _ngl_batch_combos(self) -> "list[tuple[int, int]]":
        """Return (ngl, batch) pairs to sweep."""
        ngls = self._ngl_values or [self.base_cfg.n_gpu_layers]
        batches = self._batch_values or [self.base_cfg.batch_size]
        return list(itertools.product(ngls, batches))

    def _make_cfg_for_combo(
        self, ngl: int, batch: int, ctx: int
    ) -> "BenchConfig":
        """Return a deep-copy of base_cfg with ngl/batch/ctx overridden."""
        cfg = copy.deepcopy(self.base_cfg)
        cfg.n_gpu_layers = ngl
        cfg.batch_size = batch
        cfg.ctx = ctx
        return cfg

    def _run_ngl_batch_grid(self) -> PresetResult:
        """Sweep ngl×batch at a fixed context size.

        Used by fastest_response and throughput_king presets.
        """
        from llama_bench.tuner import AdaptiveTuner, TunerBounds, TunerThresholds

        assert self.preset_cfg.fixed_ctx is not None, (
            f"Preset {self.preset_cfg.name!r} has sweep_ngl_batch=True but fixed_ctx is None"
        )
        ctx = self.preset_cfg.fixed_ctx
        combos = self._ngl_batch_combos()

        prompt_seq = self._prompt_sequence()
        all_attempts = []

        for i, (ngl, batch) in enumerate(combos):
            if self.stop_event is not None and self.stop_event.is_set():  # type: ignore[union-attr]
                logger.info("PresetRunner: stop_event set, aborting grid at combo %d", i)
                break

            cfg = self._make_cfg_for_combo(ngl, batch, ctx)

            # Create a single-candidate AdaptiveTuner (ctx_min == ctx_max == fixed_ctx).
            # This gives us OOM-adaptive retry for free, but ctx is pinned.
            bounds = TunerBounds(
                ctx_min=ctx,
                ctx_max=ctx,
                ctx_step=ctx,  # single step — no ctx sweep
                ngl_min=0,
                ngl_step=max(1, self.base_cfg.n_gpu_layers // 8),
                batch_min=256,
                batch_step=256,
                max_retries=3,  # fewer retries for grid sweeps
            )
            thresholds = TunerThresholds(min_tokens_per_sec=0.5)

            tuner = AdaptiveTuner(
                base_cfg=cfg,
                bounds=bounds,
                thresholds=thresholds,
                artifacts_dir=self.artifacts_dir,
                log_file=self.log_file,
                n_followups=self.preset_cfg.n_followups,
                max_tokens=self.preset_cfg.max_tokens,
                prompt_seq_override=prompt_seq,
                event_cb=self.event_cb,
                stop_event=self.stop_event,
            )

            attempts = tuner.run()
            all_attempts.extend(attempts)

        return self._build_result(all_attempts)

    def _run_fixed_or_ctx_sweep(self) -> PresetResult:
        """Run the preset with a context sweep (max_context) or explicit
        ctx candidate list (long_context_rag).
        """
        from llama_bench.tuner import AdaptiveTuner, TunerBounds, TunerThresholds

        all_attempts = []

        if self.preset_cfg.name == PRESET_MAX_CONTEXT:
            # Full ctx sweep with OOM adaptation — the original AdaptiveTuner use case.
            prompt_seq = self._prompt_sequence()
            cfg = copy.deepcopy(self.base_cfg)

            bounds = TunerBounds(
                ctx_min=self.preset_cfg.ctx_min,
                ctx_max=cfg.ctx,  # use whatever the caller set on base_cfg
                ctx_step=self.preset_cfg.ctx_step,
                ngl_min=0,
                ngl_step=4,
                batch_min=256,
                batch_step=256,
                max_retries=5,
            )
            thresholds = TunerThresholds(min_tokens_per_sec=0.5)

            tuner = AdaptiveTuner(
                base_cfg=cfg,
                bounds=bounds,
                thresholds=thresholds,
                artifacts_dir=self.artifacts_dir,
                log_file=self.log_file,
                n_followups=self.preset_cfg.n_followups,
                max_tokens=self.preset_cfg.max_tokens,
                prompt_seq_override=prompt_seq,
                event_cb=self.event_cb,
                stop_event=self.stop_event,
            )
            all_attempts = tuner.run()

        elif self.preset_cfg.name == PRESET_LONG_CONTEXT_RAG:
            # Explicit ctx candidates (32k, 64k, 128k), skip if above ctx_ceiling.
            candidates = list(self.preset_cfg.ctx_candidates or [32768, 65536, 131072])
            if self.ctx_ceiling is not None:
                candidates = [c for c in candidates if c <= self.ctx_ceiling]
            if not candidates:
                logger.info(
                    "long_context_rag: all ctx candidates exceed ceiling %s, skipping",
                    self.ctx_ceiling,
                )
                return PresetResult(
                    preset_name=self.preset_cfg.name,
                    primary_metric=self.preset_cfg.primary_metric,
                    skipped=True,
                )

            # Use the best ngl/batch from the caller (base_cfg already set).
            cfg = copy.deepcopy(self.base_cfg)

            for i, ctx in enumerate(sorted(candidates)):
                if self.stop_event is not None and self.stop_event.is_set():  # type: ignore[union-attr]
                    logger.info("PresetRunner/rag: stop_event set at ctx=%d", ctx)
                    break

                # Rebuild the prompt for *this* ctx size.
                overhead_tokens = 300
                padding_tokens = max(1000, ctx - overhead_tokens)
                import llama_bench.prompts as _prompts
                prompt_seq = _prompts.build_long_context_rag_prompt(
                    padding_tokens=padding_tokens
                )

                cfg_for_ctx = copy.deepcopy(cfg)
                cfg_for_ctx.ctx = ctx

                bounds = TunerBounds(
                    ctx_min=ctx,
                    ctx_max=ctx,
                    ctx_step=ctx,
                    ngl_min=0,
                    ngl_step=4,
                    batch_min=256,
                    batch_step=256,
                    max_retries=3,
                )
                thresholds = TunerThresholds(min_tokens_per_sec=0.1)

                tuner = AdaptiveTuner(
                    base_cfg=cfg_for_ctx,
                    bounds=bounds,
                    thresholds=thresholds,
                    artifacts_dir=self.artifacts_dir,
                    log_file=self.log_file,
                    n_followups=self.preset_cfg.n_followups,
                    max_tokens=self.preset_cfg.max_tokens,
                    prompt_seq_override=prompt_seq,
                    event_cb=self.event_cb,
                    stop_event=self.stop_event,
                )
                all_attempts.extend(tuner.run())

        return self._build_result(all_attempts)

    def _build_result(self, attempts: list) -> PresetResult:
        """Select the best attempt and package into PresetResult."""
        metric = self.preset_cfg.primary_metric
        lower_is_better = self.preset_cfg.lower_is_better
        successful = [a for a in attempts if a.success]

        best_attempt = None
        best_value: "Optional[float]" = None

        if successful:
            if lower_is_better:
                best_attempt = min(successful, key=lambda a: getattr(a, metric, float("inf")))
            else:
                best_attempt = max(successful, key=lambda a: getattr(a, metric, 0.0))
            best_value = float(getattr(best_attempt, metric, 0.0))

        return PresetResult(
            preset_name=self.preset_cfg.name,
            attempts=attempts,
            best_attempt=best_attempt,
            primary_metric=metric,
            best_value=best_value,
            skipped=False,
        )
