"""Adaptive configuration tuner for llama-bench.

This module implements the core tuning loop that answers:
  - What is the highest usable context for this model + hardware?
  - Which configs provide the best speed-to-context ratio near that context?
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from llama_bench.config import BenchConfig
from llama_bench.metrics import (
    RunMetrics,
    parse_memory_fit_heuristic,
)
from llama_bench.runner import BenchmarkRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_OOM_REASONS = frozenset({"out_of_vram", "server_exited"})

# Reasons that are unrecoverable; abort immediately.
_CRITICAL_REASONS = frozenset({"model_not_found"})


@dataclass
class TunerBounds:
    """Configurable bounds and step sizes for the tuner sweep."""

    ctx_min: int = 8192
    ctx_max: int = 131072
    ctx_step: int = 8192

    ngl_min: int = 0
    ngl_step: int = 4

    batch_min: int = 256
    batch_step: int = 256

    max_retries: int = 5


@dataclass
class TunerThresholds:
    """Minimum performance thresholds that define a "usable" config."""

    max_ttft_s: Optional[float] = None
    min_tokens_per_sec: float = 4.0


@dataclass
class TuneAttempt:
    """Record of a single tuner attempt (one server start + workload run)."""

    config: BenchConfig
    success: bool
    failure_reason: Optional[str] = None
    ttft_s: float = 0.0
    tokens_per_sec: float = 0.0
    cold_ttft_s: float = 0.0
    warm_ttft_s: float = 0.0
    ctx: int = 0
    n_gpu_layers: int = 0
    batch_size: int = 0
    stderr_path: Optional[str] = None
    stdout_path: Optional[str] = None
    server_error_excerpt: Optional[str] = None
    projected_mib: Optional[float] = None
    free_mib: Optional[float] = None
    run_id: str = ""
    timestamp: str = ""
    log_file: Optional[str] = None
    engine_mismatch: bool = False


# ---------------------------------------------------------------------------
# Multi-objective selection
# ---------------------------------------------------------------------------

def select_best_configs(
    attempts: list[TuneAttempt],
    ctx_pct_threshold: float = 0.90,
    top_n: int = 5,
) -> dict:
    """Multi-objective selection from a list of :class:`TuneAttempt` records.

    Primary objective: maximise usable context.
    Secondary objective: among configs within *ctx_pct_threshold* of the
    maximum usable context, maximise throughput (tokens/s).

    Returns a dict with:
      ``max_ctx_result``      — :class:`TuneAttempt` with the highest usable ctx
      ``top_throughput``      — list of up to *top_n* :class:`TuneAttempt` sorted
                                 by tokens/s (descending) within threshold of max ctx
      ``recommended``         — the config with best tokens/s that also has the
                                 highest ctx (tie-break: highest ctx first)
    """
    usable = [a for a in attempts if a.success]
    if not usable:
        return {
            "max_ctx_result": None,
            "top_throughput": [],
            "recommended": None,
        }

    max_ctx = max(a.ctx for a in usable)
    max_ctx_result = next(
        a for a in sorted(usable, key=lambda x: (-x.ctx, -x.tokens_per_sec))
        if a.ctx == max_ctx
    )

    # Configs within threshold of max ctx
    ctx_threshold = max_ctx * ctx_pct_threshold
    near_max = [a for a in usable if a.ctx >= ctx_threshold]
    top_throughput = sorted(near_max, key=lambda a: -a.tokens_per_sec)[:top_n]

    # Recommended: highest ctx first, then best throughput
    recommended = sorted(near_max, key=lambda a: (-a.ctx, -a.tokens_per_sec))
    recommended_result = recommended[0] if recommended else max_ctx_result

    return {
        "max_ctx_result": max_ctx_result,
        "top_throughput": top_throughput,
        "recommended": recommended_result,
    }


# ---------------------------------------------------------------------------
# Summary JSON writer
# ---------------------------------------------------------------------------

def _attempt_to_dict(a: TuneAttempt) -> dict:
    d = asdict(a)
    # config is a nested dataclass; already handled by asdict
    return d


def write_summary_json(path: str, attempts: list[TuneAttempt], selection: dict) -> None:
    """Write ``results/summary.json`` with tuner results and selection."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _attempt_summary(a: "Optional[TuneAttempt]") -> "Optional[dict]":
        if a is None:
            return None
        return {
            "run_id": a.run_id,
            "ctx": a.ctx,
            "n_gpu_layers": a.n_gpu_layers,
            "batch_size": a.batch_size,
            "ttft_s": a.ttft_s,
            "tokens_per_sec": a.tokens_per_sec,
            "success": a.success,
            "failure_reason": a.failure_reason,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_attempts": len(attempts),
        "successful_attempts": sum(1 for a in attempts if a.success),
        "max_ctx_result": _attempt_summary(selection.get("max_ctx_result")),
        "top_throughput_configs": [
            _attempt_summary(a) for a in selection.get("top_throughput", [])
        ],
        "recommended": _attempt_summary(selection.get("recommended")),
        "all_attempts": [_attempt_summary(a) for a in attempts],
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary written to %s", path)


# ---------------------------------------------------------------------------
# AdaptiveTuner
# ---------------------------------------------------------------------------

class AdaptiveTuner:
    """Sweep context from high to low, adapting parameters on OOM/fit failures.

    For each candidate ctx value (from *bounds.ctx_max* down to *bounds.ctx_min*)
    the tuner:

    1. Starts the server with the candidate config.
    2. Runs the reverse-engineering workload.
    3. On OOM/fit-failure: reduces ``n_gpu_layers``, ``batch_size``, or ``ctx``
       (in that order) and retries up to *bounds.max_retries* times.
    4. Records whether the config is "usable" per the given thresholds.
    5. Continues to the next ctx value even after a retry loop failure.

    Critical errors (model not found, permission errors) abort the entire run.
    """

    def __init__(
        self,
        base_cfg: BenchConfig,
        bounds: TunerBounds,
        thresholds: TunerThresholds,
        artifacts_dir: str = "results",
        log_file: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        n_followups: int = 4,
        max_tokens: int = 512,
        prompt_seq_override: Optional[list] = None,
        event_cb: Optional[Callable[[str, dict], None]] = None,
        stop_event: Optional[object] = None,  # threading.Event
    ) -> None:
        self.base_cfg = base_cfg
        self.bounds = bounds
        self.thresholds = thresholds
        self.artifacts_dir = artifacts_dir
        self.log_file = log_file
        self.progress_cb = progress_cb
        self.n_followups = n_followups
        self.max_tokens = max_tokens
        self.prompt_seq_override = prompt_seq_override
        self.event_cb = event_cb
        self.stop_event = stop_event
    def _emit(self, event: str, **data: object) -> None:
        """Fire event to registered callback (silently swallows errors)."""
        if self.event_cb is not None:
            try:
                self.event_cb(event, dict(data))
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[TuneAttempt]:
        """Execute the tuner sweep; return all attempt records."""
        from llama_bench.prompts import build_prompt_sequence

        # Emit base config so the TUI Settings panel can display it.
        self._emit(
            "bench_settings",
            model=os.path.basename(self.base_cfg.model_path),
            model_path=self.base_cfg.model_path,
            ctx_max=self.bounds.ctx_max,
            ctx_min=self.bounds.ctx_min,
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
            max_retries=self.bounds.max_retries,
        )

        if self.prompt_seq_override is not None:
            prompt_seq = self.prompt_seq_override
        else:
            prompt_seq = build_prompt_sequence(n_followups=self.n_followups)

        candidates = self._generate_candidates()
        attempts: list[TuneAttempt] = []


        total = len(candidates)
        logger.info("AdaptiveTuner: %d candidate configs (ctx %d→%d step %d)",
                    total, self.bounds.ctx_max, self.bounds.ctx_min, self.bounds.ctx_step)

        for idx, cfg in enumerate(candidates):
            # Honour cancellation request (user pressed q in TUI).
            if self.stop_event is not None and self.stop_event.is_set():  # type: ignore[union-attr]
                logger.info("AdaptiveTuner: stop_event set, aborting sweep at idx %d", idx)
                break

            if self.progress_cb:
                self.progress_cb(idx, total)

            self._emit(
                "run_start",
                ctx=cfg.ctx,
                ngl=cfg.n_gpu_layers,
                batch=cfg.batch_size,
                phase_desc=f"ctx sweep  [{idx + 1}/{total}]",
            )

            attempt = self._run_with_adaptive_retry(cfg, prompt_seq)
            attempts.append(attempt)

            logger.info(
                "Candidate %d/%d ctx=%d ngl=%d batch=%d → %s%s",
                idx + 1, total, attempt.ctx, attempt.n_gpu_layers, attempt.batch_size,
                "OK" if attempt.success else "FAIL",
                f" ({attempt.failure_reason})" if attempt.failure_reason else "",
            )

            # Abort on critical errors
            if attempt.failure_reason in _CRITICAL_REASONS:
                logger.error("Critical error (%s); aborting tuner run.", attempt.failure_reason)
                break

        if self.progress_cb:
            self.progress_cb(total, total)

        return attempts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_candidates(self) -> list[BenchConfig]:
        """Sweep ctx from max to min in ctx_step decrements."""
        candidates: list[BenchConfig] = []
        ctx = self.bounds.ctx_max
        while ctx >= self.bounds.ctx_min:
            cfg = copy.deepcopy(self.base_cfg)
            cfg.ctx = ctx
            candidates.append(cfg)
            ctx -= self.bounds.ctx_step
        return candidates

    def _run_with_adaptive_retry(
        self,
        initial_cfg: BenchConfig,
        prompt_seq: list[dict],
    ) -> TuneAttempt:
        """Run a single candidate config, retrying on OOM with adapted params."""
        current_cfg = copy.deepcopy(initial_cfg)
        last_attempt: Optional[TuneAttempt] = None
        # Track which adaptations have been tried to avoid cycles
        tried_cfgs: set[tuple] = set()

        for attempt_num in range(self.bounds.max_retries + 1):
            cfg_key = (current_cfg.ctx, current_cfg.n_gpu_layers, current_cfg.batch_size,
                       current_cfg.flash_attn)
            if cfg_key in tried_cfgs:
                logger.debug("Retry %d: already tried cfg_key=%s, stopping", attempt_num, cfg_key)
                break
            tried_cfgs.add(cfg_key)

            logger.debug(
                "Attempt %d: ctx=%d ngl=%d batch=%d",
                attempt_num + 1, current_cfg.ctx, current_cfg.n_gpu_layers, current_cfg.batch_size,
            )

            last_attempt = self._single_run(current_cfg, prompt_seq)

            if last_attempt.success:
                return last_attempt

            # Critical error — propagate immediately
            if last_attempt.failure_reason in _CRITICAL_REASONS:
                return last_attempt

            # Non-OOM failure — no point retrying with different params
            if last_attempt.failure_reason not in _OOM_REASONS:
                return last_attempt

            # OOM/fit failure — try adapting
            if attempt_num < self.bounds.max_retries:
                stderr_text = ""
                if last_attempt.stderr_path:
                    try:
                        with open(last_attempt.stderr_path, "r", encoding="utf-8",
                                  errors="replace") as fh:
                            stderr_text = fh.read()
                    except OSError:
                        pass

                next_cfg = self._adapt_for_oom(current_cfg, stderr_text, tried_cfgs)
                if next_cfg is None:
                    logger.debug("No further adaptation possible; stopping retries.")
                    break
                # Describe what changed for the TUI retry log
                changes: list[str] = []
                if next_cfg.n_gpu_layers != current_cfg.n_gpu_layers:
                    changes.append(f"ngl {current_cfg.n_gpu_layers}→{next_cfg.n_gpu_layers}")
                if next_cfg.batch_size != current_cfg.batch_size:
                    changes.append(f"batch {current_cfg.batch_size}→{next_cfg.batch_size}")
                if next_cfg.ctx != current_cfg.ctx:
                    changes.append(f"ctx {current_cfg.ctx}→{next_cfg.ctx}")
                if not next_cfg.flash_attn and current_cfg.flash_attn:
                    changes.append("flash_attn off")
                self._emit(
                    "retry",
                    attempt=attempt_num + 1,
                    max_retries=self.bounds.max_retries,
                    reason=last_attempt.failure_reason or "oom",
                    change=", ".join(changes) or "?",
                    ctx=next_cfg.ctx,
                    ngl=next_cfg.n_gpu_layers,
                    batch=next_cfg.batch_size,
                )
                current_cfg = next_cfg

        return last_attempt or TuneAttempt(
            config=initial_cfg,
            success=False,
            failure_reason="server_startup_timeout",
            ctx=initial_cfg.ctx,
            n_gpu_layers=initial_cfg.n_gpu_layers,
            batch_size=initial_cfg.batch_size,
        )

    def _adapt_for_oom(
        self,
        cfg: BenchConfig,
        stderr_text: str,
        tried_cfgs: set,
    ) -> Optional[BenchConfig]:
        """Return a new config with reduced parameters, or None if no option remains."""
        # 1. Try reducing n_gpu_layers
        new_ngl = cfg.n_gpu_layers - self.bounds.ngl_step
        if new_ngl >= self.bounds.ngl_min:
            new_cfg = copy.deepcopy(cfg)
            new_cfg.n_gpu_layers = new_ngl
            if (new_cfg.ctx, new_cfg.n_gpu_layers, new_cfg.batch_size,
                    new_cfg.flash_attn) not in tried_cfgs:
                logger.debug("OOM adaptation: ngl %d → %d", cfg.n_gpu_layers, new_ngl)
                return new_cfg

        # 2. Try reducing batch_size (reset ngl)
        new_batch = cfg.batch_size - self.bounds.batch_step
        if new_batch >= self.bounds.batch_min:
            new_cfg = copy.deepcopy(cfg)
            new_cfg.batch_size = new_batch
            new_cfg.n_gpu_layers = self.base_cfg.n_gpu_layers  # reset ngl
            if (new_cfg.ctx, new_cfg.n_gpu_layers, new_cfg.batch_size,
                    new_cfg.flash_attn) not in tried_cfgs:
                logger.debug("OOM adaptation: batch %d → %d", cfg.batch_size, new_batch)
                return new_cfg

        # 3. Try disabling flash-attn if currently enabled
        if cfg.flash_attn:
            new_cfg = copy.deepcopy(cfg)
            new_cfg.flash_attn = False
            new_cfg.n_gpu_layers = self.base_cfg.n_gpu_layers
            new_cfg.batch_size = self.base_cfg.batch_size
            if (new_cfg.ctx, new_cfg.n_gpu_layers, new_cfg.batch_size,
                    new_cfg.flash_attn) not in tried_cfgs:
                logger.debug("OOM adaptation: disabling flash-attn")
                return new_cfg

        # 4. Reduce ctx using memory-fit heuristic or default step
        fit = parse_memory_fit_heuristic(stderr_text)
        if fit is not None:
            projected_mib, free_mib = fit
            if projected_mib > 0 and free_mib < projected_mib:
                ratio = free_mib / projected_mib
                if ratio < 0.5:
                    # Memory way too low — big-step ladder: halve ctx
                    new_ctx = cfg.ctx // 2
                    new_ctx = (new_ctx // self.bounds.ctx_step) * self.bounds.ctx_step
                    logger.debug(
                        "OOM adaptation (big-step): ctx %d → %d "
                        "(projected=%.0f MiB free=%.0f MiB ratio=%.2f)",
                        cfg.ctx, new_ctx, projected_mib, free_mib, ratio,
                    )
                else:
                    new_ctx = int(cfg.ctx * ratio * 0.90)  # reduce by 10% extra to give headroom
                    # Round down to ctx_step boundary
                    new_ctx = (new_ctx // self.bounds.ctx_step) * self.bounds.ctx_step
                    logger.debug(
                        "OOM adaptation (heuristic): ctx %d → %d "
                        "(projected=%.0f MiB free=%.0f MiB ratio=%.2f)",
                        cfg.ctx, new_ctx, projected_mib, free_mib, ratio,
                    )
                # Round down to ctx_step boundary
                new_ctx = (new_ctx // self.bounds.ctx_step) * self.bounds.ctx_step
                logger.debug(
                    "OOM adaptation (heuristic): ctx %d → %d "
                    "(projected=%.0f MiB free=%.0f MiB ratio=%.2f)",
                    cfg.ctx, new_ctx, projected_mib, free_mib, ratio,
                )
                if new_ctx >= self.bounds.ctx_min and new_ctx < cfg.ctx:
                    new_cfg = copy.deepcopy(cfg)
                    new_cfg.ctx = new_ctx
                    new_cfg.n_gpu_layers = self.base_cfg.n_gpu_layers
                    new_cfg.batch_size = self.base_cfg.batch_size
                    new_cfg.flash_attn = self.base_cfg.flash_attn
                    if (new_cfg.ctx, new_cfg.n_gpu_layers, new_cfg.batch_size,
                            new_cfg.flash_attn) not in tried_cfgs:
                        return new_cfg
        else:
            # Default ctx reduction
            new_ctx = cfg.ctx - self.bounds.ctx_step
            if new_ctx >= self.bounds.ctx_min:
                new_cfg = copy.deepcopy(cfg)
                new_cfg.ctx = new_ctx
                new_cfg.n_gpu_layers = self.base_cfg.n_gpu_layers
                new_cfg.batch_size = self.base_cfg.batch_size
                new_cfg.flash_attn = self.base_cfg.flash_attn
                if (new_cfg.ctx, new_cfg.n_gpu_layers, new_cfg.batch_size,
                        new_cfg.flash_attn) not in tried_cfgs:
                    logger.debug("OOM adaptation (default): ctx %d → %d", cfg.ctx, new_ctx)
                    return new_cfg

        return None  # no further adaptation possible

    def _single_run(
        self,
        cfg: BenchConfig,
        prompt_seq: list[dict],
    ) -> TuneAttempt:
        """Run one server instance and return a :class:`TuneAttempt`."""
        runner = BenchmarkRunner(cfg, artifacts_dir=self.artifacts_dir,
            log_file=self.log_file,
            max_tokens=self.max_tokens,
            event_cb=lambda ev, d: self._emit(ev, **d),
        )
        run_metrics_list: list[RunMetrics] = runner.run_single(prompt_seq, n_followups=self.n_followups)

        ts = datetime.now(timezone.utc).isoformat()

        if not run_metrics_list:
            return TuneAttempt(
                config=cfg,
                success=False,
                failure_reason="server_startup_timeout",
                ctx=cfg.ctx,
                n_gpu_layers=cfg.n_gpu_layers,
                batch_size=cfg.batch_size,
                run_id=f"tune_{ts}",
                timestamp=ts,
                log_file=self.log_file,
            )

        first = run_metrics_list[0]

        # If startup failed, record it immediately
        if not first.success:
            # Try to extract memory-fit numbers for ctx heuristic
            projected_mib: Optional[float] = None
            free_mib: Optional[float] = None
            if first.stderr_path:
                try:
                    with open(first.stderr_path, "r", encoding="utf-8",
                              errors="replace") as fh:
                        stderr_text = fh.read()
                    fit = parse_memory_fit_heuristic(stderr_text)
                    if fit:
                        projected_mib, free_mib = fit
                except OSError:
                    pass

            return TuneAttempt(
                config=cfg,
                success=False,
                failure_reason=first.failure_reason,
                ctx=cfg.ctx,
                n_gpu_layers=cfg.n_gpu_layers,
                batch_size=cfg.batch_size,
                stderr_path=first.stderr_path,
                stdout_path=first.stdout_path,
                server_error_excerpt=first.server_error_excerpt,
                projected_mib=projected_mib,
                free_mib=free_mib,
                run_id=first.run_id,
                timestamp=first.timestamp,
                log_file=self.log_file,
            )

        # Collect metrics from successful runs
        successful = [m for m in run_metrics_list if m.success]
        if not successful:
            # All prompt requests failed after server startup
            failed = run_metrics_list[0]
            return TuneAttempt(
                config=cfg,
                success=False,
                failure_reason=failed.failure_reason or "http_error",
                ctx=cfg.ctx,
                n_gpu_layers=cfg.n_gpu_layers,
                batch_size=cfg.batch_size,
                stderr_path=failed.stderr_path,
                stdout_path=failed.stdout_path,
                run_id=failed.run_id,
                timestamp=failed.timestamp,
                log_file=self.log_file,
            )

        # Cold TTFT = first (initial) request; Warm TTFT = avg of follow-ups
        cold_ttft_s = successful[0].client.ttft_ms / 1000.0 if successful else 0.0
        warm_runs = successful[1:] if len(successful) > 1 else []
        warm_ttft_s = (
            sum(m.client.ttft_ms for m in warm_runs) / len(warm_runs) / 1000.0
            if warm_runs else 0.0
        )

        # Overall average (kept for backward compat / threshold check)
        avg_ttft_s = (
            sum(m.client.ttft_ms for m in successful) / len(successful) / 1000.0
        )
        avg_tps = (
            sum(m.client.streaming_tok_per_s for m in successful) / len(successful)
        )

        # Check performance thresholds
        usable = (
            (self.thresholds.max_ttft_s is None or avg_ttft_s <= self.thresholds.max_ttft_s)
            and avg_tps >= self.thresholds.min_tokens_per_sec
        )
        failure_reason: Optional[str] = None
        if not usable:
            if self.thresholds.max_ttft_s is not None and avg_ttft_s > self.thresholds.max_ttft_s:
                failure_reason = "ttft_exceeded"
            else:
                failure_reason = "throughput_too_low"

        best = successful[0]
        attempt = TuneAttempt(
            config=cfg,
            success=usable,
            failure_reason=failure_reason,
            ttft_s=avg_ttft_s,
            tokens_per_sec=avg_tps,
            cold_ttft_s=cold_ttft_s,
            warm_ttft_s=warm_ttft_s,
            ctx=cfg.ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            batch_size=cfg.batch_size,
            stderr_path=best.stderr_path,
            stdout_path=best.stdout_path,
            run_id=best.run_id,
            timestamp=best.timestamp,
            log_file=self.log_file,
        )
        self._emit(
            "run_done",
            ctx=attempt.ctx,
            ngl=attempt.n_gpu_layers,
            batch=attempt.batch_size,
            cold_ttft_s=cold_ttft_s,
            warm_ttft_s=warm_ttft_s,
            tokens_per_sec=avg_tps,
            success=usable,
            reason=failure_reason,
        )
        return attempt
