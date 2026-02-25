"""Continuous multi-objective parameter explorer for llama-bench.

The :class:`ContinuousExplorer` sweeps the full (ctx, ngl, batch) grid
indefinitely until ``stop_event`` is set, tracking the best config for each
of five objectives in a :class:`HallOfFame`.

Usage::

    explorer = ContinuousExplorer(
        base_cfg=base_cfg,
        bounds=bounds,
        thresholds=thresholds,
        ngl_values=[37, 41, 45],
        batch_values=[1024, 1536],
        event_cb=tui.handle_event,
        stop_event=tui._stop_event,
        output_path="results/explore_xxx.jsonl",
    )
    attempts = explorer.run()   # blocks; returns when stop_event is set
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from llama_bench.config import BenchConfig
from llama_bench.tuner import (
    AdaptiveTuner,
    TuneAttempt,
    TunerBounds,
    TunerThresholds,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hall of Fame
# ---------------------------------------------------------------------------

@dataclass
class HallOfFameEntry:
    """A single best-in-class config record."""
    label: str
    ctx: int
    ngl: int
    batch: int
    cold_ttft_s: float
    warm_ttft_s: float
    tokens_per_sec: float
    run_id: str = ""


def _bal_score(ctx: int, tps: float, cold_ttft: float) -> float:
    """Balanced objective: rewards large context + high throughput + low TTFT."""
    if tps <= 0 or cold_ttft <= 0:
        return 0.0
    return (ctx ** 0.5) * tps / (1.0 + cold_ttft)


@dataclass
class HallOfFame:
    """Tracks the best config for each of five objectives."""
    max_ctx: Optional[HallOfFameEntry] = None
    fastest_ttft: Optional[HallOfFameEntry] = None
    best_warm: Optional[HallOfFameEntry] = None
    best_throughput: Optional[HallOfFameEntry] = None
    best_balanced: Optional[HallOfFameEntry] = None
    total_tested: int = 0
    round_num: int = 1

    def update(self, attempt: TuneAttempt) -> bool:
        """Incorporate a new attempt. Returns True if any record improved."""
        if not attempt.success:
            return False

        changed = False

        def _entry(label: str) -> HallOfFameEntry:
            return HallOfFameEntry(
                label=label,
                ctx=attempt.ctx,
                ngl=attempt.n_gpu_layers,
                batch=attempt.batch_size,
                cold_ttft_s=attempt.cold_ttft_s,
                warm_ttft_s=attempt.warm_ttft_s,
                tokens_per_sec=attempt.tokens_per_sec,
                run_id=attempt.run_id,
            )

        # 1. Max Context
        if self.max_ctx is None or attempt.ctx > self.max_ctx.ctx:
            self.max_ctx = _entry("Max Context")
            changed = True

        # 2. Fastest cold TTFT (first response latency)
        if attempt.cold_ttft_s > 0 and (
            self.fastest_ttft is None
            or attempt.cold_ttft_s < self.fastest_ttft.cold_ttft_s
        ):
            self.fastest_ttft = _entry("Fastest TTFT")
            changed = True

        # 3. Best warm TTFT (KV cache hit — subsequent requests)
        if attempt.warm_ttft_s > 0 and (
            self.best_warm is None
            or attempt.warm_ttft_s < self.best_warm.warm_ttft_s
        ):
            self.best_warm = _entry("Best Warm TTFT")
            changed = True

        # 4. Best throughput (tokens/s)
        if attempt.tokens_per_sec > 0 and (
            self.best_throughput is None
            or attempt.tokens_per_sec > self.best_throughput.tokens_per_sec
        ):
            self.best_throughput = _entry("Best Throughput")
            changed = True

        # 5. Best balanced: sqrt(ctx) * tps / (1 + cold_ttft)
        score = _bal_score(attempt.ctx, attempt.tokens_per_sec, attempt.cold_ttft_s)
        if score > 0:
            if self.best_balanced is None:
                existing_score = 0.0
            else:
                existing_score = _bal_score(
                    self.best_balanced.ctx,
                    self.best_balanced.tokens_per_sec,
                    self.best_balanced.cold_ttft_s,
                )
            if score > existing_score:
                self.best_balanced = _entry("Best Overall")
                changed = True

        return changed

    def entries(self) -> list[HallOfFameEntry]:
        """Return all populated entries in display order."""
        return [
            e for e in [
                self.max_ctx,
                self.fastest_ttft,
                self.best_warm,
                self.best_throughput,
                self.best_balanced,
            ]
            if e is not None
        ]

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for TUI events."""

        def _ed(e: Optional[HallOfFameEntry]) -> Optional[dict]:
            if e is None:
                return None
            return {
                "label": e.label,
                "ctx": e.ctx,
                "ngl": e.ngl,
                "batch": e.batch,
                "cold_ttft_s": e.cold_ttft_s,
                "warm_ttft_s": e.warm_ttft_s,
                "tokens_per_sec": e.tokens_per_sec,
            }

        return {
            "max_ctx": _ed(self.max_ctx),
            "fastest_ttft": _ed(self.fastest_ttft),
            "best_warm": _ed(self.best_warm),
            "best_throughput": _ed(self.best_throughput),
            "best_balanced": _ed(self.best_balanced),
            "total_tested": self.total_tested,
            "round_num": self.round_num,
        }


# ---------------------------------------------------------------------------
# ContinuousExplorer
# ---------------------------------------------------------------------------

class ContinuousExplorer:
    """Sweep the (ctx, ngl, batch) space indefinitely.

    Each round generates the full Cartesian product of
    ``ctx_values × ngl_values × batch_values``.  Round 1 runs them
    high-to-low (most likely to succeed first).  Subsequent rounds shuffle
    the order so different regions get covered over time.

    OOM retry is delegated to :class:`~llama_bench.tuner.AdaptiveTuner`'s
    ``_run_with_adaptive_retry`` so all the same reduction logic applies.

    Results are appended to *output_path* immediately after each run so
    nothing is lost if the user quits mid-sweep.
    """

    def __init__(
        self,
        base_cfg: BenchConfig,
        bounds: TunerBounds,
        thresholds: TunerThresholds,
        ngl_values: list[int],
        batch_values: list[int],
        artifacts_dir: str = "results",
        log_file: Optional[str] = None,
        n_followups: int = 2,
        max_tokens: int = 512,
        prompt_seq_override: Optional[list] = None,
        event_cb: Optional[Callable[[str, dict], None]] = None,
        stop_event: Optional[object] = None,
        output_path: Optional[str] = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.bounds = bounds
        self.thresholds = thresholds
        self.ngl_values = sorted(set(ngl_values), reverse=True)
        self.batch_values = sorted(set(batch_values), reverse=True)
        self.artifacts_dir = artifacts_dir
        self.log_file = log_file
        self.n_followups = n_followups
        self.max_tokens = max_tokens
        self.prompt_seq_override = prompt_seq_override
        self.event_cb = event_cb
        self.stop_event = stop_event
        self.output_path = output_path
        self._hof = HallOfFame()
        # Accumulated results — accessible even if run() hasn't returned yet.
        # This allows callers to retrieve partial results after an interrupt.
        self._accumulated_attempts: list[TuneAttempt] = []
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def hall_of_fame(self) -> HallOfFame:
        return self._hof

    @property
    def accumulated_attempts(self) -> list[TuneAttempt]:
        """Return a copy of all attempts recorded so far (thread-safe read)."""
        return list(self._accumulated_attempts)

    def run(self) -> list[TuneAttempt]:
        """Run indefinitely until *stop_event* is set; return all attempts."""
        from llama_bench.prompts import build_prompt_sequence

        # Announce settings to the TUI Settings panel.
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

        prompt_seq = (
            self.prompt_seq_override
            if self.prompt_seq_override is not None
            else build_prompt_sequence(n_followups=self.n_followups)
        )

        all_attempts: list[TuneAttempt] = []
        round_num = 0

        while True:
            if self.stop_event is not None and self.stop_event.is_set():  # type: ignore[union-attr]
                break

            round_num += 1
            self._hof.round_num = round_num
            candidates = self._generate_round(round_num)
            total = len(candidates)

            logger.info("Explorer: round %d — %d candidates", round_num, total)
            self._emit(
                "explore_round",
                round=round_num,
                total=total,
                progress_current=0,
                progress_total=total,
                phase_desc=f"Explore round {round_num}  (0/{total})",
            )

            for idx, cfg in enumerate(candidates):
                if self.stop_event is not None and self.stop_event.is_set():  # type: ignore[union-attr]
                    break

                self._emit(
                    "run_start",
                    ctx=cfg.ctx,
                    ngl=cfg.n_gpu_layers,
                    batch=cfg.batch_size,
                    phase_desc=f"Explore round {round_num}  [{idx + 1}/{total}]",
                    progress_current=idx,
                    progress_total=total,
                )

                attempt = self._run_candidate(cfg, prompt_seq)
                all_attempts.append(attempt)
                self._accumulated_attempts.append(attempt)
                self._hof.total_tested += 1
                self._hof.update(attempt)

                self._emit(
                    "run_done",
                    ctx=attempt.ctx,
                    ngl=attempt.n_gpu_layers,
                    batch=attempt.batch_size,
                    cold_ttft_s=attempt.cold_ttft_s,
                    warm_ttft_s=attempt.warm_ttft_s,
                    tokens_per_sec=attempt.tokens_per_sec,
                    success=attempt.success,
                    reason=attempt.failure_reason,
                )
                # Always push the latest hall of fame to the TUI.
                self._emit("hall_of_fame", **self._hof.to_dict())

                if self.output_path:
                    self._append_result(attempt)

        return all_attempts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(self, event: str, **data: object) -> None:
        if self.event_cb is not None:
            try:
                self.event_cb(event, dict(data))
            except Exception:  # noqa: BLE001
                pass

    def _run_candidate(self, cfg: BenchConfig, prompt_seq: list) -> TuneAttempt:
        """Run one starting config with OOM-adaptive retry via AdaptiveTuner."""
        per_cfg_tuner = AdaptiveTuner(
            base_cfg=cfg,
            bounds=TunerBounds(
                # Lock ctx to this specific value — only ngl/batch may be reduced.
                ctx_min=cfg.ctx,
                ctx_max=cfg.ctx,
                ctx_step=self.bounds.ctx_step,
                ngl_min=self.bounds.ngl_min,
                ngl_step=self.bounds.ngl_step,
                batch_min=self.bounds.batch_min,
                batch_step=self.bounds.batch_step,
                max_retries=self.bounds.max_retries,
            ),
            thresholds=self.thresholds,
            artifacts_dir=self.artifacts_dir,
            log_file=self.log_file,
            n_followups=self.n_followups,
            max_tokens=self.max_tokens,
            prompt_seq_override=prompt_seq,
            event_cb=self.event_cb,
            stop_event=self.stop_event,
        )
        return per_cfg_tuner._run_with_adaptive_retry(cfg, prompt_seq)

    def _generate_round(self, round_num: int) -> list[BenchConfig]:
        """Build the candidate list for one round."""
        ctx_vals: list[int] = []
        c = self.bounds.ctx_max
        while c >= self.bounds.ctx_min:
            ctx_vals.append(c)
            c -= self.bounds.ctx_step

        combos = list(itertools.product(ctx_vals, self.ngl_values, self.batch_values))
        if round_num > 1:
            # Shuffle later rounds to cover the space in different orders.
            random.shuffle(combos)

        candidates: list[BenchConfig] = []
        for ctx, ngl, batch in combos:
            cfg = copy.deepcopy(self.base_cfg)
            cfg.ctx = ctx
            cfg.n_gpu_layers = ngl
            cfg.batch_size = batch
            candidates.append(cfg)
        return candidates

    def _append_result(self, attempt: TuneAttempt) -> None:
        """Append one result line to the JSONL output (never overwrites).

        ``stop_requested`` attempts are noise — the server was interrupted
        mid-startup, so no measurements were taken.  Skip them.
        """
        if attempt.failure_reason == "stop_requested":
            return
        """Append one result line to the JSONL output (never overwrites)."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
                fh.write(json.dumps(asdict(attempt)) + "\n")
        except OSError as exc:
            logger.warning("Failed to append result to %s: %s", self.output_path, exc)
