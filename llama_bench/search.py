"""Staged parameter search for llama-bench."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from llama_bench.config import BenchConfig, SearchSpace, generate_configs
from llama_bench.metrics import RunMetrics, score_run
from llama_bench.runner import BenchmarkRunner


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    config: BenchConfig
    metrics: list[RunMetrics]
    best_score: float
    phase: int
    run_id: str


class EarlyStopReason(Enum):
    FAILED = auto()
    TIMEOUT = auto()
    MUCH_WORSE_THAN_BEST = auto()
    OOM = auto()


# ---------------------------------------------------------------------------
# StagedSearcher
# ---------------------------------------------------------------------------

class StagedSearcher:
    """Two-phase staged search over a :class:`SearchSpace`.

    Phase 1 — coarse sweep at low fidelity (2 follow-ups).
    Phase 2 — refine the top 25% at full fidelity (4 follow-ups).
    """

    def __init__(
        self,
        space: SearchSpace,
        base_cfg: BenchConfig,
        artifacts_dir: str = "results",
        max_configs: int = 50,
        timeout_factor: float = 3.0,
        progress_cb: Optional[Callable[[int, int, int, str], None]] = None,
        log_file: Optional[str] = None,
    ) -> None:
        self.space = space
        self.base_cfg = base_cfg
        self.artifacts_dir = artifacts_dir
        self.max_configs = max_configs
        self.timeout_factor = timeout_factor
        self.progress_cb = progress_cb
        self.log_file = log_file

        self._results: list[SearchResult] = []
        self._best_score: float = float("inf")
        self._best_config: Optional[BenchConfig] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[SearchResult]:
        """Execute staged search and return all results sorted by score."""
        all_configs = generate_configs(self.space, self.base_cfg)
        # Limit to max_configs
        if len(all_configs) > self.max_configs:
            # Sample evenly
            step = len(all_configs) / self.max_configs
            all_configs = [all_configs[int(i * step)] for i in range(self.max_configs)]

        # ---- Phase 1: coarse ----
        phase1_results = self._run_phase(
            configs=all_configs,
            phase=1,
            phase_name="coarse sweep",
            n_followups=2,
        )

        # ---- Phase 2: refine top 25% ----
        successful = [r for r in phase1_results if r.best_score < float("inf")]
        successful.sort(key=lambda r: r.best_score)
        top_n = max(1, len(successful) // 4)
        top_configs = [r.config for r in successful[:top_n]]

        phase2_results = self._run_phase(
            configs=top_configs,
            phase=2,
            phase_name="refinement",
            n_followups=4,
        )

        all_results = phase1_results + phase2_results
        all_results.sort(key=lambda r: r.best_score)
        self._results = all_results
        return all_results

    def best_config(self) -> Optional[BenchConfig]:
        """Return the config with the best score, or None if no runs succeeded."""
        successful = [r for r in self._results if r.best_score < float("inf")]
        if not successful:
            return None
        return min(successful, key=lambda r: r.best_score).config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        configs: list[BenchConfig],
        phase: int,
        phase_name: str,
        n_followups: int,
    ) -> list[SearchResult]:
        from llama_bench.prompts import build_prompt_sequence

        results: list[SearchResult] = []
        total = len(configs)

        for idx, cfg in enumerate(configs):
            if self.progress_cb:
                self.progress_cb(idx, total, phase, phase_name)

            prompt_seq = build_prompt_sequence(n_followups=n_followups)
            runner = BenchmarkRunner(cfg, artifacts_dir=self.artifacts_dir,
                                     log_file=self.log_file)

            run_start = time.monotonic()
            try:
                run_metrics = runner.run_single(prompt_seq, n_followups=n_followups)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("Config %d/%d raised %s: %s", idx + 1, total, type(exc).__name__, exc)
                run_metrics = []

            elapsed_s = time.monotonic() - run_start

            if not run_metrics:
                result = SearchResult(
                    config=cfg,
                    metrics=[],
                    best_score=float("inf"),
                    phase=phase,
                    run_id=f"phase{phase}_cfg{idx}",
                )
            else:
                scores = [score_run(m) for m in run_metrics]
                best = min(scores)

                # Check early stop
                stop, reason = self._should_early_stop_score(best, elapsed_s)
                if stop:
                    result = SearchResult(
                        config=cfg,
                        metrics=run_metrics,
                        best_score=float("inf"),
                        phase=phase,
                        run_id=f"phase{phase}_cfg{idx}_stopped_{reason.name.lower() if isinstance(reason, EarlyStopReason) else 'unknown'}",
                    )
                else:
                    if best < self._best_score:
                        self._best_score = best
                        self._best_config = cfg

                    result = SearchResult(
                        config=cfg,
                        metrics=run_metrics,
                        best_score=best,
                        phase=phase,
                        run_id=f"phase{phase}_cfg{idx}",
                    )

            results.append(result)

        if self.progress_cb:
            self.progress_cb(total, total, phase, phase_name)

        return results

    def _should_early_stop(
        self,
        result: SearchResult,
        best_score: float,
    ) -> tuple[bool, Optional[EarlyStopReason]]:
        """Decide whether to early-stop based on a completed SearchResult."""
        if result.best_score == float("inf"):
            return True, EarlyStopReason.FAILED

        if best_score < float("inf") and result.best_score > self.timeout_factor * best_score:
            return True, EarlyStopReason.MUCH_WORSE_THAN_BEST

        for m in result.metrics:
            if m.failure_reason == "oom":
                return True, EarlyStopReason.OOM

        return False, None

    def _should_early_stop_score(
        self,
        best: float,
        elapsed_s: float,
    ) -> tuple[bool, Optional[EarlyStopReason]]:
        if best == float("inf"):
            return True, EarlyStopReason.FAILED
        if (
            self._best_score < float("inf")
            and best > self.timeout_factor * self._best_score
        ):
            return True, EarlyStopReason.MUCH_WORSE_THAN_BEST
        return False, None


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

def save_results(results: list[SearchResult], output_file: str) -> None:
    """Serialise *results* to a JSONL file, one JSON object per line."""
    import dataclasses

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fh:
        for result in results:
            obj = {
                "run_id": result.run_id,
                "phase": result.phase,
                "best_score": result.best_score if result.best_score != float("inf") else None,
                "config": dataclasses.asdict(result.config),
                "metrics": [
                    {
                        "run_id": m.run_id,
                        "success": m.success,
                        "failure_reason": m.failure_reason,
                        "timestamp": m.timestamp,
                        "config_hash": m.config_hash,
                        "log_file": m.log_file,
                        "client": {
                            "ttft_ms": m.client.ttft_ms,
                            "end_to_end_latency_ms": m.client.end_to_end_latency_ms,
                            "streaming_tok_per_s": m.client.streaming_tok_per_s,
                            "total_tokens": m.client.total_tokens,
                            "is_streaming": m.client.is_streaming,
                        },
                        "server": (
                            {
                                "prompt_eval_time_ms": m.server.prompt_eval_time_ms,
                                "prompt_eval_count": m.server.prompt_eval_count,
                                "eval_time_ms": m.server.eval_time_ms,
                                "eval_count": m.server.eval_count,
                                "prompt_tok_per_s": m.server.prompt_tok_per_s,
                                "decode_tok_per_s": m.server.decode_tok_per_s,
                                "total_time_ms": m.server.total_time_ms,
                                "has_memory_info": m.server.has_memory_info,
                                "memory_used_mb": m.server.memory_used_mb,
                            }
                            if m.server is not None
                            else None
                        ),
                    }
                    for m in result.metrics
                ],
            }
            fh.write(json.dumps(obj) + "\n")
