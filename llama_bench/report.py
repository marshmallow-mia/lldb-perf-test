"""Report generation for llama-bench."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------

def load_results(jsonl_path: str) -> list[dict]:
    """Load a JSONL file and return one dict per non-empty line."""
    results: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


# ---------------------------------------------------------------------------
# generate_markdown_report
# ---------------------------------------------------------------------------

def _config_summary(config: dict) -> str:
    """Short human-readable summary of a config dict."""
    parts = []
    for key in ("np", "ctx", "n_gpu_layers", "flash_attn", "batch_size", "ubatch_size",
                 "cache_type_k", "cache_type_v", "kv_unified", "cache_reuse",
                 "threads", "threads_batch", "split_mode"):
        if key in config:
            parts.append(f"{key}={config[key]}")
    return ", ".join(parts)


def _config_cli_flags(config: dict) -> str:
    """Render a config dict as llama-server CLI flags."""
    lines = [
        f"  --model {config.get('model_path', '<model>')}",
        f"  --host {config.get('host', '0.0.0.0')}",
        f"  --port {config.get('port', 5001)}",
        f"  -np {config.get('np', 1)}",
        f"  -c {config.get('ctx', 49152)}",
        f"  --n-gpu-layers {config.get('n_gpu_layers', 45)}",
        f"  --flash-attn {'on' if config.get('flash_attn', True) else 'off'}",
        f"  --batch-size {config.get('batch_size', 1536)}",
        f"  --ubatch-size {config.get('ubatch_size', 512)}",
        f"  --cache-type-k {config.get('cache_type_k', 'q8_0')}",
        f"  --cache-type-v {config.get('cache_type_v', 'q8_0')}",
        f"  --cache-reuse {config.get('cache_reuse', 512)}",
        f"  --threads {config.get('threads', 8)}",
        f"  --threads-batch {config.get('threads_batch', 8)}",
        f"  --split-mode {config.get('split_mode', 'none')}",
    ]
    if config.get("kv_unified", True):
        lines.append("  --kv-unified")
    if config.get("cont_batching", True):
        lines.append("  --cont-batching")
    return "llama-server \\\n" + " \\\n".join(lines)


def generate_markdown_report(results: list[dict], output_path: str) -> str:
    """Generate a Markdown report from *results* and write to *output_path*.

    Returns the report content as a string.
    """
    now = datetime.now(timezone.utc).isoformat()
    total_runs = len(results)

    # Flatten all individual metric runs from search results
    all_runs: list[dict] = []
    for r in results:
        metrics_list = r.get("metrics", [])
        if not metrics_list:
            # Treat the result itself as a run summary
            all_runs.append(r)
        else:
            for m in metrics_list:
                m = dict(m)
                m["_config"] = r.get("config", {})
                m["_phase"] = r.get("phase", 0)
                all_runs.append(m)

    success_count = sum(1 for r in all_runs if r.get("success", False))
    success_rate = (success_count / total_runs * 100) if total_runs else 0.0

    # Sort by e2e latency ascending (failures last)
    def _sort_key(r: dict) -> float:
        client = r.get("client", {})
        if not r.get("success", False):
            return float("inf")
        return float(client.get("end_to_end_latency_ms", float("inf")))

    sorted_runs = sorted(all_runs, key=_sort_key)

    # Best config
    best_run = next((r for r in sorted_runs if r.get("success", False)), None)

    # Failure summary
    failure_counts: dict[str, int] = {}
    for r in all_runs:
        if not r.get("success", False):
            reason = r.get("failure_reason") or "unknown"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    lines: list[str] = []
    lines.append("# llama-bench Report\n")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Total runs:** {total_runs}  ")
    lines.append(f"**Success rate:** {success_rate:.1f}%  ")
    lines.append(f"**Source:** `{output_path}`\n")

    # Best configuration
    lines.append("## Best Configuration\n")
    if best_run:
        config = best_run.get("_config", {})
        client = best_run.get("client", {})
        lines.append(f"**End-to-end latency:** {client.get('end_to_end_latency_ms', 0):.1f} ms  ")
        lines.append(f"**TTFT:** {client.get('ttft_ms', 0):.1f} ms  ")
        lines.append(f"**Decode tok/s:** {client.get('streaming_tok_per_s', 0):.2f}  \n")
        lines.append("```bash")
        lines.append(_config_cli_flags(config))
        lines.append("```\n")
    else:
        lines.append("_No successful runs._\n")

    # Top 10 results table
    lines.append("## Top 10 Results\n")
    lines.append(
        "| Rank | Config Summary | E2E Latency (ms) | TTFT (ms) | Decode tok/s | Success |"
    )
    lines.append(
        "|------|---------------|-----------------|-----------|-------------|---------|"
    )
    for rank, run in enumerate(sorted_runs[:10], start=1):
        config = run.get("_config", {})
        client = run.get("client", {})
        success = run.get("success", False)
        e2e = f"{client.get('end_to_end_latency_ms', 0):.1f}" if success else "—"
        ttft = f"{client.get('ttft_ms', 0):.1f}" if success else "—"
        tps = f"{client.get('streaming_tok_per_s', 0):.2f}" if success else "—"
        summary = _config_summary(config)[:80]
        lines.append(
            f"| {rank} | `{summary}` | {e2e} | {ttft} | {tps} | {'✓' if success else '✗'} |"
        )
    lines.append("")

    # Failure summary
    if failure_counts:
        lines.append("## Failure Summary\n")
        lines.append("| Failure Reason | Count |")
        lines.append("|---------------|-------|")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")

    # JSONL reference
    lines.append(f"## Source Data\n\nFull results: `{output_path}`\n")

    content = "\n".join(lines)

    abs_output = os.path.abspath(output_path)
    parent_dir = os.path.dirname(abs_output)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(abs_output, "w", encoding="utf-8") as fh:
        fh.write(content)

    return content


# ---------------------------------------------------------------------------
# print_summary_table
# ---------------------------------------------------------------------------

def print_summary_table(results: list[dict]) -> None:
    """Print a rich table summarising *results* to the console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="llama-bench Results Summary", show_lines=False)
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Phase", justify="right")
    table.add_column("Success", justify="center")
    table.add_column("E2E ms", justify="right")
    table.add_column("TTFT ms", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Failure Reason")

    for r in results:
        metrics_list = r.get("metrics", [r])
        for m in metrics_list:
            client = m.get("client", {})
            success = m.get("success", False)
            e2e = f"{client.get('end_to_end_latency_ms', 0):.1f}" if success else "—"
            ttft = f"{client.get('ttft_ms', 0):.1f}" if success else "—"
            tps = f"{client.get('streaming_tok_per_s', 0):.2f}" if success else "—"
            reason = m.get("failure_reason") or ""
            table.add_row(
                m.get("run_id", r.get("run_id", "")),
                str(r.get("phase", "")),
                "[green]✓[/green]" if success else "[red]✗[/red]",
                e2e,
                ttft,
                tps,
                reason,
            )

    console.print(table)



# ---------------------------------------------------------------------------
# generate_bench_report  (preset-bench / 4-phase characterization)
# ---------------------------------------------------------------------------


def generate_bench_report(result: "OrchestratorResult", output_path: str) -> str:  # noqa: F821
    """Generate a Markdown report from a preset-bench OrchestratorResult.

    Args:
        result:      :class:`~llama_bench.orchestrator.OrchestratorResult`.
        output_path: Path to write the Markdown file (parent dirs created).

    Returns:
        The report content as a string.
    """
    from llama_bench.presets import ALL_PRESET_NAMES, PRESET_REGISTRY

    now = datetime.now(timezone.utc).isoformat()
    goal = result.goal
    sr = result.scoring_result

    lines: list[str] = []
    lines.append("# llama-bench Preset Bench Report\n")
    lines.append(f"**Generated:** {now}  ")
    if goal is not None:
        lines.append(f"**Goal preset:** {goal.name}  ")
        weight_parts = ", ".join(
            f"{name}={goal.weight_for(name)}" for name in ALL_PRESET_NAMES
        )
        lines.append(f"**Weights (out of 20):** {weight_parts}  ")
    if result.interrupted:
        lines.append("**Status:** Interrupted (partial results)  ")
    else:
        lines.append("**Status:** Complete  ")
    lines.append("")

    # ---- Per-preset summary ----
    lines.append("## Phase Results\n")
    lines.append("| Preset | Primary Metric | Best Value | Status |")
    lines.append("|--------|---------------|------------|--------|")
    for name in ALL_PRESET_NAMES:
        pr = result.preset_results.get(name)
        if pr is None:
            lines.append(f"| {name} | — | — | not run |")
            continue
        metric = pr.primary_metric
        if pr.skipped:
            lines.append(f"| {name} | {metric} | — | skipped (weight=0) |")
        elif pr.best_attempt is None:
            lines.append(f"| {name} | {metric} | — | ✗ no success |")
        else:
            val = f"{pr.best_value:.3f}" if pr.best_value is not None else "—"
            lines.append(f"| {name} | {metric} | {val} | ✓ |")
    lines.append("")

    # ---- Weighted scoring ----
    if sr is not None and sr.config_scores:
        lines.append("## Top Configurations (Weighted Score)\n")
        lines.append("| Rank | ngl | batch | Weighted Score |" + "".join(
            f" {name} (w={goal.weight_for(name) if goal else '?'}) |"
            for name in ALL_PRESET_NAMES
        ))
        sep_cols = "|------|-----|-------|----------------|" + "".join(
            "---|" for _ in ALL_PRESET_NAMES
        )
        lines.append(sep_cols)
        for rank, cs in enumerate(sr.config_scores[:10], 1):
            preset_cols = "".join(
                f" {cs.normalized_scores.get(n, 0.0):.3f} |" for n in ALL_PRESET_NAMES
            )
            lines.append(
                f"| {rank} | {cs.ngl} | {cs.batch_size} "
                f"| {cs.weighted_score:.3f} |{preset_cols}"
            )
        lines.append("")

        # ---- Recommendation ----
        if sr.recommendation:
            lines.append("## Recommendation\n")
            lines.append("```")
            lines.append(sr.recommendation)
            lines.append("```\n")

        # ---- Optimal server command ----
        if sr.optimal is not None:
            opt = sr.optimal
            # Find the base config from any attempt with matching ngl/batch
            base_cfg: Optional[dict] = None
            for pr in result.preset_results.values():
                if pr.skipped or not pr.attempts:
                    continue
                for att in pr.attempts:
                    if (getattr(att, "n_gpu_layers", None) == opt.ngl and
                            getattr(att, "batch_size", None) == opt.batch_size and
                            getattr(att, "success", False)):
                        # Build a config dict from the attempt
                        base_cfg = {
                            "n_gpu_layers": opt.ngl,
                            "batch_size": opt.batch_size,
                            "ctx": getattr(att, "ctx", "?"),
                            "flash_attn": True,
                            "kv_unified": True,
                            "cont_batching": True,
                            "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0",
                            "cache_reuse": 512,
                            "threads": 8,
                            "threads_batch": 8,
                            "split_mode": "none",
                            "np": 1,
                        }
                        break
                if base_cfg is not None:
                    break
            if base_cfg:
                lines.append("## Recommended llama-server Command\n")
                lines.append("```bash")
                lines.append(_config_cli_flags(base_cfg))
                lines.append("```\n")
    else:
        lines.append("\n_Scoring unavailable (insufficient results)._\n")

    lines.append(f"## Source\n\nReport generated by llama-bench preset bench. Output: `{output_path}`\n")

    content = "\n".join(lines)

    abs_output = os.path.abspath(output_path)
    parent_dir = os.path.dirname(abs_output)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(abs_output, "w", encoding="utf-8") as fh:
        fh.write(content)

    return content