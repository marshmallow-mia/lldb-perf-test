"""Click CLI entry points for llama-bench."""
from __future__ import annotations

import logging
import os
import signal
import shutil
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

import click
from rich.console import Console

console = Console(stderr=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REASON_LABELS: dict[str, str] = {
    "out_of_vram": "Out of VRAM",
    "server_exited": "Server crashed",
    "server_startup_timeout": "Startup timeout",
    "ttft_exceeded": "TTFT too slow",
    "throughput_too_low": "Throughput too low",
    "model_not_found": "Model not found",
    "http_error": "HTTP error",
    "oom": "OOM",
}

_REASON_HINTS: dict[str, str] = {
    "out_of_vram": "Reduce --n-gpu-layers or --ctx",
    "server_exited": "Check server log for missing libs or wrong binary",
    "server_startup_timeout": "Model loading slowly \u2014 try smaller ctx or faster storage",
    "ttft_exceeded": "Reduce --ctx or lower --n-gpu-layers to free compute headroom",
    "throughput_too_low": "Reduce --ctx or increase --n-gpu-layers to offload more layers",
    "model_not_found": "Verify the model path exists and is readable",
    "http_error": "Server returned unexpected HTTP error \u2014 check server log",
    "oom": "Reduce --n-gpu-layers or --batch-size",
}

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _default_output() -> str:
    return os.path.join("results", f"bench_{_timestamp()}.jsonl")


def _discover_and_print_gpus() -> str:
    from llama_bench.gpu import default_vk_devices, discover_vulkan_gpus
    gpus = discover_vulkan_gpus()
    if gpus:
        console.print(f"[green]Vulkan GPUs discovered:[/] {len(gpus)}")
        for g in gpus:
            console.print(f"  [{g['index']}] {g['name']}")
    else:
        console.print("[yellow]No Vulkan GPUs discovered (vulkaninfo not available).[/]")
    return default_vk_devices(gpus)


def _resolve_server_path(server: str) -> str:
    """Resolve *server* to an absolute path and validate it.

    Handles:

    * ``~`` expansion (``~/bin/llama-server`` → ``/home/user/bin/llama-server``)
    * relative paths (``./llama-server`` → ``/cwd/llama-server``)
    * bare names on PATH (``llama-server`` → ``/venv/bin/llama-server``)

    Exits with a clear error message if the binary is not found or not
    executable.
    """
    path = os.path.expanduser(server)
    if not os.path.isabs(path):
        # Try resolving as a relative path first
        candidate = os.path.abspath(path)
        if os.path.isfile(candidate):
            path = candidate
        else:
            # Fall back to PATH lookup for bare names (e.g. "llama-server")
            resolved = shutil.which(server)
            if resolved:
                path = resolved
            else:
                path = candidate  # keep absolute path so error message is clear

    if not os.path.isfile(path):
        console.print(f"[red]Error:[/] llama-server binary not found: {path!r}")
        console.print(
            "[yellow]Tip:[/] Provide an absolute path, e.g. "
            "--server /usr/local/bin/llama-server"
        )
        sys.exit(1)

    if not os.access(path, os.X_OK):
        console.print(f"[red]Error:[/] llama-server binary is not executable: {path!r}")
        sys.exit(1)

    logger.info("Resolved server path: %s", path)
    return path


def _check_version(server: str, use_sudo: bool = False) -> None:
    from llama_bench.runner import check_version_mismatch
    logger.info("Running version check for %s", server)
    warning = check_version_mismatch(server, use_sudo)
    if warning:
        logger.info("Version check result: %s", warning)
        console.print(f"[yellow]Version warning:[/] {warning}")
    else:
        logger.info("Version check passed")


def _print_validation(cfg) -> None:
    from llama_bench.config import validate_config
    warnings = validate_config(cfg)
    for w in warnings:
        console.print(f"[yellow]Config warning:[/] {w}")


def _setup_graceful_shutdown() -> tuple[threading.Event, object]:
    """Install a SIGINT handler for graceful shutdown in no-TUI mode.

    Returns ``(stop_event, original_handler)``.  First Ctrl+C sets the event
    and restores the original handler so a second Ctrl+C force-kills.
    """
    stop_event = threading.Event()
    original = signal.getsignal(signal.SIGINT)

    def _handler(sig: int, frame: object) -> None:
        console.print(
            "\n[yellow]Stopping after current run\u2026 generating final report.[/]"
        )
        stop_event.set()
        # Restore original handler so a second Ctrl+C force-kills.
        signal.signal(signal.SIGINT, original)

    signal.signal(signal.SIGINT, _handler)
    return stop_event, original


def _restore_signal(original: object) -> None:
    """Restore the original SIGINT handler."""
    signal.signal(signal.SIGINT, original)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.version_option("0.1.0", prog_name="llama-bench")
@click.option("-v", "--verbose", "verbosity", count=True,
              help="Verbose output; use -vv for debug-level logging.")
@click.pass_context
def main(ctx: click.Context, verbosity: int) -> None:
    """llama-bench: benchmarking and optimizer CLI for llama.cpp llama-server."""
    ctx.ensure_object(dict)
    ctx.obj["verbosity"] = verbosity
    if ctx.invoked_subcommand is None:
        from llama_bench.menu_app import run_menu
        run_menu()

# ---------------------------------------------------------------------------
# bench command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--server", "-s", default="./llama-server", show_default=True,
              help="Path to llama-server binary.")
@click.option("--model", "-m", required=True, help="Path to model file (.gguf).")
@click.option("--host", default="0.0.0.0", show_default=True, help="Server host.")
@click.option("--port", "-p", default=5001, show_default=True, help="Server port.")
@click.option("--np", default=1, show_default=True, help="Number of parallel slots.")
@click.option("--ctx", "-c", default=49152, show_default=True,
              help="Starting context size (tokens); used as ctx_max if --ctx-max not set.")
@click.option("--n-gpu-layers", "-ngl", "n_gpu_layers", default=45, show_default=True,
              help="Layers to offload to GPU.")
@click.option("--flash-attn/--no-flash-attn", default=True, show_default=True,
              help="Enable Flash Attention.")
@click.option("--batch-size", default=1536, show_default=True, help="Logical batch size.")
@click.option("--ubatch-size", default=512, show_default=True, help="Micro batch size.")
@click.option("--cache-type-k", default="q8_0", show_default=True, help="KV cache K quantisation.")
@click.option("--cache-type-v", default="q8_0", show_default=True, help="KV cache V quantisation.")
@click.option("--kv-unified/--no-kv-unified", default=True, show_default=True,
              help="Unified KV cache pool.")
@click.option("--cache-reuse", default=512, show_default=True,
              help="Min prefix tokens for KV cache reuse.")
@click.option("--cont-batching/--no-cont-batching", default=True, show_default=True,
              help="Enable continuous batching.")
@click.option("--threads", default=8, show_default=True, help="CPU inference threads.")
@click.option("--threads-batch", default=8, show_default=True, help="CPU batch threads.")
@click.option("--split-mode", default="none", show_default=True,
              type=click.Choice(["none", "layer", "row"]), help="GPU split mode.")
@click.option("--vk-devices", default=None,
              help="GGML_VK_VISIBLE_DEVICES value (auto-discover if omitted).")
@click.option("--sudo/--no-sudo", default=True, show_default=True,
              help="Launch server with sudo.")
# --- Tuner bounds ---
@click.option("--ctx-min", default=8192, show_default=True,
              help="Minimum context size to try.")
@click.option("--ctx-max", default=None, type=int,
              help="Maximum context size to try (defaults to --ctx value).")
@click.option("--ctx-step", default=8192, show_default=True,
              help="Context step size between candidates.")
@click.option("--ngl-step", default=4, show_default=True,
              help="n-gpu-layers reduction step on OOM.")
@click.option("--batch-step", default=256, show_default=True,
              help="batch-size reduction step on OOM.")
@click.option("--max-retries", default=5, show_default=True,
              help="Max retry attempts per candidate config on OOM.")
# --- Performance thresholds ---
@click.option("--max-ttft", default=60.0, show_default=True,
              help="Maximum acceptable TTFT in seconds.")
@click.option("--min-tokens-per-sec", default=1.0, show_default=True,
              help="Minimum acceptable decode throughput (tokens/s).")
@click.option("--ctx-pct-threshold", default=0.90, show_default=True,
              help="Fraction of max usable ctx for secondary throughput ranking.")
# --- Output ---
@click.option("--output", "-o", default=None,
              help="JSONL output path (default: results/bench_<timestamp>.jsonl).")
@click.option("--summary", default=None,
              help="Summary JSON path (default: results/summary.json).")
@click.option("--n-followups", default=4, show_default=True,
              help="Number of follow-up prompts per configuration run.")
@click.option("--max-tokens", "max_tokens", default=512, show_default=True,
              help="Max tokens the model generates per request (affects speed).")

@click.option("--prompt-pack", default=None, type=click.Path(exists=True),
              help="Path to a custom prompt pack (JSON/YAML).")
@click.option("--no-tui", is_flag=True, default=False, help="Disable TUI, use plain output.")
@click.option("-v", "--verbose", "verbosity", count=True,
              help="Verbose output; use -vv for debug-level logging.")
@click.option("--tensor-split", default="", show_default=True,
              help="Tensor split fractions across GPUs, comma-separated (e.g. '3,7').")
@click.option("--main-gpu", "main_gpu", default=0, show_default=True,
              help="Index of the primary GPU (for split-mode row/layer).")
@click.option("--kv-offload/--no-kv-offload", default=True, show_default=True,
              help="Offload KV cache to GPU memory.")
@click.option("--mmap/--no-mmap", default=True, show_default=True,
              help="Memory-map the model file (disable to load fully into RAM, no page faults).")
@click.option("--mlock", is_flag=True, default=False,
              help="Lock model in physical RAM (prevent swapping).")
@click.option("--numa", default=None, type=click.Choice(["distribute", "isolate", "numactl"]),
              help="NUMA affinity mode for multi-socket systems.")
@click.option("--prio", default=0, show_default=True,
              type=click.IntRange(-1, 3),
              help="Process/thread priority: -1=low, 0=normal, 1=medium, 2=high, 3=realtime.")
@click.option("--poll", default=-1, show_default=True,
              help="CPU busy-wait poll level 0-100 (0=sleep, 50=default, 100=spin); -1=server default.")
@click.option("--cpu-mask", "cpu_mask", default=None,
              help="CPU affinity hex mask for inference threads.")
@click.option("--cpu-range", "cpu_range", default=None,
              help="CPU core range for inference threads, e.g. '0-7'.")
@click.option("--threads-http", "threads_http", default=-1, show_default=True,
              help="HTTP server thread count (-1 = server default).")
@click.option("--defrag-thold", "defrag_thold", default=-1.0, show_default=True,
              help="KV cache defragmentation threshold 0.0-1.0 (-1 = server default).")
@click.option("--slot-prompt-similarity", "slot_prompt_similarity", default=-1.0, show_default=True,
              help="Slot reuse similarity threshold 0.0-1.0 (-1 = server default 0.10).")
@click.option("--grp-attn-n", "grp_attn_n", default=-1, show_default=True,
              help="Grouped-attention factor N for Self-Extend (-1 = off).")
@click.option("--grp-attn-w", "grp_attn_w", default=-1, show_default=True,
              help="Grouped-attention context width W for Self-Extend (-1 = off).")
@click.option("--context-shift/--no-context-shift", "context_shift", default=False, show_default=True,
              help="Enable automatic KV cache context shifting for long generation.")
@click.option("--rope-scaling", "rope_scaling", default=None,
              type=click.Choice(["none", "linear", "yarn"]),
              help="RoPE scaling type for extended context.")
@click.option("--rope-freq-base", "rope_freq_base", default=-1.0, show_default=True,
              help="RoPE base frequency override (-1 = model default).")
@click.option("--rope-freq-scale", "rope_freq_scale", default=-1.0, show_default=True,
              help="RoPE frequency scale factor for context extension (-1 = model default).")
@click.option("--model-draft", "model_draft", default=None, type=click.Path(),
              help="Path to draft model .gguf for speculative decoding.")
@click.option("--draft-n", "draft_n", default=-1, show_default=True,
              help="Number of tokens to draft per speculative cycle (-1 = server default 16).")
@click.option("--draft-n-min", "draft_n_min", default=-1, show_default=True,
              help="Minimum accepted draft tokens per cycle (-1 = server default 0).")
@click.option("--n-gpu-layers-draft", "n_gpu_layers_draft", default=-1, show_default=True,
              help="GPU layers to offload for the draft model (-1 = server default).")
@click.option("--goal", default="general",
              type=click.Choice(["reverse_engineering", "coding", "chatting", "rag_research", "general", "custom"]),
              show_default=True,
              help="Optimization goal preset that weights the 4 benchmark phases.")
@click.option("--w-max-context", "w_max_context", default=5, show_default=True, type=int,
              help="Weight for max-context phase (goal=custom only; all 4 weights must sum to 20).")
@click.option("--w-fastest-response", "w_fastest_response", default=5, show_default=True, type=int,
              help="Weight for fastest-response phase (goal=custom only).")
@click.option("--w-throughput", "w_throughput", default=5, show_default=True, type=int,
              help="Weight for throughput phase (goal=custom only).")
@click.option("--w-long-context-rag", "w_long_context_rag", default=5, show_default=True, type=int,
              help="Weight for long-context RAG phase (goal=custom only).")
@click.option("--ngl-tests", "ngl_tests", default=None,
              help="ngl values to sweep in grid presets, comma-separated (e.g. '40,43,45'). Default: base ngl only.")
@click.option("--batch-tests", "batch_tests", default=None,
              help="batch_size values to sweep in grid presets, comma-separated (e.g. '1024,1536'). Default: base batch only.")
@click.pass_context
def bench(
    click_ctx, server, model, host, port, np, ctx, n_gpu_layers, flash_attn,
    batch_size, ubatch_size, cache_type_k, cache_type_v, kv_unified,
    cache_reuse, cont_batching, threads, threads_batch, split_mode,
    vk_devices, sudo,
    ctx_min, ctx_max, ctx_step, ngl_step, batch_step, max_retries,
    max_ttft, min_tokens_per_sec, ctx_pct_threshold,
    output, summary, prompt_pack, no_tui, verbosity,
    n_followups, max_tokens,
    # New args
    tensor_split, main_gpu, kv_offload,
    mmap, mlock, numa, prio, poll, cpu_mask, cpu_range,
    threads_http,
    defrag_thold, slot_prompt_similarity,
    grp_attn_n, grp_attn_w, context_shift,
    rope_scaling, rope_freq_base, rope_freq_scale,
    model_draft, draft_n, draft_n_min, n_gpu_layers_draft,
    # Preset bench args
    goal, w_max_context, w_fastest_response, w_throughput, w_long_context_rag,
    ngl_tests, batch_tests,
) -> None:
    """4-preset characterization bench: finds the optimal server config for your use-case goal."""
    import json as _json
    from llama_bench.config import configs_from_args, parse_range
    from llama_bench.logging_setup import setup_logging
    from llama_bench.orchestrator import OrchestratorResult, PresetBenchOrchestrator
    from llama_bench.presets import get_goal_preset

    output_path = output or _default_output()
    artifacts_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(artifacts_dir, exist_ok=True)
    log_file = setup_logging(max(verbosity, click_ctx.obj.get("verbosity", 0)), artifacts_dir)
    if log_file:
        console.print(f"[dim]Verbose log: {log_file}[/]")

    server = _resolve_server_path(server)

    if vk_devices is None:
        vk_devices = _discover_and_print_gpus()
    else:
        console.print(f"Using VK devices: {vk_devices}")

    if split_mode == "none" and vk_devices and "," in str(vk_devices):
        split_mode = "layer"
        console.print("[dim]Auto-set split-mode=layer for multi-GPU[/]")

    _check_version(server, sudo)

    effective_ctx_max = ctx_max if ctx_max is not None else ctx

    base_cfg = configs_from_args(
        server=server, model=model, host=host, port=port, np=np, ctx=effective_ctx_max,
        n_gpu_layers=n_gpu_layers, flash_attn=flash_attn, batch_size=batch_size,
        ubatch_size=ubatch_size, cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        kv_unified=kv_unified, kv_offload=kv_offload, cache_reuse=cache_reuse,
        cont_batching=cont_batching, threads=threads, threads_batch=threads_batch,
        threads_http=threads_http, split_mode=split_mode, tensor_split=tensor_split,
        main_gpu=main_gpu, vk_devices=vk_devices, use_sudo=sudo,
        max_predict_tokens=max_tokens,
        mmap=mmap, mlock=mlock, numa=numa, prio=prio, poll=poll,
        cpu_mask=cpu_mask, cpu_range=cpu_range,
        defrag_thold=defrag_thold, slot_prompt_similarity=slot_prompt_similarity,
        grp_attn_n=grp_attn_n, grp_attn_w=grp_attn_w, context_shift=context_shift,
        rope_scaling=rope_scaling, rope_freq_base=rope_freq_base, rope_freq_scale=rope_freq_scale,
        model_draft=model_draft, draft_n=draft_n, draft_n_min=draft_n_min,
        n_gpu_layers_draft=n_gpu_layers_draft,
    )
    _print_validation(base_cfg)

    # Resolve goal preset
    try:
        goal_preset = get_goal_preset(
            goal,
            w_max_context=w_max_context,
            w_fastest_response=w_fastest_response,
            w_throughput=w_throughput,
            w_long_context_rag=w_long_context_rag,
        )
    except (ValueError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc

    # Parse ngl/batch sweep lists (reuse existing parse_range from config.py)
    ngl_list: Optional[list[int]] = (
        [int(x) for x in parse_range(ngl_tests)] if ngl_tests else None
    )
    batch_list: Optional[list[int]] = (
        [int(x) for x in parse_range(batch_tests)] if batch_tests else None
    )

    summary_path = summary or os.path.join(artifacts_dir, "summary.json")

    console.print(
        f"[bold]Preset bench[/] goal=[cyan]{goal_preset.name}[/] "
        f"weights={goal_preset.as_dict()} | output → {output_path}"
    )

    if no_tui:
        stop_event, orig_handler = _setup_graceful_shutdown()
        orchestrator = PresetBenchOrchestrator(
            base_cfg=base_cfg,
            goal=goal_preset,
            event_cb=None,
            stop_event=stop_event,
            artifacts_dir=artifacts_dir,
            log_file=log_file,
            ngl_values=ngl_list,
            batch_values=batch_list,
        )
        result = orchestrator.run()
        _restore_signal(orig_handler)
    else:
        from llama_bench.hw_monitor import HWMonitor
        from llama_bench.tui import BenchTUI
        hw_monitor = HWMonitor()
        hw_monitor.start()
        tui = BenchTUI(hw_monitor=hw_monitor, bench_log_path=log_file)

        orchestrator = PresetBenchOrchestrator(
            base_cfg=base_cfg,
            goal=goal_preset,
            event_cb=tui.handle_event,
            stop_event=tui._stop_event,
            artifacts_dir=artifacts_dir,
            log_file=log_file,
            ngl_values=ngl_list,
            batch_values=batch_list,
        )
        tui_result = tui.run(orchestrator.run)
        hw_monitor.stop()
        # tui.run() returns the orchestrator's return value, or None on early exit.
        if tui_result is None:
            partial = orchestrator.accumulated_results
            result = OrchestratorResult(
                preset_results=partial,
                scoring_result=None,
                goal=goal_preset,
                completed_presets=list(partial.keys()),
                interrupted=True,
            )
        else:
            result = tui_result

    # --- Final report (always runs, even on early quit) ---
    _finalize_preset_bench(result, output_path, summary_path, _json)


def _finalize_preset_bench(result, output_path, summary_path, _json) -> None:
    """Write JSONL + print scoring summary for a PresetBenchOrchestrator result."""
    from llama_bench.tuner import _attempt_to_dict as _atd

    # Collect all TuneAttempt records across all presets
    all_attempts = []
    for preset_result in result.preset_results.values():
        if not preset_result.skipped:
            all_attempts.extend(preset_result.attempts)

    real_attempts = [
        a for a in all_attempts
        if getattr(a, "failure_reason", None) != "stop_requested"
    ]

    if not real_attempts:
        console.print("[yellow]Run stopped before any measurement completed — no results.[/]")
    else:
        with open(output_path, "w", encoding="utf-8") as fh:
            for a in real_attempts:
                fh.write(_json.dumps(_atd(a)) + "\n")
        console.print(f"[green]Results saved to[/] {output_path}")

    # Write Markdown report
    report_path = output_path.replace(".jsonl", "_report.md") if output_path.endswith(".jsonl") else output_path + "_report.md"
    try:
        from llama_bench.report import generate_bench_report
        generate_bench_report(result, report_path)
        console.print(f"[green]Report saved to[/] {report_path}")
    except Exception as _exc:
        console.print(f"[yellow]Report generation failed:[/] {_exc}")

    _print_preset_bench_summary(result)

    # Check for model-not-found (critical error — abort after printing results)
    for a in all_attempts:
        if getattr(a, "failure_reason", None) == "model_not_found":
            import textwrap
            stderr_hint = (
                f"\n  Server stderr log: {a.stderr_path}" if getattr(a, 'stderr_path', None) else ""
            )
            excerpt_hint = ""
            if getattr(a, 'server_error_excerpt', None):
                indented = textwrap.indent(a.server_error_excerpt, "    ")
                excerpt_hint = f"\n  Excerpt:\n{indented}"
            raise click.ClickException(
                f"Critical error: model not found.{stderr_hint}{excerpt_hint}"
            )


def _print_preset_bench_summary(result) -> None:
    """Print per-preset results and weighted scoring recommendation."""
    from rich.panel import Panel
    from rich.table import Table

    console.print()
    if result.goal:
        weights = result.goal.as_dict()
        w_str = "  ".join(f"{k}:{v}" for k, v in weights.items())
        console.print(f"[bold]Goal:[/] {result.goal.name}  [dim]{w_str}[/]")
    if result.interrupted:
        console.print("[yellow]Run was interrupted — results may be partial.[/]")

    # Per-preset summary table
    table = Table(title="Preset Results", show_lines=False, expand=False)
    table.add_column("Preset", style="bold")
    table.add_column("Metric", justify="left")
    table.add_column("Best Value", justify="right")
    table.add_column("Status", justify="center")

    for name, pr in result.preset_results.items():
        if pr.skipped:
            table.add_row(name, pr.primary_metric, "\u2014", "[dim]skipped[/]")
        elif pr.best_attempt is None:
            table.add_row(name, pr.primary_metric, "\u2014", "[red]\u2717 no success[/]")
        else:
            val = f"{pr.best_value:.3f}" if pr.best_value is not None else "\u2014"
            table.add_row(name, pr.primary_metric, val, "[green]\u2713[/]")
    console.print(table)

    sr = result.scoring_result
    if sr is None:
        console.print("[yellow]Scoring unavailable (insufficient results).[/]")
        return

    scores_table = Table(title="Top Configurations (Weighted Score)", show_lines=False)
    scores_table.add_column("Rank", justify="right", style="dim")
    scores_table.add_column("ngl", justify="right")
    scores_table.add_column("batch", justify="right")
    scores_table.add_column("Score", justify="right")
    for rank, cs in enumerate(sr.config_scores[:5], 1):
        scores_table.add_row(
            str(rank), str(cs.ngl), str(cs.batch_size),
            f"{cs.weighted_score:.3f}",
        )
    console.print(scores_table)

    console.print(Panel(
        f"[bold green]{sr.recommendation}[/]",
        title="[bold]Recommendation[/]",
        expand=False,
    ))

def _finalize_bench(attempts, output_path, summary_path, ctx_pct_threshold, _json):
    """Write JSONL + summary + print table.  Runs even on early quit."""
    from llama_bench.tuner import (
        _attempt_to_dict as _atd,
        select_best_configs,
        write_summary_json,
    )

    if not attempts:
        console.print("[yellow]Run stopped before any measurement started — no results.[/]")
        return

    # Separate true results from stop_requested noise.
    real_attempts = [a for a in attempts if getattr(a, 'failure_reason', None) != 'stop_requested']
    if not real_attempts:
        console.print(
            "[yellow]Run was stopped before the first measurement completed — no results.[/]\n"
            "[dim]The server was still starting up when you pressed q.  "
            "Let the server finish loading before quitting to get results.[/]"
        )
        return

    # Always write JSONL (successful + failed-for-real, not stop_requested)
    with open(output_path, "w", encoding="utf-8") as fh:
        for a in real_attempts:
            fh.write(_json.dumps(_atd(a)) + "\n")
    console.print(f"[green]Results saved to[/] {output_path}")

    # Multi-objective selection + summary
    selection = select_best_configs(real_attempts, ctx_pct_threshold=ctx_pct_threshold)
    write_summary_json(summary_path, real_attempts, selection)
    console.print(f"[green]Summary written to[/] {summary_path}")

    # Print final summary table
    # Print final summary table
    _print_tuner_summary(selection, real_attempts)


def _print_tuner_summary(selection, all_attempts=None):
    """Print a Rich table with ALL results + multi-objective best-config suggestions."""
    from rich.table import Table
    from rich.panel import Panel

    max_ctx = selection.get("max_ctx_result")
    top = selection.get("top_throughput", [])
    recommended = selection.get("recommended")

    if max_ctx is None:
        console.print("[yellow]No usable configurations found (all runs failed).[/]")
        return

    # ── All results table ────────────────────────────────────────────────────
    if all_attempts:
        table = Table(title="All Results", show_lines=False, expand=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("ctx", justify="right")
        table.add_column("ngl", justify="right")
        table.add_column("batch", justify="right")
        table.add_column("Cold TTFT", justify="right")
        table.add_column("Warm TTFT", justify="right")
        table.add_column("tok/s", justify="right")
        table.add_column("Status", justify="center")
        for i, a in enumerate(all_attempts, 1):
            ok = getattr(a, "success", False)
            cold = f"{a.ttft_s:.2f}s" if ok and a.ttft_s else "—"
            warm = f"{a.warm_ttft_s:.3f}s" if ok and getattr(a, "warm_ttft_s", None) else "—"
            tps = f"{a.tokens_per_sec:.1f}" if ok and a.tokens_per_sec else "—"
            reason = getattr(a, "failure_reason", None) or ""
            if ok:
                status = "[green]\u2713[/]"
            else:
                label = _REASON_LABELS.get(reason, reason) if reason else "failed"
                hint = _REASON_HINTS.get(reason, "")
                status = f"[red]\u2717 {label}[/]" + (f"  [dim]{hint}[/]" if hint else "")
            table.add_row(
                str(i), str(a.ctx), str(a.n_gpu_layers), str(a.batch_size),
                cold, warm, tps, status,
                style="" if ok else "dim red",
            )
        console.print(table)

    # ── Best-config suggestions ───────────────────────────────────────────────
    lines = []
    lines.append(f"[bold green]Max usable context:[/] {max_ctx.ctx:,} tokens  "
                 f"(ngl={max_ctx.n_gpu_layers}, batch={max_ctx.batch_size}, "
                 f"tok/s={max_ctx.tokens_per_sec:.1f})")

    if top:
        best_tps = max(top, key=lambda a: a.tokens_per_sec)
        lines.append(f"[bold yellow]Best throughput:[/] {best_tps.tokens_per_sec:.1f} tok/s  "
                     f"ctx={best_tps.ctx:,}, ngl={best_tps.n_gpu_layers}, "
                     f"batch={best_tps.batch_size}, TTFT={best_tps.ttft_s:.2f}s")

    if all_attempts:
        successful = [a for a in all_attempts if getattr(a, "success", False) and a.ttft_s]
        if successful:
            fastest = min(successful, key=lambda a: a.ttft_s)
            lines.append(f"[bold cyan]Fastest TTFT:[/] {fastest.ttft_s:.2f}s  "
                         f"ctx={fastest.ctx:,}, ngl={fastest.n_gpu_layers}, "
                         f"batch={fastest.batch_size}, tok/s={fastest.tokens_per_sec:.1f}")

    if recommended:
        lines.append(f"[bold blue]Recommended (balanced):[/] "
                     f"ctx={recommended.ctx:,}, ngl={recommended.n_gpu_layers}, "
                     f"batch={recommended.batch_size}, "
                     f"tok/s={recommended.tokens_per_sec:.1f}, TTFT={recommended.ttft_s:.2f}s")

    console.print(Panel("\n".join(lines), title="Best Configs", border_style="green"))

# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--server", "-s", default="./llama-server", show_default=True,
              help="Path to llama-server binary.")
@click.option("--model", "-m", required=True, help="Path to model file (.gguf).")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", "-p", default=5001, show_default=True)
@click.option("--np", default=1, show_default=True)
@click.option("--ctx", "-c", default=49152, show_default=True)
@click.option("--n-gpu-layers", "-ngl", "n_gpu_layers", default=45, show_default=True)
@click.option("--flash-attn/--no-flash-attn", default=True, show_default=True)
@click.option("--batch-size", default=1536, show_default=True)
@click.option("--ubatch-size", default=512, show_default=True)
@click.option("--cache-type-k", default="q8_0", show_default=True)
@click.option("--cache-type-v", default="q8_0", show_default=True)
@click.option("--kv-unified/--no-kv-unified", default=True, show_default=True)
@click.option("--cache-reuse", default=512, show_default=True)
@click.option("--cont-batching/--no-cont-batching", default=True, show_default=True)
@click.option("--threads", default=8, show_default=True)
@click.option("--threads-batch", default=8, show_default=True)
@click.option("--split-mode", default="none", show_default=True,
              type=click.Choice(["none", "layer", "row"]))
@click.option("--vk-devices", default=None)
@click.option("--sudo/--no-sudo", default=True, show_default=True)
@click.option("--np-tests", default="1", show_default=True,
              help="Range spec for -np values, e.g. '1-2' or '1,2'.")
@click.option("--ctx-tests", default="49152", show_default=True,
              help="Range spec for -c values.")
@click.option("--ngl-tests", default="45", show_default=True,
              help="Range spec for --n-gpu-layers values.")
@click.option("--max-configs", default=50, show_default=True,
              help="Maximum number of configs to test.")
@click.option("--output", "-o", default=None,
              help="JSONL output path.")
@click.option("--no-tui", is_flag=True, default=False, help="Disable TUI.")
@click.option("-v", "--verbose", "verbosity", count=True,
              help="Verbose output; use -vv for debug-level logging.")
@click.option("--tensor-split", default="", show_default=True)
@click.option("--main-gpu", "main_gpu", default=0, show_default=True)
@click.option("--kv-offload/--no-kv-offload", default=True, show_default=True)
@click.option("--mmap/--no-mmap", default=True, show_default=True)
@click.option("--mlock", is_flag=True, default=False)
@click.option("--numa", default=None, type=click.Choice(["distribute", "isolate", "numactl"]))
@click.option("--prio", default=0, show_default=True, type=click.IntRange(-1, 3))
@click.option("--poll", default=-1, show_default=True)
@click.option("--cpu-mask", "cpu_mask", default=None)
@click.option("--cpu-range", "cpu_range", default=None)
@click.option("--threads-http", "threads_http", default=-1, show_default=True)
@click.option("--defrag-thold", "defrag_thold", default=-1.0, show_default=True)
@click.option("--slot-prompt-similarity", "slot_prompt_similarity", default=-1.0, show_default=True)
@click.option("--grp-attn-n", "grp_attn_n", default=-1, show_default=True)
@click.option("--grp-attn-w", "grp_attn_w", default=-1, show_default=True)
@click.option("--context-shift/--no-context-shift", "context_shift", default=False, show_default=True)
@click.option("--rope-scaling", "rope_scaling", default=None,
              type=click.Choice(["none", "linear", "yarn"]))
@click.option("--rope-freq-base", "rope_freq_base", default=-1.0, show_default=True)
@click.option("--rope-freq-scale", "rope_freq_scale", default=-1.0, show_default=True)
@click.option("--model-draft", "model_draft", default=None, type=click.Path())
@click.option("--draft-n", "draft_n", default=-1, show_default=True)
@click.option("--draft-n-min", "draft_n_min", default=-1, show_default=True)
@click.option("--n-gpu-layers-draft", "n_gpu_layers_draft", default=-1, show_default=True)
@click.pass_context
def search(
    click_ctx, server, model, host, port, np, ctx, n_gpu_layers, flash_attn,
    batch_size, ubatch_size, cache_type_k, cache_type_v, kv_unified,
    cache_reuse, cont_batching, threads, threads_batch, split_mode,
    vk_devices, sudo,
    np_tests, ctx_tests, ngl_tests, max_configs, output, no_tui, verbosity,
    # New args
    tensor_split, main_gpu, kv_offload,
    mmap, mlock, numa, prio, poll, cpu_mask, cpu_range,
    threads_http,
    defrag_thold, slot_prompt_similarity,
    grp_attn_n, grp_attn_w, context_shift,
    rope_scaling, rope_freq_base, rope_freq_scale,
    model_draft, draft_n, draft_n_min, n_gpu_layers_draft,
) -> None:
    """Run a staged parameter search over the search space."""
    from llama_bench.config import (
        SearchSpace,
        configs_from_args,
        default_search_space,
        parse_range,
    )
    from llama_bench.search import StagedSearcher, save_results
    from llama_bench.logging_setup import setup_logging

    output_path = output or _default_output()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    log_file = setup_logging(max(verbosity, click_ctx.obj.get("verbosity", 0)), os.path.dirname(os.path.abspath(output_path)))
    if log_file:
        console.print(f"[dim]Verbose log: {log_file}[/]")

    server = _resolve_server_path(server)

    if vk_devices is None:
        vk_devices = _discover_and_print_gpus()

    _check_version(server, sudo)

    base_cfg = configs_from_args(
        server=server, model=model, host=host, port=port, np=np, ctx=ctx,
        n_gpu_layers=n_gpu_layers, flash_attn=flash_attn, batch_size=batch_size,
        ubatch_size=ubatch_size, cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        kv_unified=kv_unified, kv_offload=kv_offload, cache_reuse=cache_reuse,
        cont_batching=cont_batching, threads=threads, threads_batch=threads_batch,
        threads_http=threads_http, split_mode=split_mode, tensor_split=tensor_split,
        main_gpu=main_gpu, vk_devices=vk_devices, use_sudo=sudo,
        mmap=mmap, mlock=mlock, numa=numa, prio=prio, poll=poll,
        cpu_mask=cpu_mask, cpu_range=cpu_range,
        defrag_thold=defrag_thold, slot_prompt_similarity=slot_prompt_similarity,
        grp_attn_n=grp_attn_n, grp_attn_w=grp_attn_w, context_shift=context_shift,
        rope_scaling=rope_scaling, rope_freq_base=rope_freq_base, rope_freq_scale=rope_freq_scale,
        model_draft=model_draft, draft_n=draft_n, draft_n_min=draft_n_min,
        n_gpu_layers_draft=n_gpu_layers_draft,
    )
    _print_validation(base_cfg)

    # Build search space from range specs
    space = default_search_space()
    try:
        space.np_values = parse_range(np_tests)
        space.ctx_values = parse_range(ctx_tests)
        space.n_gpu_layers_values = parse_range(ngl_tests)
    except ValueError as exc:
        console.print(f"[red]Invalid range spec:[/] {exc}")
        sys.exit(1)

    def _progress_cb(current: int, total: int, phase: int, phase_name: str) -> None:
        console.print(f"[dim]Phase {phase} ({phase_name}): {current}/{total}[/]")

    if no_tui:
        stop_event, orig_handler = _setup_graceful_shutdown()
        searcher = StagedSearcher(
            space=space,
            base_cfg=base_cfg,
            artifacts_dir=os.path.dirname(os.path.abspath(output_path)),
            max_configs=max_configs,
            progress_cb=_progress_cb,
            log_file=log_file,
            stop_event=stop_event,
        )
        results = searcher.run()
        _restore_signal(orig_handler)
        # If run() returned empty but partial results exist, use those.
        if not results:
            results = searcher.accumulated_results
    else:
        from llama_bench.tui import BenchTUI
        tui = BenchTUI(bench_log_path=log_file)

        def _tui_cb(current: int, total: int, phase: int, phase_name: str) -> None:
            tui.update_progress(current, total, phase, phase_name)

        searcher = StagedSearcher(
            space=space,
            base_cfg=base_cfg,
            artifacts_dir=os.path.dirname(os.path.abspath(output_path)),
            max_configs=max_configs,
            progress_cb=_tui_cb,
            log_file=log_file,
            stop_event=tui._stop_event,
        )

        def _search_work():
            r = searcher.run()
            best = searcher.best_config()
            if best:
                scores = [x.best_score for x in r if x.best_score < float("inf")]
                tui.set_best(f"np={best.np}, ctx={best.ctx}, ngl={best.n_gpu_layers}",
                    min(scores) if scores else float("inf"),
                )
            return r

        results = tui.run(_search_work)
        # If TUI returned empty but partial results exist, use those.
        if not results:
            results = searcher.accumulated_results

    # --- Final report generation (always runs, even on early quit) ---
    if results:
        save_results(results, output_path)
        console.print(f"[green]Search results saved to[/] {output_path}")
        from llama_bench.report import load_results, print_summary_table
        loaded = load_results(output_path)
        print_summary_table(loaded)
    else:
        console.print("[yellow]No results to report (stopped before any runs completed).[/]")

# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None,
              help="Markdown output path (derived from first input if omitted).")
@click.option("--top-n", default=10, show_default=True, help="Number of top results to show.")
def report(input_files: tuple[str, ...], output: "Optional[str]", top_n: int) -> None:
    """Generate a Markdown report from one or more JSONL result files."""
    from llama_bench.report import generate_markdown_report, load_results, print_summary_table

    all_results: list[dict] = []
    for path in input_files:
        all_results.extend(load_results(path))

    if not all_results:
        console.print("[yellow]No results found in input files.[/]")
        return

    if output is None:
        base = input_files[0].replace(".jsonl", "")
        output = base + "_report.md"

    content = generate_markdown_report(all_results, output)
    console.print(f"[green]Report written to[/] {output}")
    print_summary_table(all_results[:top_n])


# ---------------------------------------------------------------------------
# explore command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--server", "-s", default="./llama-server", show_default=True)
@click.option("--model", "-m", required=True)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", "-p", default=5001, show_default=True)
@click.option("--np", default=1, show_default=True)
@click.option("--ctx", "-c", default=131072, show_default=True,
              help="Maximum context size for the sweep.")
@click.option("--ctx-min", default=8192, show_default=True)
@click.option("--ctx-step", default=8192, show_default=True)
@click.option("--n-gpu-layers", "-ngl", "n_gpu_layers", default=45, show_default=True)
@click.option("--ngl-tests", default=None,
              help="Comma-separated ngl values to sweep, e.g. '37,41,45'. Default: just --n-gpu-layers.")
@click.option("--flash-attn/--no-flash-attn", default=True, show_default=True)
@click.option("--batch-size", default=1536, show_default=True)
@click.option("--batch-tests", default=None,
              help="Comma-separated batch values to sweep, e.g. '1024,1536'. Default: just --batch-size.")
@click.option("--ubatch-size", default=512, show_default=True)
@click.option("--cache-type-k", default="q8_0", show_default=True)
@click.option("--cache-type-v", default="q8_0", show_default=True)
@click.option("--kv-unified/--no-kv-unified", default=True, show_default=True)
@click.option("--cache-reuse", default=512, show_default=True)
@click.option("--cont-batching/--no-cont-batching", default=True, show_default=True)
@click.option("--threads", default=8, show_default=True)
@click.option("--threads-batch", default=8, show_default=True)
@click.option("--split-mode", default="none", show_default=True,
              type=click.Choice(["none", "layer", "row"]))
@click.option("--vk-devices", default=None)
@click.option("--sudo/--no-sudo", default=True, show_default=True)
@click.option("--ngl-step", default=4, show_default=True)
@click.option("--batch-step", default=256, show_default=True)
@click.option("--ngl-min", default=0, show_default=True)
@click.option("--max-retries", default=5, show_default=True)
@click.option("--max-ttft", default=60.0, show_default=True)
@click.option("--min-tokens-per-sec", default=1.0, show_default=True)
@click.option("--n-followups", default=2, show_default=True,
              help="Follow-up prompts per run (default 2 for faster exploration).")
@click.option("--max-tokens", "max_tokens", default=512, show_default=True)
@click.option("--output", "-o", default=None)
@click.option("--no-tui", is_flag=True, default=False)
@click.option("-v", "--verbose", "verbosity", count=True)
@click.option("--tensor-split", default="", show_default=True)
@click.option("--main-gpu", "main_gpu", default=0, show_default=True)
@click.option("--kv-offload/--no-kv-offload", default=True, show_default=True)
@click.option("--mmap/--no-mmap", default=True, show_default=True)
@click.option("--mlock", is_flag=True, default=False)
@click.option("--numa", default=None, type=click.Choice(["distribute", "isolate", "numactl"]))
@click.option("--prio", default=0, show_default=True, type=click.IntRange(-1, 3))
@click.option("--poll", default=-1, show_default=True)
@click.option("--cpu-mask", "cpu_mask", default=None)
@click.option("--cpu-range", "cpu_range", default=None)
@click.option("--threads-http", "threads_http", default=-1, show_default=True)
@click.option("--defrag-thold", "defrag_thold", default=-1.0, show_default=True)
@click.option("--slot-prompt-similarity", "slot_prompt_similarity", default=-1.0, show_default=True)
@click.option("--grp-attn-n", "grp_attn_n", default=-1, show_default=True)
@click.option("--grp-attn-w", "grp_attn_w", default=-1, show_default=True)
@click.option("--context-shift/--no-context-shift", "context_shift", default=False, show_default=True)
@click.option("--rope-scaling", "rope_scaling", default=None,
              type=click.Choice(["none", "linear", "yarn"]))
@click.option("--rope-freq-base", "rope_freq_base", default=-1.0, show_default=True)
@click.option("--rope-freq-scale", "rope_freq_scale", default=-1.0, show_default=True)
@click.option("--model-draft", "model_draft", default=None, type=click.Path())
@click.option("--draft-n", "draft_n", default=-1, show_default=True)
@click.option("--draft-n-min", "draft_n_min", default=-1, show_default=True)
@click.option("--n-gpu-layers-draft", "n_gpu_layers_draft", default=-1, show_default=True)
@click.pass_context
def explore(
    click_ctx, server, model, host, port, np, ctx, ctx_min, ctx_step,
    n_gpu_layers, ngl_tests, flash_attn, batch_size, batch_tests, ubatch_size,
    cache_type_k, cache_type_v, kv_unified, cache_reuse, cont_batching,
    threads, threads_batch, split_mode, vk_devices, sudo,
    ngl_step, batch_step, ngl_min, max_retries,
    max_ttft, min_tokens_per_sec, n_followups, max_tokens, output, no_tui, verbosity,
    # New args
    tensor_split, main_gpu, kv_offload,
    mmap, mlock, numa, prio, poll, cpu_mask, cpu_range,
    threads_http,
    defrag_thold, slot_prompt_similarity,
    grp_attn_n, grp_attn_w, context_shift,
    rope_scaling, rope_freq_base, rope_freq_scale,
    model_draft, draft_n, draft_n_min, n_gpu_layers_draft,
) -> None:
    """Continuous multi-objective explorer: runs until you press q/Ctrl+C.

    Sweeps the (ctx, ngl, batch) space indefinitely and tracks the best config
    for each of five objectives: Max Context, Fastest TTFT, Best Warm TTFT,
    Best Throughput, Best Overall.
    """
    from llama_bench.config import configs_from_args, parse_range
    from llama_bench.explorer import ContinuousExplorer
    from llama_bench.logging_setup import setup_logging
    from llama_bench.tuner import TunerBounds, TunerThresholds

    ts = _timestamp()
    output_path = output or os.path.join("results", f"explore_{ts}.jsonl")
    artifacts_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(artifacts_dir, exist_ok=True)
    log_file = setup_logging(max(verbosity, click_ctx.obj.get("verbosity", 0)), artifacts_dir)
    if log_file:
        console.print(f"[dim]Verbose log: {log_file}[/]")

    server = _resolve_server_path(server)

    if vk_devices is None:
        vk_devices = _discover_and_print_gpus()
    else:
        console.print(f"Using VK devices: {vk_devices}")

    if split_mode == "none" and vk_devices and "," in str(vk_devices):
        split_mode = "layer"
        console.print("[dim]Auto-set split-mode=layer for multi-GPU[/]")

    _check_version(server, sudo)

    base_cfg = configs_from_args(
        server=server, model=model, host=host, port=port, np=np, ctx=ctx,
        n_gpu_layers=n_gpu_layers, flash_attn=flash_attn, batch_size=batch_size,
        ubatch_size=ubatch_size, cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        kv_unified=kv_unified, kv_offload=kv_offload, cache_reuse=cache_reuse,
        cont_batching=cont_batching, threads=threads, threads_batch=threads_batch,
        threads_http=threads_http, split_mode=split_mode, tensor_split=tensor_split,
        main_gpu=main_gpu, vk_devices=vk_devices, use_sudo=sudo,
        max_predict_tokens=max_tokens,
        mmap=mmap, mlock=mlock, numa=numa, prio=prio, poll=poll,
        cpu_mask=cpu_mask, cpu_range=cpu_range,
        defrag_thold=defrag_thold, slot_prompt_similarity=slot_prompt_similarity,
        grp_attn_n=grp_attn_n, grp_attn_w=grp_attn_w, context_shift=context_shift,
        rope_scaling=rope_scaling, rope_freq_base=rope_freq_base, rope_freq_scale=rope_freq_scale,
        model_draft=model_draft, draft_n=draft_n, draft_n_min=draft_n_min,
        n_gpu_layers_draft=n_gpu_layers_draft,
    )
    _print_validation(base_cfg)

    try:
        ngl_values = parse_range(ngl_tests) if ngl_tests else [n_gpu_layers]
        batch_values = parse_range(batch_tests) if batch_tests else [batch_size]
    except ValueError as exc:
        console.print(f"[red]Invalid range spec:[/] {exc}")
        sys.exit(1)

    bounds = TunerBounds(
        ctx_min=ctx_min, ctx_max=ctx, ctx_step=ctx_step,
        ngl_min=ngl_min, ngl_step=ngl_step,
        batch_min=256, batch_step=batch_step,
        max_retries=max_retries,
    )
    thresholds = TunerThresholds(max_ttft_s=max_ttft, min_tokens_per_sec=min_tokens_per_sec)

    ctx_steps = list(range(ctx, ctx_min - 1, -ctx_step))
    total_per_round = len(ngl_values) * len(batch_values) * len(ctx_steps)
    console.print(
        f"[bold]Continuous explore[/] ctx {ctx}→{ctx_min} step {ctx_step} | "
        f"ngl {ngl_values} | batch {batch_values} | "
        f"{total_per_round} configs/round | Press [bold]q[/] to stop"
    )

    if no_tui:
        stop_event, orig_handler = _setup_graceful_shutdown()
        explorer = ContinuousExplorer(
            base_cfg=base_cfg, bounds=bounds, thresholds=thresholds,
            ngl_values=ngl_values, batch_values=batch_values,
            artifacts_dir=artifacts_dir, log_file=log_file,
            n_followups=n_followups, max_tokens=max_tokens,
            output_path=output_path,
            stop_event=stop_event,
        )
        attempts = explorer.run()
        _restore_signal(orig_handler)
        # If run() returned empty but partial results exist, use those.
        if not attempts:
            attempts = explorer.accumulated_attempts
        hof = explorer.hall_of_fame
    else:
        from llama_bench.hw_monitor import HWMonitor
        from llama_bench.tui import BenchTUI
        hw_monitor = HWMonitor()
        hw_monitor.start()
        tui = BenchTUI(hw_monitor=hw_monitor, bench_log_path=log_file)

        explorer = ContinuousExplorer(
            base_cfg=base_cfg, bounds=bounds, thresholds=thresholds,
            ngl_values=ngl_values, batch_values=batch_values,
            artifacts_dir=artifacts_dir, log_file=log_file,
            n_followups=n_followups, max_tokens=max_tokens,
            event_cb=tui.handle_event,
            stop_event=tui._stop_event,
            output_path=output_path,
        )
        attempts = tui.run(explorer.run)
        hw_monitor.stop()
        # If TUI returned empty but partial results exist, use those.
        if not attempts:
            attempts = explorer.accumulated_attempts
        hof = explorer.hall_of_fame

    # --- Final report generation (always runs, even on early quit) ---
    # Filter out stop_requested noise — those are mid-startup interrupts with no data.
    real_attempts = [a for a in attempts if getattr(a, 'failure_reason', None) != 'stop_requested']

    console.print(
        f"\n[bold]Explorer stopped[/] after {hof.total_tested} configs "
        f"across {hof.round_num} round(s)."
    )

    if real_attempts:
        _print_hof_summary(hof)
        console.print(f"[green]Results saved to[/] {output_path}")
    else:
        console.print(
            "[yellow]Run was stopped before the first measurement completed — no results.[/]\n"
            "[dim]The server was still loading when you pressed q.  "
            "Wait for the server to finish loading (watch the Activity log) before quitting.[/]"
        )

def _print_hof_summary(hof) -> None:
    """Print the Hall of Fame as a Rich table."""
    from rich.table import Table
    entries = hof.entries()
    if not entries:
        console.print("[yellow]No successful configs found.[/]")
        return
    table = Table(title="Hall of Fame — Best Configs by Objective", show_lines=False)
    table.add_column("Objective", style="bold")
    table.add_column("ctx", justify="right")
    table.add_column("ngl", justify="right")
    table.add_column("batch", justify="right")
    table.add_column("Cold TTFT", justify="right")
    table.add_column("Warm TTFT", justify="right")
    table.add_column("tok/s", justify="right")
    for e in entries:
        table.add_row(
            e.label, str(e.ctx), str(e.ngl), str(e.batch),
            f"{e.cold_ttft_s:.2f}s", f"{e.warm_ttft_s:.3f}s",
            f"{e.tokens_per_sec:.1f}",
        )
    console.print(table)
