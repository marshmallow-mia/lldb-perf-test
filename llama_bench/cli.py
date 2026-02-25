"""Click CLI entry points for llama-bench."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Optional

import click
from rich.console import Console

console = Console(stderr=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("0.1.0", prog_name="llama-bench")
def main() -> None:
    """llama-bench: benchmarking and optimizer CLI for llama.cpp llama-server."""


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
def bench(
    server, model, host, port, np, ctx, n_gpu_layers, flash_attn,
    batch_size, ubatch_size, cache_type_k, cache_type_v, kv_unified,
    cache_reuse, cont_batching, threads, threads_batch, split_mode,
    vk_devices, sudo,
    ctx_min, ctx_max, ctx_step, ngl_step, batch_step, max_retries,
    max_ttft, min_tokens_per_sec, ctx_pct_threshold,
    output, summary, prompt_pack, no_tui, verbosity,
    n_followups, max_tokens,
) -> None:
    """Adaptive configuration tuner: find the highest usable context for your model.

    Iterates through multiple configurations automatically, adjusts parameters
    on VRAM OOM / fit failures, and emits a ranked summary of usable configs.
    """
    import json as _json
    from llama_bench.config import configs_from_args
    from llama_bench.logging_setup import setup_logging
    from llama_bench.prompts import build_prompt_sequence, load_prompt_pack
    from llama_bench.tuner import (
        AdaptiveTuner, TunerBounds, TunerThresholds, select_best_configs, write_summary_json,
    )

    output_path = output or _default_output()
    artifacts_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(artifacts_dir, exist_ok=True)
    log_file = setup_logging(verbosity, artifacts_dir)
    if log_file:
        console.print(f"[dim]Verbose log: {log_file}[/]")

    server = _resolve_server_path(server)

    if vk_devices is None:
        vk_devices = _discover_and_print_gpus()
    else:
        console.print(f"Using VK devices: {vk_devices}")

    # Auto-enable layer-split when multiple Vulkan devices are in use and the
    # user has not explicitly requested a different mode.
    if split_mode == "none" and vk_devices and "," in str(vk_devices):
        split_mode = "layer"
        console.print("[dim]Auto-set split-mode=layer for multi-GPU[/]")

    _check_version(server, sudo)

    # Use --ctx as the starting/max context if --ctx-max not explicitly provided
    effective_ctx_max = ctx_max if ctx_max is not None else ctx

    base_cfg = configs_from_args(
        server=server, model=model, host=host, port=port, np=np, ctx=effective_ctx_max,
        n_gpu_layers=n_gpu_layers, flash_attn=flash_attn, batch_size=batch_size,
        ubatch_size=ubatch_size, cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        kv_unified=kv_unified, cache_reuse=cache_reuse, cont_batching=cont_batching,
        threads=threads, threads_batch=threads_batch, split_mode=split_mode,
        vk_devices=vk_devices, use_sudo=sudo,
        max_predict_tokens=max_tokens,
    )
    _print_validation(base_cfg)

    bounds = TunerBounds(
        ctx_min=ctx_min,
        ctx_max=effective_ctx_max,
        ctx_step=ctx_step,
        ngl_step=ngl_step,
        batch_step=batch_step,
        max_retries=max_retries,
    )
    thresholds = TunerThresholds(
        max_ttft_s=max_ttft,
        min_tokens_per_sec=min_tokens_per_sec,
    )

    summary_path = summary or os.path.join(artifacts_dir, "summary.json")

    console.print(
        f"[bold]Tuner sweep[/] ctx {effective_ctx_max}→{ctx_min} step {ctx_step} "
        f"| max_retries={max_retries} | output → {output_path}"
    )

    def _plain_progress(current: int, total: int) -> None:
        console.print(f"[dim]  [{current}/{total}][/]")

    # Build the prompt sequence (use custom pack if provided, else built-in workload)
    if prompt_pack:
        from llama_bench.prompts import load_prompt_pack
        prompt_seq_override = load_prompt_pack(prompt_pack)
    else:
        prompt_seq_override = None

    if no_tui:
        tuner = AdaptiveTuner(
            base_cfg=base_cfg, bounds=bounds, thresholds=thresholds,
            artifacts_dir=artifacts_dir, log_file=log_file,
            progress_cb=_plain_progress,
            n_followups=n_followups, max_tokens=max_tokens,
            prompt_seq_override=prompt_seq_override,
        )
        attempts = tuner.run()
    else:
        from llama_bench.hw_monitor import HWMonitor
        from llama_bench.tui import BenchTUI
        hw_monitor = HWMonitor()
        hw_monitor.start()
        tui = BenchTUI(hw_monitor=hw_monitor, bench_log_path=log_file)

        def _tui_progress(current: int, total: int) -> None:
            tui.update_progress(current, total, 1, "tuner sweep")

        tuner = AdaptiveTuner(
            base_cfg=base_cfg, bounds=bounds, thresholds=thresholds,
            artifacts_dir=artifacts_dir, log_file=log_file,
            progress_cb=_tui_progress,
            n_followups=n_followups, max_tokens=max_tokens,
            prompt_seq_override=prompt_seq_override,
            event_cb=tui.handle_event,
            stop_event=tui._stop_event,
        )
        attempts = tui.run(tuner.run)
        hw_monitor.stop()
    # Write JSONL (attempts may be [] if user quit early)
    if attempts:
        from llama_bench.tuner import _attempt_to_dict as _atd
        with open(output_path, "w", encoding="utf-8") as fh:
            for a in attempts:
                fh.write(_json.dumps(_atd(a)) + "\n")
        console.print(f"[green]Results saved to[/] {output_path}")
    else:
        console.print("[yellow]Bench stopped early — no results saved.[/]")
        return

    # Multi-objective selection
    selection = select_best_configs(attempts, ctx_pct_threshold=ctx_pct_threshold)
    write_summary_json(summary_path, attempts, selection)
    console.print(f"[green]Summary written to[/] {summary_path}")

    # Print final TUI summary
    _print_tuner_summary(selection)

    # Abort on critical errors (abort run if no usable configs found due to model not found)
    for a in attempts:
        if a.failure_reason == "model_not_found":
            import textwrap
            stderr_hint = f"\n  Server stderr log: {a.stderr_path}" if a.stderr_path else ""
            if a.server_error_excerpt:
                indented = textwrap.indent(a.server_error_excerpt, "    ")
                excerpt_hint = f"\n  Excerpt:\n{indented}"
            else:
                excerpt_hint = ""
            raise click.ClickException(
                f"Critical error: model not found.{stderr_hint}{excerpt_hint}"
            )


def _print_tuner_summary(selection: dict) -> None:
    """Print a Rich table summarising the tuner multi-objective selection."""
    from rich.table import Table

    max_ctx = selection.get("max_ctx_result")
    top = selection.get("top_throughput", [])
    recommended = selection.get("recommended")

    if max_ctx is None:
        console.print("[yellow]No usable configurations found.[/]")
        return

    console.print(f"\n[bold green]Max usable context:[/] {max_ctx.ctx} tokens "
                  f"(ngl={max_ctx.n_gpu_layers}, batch={max_ctx.batch_size}, "
                  f"tok/s={max_ctx.tokens_per_sec:.1f})")

    if top:
        table = Table(title="Top throughput configs near max ctx", show_lines=False)
        table.add_column("Rank", justify="right")
        table.add_column("ctx", justify="right")
        table.add_column("ngl", justify="right")
        table.add_column("batch", justify="right")
        table.add_column("tok/s", justify="right")
        table.add_column("TTFT (s)", justify="right")
        for rank, a in enumerate(top, 1):
            table.add_row(
                str(rank),
                str(a.ctx),
                str(a.n_gpu_layers),
                str(a.batch_size),
                f"{a.tokens_per_sec:.1f}",
                f"{a.ttft_s:.2f}",
            )
        console.print(table)

    if recommended:
        console.print(
            f"\n[bold cyan]Recommended stable config:[/] "
            f"ctx={recommended.ctx} ngl={recommended.n_gpu_layers} "
            f"batch={recommended.batch_size} tok/s={recommended.tokens_per_sec:.1f}"
        )


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
def search(
    server, model, host, port, np, ctx, n_gpu_layers, flash_attn,
    batch_size, ubatch_size, cache_type_k, cache_type_v, kv_unified,
    cache_reuse, cont_batching, threads, threads_batch, split_mode,
    vk_devices, sudo,
    np_tests, ctx_tests, ngl_tests, max_configs, output, no_tui, verbosity,
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
    log_file = setup_logging(verbosity, os.path.dirname(os.path.abspath(output_path)))
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
        kv_unified=kv_unified, cache_reuse=cache_reuse, cont_batching=cont_batching,
        threads=threads, threads_batch=threads_batch, split_mode=split_mode,
        vk_devices=vk_devices, use_sudo=sudo,
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
        searcher = StagedSearcher(
            space=space,
            base_cfg=base_cfg,
            artifacts_dir=os.path.dirname(os.path.abspath(output_path)),
            max_configs=max_configs,
            progress_cb=_progress_cb,
            log_file=log_file,
        )
        results = searcher.run()
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
    save_results(results, output_path)
    console.print(f"[green]Search complete. Results saved to[/] {output_path}")

    from llama_bench.report import load_results, print_summary_table
    loaded = load_results(output_path)
    print_summary_table(loaded)


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
def explore(
    server, model, host, port, np, ctx, ctx_min, ctx_step,
    n_gpu_layers, ngl_tests, flash_attn, batch_size, batch_tests, ubatch_size,
    cache_type_k, cache_type_v, kv_unified, cache_reuse, cont_batching,
    threads, threads_batch, split_mode, vk_devices, sudo,
    ngl_step, batch_step, ngl_min, max_retries,
    max_ttft, min_tokens_per_sec, n_followups, max_tokens, output, no_tui, verbosity,
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
    log_file = setup_logging(verbosity, artifacts_dir)
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
        kv_unified=kv_unified, cache_reuse=cache_reuse, cont_batching=cont_batching,
        threads=threads, threads_batch=threads_batch, split_mode=split_mode,
        vk_devices=vk_devices, use_sudo=sudo, max_predict_tokens=max_tokens,
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
        explorer = ContinuousExplorer(
            base_cfg=base_cfg, bounds=bounds, thresholds=thresholds,
            ngl_values=ngl_values, batch_values=batch_values,
            artifacts_dir=artifacts_dir, log_file=log_file,
            n_followups=n_followups, max_tokens=max_tokens,
            output_path=output_path,
        )
        attempts = explorer.run()
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
        hof = explorer.hall_of_fame

    console.print(
        f"\n[bold]Explorer stopped[/] after {hof.total_tested} configs "
        f"across {hof.round_num} round(s)."
    )
    _print_hof_summary(hof)
    if attempts:
        console.print(f"[green]Results saved to[/] {output_path}")
    else:
        console.print("[yellow]No results saved (stopped before any completed).[/]")


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
