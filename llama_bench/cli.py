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
@click.option("--ctx", "-c", default=49152, show_default=True, help="Context size (tokens).")
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
@click.option("--n-followups", default=4, show_default=True,
              help="Number of follow-up prompts per run.")
@click.option("--output", "-o", default=None,
              help="JSONL output path (default: results/bench_<timestamp>.jsonl).")
@click.option("--prompt-pack", default=None, type=click.Path(exists=True),
              help="Path to a custom prompt pack (JSON/YAML).")
@click.option("--no-tui", is_flag=True, default=False, help="Disable TUI, use plain output.")
@click.option("-v", "--verbose", "verbosity", count=True,
              help="Verbose output; use -vv for debug-level logging.")
def bench(
    server, model, host, port, np, ctx, n_gpu_layers, flash_attn,
    batch_size, ubatch_size, cache_type_k, cache_type_v, kv_unified,
    cache_reuse, cont_batching, threads, threads_batch, split_mode,
    vk_devices, sudo, n_followups, output, prompt_pack, no_tui, verbosity,
) -> None:
    """Run a single benchmark configuration."""
    from llama_bench.config import configs_from_args
    from llama_bench.metrics import metrics_to_dict
    from llama_bench.prompts import build_prompt_sequence, load_prompt_pack
    from llama_bench.report import print_summary_table
    from llama_bench.runner import BenchmarkRunner
    from llama_bench.logging_setup import setup_logging

    output_path = output or _default_output()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    log_file = setup_logging(verbosity, os.path.dirname(os.path.abspath(output_path)))
    if log_file:
        console.print(f"[dim]Verbose log: {log_file}[/]")

    server = _resolve_server_path(server)

    if vk_devices is None:
        vk_devices = _discover_and_print_gpus()
    else:
        console.print(f"Using VK devices: {vk_devices}")

    _check_version(server, sudo)

    cfg = configs_from_args(
        server=server, model=model, host=host, port=port, np=np, ctx=ctx,
        n_gpu_layers=n_gpu_layers, flash_attn=flash_attn, batch_size=batch_size,
        ubatch_size=ubatch_size, cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        kv_unified=kv_unified, cache_reuse=cache_reuse, cont_batching=cont_batching,
        threads=threads, threads_batch=threads_batch, split_mode=split_mode,
        vk_devices=vk_devices, use_sudo=sudo,
    )
    _print_validation(cfg)

    if prompt_pack:
        prompt_seq = load_prompt_pack(prompt_pack)
    else:
        prompt_seq = build_prompt_sequence(n_followups=n_followups)

    runner = BenchmarkRunner(cfg, artifacts_dir=os.path.dirname(os.path.abspath(output_path)),
                             log_file=log_file)

    console.print(f"[bold]Running benchmark[/] → {output_path}")

    if no_tui:
        run_metrics = runner.run_single(prompt_seq, n_followups=n_followups)
    else:
        from llama_bench.tui import BenchTUI
        with BenchTUI() as tui:
            tui.update_progress(0, 1, 1, "benchmark")
            run_metrics = runner.run_single(prompt_seq, n_followups=n_followups)
            for m in run_metrics:
                from llama_bench.metrics import score_run
                sc = score_run(m)
                tui.add_result(m.run_id, f"np={cfg.np}, ctx={cfg.ctx}", sc, m.success,
                               m.failure_reason)
            tui.update_progress(1, 1, 1, "benchmark")

    # Save JSONL
    import json
    with open(output_path, "w", encoding="utf-8") as fh:
        for m in run_metrics:
            fh.write(json.dumps(metrics_to_dict(m)) + "\n")

    console.print(f"[green]Results saved to[/] {output_path}")

    # Fail fast for startup errors in bench mode (model not found / OOM VRAM).
    _STARTUP_FAILURE_REASONS = {"model_not_found", "out_of_vram", "server_exited", "server_startup_timeout"}
    if run_metrics and not run_metrics[0].success and run_metrics[0].failure_reason in _STARTUP_FAILURE_REASONS:
        import textwrap
        m = run_metrics[0]
        stderr_hint = f"\n  Server stderr log: {m.stderr_path}" if m.stderr_path else ""
        if m.server_error_excerpt:
            indented = textwrap.indent(m.server_error_excerpt, "    ")
            excerpt_hint = f"\n  Excerpt:\n{indented}"
        else:
            excerpt_hint = ""
        raise click.ClickException(
            f"Server startup failed ({m.failure_reason}).{stderr_hint}{excerpt_hint}"
        )

    # Print summary
    from llama_bench.report import load_results
    loaded = load_results(output_path)
    print_summary_table(loaded)


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
        with BenchTUI() as tui:
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
            results = searcher.run()

            best = searcher.best_config()
            if best:
                tui.set_best(f"np={best.np}, ctx={best.ctx}, ngl={best.n_gpu_layers}", 
                             min(r.best_score for r in results if r.best_score < float("inf")))

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
