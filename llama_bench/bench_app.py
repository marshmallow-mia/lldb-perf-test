"""Textual dashboard UI for llama-bench.

Each panel is an independent widget that only re-renders when its own
state changes — true differential updates, no full-screen flicker.

Architecture
------------
- ``BenchApp`` is a Textual ``App`` running on a dedicated thread.
- The main (tuner) thread pushes updates via ``app.call_from_thread()``,
  which schedules the callback on Textual's event loop.
- Periodic updates (hardware, elapsed time, log tail) are driven by
  ``set_interval`` timers on Textual's own thread — no extra threads.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Static

from llama_bench.hw_monitor import HWMonitor, HWSnapshot, shorten_gpu_name


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SPARK = "▁▂▃▄▅▆▇█"
_BAR_W = 14


def _sparkline(values: list[float], width: int = 8) -> str:
    if not values:
        return "─" * width
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return _SPARK[3] * len(vals)
    return "".join(_SPARK[int((v - mn) / (mx - mn) * 7)] for v in vals)


def _pct_bar(pct: float, width: int = _BAR_W) -> Text:
    p = min(max(pct, 0.0), 100.0)
    color = "red" if p >= 85 else "yellow" if p >= 65 else "green"
    filled = int(p / 100.0 * width)
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="dim white")
    return t


def _vram_bar(used_mb: float, total_mb: float, width: int = _BAR_W) -> Text:
    t = Text()
    if total_mb <= 0:
        t.append("N/A", style="dim")
        return t
    pct = used_mb / total_mb * 100
    t.append_text(_pct_bar(pct, width))
    t.append(f"  {used_mb/1024:.1f}/{total_mb/1024:.1f} GB", style="white")
    return t


def _fmt_dur(seconds: float) -> str:
    s = int(max(seconds, 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _read_log_tail(path: str, n: int = 7) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            raw = f.read().decode("utf-8", errors="replace")
        return [ln for ln in raw.splitlines() if ln.strip()][-n:]
    except OSError:
        return []


def _compute_eta(state: "TUIState") -> Optional[float]:
    if state.progress_current <= 0 or state.sweep_start_time == 0.0:
        return None
    elapsed = time.monotonic() - state.sweep_start_time
    remaining = state.progress_total - state.progress_current
    if remaining <= 0:
        return None
    return elapsed / state.progress_current * remaining


# ---------------------------------------------------------------------------
# Reason labels — shared by Retries widget and Activity log
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


# ---------------------------------------------------------------------------
# Shared state (written by main thread, read by Textual thread)
# ---------------------------------------------------------------------------

@dataclass
class TUIState:
    phase_desc: str = ""
    current_config: dict = field(default_factory=dict)
    request_step: str = ""
    server_command: str = ""
    server_log_path: Optional[str] = None
    run_start_time: float = 0.0
    sweep_start_time: float = 0.0
    progress_current: int = 0
    progress_total: int = 1
    current_run_peak_vram_mb: float = 0.0
    retry_log: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    tps_history: list[float] = field(default_factory=list)
    hw_snapshot: HWSnapshot = field(default_factory=HWSnapshot)
    bench_log_path: Optional[str] = None
    settings: dict = field(default_factory=dict)
    activity_log: list[str] = field(default_factory=list)
    hall_of_fame: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Widgets — each re-renders only when explicitly told to
# ---------------------------------------------------------------------------

class CurrentRunWidget(Static):
    """Phase / config / step / elapsed / ETA."""

    BORDER_TITLE = "Current Run"

    DEFAULT_CSS = """
    CurrentRunWidget {
        border: solid green;
        padding: 0 1;
        height: auto;
        min-height: 8;
        width: 1fr;
    }
    """

    def refresh_content(self, state: TUIState) -> None:
        t = Text()
        if state.phase_desc:
            t.append("Phase   ", style="dim")
            t.append(state.phase_desc + "\n", style="bold cyan")
        cfg = state.current_config
        if cfg:
            t.append("Config  ", style="dim")
            t.append(
                f"ctx={cfg.get('ctx','?')}  "
                f"ngl={cfg.get('ngl','?')}  "
                f"batch={cfg.get('batch','?')}\n",
                style="white",
            )
        if state.request_step:
            t.append("        ", style="dim")
            t.append("► ", style="bold yellow")
            t.append(state.request_step + "\n", style="yellow")
        if state.run_start_time > 0:
            elapsed = time.monotonic() - state.run_start_time
            t.append("Elapsed ", style="dim")
            t.append(_fmt_dur(elapsed) + "\n", style="green")
        eta = _compute_eta(state)
        if eta is not None:
            t.append("ETA     ", style="dim")
            t.append("~" + _fmt_dur(eta), style="green")
        self.update(t if t.plain else Text("(waiting for first run…)", style="dim"))


class HardwareWidget(Static):
    """GPU0 / GPU1 VRAM bars, utilisation, temperature; CPU %; RAM."""

    BORDER_TITLE = "Hardware"

    DEFAULT_CSS = """
    HardwareWidget {
        border: solid cyan;
        padding: 0 1;
        height: auto;
        min-height: 8;
        width: 1fr;
    }
    """

    def refresh_content(self, snap: HWSnapshot) -> None:
        t = Text()
        for i, gpu in enumerate(snap.gpus):
            short = shorten_gpu_name(gpu.name)
            t.append(f"GPU{i} {short:<16}", style="cyan")
            t.append_text(_vram_bar(gpu.vram_used_mb, gpu.vram_total_mb))
            t.append(f"  {gpu.util_pct:.0f}%", style="white")
            if gpu.temp_c > 0:
                tc = "red" if gpu.temp_c >= 85 else "yellow" if gpu.temp_c >= 75 else "dim white"
                t.append(f"  {gpu.temp_c:.0f}°C", style=tc)
            t.append("\n")

        t.append("CPU              ", style="cyan")
        t.append_text(_pct_bar(snap.cpu_pct))
        t.append(f"  {snap.cpu_pct:.0f}%\n", style="white")

        ram_pct = (snap.ram_used_gb / snap.ram_total_gb * 100) if snap.ram_total_gb > 0 else 0.0
        t.append("RAM              ", style="cyan")
        t.append_text(_pct_bar(ram_pct))
        t.append(f"  {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB", style="white")

        self.update(t if t.plain else Text("(no GPU data)", style="dim"))


class SettingsWidget(Static):
    """Shows the base benchmark configuration — model, GPU flags, cache settings, etc."""

    BORDER_TITLE = "Settings"

    DEFAULT_CSS = """
    SettingsWidget {
        border: solid $panel-lighten-2;
        padding: 0 1;
        height: auto;
        min-height: 8;
        width: 1fr;
    }
    """

    def refresh_content(self, settings: dict) -> None:
        if not settings:
            self.update(Text("(waiting…)", style="dim"))
            return

        t = Text()

        # Model name (basename only, no path)
        model_name = settings.get("model", "?")
        t.append("Model    ", style="dim")
        t.append(model_name + "\n", style="bold white")

        # Context sweep range
        ctx_max = settings.get("ctx_max", "?")
        ctx_min = settings.get("ctx_min", "?")
        t.append("Ctx      ", style="dim")
        if isinstance(ctx_max, int) and isinstance(ctx_min, int):
            t.append(f"{ctx_min:,} → {ctx_max:,}  (sweep range)\n", style="white")
        else:
            t.append(f"{ctx_min} → {ctx_max}\n", style="white")

        # GPU layers / batch sizes
        t.append("NGL      ", style="dim")
        t.append(f"{settings.get('ngl_initial', '?')}    ", style="white")
        t.append("Batch  ", style="dim")
        t.append(f"{settings.get('batch', '?')}    ", style="white")
        t.append("µBatch  ", style="dim")
        t.append(f"{settings.get('ubatch', '?')}\n", style="white")

        # Feature flags
        flash_attn = settings.get("flash_attn", False)
        kv_unified = settings.get("kv_unified", False)
        cont_batch = settings.get("cont_batching", False)
        t.append("Flash    ", style="dim")
        t.append(("✓" if flash_attn else "✗") + "    ", style="green" if flash_attn else "red")
        t.append("KV-Unified  ", style="dim")
        t.append(("✓" if kv_unified else "✗") + "    ", style="green" if kv_unified else "red")
        t.append("ContBatch  ", style="dim")
        t.append(("✓" if cont_batch else "✗") + "\n", style="green" if cont_batch else "red")

        # KV cache quantisation + reuse threshold
        t.append("KV cache ", style="dim")
        t.append(
            f"K:{settings.get('cache_type_k','?')}  "
            f"V:{settings.get('cache_type_v','?')}  "
            f"reuse:{settings.get('cache_reuse','?')}\n",
            style="white",
        )

        # GPU split mode / Vulkan devices / parallel slots
        t.append("Split    ", style="dim")
        t.append(f"{settings.get('split_mode','?')}    ", style="white")
        t.append("VK  ", style="dim")
        t.append(f"{settings.get('vk_devices','?')}    ", style="white")
        t.append("NP  ", style="dim")
        t.append(f"{settings.get('np','?')}\n", style="white")

        # Thread counts
        t.append("Threads  ", style="dim")
        t.append(
            f"{settings.get('threads','?')} infer  /  "
            f"{settings.get('threads_batch','?')} batch\n",
            style="white",
        )

        self.update(t)


class CommandWidget(Static):
    """Shows the current llama-server command."""

    BORDER_TITLE = "Server Command"

    DEFAULT_CSS = """
    CommandWidget {
        border: solid $panel-lighten-2;
        padding: 0 1;
        height: auto;
        max-height: 7;
    }
    """

    def refresh_content(self, cmd: str) -> None:
        if not cmd:
            self.update(Text("(waiting for server launch…)", style="dim"))
            return
        self.update(Text(cmd, style="dim white"))


class ActivityWidget(Static):
    """Chronological event log — what is the tool doing right now?

    Shows timestamped entries for every significant lifecycle event:
    server start/ready, health polling, prompt requests, results, retries.
    """

    BORDER_TITLE = "Activity"

    DEFAULT_CSS = """
    ActivityWidget {
        border: solid $accent;
        padding: 0 1;
        height: auto;
        min-height: 6;
        max-height: 12;
    }
    """

    def refresh_content(self, activity_log: list[str]) -> None:
        if not activity_log:
            self.update(Text("(waiting to start…)", style="dim"))
            return
        t = Text()
        for entry in activity_log[-12:]:
            if "✓" in entry or "✅" in entry:
                style = "green"
            elif "✗" in entry:
                style = "red"
            elif "⟳" in entry or "🔄" in entry:
                style = "yellow"
            elif "🚀" in entry:
                style = "cyan"
            else:
                style = "white"
            t.append(entry + "\n", style=style)
        self.update(t)


class ProgressWidget(Static):
    """Sweep progress bar with candidate count and ETA."""

    BORDER_TITLE = "Progress"

    DEFAULT_CSS = """
    ProgressWidget {
        border: solid blue;
        padding: 0 1;
        height: 5;
    }
    """

    def refresh_content(self, state: TUIState) -> None:
        total = max(state.progress_total, 1)
        current = state.progress_current
        pct = current / total
        filled = int(pct * 44)

        t = Text()
        if state.phase_desc:
            t.append(state.phase_desc + "\n", style="dim")
        t.append("█" * filled, style="blue bold")
        t.append("░" * (44 - filled), style="dim")
        t.append(f"  {current}/{total}  {pct * 100:.0f}%", style="white")
        eta = _compute_eta(state)
        if eta is not None:
            t.append(f"  ~{_fmt_dur(eta)} remaining", style="dim")
        self.update(t)


class ResultsWidget(Static):
    """Live results table with cold/warm TTFT, tok/s sparkline, peak VRAM."""

    BORDER_TITLE = "Results"

    DEFAULT_CSS = """
    ResultsWidget {
        border: solid magenta;
        padding: 0 1;
        height: auto;
        min-height: 4;
    }
    """

    def refresh_content(self, state: TUIState) -> None:
        if not state.results:
            self.update(Text("(no results yet)", style="dim"))
            return

        table = Table(
            show_header=True,
            header_style="bold magenta",
            padding=(0, 1),
            show_lines=False,
            expand=True,
        )
        table.add_column("ctx", justify="right", min_width=6)
        table.add_column("ngl", justify="right", min_width=4)
        table.add_column("batch", justify="right", min_width=5)
        table.add_column("Cold TTFT", justify="right", min_width=9)
        table.add_column("Warm TTFT", justify="right", min_width=9)
        table.add_column("tok/s", justify="right", min_width=6)
        table.add_column("Trend", min_width=8)
        table.add_column("Peak VRAM", justify="right", min_width=9)
        table.add_column("", justify="center", min_width=2)

        for i, r in enumerate(state.results[-12:]):
            ok = r.get("success", False)
            cold = f"{r['cold_ttft_s']:.2f}s" if r.get("cold_ttft_s") else "—"
            warm = f"{r['warm_ttft_s']:.3f}s" if r.get("warm_ttft_s") else "—"
            tps = f"{r['tokens_per_sec']:.1f}" if r.get("tokens_per_sec") else "—"
            trend = _sparkline(state.tps_history[: i + 1])
            peak_gb = r.get("peak_vram_mb", 0.0) / 1024
            peak = f"{peak_gb:.1f} GB" if peak_gb > 0 else "—"
            status = "[green]✓[/]" if ok else "[red]✗[/]"
            table.add_row(
                str(r["ctx"]), str(r["ngl"]), str(r["batch"]),
                cold, warm, tps, trend, peak, status,
                style="" if ok else "dim red",
            )

        self.update(table)


class HallOfFameWidget(Static):
    """Best config per objective — populated by the 'explore' command."""

    BORDER_TITLE = "Best Configs Found (Hall of Fame)"

    DEFAULT_CSS = """
    HallOfFameWidget {
        border: solid $success;
        padding: 0 1;
        height: auto;
        min-height: 4;
    }
    """

    def refresh_content(self, hof: dict) -> None:
        keys = ["max_ctx", "fastest_ttft", "best_warm", "best_throughput", "best_balanced"]
        if not hof or not any(hof.get(k) for k in keys):
            self.update(Text(
                "(run 'llama-bench explore' for continuous multi-objective optimization)",
                style="dim",
            ))
            return
        table = Table(
            show_header=True,
            header_style="bold green",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Objective", min_width=16)
        table.add_column("ctx", justify="right", min_width=7)
        table.add_column("ngl", justify="right", min_width=4)
        table.add_column("batch", justify="right", min_width=5)
        table.add_column("Cold TTFT", justify="right", min_width=9)
        table.add_column("Warm TTFT", justify="right", min_width=9)
        table.add_column("tok/s", justify="right", min_width=6)
        for key in keys:
            e = hof.get(key)
            if not e:
                continue
            table.add_row(
                e["label"],
                str(e["ctx"]),
                str(e["ngl"]),
                str(e["batch"]),
                f"{e['cold_ttft_s']:.2f}s",
                f"{e['warm_ttft_s']:.3f}s",
                f"{e['tokens_per_sec']:.1f}",
            )
        total = hof.get("total_tested", 0)
        rnd = hof.get("round_num", 1)
        caption = Text(f"Round {rnd}  ·  {total} configs tested", style="dim")
        self.update(table)


class RetriesWidget(Static):
    """OOM retry history."""

    BORDER_TITLE = "Retries"

    DEFAULT_CSS = """
    RetriesWidget {
        border: solid yellow;
        padding: 0 1;
        height: auto;
        min-height: 5;
        width: 1fr;
    }
    """

    def refresh_content(self, retry_log: list[str]) -> None:
        if not retry_log:
            self.update(Text("(none)", style="dim"))
            return
        t = Text()
        for msg in retry_log[-4:]:
            lines = msg.split("\n")
            t.append("⟳ ", style="bold yellow")
            t.append(lines[0] + "\n", style="yellow")
            if len(lines) > 1:
                t.append(lines[1] + "\n", style="dim yellow")
        self.update(t)


class ServerLogWidget(Static):
    """Live tail of the server stderr log."""

    BORDER_TITLE = "Server Log"

    DEFAULT_CSS = """
    ServerLogWidget {
        border: solid $panel-lighten-1;
        padding: 0 1;
        height: auto;
        min-height: 5;
        width: 2fr;
    }
    """

    def refresh_content(self, lines: list[str]) -> None:
        if not lines:
            self.update(Text("(no server log)", style="dim"))
            return
        t = Text()
        for line in lines:
            t.append(line + "\n", style="dim white")
        self.update(t)


class VerboseLogWidget(Static):
    """Tail of the bench tool's own verbose log (-v / -vv)."""

    BORDER_TITLE = "Verbose Log"

    DEFAULT_CSS = """
    VerboseLogWidget {
        border: solid $panel;
        padding: 0 1;
        height: auto;
        min-height: 5;
        width: 2fr;
    }
    """

    def refresh_content(self, lines: list[str]) -> None:
        if not lines:
            self.update(Text("(no verbose log — run with -v or -vv)", style="dim"))
            return
        t = Text()
        for line in lines:
            t.append(line + "\n", style="dim cyan")
        self.update(t)


# ---------------------------------------------------------------------------
# BenchApp
# ---------------------------------------------------------------------------

class BenchApp(App):
    """Textual application. Runs on a dedicated thread; updated via call_from_thread."""

    TITLE = "llama-bench"
    SUB_TITLE = "GPU benchmarking & context tuner"

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #top-row {
        layout: horizontal;
        height: auto;
        min-height: 9;
    }

    #bottom-row {
        layout: horizontal;
        height: auto;
        min-height: 6;
    }
    """

    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        hw_monitor: Optional[HWMonitor] = None,
        bench_log_path: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._hw_monitor = hw_monitor
        self._state = TUIState()
        self._state.bench_log_path = bench_log_path
        self._ready_event: Optional[object] = None  # set by BenchTUI.run()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            yield CurrentRunWidget()
            yield HardwareWidget()
            yield SettingsWidget()
        yield CommandWidget()
        yield ActivityWidget()
        yield ProgressWidget()
        yield ResultsWidget()
        yield HallOfFameWidget()
        with Horizontal(id="bottom-row"):
            yield RetriesWidget()
            yield ServerLogWidget()
            yield VerboseLogWidget()
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        if self._hw_monitor is not None:
            self.set_interval(2.0, self._refresh_hardware)
        self.set_interval(0.5, self._tick)
        self.set_interval(0.5, self._refresh_log)
        # Signal the worker thread that Textual is ready to receive events.
        if self._ready_event is not None:
            import threading
            if isinstance(self._ready_event, threading.Event):
                self._ready_event.set()

    # ------------------------------------------------------------------
    # Periodic refresh (run on Textual's thread via set_interval)
    # ------------------------------------------------------------------

    def _refresh_hardware(self) -> None:
        snap = self._hw_monitor.latest()  # type: ignore[union-attr]
        self._state.hw_snapshot = snap
        # Track peak VRAM for current run
        if self._state.run_start_time > 0:
            for gpu in snap.gpus:
                if gpu.vram_used_mb > self._state.current_run_peak_vram_mb:
                    self._state.current_run_peak_vram_mb = gpu.vram_used_mb
        self.query_one(HardwareWidget).refresh_content(snap)

    def _tick(self) -> None:
        """Update widgets that depend on wall-clock time."""
        self.query_one(CurrentRunWidget).refresh_content(self._state)
        self.query_one(ProgressWidget).refresh_content(self._state)

    def _refresh_log(self) -> None:
        if self._state.server_log_path:
            lines = _read_log_tail(self._state.server_log_path)
            self.query_one(ServerLogWidget).refresh_content(lines)
        if self._state.bench_log_path:
            vlines = _read_log_tail(self._state.bench_log_path, 10)
            self.query_one(VerboseLogWidget).refresh_content(vlines)

    # ------------------------------------------------------------------
    # Activity log helper
    # ------------------------------------------------------------------

    def _activity(self, msg: str, replace_if: Optional[str] = None) -> None:
        """Append a timestamped entry to the activity log.

        If *replace_if* is set and the last existing entry contains that
        substring, the last entry is updated in-place instead of appending.
        This prevents health-poll lines from flooding the log.
        """
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}]  {msg}"
        if (
            replace_if
            and self._state.activity_log
            and replace_if in self._state.activity_log[-1]
        ):
            self._state.activity_log[-1] = entry
        else:
            self._state.activity_log.append(entry)
        self.query_one(ActivityWidget).refresh_content(self._state.activity_log)

    # ------------------------------------------------------------------
    # Event handlers (called from main thread via call_from_thread)
    # ------------------------------------------------------------------

    def _on_bench_event(self, event: str, data: dict) -> None:
        """Process a benchmark lifecycle event on Textual's thread."""
        s = self._state

        if event == "bench_settings":
            s.settings = dict(data)
            self.query_one(SettingsWidget).refresh_content(s.settings)

        elif event == "server_starting":
            s.server_command = data.get("command", "")
            s.server_log_path = data.get("stderr_path")
            s.request_step = "Starting server…"
            s.run_start_time = time.monotonic()
            s.current_run_peak_vram_mb = 0.0
            self.query_one(CommandWidget).refresh_content(s.server_command)
            self.query_one(ServerLogWidget).refresh_content([])
            pid = data.get("pid", "?")
            self._activity(f"🚀  Server starting  (PID {pid})")

        elif event == "health_poll":
            attempt = data.get("attempt", "?")
            elapsed_s = data.get("elapsed_s")
            elapsed_str = (
                f"  {elapsed_s:.0f}s elapsed"
                if isinstance(elapsed_s, (int, float))
                else ""
            )
            self._activity(
                f"🔄  Polling /health…  attempt {attempt}{elapsed_str}",
                replace_if="Polling /health",
            )

        elif event == "server_ready":
            s.request_step = "Server ready — sending initial request…"
            self._activity("✅  Server ready")

        elif event == "request_step":
            s.request_step = data.get("step", "")
            self._activity(f"▶  {s.request_step}")

        elif event == "run_start":
            s.phase_desc = data.get("phase_desc", "")
            s.current_config = {
                "ctx": data.get("ctx", "?"),
                "ngl": data.get("ngl", "?"),
                "batch": data.get("batch", "?"),
            }
            # Sweep timing starts on first run_start
            if s.sweep_start_time == 0.0:
                s.sweep_start_time = time.monotonic()
            ctx = data.get("ctx", "?")
            ngl = data.get("ngl", "?")
            batch = data.get("batch", "?")
            phase = data.get("phase_desc", "")
            # Update progress if explorer provided it
            pc = data.get("progress_current")
            pt = data.get("progress_total")
            if pc is not None and pt is not None:
                s.progress_current = int(pc)
                s.progress_total = int(pt)
            self._activity(f"▷  ctx={ctx}  ngl={ngl}  batch={batch}  — {phase}")
            s.phase_desc = data.get("phase_desc", "")
            s.current_config = {
                "ctx": data.get("ctx", "?"),
                "ngl": data.get("ngl", "?"),
                "batch": data.get("batch", "?"),
            }
            # Sweep timing starts on first run_start
            if s.sweep_start_time == 0.0:
                s.sweep_start_time = time.monotonic()
            ctx = data.get("ctx", "?")
            ngl = data.get("ngl", "?")
            batch = data.get("batch", "?")
            phase = data.get("phase_desc", "")
            self._activity(f"▷  ctx={ctx}  ngl={ngl}  batch={batch}  — {phase}")

        elif event == "retry":
            attempt = data.get("attempt", "?")
            max_r = data.get("max_retries", "?")
            reason_label = _REASON_LABELS.get(data.get("reason", ""), data.get("reason", "?"))
            change = data.get("change", "?")
            ctx = data.get("ctx", "")
            ngl = data.get("ngl", "")
            batch = data.get("batch", "")
            msg = (
                f"#{attempt}/{max_r} — {reason_label}  →  {change}\n"
                f"  ctx={ctx} ngl={ngl} batch={batch}"
            )
            s.retry_log.append(msg)
            self.query_one(RetriesWidget).refresh_content(s.retry_log)
            self._activity(f"⟳  #{attempt}/{max_r}  {reason_label}  →  {change}")
        elif event == "hall_of_fame":
            s.hall_of_fame = dict(data)
            self.query_one(HallOfFameWidget).refresh_content(s.hall_of_fame)

        elif event == "explore_round":
            s.phase_desc = data.get("phase_desc", "")
            s.progress_current = data.get("progress_current", 0)
            s.progress_total = data.get("progress_total", 1)
            attempt = data.get("attempt", "?")
            max_r = data.get("max_retries", "?")
            reason_label = _REASON_LABELS.get(
                data.get("reason", ""), data.get("reason", "?")
            )
            change = data.get("change", "?")
            ctx = data.get("ctx", "")
            ngl = data.get("ngl", "")
            batch = data.get("batch", "")
            msg = (
                f"#{attempt}/{max_r} — {reason_label}  →  {change}\n"
                f"  ctx={ctx} ngl={ngl} batch={batch}"
            )
            s.retry_log.append(msg)
            self.query_one(RetriesWidget).refresh_content(s.retry_log)
            self._activity(f"⟳  #{attempt}/{max_r}  {reason_label}  →  {change}")

        elif event == "run_done":
            tps = data.get("tokens_per_sec", 0.0)
            if tps > 0:
                s.tps_history.append(tps)
            ok = data.get("success", False)
            s.results.append({
                "ctx": data.get("ctx", 0),
                "ngl": data.get("ngl", 0),
                "batch": data.get("batch", 0),
                "cold_ttft_s": data.get("cold_ttft_s", 0.0),
                "warm_ttft_s": data.get("warm_ttft_s", 0.0),
                "tokens_per_sec": tps,
                "success": ok,
                "reason": data.get("reason"),
                "peak_vram_mb": s.current_run_peak_vram_mb,
            })
            self.query_one(ResultsWidget).refresh_content(s)
            ctx = data.get("ctx", "?")
            if ok:
                cold = data.get("cold_ttft_s", 0.0)
                self._activity(
                    f"✓  ctx={ctx}  TTFT {cold:.2f}s  {tps:.1f} tok/s"
                )
            else:
                reason = data.get("reason", "failed")
                self._activity(
                    f"✗  ctx={ctx}  {_REASON_LABELS.get(reason, reason)}"
                )

    def _on_progress(self, current: int, total: int, phase: int, phase_name: str) -> None:
        s = self._state
        if s.sweep_start_time == 0.0 and current == 0:
            s.sweep_start_time = time.monotonic()
        s.progress_current = current
        s.progress_total = total
        if not s.phase_desc:
            s.phase_desc = f"Phase {phase} — {phase_name}"
        self.query_one(ProgressWidget).refresh_content(s)
