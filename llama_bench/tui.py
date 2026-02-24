"""Rich-based TUI for llama-bench progress display.

Sections displayed (top → bottom):
  - To-Dos        remaining candidate configurations
  - Current state what is being tested right now
  - Progress      progress bar with ETA
  - Errors        errors and warnings (critical ones in red)
  - Results       rolling table of completed runs
"""
from __future__ import annotations

import time
from typing import Optional

from rich.console import Console, Group as RichGroup, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


class BenchTUI:
    """Context-manager TUI backed by :class:`rich.live.Live`.

    Usage::

        with BenchTUI() as tui:
            tui.set_todos(["ctx=65536", "ctx=49152", "ctx=32768"])
            tui.set_current_state("Testing ctx=65536, ngl=45, batch=1536")
            tui.update_progress(current=1, total=3, phase=1, phase_name="tuner sweep")
            tui.add_error("Engine mismatch: vulkan requested but not active", red=True)
            tui.add_result("run_001", "ctx=65536 ngl=45 batch=1536", 12345.0, True)
    """

    def __init__(self) -> None:
        self._console = Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]ETA: {task.fields[eta]}"),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "Initialising…", total=1, completed=0, eta="?"
        )

        # To-Dos
        self._todos: list[str] = []

        # Current state
        self._current_state: Optional[str] = None

        # Errors / warnings (msg, is_red)
        self._errors: list[tuple[str, bool]] = []

        # Results table entries (run_id, config_summary, score, success, reason)
        self._results: list[tuple[str, str, float, bool, Optional[str]]] = []

        # Timing for ETA
        self._start_time: float = time.monotonic()

        self._live = Live(
            self.render(),
            console=self._console,
            refresh_per_second=4,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BenchTUI":
        self._start_time = time.monotonic()
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._live.update(self.render())
        self._live.__exit__(*args)

    # ------------------------------------------------------------------
    # Update API
    # ------------------------------------------------------------------

    def set_todos(self, todos: list[str]) -> None:
        """Replace the To-Dos list."""
        self._todos = list(todos)
        self._live.update(self.render())

    def set_current_state(self, state: str) -> None:
        """Update the current-state description."""
        self._current_state = state
        self._live.update(self.render())

    def update_progress(
        self,
        current: int,
        total: int,
        phase: int,
        phase_name: str,
    ) -> None:
        """Update the progress bar and recompute ETA."""
        elapsed = time.monotonic() - self._start_time
        if current > 0 and total > 0:
            rate = elapsed / current  # seconds per item
            remaining = max(0, total - current)
            eta_s = rate * remaining
            if eta_s < 60:
                eta_str = f"{eta_s:.0f}s"
            else:
                eta_str = f"{eta_s / 60:.1f}m"
        else:
            eta_str = "?"

        self._progress.update(
            self._task_id,
            description=f"[bold cyan]Phase {phase}[/] {phase_name}",
            total=total,
            completed=current,
            eta=eta_str,
        )
        self._live.update(self.render())

    def add_error(self, msg: str, red: bool = False) -> None:
        """Queue an error or warning message for display.

        Pass ``red=True`` for critical errors (e.g. engine mismatch).
        """
        self._errors.append((msg, red))
        self._live.update(self.render())

    def show_warning(self, msg: str) -> None:
        """Queue a non-critical warning message (yellow)."""
        self.add_error(msg, red=False)

    def add_result(
        self,
        run_id: str,
        config_summary: str,
        score: float,
        success: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Record a completed run result."""
        self._results.append((run_id, config_summary, score, success, reason))
        self._live.update(self.render())

    def set_best(self, config_summary: str, score: float) -> None:
        """Legacy helper — no-op kept for backward compatibility."""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> RenderableType:
        """Build the full TUI renderable."""
        parts: list[RenderableType] = []

        # --- To-Dos ---
        if self._todos:
            todo_text = Text()
            for item in self._todos[:10]:  # show up to 10
                todo_text.append("  • ", style="dim")
                todo_text.append(item + "\n", style="white")
            if len(self._todos) > 10:
                todo_text.append(f"  … and {len(self._todos) - 10} more\n", style="dim")
            parts.append(Panel(todo_text, title="To-Dos", border_style="blue"))

        # --- Current state ---
        if self._current_state is not None:
            state_text = Text(self._current_state, style="bold white")
            parts.append(Panel(state_text, title="Current State", border_style="cyan"))

        # --- Progress bar ---
        parts.append(Panel(self._progress, title="Progress", border_style="blue"))

        # --- Errors / warnings ---
        if self._errors:
            err_text = Text()
            for msg, is_red in self._errors[-8:]:  # show last 8
                if is_red:
                    err_text.append("✖ ", style="bold red")
                    err_text.append(msg + "\n", style="red")
                else:
                    err_text.append("⚠ ", style="bold yellow")
                    err_text.append(msg + "\n", style="yellow")
            parts.append(Panel(err_text, title="Errors / Warnings", border_style="red"))

        # --- Results table (last 20) ---
        if self._results:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Run ID", no_wrap=True, style="dim")
            table.add_column("Config", overflow="ellipsis", max_width=50)
            table.add_column("tok/s", justify="right")
            table.add_column("Status", justify="center")
            table.add_column("Reason")

            for run_id, cfg_sum, sc, ok, reason in self._results[-20:]:
                score_str = f"{sc:.1f}" if sc < float("inf") else "—"
                status = "[green]✓[/green]" if ok else "[red]✗[/red]"
                table.add_row(run_id, cfg_sum, score_str, status, reason or "")

            parts.append(Panel(table, title="Results", border_style="magenta"))

        # Stack vertically
        return Panel(
            RichGroup(*parts),
            title="[bold]llama-bench[/bold]",
            border_style="white",
        )

