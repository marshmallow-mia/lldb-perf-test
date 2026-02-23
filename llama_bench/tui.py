"""Rich-based TUI for llama-bench progress display."""
from __future__ import annotations

from typing import Optional

from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text


class BenchTUI:
    """Context-manager TUI backed by :class:`rich.live.Live`.

    Usage::

        with BenchTUI() as tui:
            tui.update_progress(current=0, total=50, phase=1, phase_name="coarse sweep")
            tui.add_result("run_001", "np=1, ctx=49152", 12345.0, True)
    """

    def __init__(self) -> None:
        self._console = Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task("Initialising…", total=1, completed=0)

        self._results: list[tuple[str, str, float, bool, Optional[str]]] = []
        # (run_id, config_summary, score, success, reason)

        self._best_summary: Optional[str] = None
        self._best_score: float = float("inf")
        self._warnings: list[str] = []

        self._live = Live(
            self.render(),
            console=self._console,
            refresh_per_second=4,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BenchTUI":
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._live.update(self.render())
        self._live.__exit__(*args)

    # ------------------------------------------------------------------
    # Update API
    # ------------------------------------------------------------------

    def update_progress(
        self,
        current: int,
        total: int,
        phase: int,
        phase_name: str,
    ) -> None:
        """Update the progress bar."""
        self._progress.update(
            self._task_id,
            description=f"[bold cyan]Phase {phase}[/] {phase_name}",
            total=total,
            completed=current,
        )
        self._live.update(self.render())

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
        """Update the best-known configuration display."""
        self._best_summary = config_summary
        self._best_score = score
        self._live.update(self.render())

    def show_warning(self, msg: str) -> None:
        """Queue a warning message for display."""
        self._warnings.append(msg)
        self._live.update(self.render())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> RenderableType:
        """Build the full TUI renderable."""
        parts: list[RenderableType] = []

        # Progress bar
        parts.append(Panel(self._progress, title="Progress", border_style="blue"))

        # Warnings
        if self._warnings:
            warn_text = Text()
            for w in self._warnings[-5:]:  # show last 5
                warn_text.append("⚠ ", style="bold yellow")
                warn_text.append(w + "\n", style="yellow")
            parts.append(Panel(warn_text, title="Warnings", border_style="yellow"))

        # Best config so far
        if self._best_summary is not None:
            best_text = Text()
            best_text.append(f"Score: {self._best_score:.1f} ms\n", style="bold green")
            best_text.append(self._best_summary, style="green")
            parts.append(Panel(best_text, title="Best So Far", border_style="green"))

        # Results table (last 20)
        if self._results:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Run ID", no_wrap=True, style="dim")
            table.add_column("Config", overflow="ellipsis", max_width=50)
            table.add_column("Score (ms)", justify="right")
            table.add_column("Status", justify="center")
            table.add_column("Reason")

            for run_id, cfg_sum, sc, ok, reason in self._results[-20:]:
                score_str = f"{sc:.1f}" if sc < float("inf") else "—"
                status = "[green]✓[/green]" if ok else "[red]✗[/red]"
                table.add_row(run_id, cfg_sum, score_str, status, reason or "")

            parts.append(Panel(table, title="Results", border_style="magenta"))

        # Stack vertically
        from rich.console import Group as RichGroup
        return Panel(
            RichGroup(*parts),
            title="[bold]llama-bench[/bold]",
            border_style="white",
        )
