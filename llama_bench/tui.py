"""Thin wrapper around the Textual BenchApp for llama-bench."""
from __future__ import annotations

import os
import signal
import sys
import threading
from typing import Any, Callable, Optional, TypeVar

from llama_bench.hw_monitor import HWMonitor

_T = TypeVar("_T")


class BenchTUI:
    """TUI wrapper that runs Textual on the **main** thread.

    Attributes exposed to CLI callers:

    * ``_stop_event`` — set when the user quits early; pass to
      ``AdaptiveTuner(stop_event=...)`` so the tuner loop exits between runs.

    Usage::

        hw = HWMonitor(); hw.start()
        tui = BenchTUI(hw_monitor=hw, bench_log_path=log_file)
        tuner = AdaptiveTuner(..., event_cb=tui.handle_event,
                              stop_event=tui._stop_event)
        attempts = tui.run(tuner.run)   # blocks until bench done or 'q'
        hw.stop()
    """

    def __init__(
        self,
        hw_monitor: Optional[HWMonitor] = None,
        bench_log_path: Optional[str] = None,
    ) -> None:
        from llama_bench.bench_app import BenchApp

        # Stop event: set by action_quit (via BenchApp) or run() when user quits.
        # Must be created before BenchApp so it can be passed in.
        self._stop_event: threading.Event = threading.Event()
        self._app = BenchApp(hw_monitor=hw_monitor, bench_log_path=bench_log_path, stop_event=self._stop_event)
        # PID of the current llama-server subprocess (or its sudo wrapper).
        # Tracked via the server_starting / run_done events so we can SIGTERM
        # it immediately when the user quits.
        self._current_server_pid: Optional[int] = None

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(self, work_fn: Callable[[], _T]) -> _T:
        """Run *work_fn* on a background thread, Textual on the calling thread.

        Blocks until the benchmark completes (or the user presses ``q``/Ctrl+C).

        * Normal completion: returns work_fn's return value (or re-raises its
          exception).
        * User quit: signals the benchmark to stop, gives it 20 s to clean up
          (so llama-server can be shut down by the runner's finally-block), then
          calls sys.exit(0) if the worker is still alive.
        """
        result_holder: list[Any] = [None]
        exc_holder: list[Optional[BaseException]] = [None]
        ready = threading.Event()         # signals Textual ready → worker starts
        completed = threading.Event()    # set when work_fn returns/raises

        def _worker() -> None:
            ready.wait(timeout=10.0)
            try:
                result_holder[0] = work_fn()
            except BaseException as exc:
                exc_holder[0] = exc
            finally:
                completed.set()
                try:
                    self._app.call_from_thread(self._app.exit)
                except Exception:
                    pass

        worker = threading.Thread(target=_worker, daemon=True, name="bench-worker")
        worker.start()
        self._app._ready_event = ready  # type: ignore[attr-defined]
        saved = self._suppress_console_logging()
        self._app.run()  # blocks on main thread until Textual exits
        self._restore_console_logging(saved)
        # Textual uses an alternate screen; when it exits it sends \x1b[?1049l
        # which restores the primary screen.  That primary screen may show stale
        # content or log lines that bled through during the TUI session, so we
        # explicitly clear it.  We write directly to the stdout fd (same fd
        # Textual uses) to avoid any Python-level buffering delay.
        # The writer thread has already been join()ed inside App.run() so all
        # Textual escape sequences are guaranteed to be in the kernel by now.
        # \033[2J   = erase entire display
        # \033[3J   = erase scrollback buffer (supported by xterm/VTE/kitty)
        # \033[H    = move cursor to top-left
        try:
            os.write(sys.stdout.fileno(), b"\033[2J\033[3J\033[H")
        except OSError:
            pass

        if completed.is_set():
            # Normal completion — work_fn finished before (or triggered) the exit.
            worker.join(timeout=5.0)
            if exc_holder[0] is not None:
                raise exc_holder[0]
            return result_holder[0]  # type: ignore[return-value]

        # ---- User quit early ----
        # 1. Signal the tuner loop to stop between candidates.
        self._stop_event.set()
        # 2. SIGTERM the current llama-server subprocess so it dies fast and the
        #    runner's finally-block runs stop_server() for a clean shutdown.
        pid = self._current_server_pid
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # 3. If the server was launched with sudo, the PID tracked is the sudo
        #    wrapper. Kill its children too by iterating /proc (best-effort).
        #    Do NOT use os.killpg — the server may share our process group,
        #    which would kill this Python process and prevent the final report.
        if pid is not None:
            try:
                import subprocess
                # Use pkill to kill children of pid without killing our own group.
                subprocess.run(
                    ["pkill", "-TERM", "-P", str(pid)],
                    capture_output=True,
                    timeout=2,
                )
            except Exception:
                pass
        # 4. Give the worker a window to finish its current cleanup.
        worker.join(timeout=20.0)
        if worker.is_alive():
            # Worker is still alive — try harder to interrupt it.
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            worker.join(timeout=5.0)
        # 5. Return whatever results we have — NEVER sys.exit(0).
        #    The caller (cli.py) is responsible for generating the final
        #    report from whatever data is available, including partial results
        #    accessible via the tuner/explorer/searcher's accumulated_* property.
        if result_holder[0] is not None:
            return result_holder[0]  # type: ignore[return-value]
        return []  # type: ignore[return-value]

    def _suppress_console_logging(self) -> list:
        """Remove StreamHandlers from root logger so they don't bleed into TUI."""
        import logging
        root = logging.getLogger()
        to_remove = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        for h in to_remove:
            root.removeHandler(h)
        return to_remove

    def _restore_console_logging(self, handlers: list) -> None:
        """Re-attach console handlers after TUI exits."""
        import logging
        root = logging.getLogger()
        for h in handlers:
            root.addHandler(h)

    # ------------------------------------------------------------------
    # Event API (called from the benchmark worker thread)
    # ------------------------------------------------------------------

    def handle_event(self, event: str, data: dict) -> None:
        """Dispatch a benchmark lifecycle event to the Textual app."""
        # Track current server PID for emergency SIGTERM on user-quit.
        if event == "server_starting":
            self._current_server_pid = data.get("pid")
        elif event == "run_done":
            self._current_server_pid = None
        try:
            self._app.call_from_thread(self._app._on_bench_event, event, data)
        except Exception:
            pass

    def update_progress(
        self,
        current: int,
        total: int,
        phase: int,
        phase_name: str,
    ) -> None:
        """Update the sweep progress bar."""
        try:
            self._app.call_from_thread(
                self._app._on_progress, current, total, phase, phase_name
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Legacy no-ops (backward-compatible with search command)
    # ------------------------------------------------------------------

    def add_result(
        self,
        run_id: str,
        config_summary: str,
        score: float,
        success: bool,
        reason: Optional[str] = None,
    ) -> None:
        pass

    def set_best(self, config_summary: str, score: float) -> None:
        pass

    def show_warning(self, msg: str) -> None:
        pass
