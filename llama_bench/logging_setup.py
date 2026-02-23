"""Logging configuration for llama-bench verbose mode."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone


def setup_logging(verbosity: int, results_dir: str = "results") -> str | None:
    """Configure logging based on *verbosity* level.

    Args:
        verbosity: 0 = warnings only (default), 1 = INFO (-v), 2+ = DEBUG (-vv).
        results_dir: Directory where the log file should be saved.

    Returns:
        The path to the log file if ``verbosity > 0``, otherwise ``None``.
    """
    if verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
        return None

    level = logging.DEBUG if verbosity >= 2 else logging.INFO

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = os.path.join(results_dir, f"llama_bench_{timestamp}.log")

    # Root logger configuration
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Stderr handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    return log_path
