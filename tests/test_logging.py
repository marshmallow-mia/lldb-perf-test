"""Tests for llama_bench.logging_setup."""
from __future__ import annotations

import logging
import os

import pytest

from llama_bench.logging_setup import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_root_logger():
    """Remove all handlers from the root logger to avoid cross-test pollution."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSetupLogging:
    """Unit tests for setup_logging."""

    def test_verbosity_zero_returns_none(self, tmp_path):
        _reset_root_logger()
        result = setup_logging(0, str(tmp_path))
        assert result is None

    def test_verbosity_zero_creates_no_log_file(self, tmp_path):
        _reset_root_logger()
        setup_logging(0, str(tmp_path))
        log_files = [f for f in os.listdir(tmp_path) if f.endswith(".log")]
        assert log_files == []

    def test_verbosity_one_returns_log_path(self, tmp_path):
        _reset_root_logger()
        log_path = setup_logging(1, str(tmp_path))
        assert log_path is not None
        assert log_path.startswith(str(tmp_path))
        assert log_path.endswith(".log")
        _reset_root_logger()

    def test_verbosity_one_creates_log_file(self, tmp_path):
        _reset_root_logger()
        log_path = setup_logging(1, str(tmp_path))
        assert os.path.exists(log_path)
        _reset_root_logger()

    def test_verbosity_two_creates_log_file(self, tmp_path):
        _reset_root_logger()
        log_path = setup_logging(2, str(tmp_path))
        assert log_path is not None
        assert os.path.exists(log_path)
        _reset_root_logger()

    def test_log_file_named_with_timestamp(self, tmp_path):
        _reset_root_logger()
        log_path = setup_logging(1, str(tmp_path))
        basename = os.path.basename(log_path)
        assert basename.startswith("llama_bench_")
        assert basename.endswith(".log")
        _reset_root_logger()

    def test_verbosity_one_sets_info_level(self, tmp_path):
        _reset_root_logger()
        setup_logging(1, str(tmp_path))
        assert logging.getLogger().level == logging.INFO
        _reset_root_logger()

    def test_verbosity_two_sets_debug_level(self, tmp_path):
        _reset_root_logger()
        setup_logging(2, str(tmp_path))
        assert logging.getLogger().level == logging.DEBUG
        _reset_root_logger()

    def test_verbosity_three_sets_debug_level(self, tmp_path):
        """Any verbosity >= 2 should map to DEBUG."""
        _reset_root_logger()
        setup_logging(3, str(tmp_path))
        assert logging.getLogger().level == logging.DEBUG
        _reset_root_logger()

    def test_log_file_written_on_info_message(self, tmp_path):
        _reset_root_logger()
        log_path = setup_logging(1, str(tmp_path))
        logging.getLogger("test_logger").info("hello verbose world")
        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()
        with open(log_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "hello verbose world" in content
        _reset_root_logger()

    def test_results_dir_created_if_missing(self, tmp_path):
        _reset_root_logger()
        new_dir = str(tmp_path / "sub" / "results")
        log_path = setup_logging(1, new_dir)
        assert os.path.isdir(new_dir)
        assert os.path.exists(log_path)
        _reset_root_logger()
