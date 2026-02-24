"""Tests for llama_bench.prompts."""
import os
import pytest

from llama_bench.prompts import (
    CORPUS_FILES,
    EXCLUDED_CORPUS_FILES,
    FOLLOWUP_QUESTIONS,
    REVERSE_ENGINEERING_SYSTEM_PROMPT,
    SHARED_PREFIX_TEMPLATE,
    build_corpus_context,
    build_prompt_sequence,
    estimate_token_count,
    load_corpus_files,
)


def test_estimate_token_count_basic():
    result = estimate_token_count("hello world")
    assert isinstance(result, int)
    assert result >= 1


def test_estimate_token_count_empty():
    assert estimate_token_count("") == 0


def test_estimate_token_count_proportional():
    short = estimate_token_count("hi")
    long = estimate_token_count("hi " * 100)
    assert long > short


# ---------------------------------------------------------------------------
# build_prompt_sequence
# ---------------------------------------------------------------------------

def test_build_prompt_sequence_length_2():
    seq = build_prompt_sequence(n_followups=2)
    assert len(seq) == 3  # 1 initial + 2 followups


def test_build_prompt_sequence_length_4():
    seq = build_prompt_sequence(n_followups=4)
    assert len(seq) == 5


def test_build_prompt_sequence_first_not_followup():
    seq = build_prompt_sequence(n_followups=2)
    assert seq[0]["is_followup"] is False


def test_build_prompt_sequence_followups_are_followup():
    seq = build_prompt_sequence(n_followups=2)
    assert seq[1]["is_followup"] is True
    assert seq[2]["is_followup"] is True


def test_build_prompt_sequence_has_messages_key():
    seq = build_prompt_sequence(n_followups=2)
    for item in seq:
        assert "messages" in item
        assert isinstance(item["messages"], list)


def test_build_prompt_sequence_messages_have_role_content():
    seq = build_prompt_sequence(n_followups=1)
    for item in seq:
        for msg in item["messages"]:
            assert "role" in msg
            assert "content" in msg


def test_build_prompt_sequence_no_system():
    seq = build_prompt_sequence(n_followups=1, use_system=False)
    # No system message
    first = seq[0]
    roles = [m["role"] for m in first["messages"]]
    assert "system" not in roles


def test_build_prompt_sequence_with_system():
    seq = build_prompt_sequence(n_followups=1, use_system=True)
    first = seq[0]
    roles = [m["role"] for m in first["messages"]]
    assert "system" in roles


def test_build_prompt_sequence_expected_prefix_len():
    seq = build_prompt_sequence(n_followups=1)
    for item in seq:
        assert "expected_prefix_len_tokens" in item
        assert item["expected_prefix_len_tokens"] > 0


# ---------------------------------------------------------------------------
# Content checks
# ---------------------------------------------------------------------------

def test_system_prompt_nonempty():
    assert len(REVERSE_ENGINEERING_SYSTEM_PROMPT) > 100


def test_system_prompt_contains_keywords():
    text = REVERSE_ENGINEERING_SYSTEM_PROMPT.lower()
    assert "assembly" in text
    assert "ghidra" in text or "ida" in text


def test_shared_prefix_contains_code():
    text = SHARED_PREFIX_TEMPLATE
    assert "memcpy" in text or "staging" in text


def test_followup_questions_count():
    assert len(FOLLOWUP_QUESTIONS) >= 6


# ---------------------------------------------------------------------------
# Corpus file loading
# ---------------------------------------------------------------------------

def test_corpus_files_constant():
    """CORPUS_FILES must include README.md, agent.md, artifacts.md."""
    assert "README.md" in CORPUS_FILES
    assert "agent.md" in CORPUS_FILES
    assert "artifacts.md" in CORPUS_FILES


def test_excluded_corpus_files_constant():
    """solution.md must be excluded from prompts."""
    assert "solution.md" in EXCLUDED_CORPUS_FILES


def test_load_corpus_files_from_repo(tmp_path):
    """load_corpus_files returns dict with expected keys when files exist."""
    # Create minimal corpus files in a temp dir that resembles a repo root
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "README.md").write_text("# Test README\n")
    (tmp_path / "agent.md").write_text("# Agent context\n")
    (tmp_path / "artifacts.md").write_text("# Binary artifacts\n")

    corpus = load_corpus_files(repo_root=str(tmp_path))
    assert "README.md" in corpus
    assert "agent.md" in corpus
    assert "artifacts.md" in corpus
    assert "solution.md" not in corpus


def test_load_corpus_files_skips_missing(tmp_path):
    """load_corpus_files skips files that do not exist gracefully."""
    corpus = load_corpus_files(repo_root=str(tmp_path))
    # No corpus files in tmp_path; result should be empty
    assert isinstance(corpus, dict)


def test_build_corpus_context_returns_string():
    ctx = build_corpus_context()
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_build_corpus_context_uses_fallback_when_no_files(tmp_path):
    """Falls back to SHARED_PREFIX_TEMPLATE when corpus files are missing."""
    ctx = build_corpus_context(repo_root=str(tmp_path))
    # Should fall back gracefully (either use template or empty dir content)
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_solution_md_not_in_corpus():
    """solution.md must NOT appear in the prompt corpus context."""
    ctx = build_corpus_context()
    # The section header must not appear
    assert "=== solution.md ===" not in ctx
    # Unique content from solution.md (Vulnerability Summary header) must not appear
    assert "Vulnerability Summary" not in ctx
    assert "CVE-candidate" not in ctx
