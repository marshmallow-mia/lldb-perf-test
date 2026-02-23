"""Tests for llama_bench.prompts."""
import pytest

from llama_bench.prompts import (
    FOLLOWUP_QUESTIONS,
    REVERSE_ENGINEERING_SYSTEM_PROMPT,
    SHARED_PREFIX_TEMPLATE,
    build_prompt_sequence,
    estimate_token_count,
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
