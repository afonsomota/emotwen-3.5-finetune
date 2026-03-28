"""
Tests for data conversion functions in src/data_prep.py.
All tests use hand-crafted in-memory mock data — no load_dataset, no W&B.
"""

import random
import pytest

from tests.conftest import (
    MOCK_SYSTEM_PROMPT,
    MOCK_SYSTEM_PROMPT_RAG,
    make_ed_row,
    make_dd_row,
    make_dair_row,
)
from src.data_prep import (
    _ed_split_to_messages,
    _daily_dialog_to_messages,
    _dair_emotion_to_messages,
    _counsel_chat_to_messages,
    _let_me_explain_conversations,
    _get_reflection,
    _go_emotions_to_messages,
    _LET_ME_EXPLAIN_EXAMPLES,
    _REFLECTION_TEMPLATES,
)


# ─── _ed_split_to_messages ────────────────────────────────────────────────────

def _make_clean_ed_row():
    return make_ed_row([
        {"role": "user", "content": "I feel sad."},
        {"role": "assistant", "content": "That sounds heavy."},
    ])


def test_ed_normal_row():
    result = _ed_split_to_messages([_make_clean_ed_row()], MOCK_SYSTEM_PROMPT)
    assert len(result) == 1
    conv = result[0]
    msgs = conv["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert conv["source"] == "empathetic_dialogues"


def test_ed_first_message_is_system():
    result = _ed_split_to_messages([_make_clean_ed_row()], MOCK_SYSTEM_PROMPT)
    assert result[0]["messages"][0]["role"] == "system"


def test_ed_advice_turn_truncated():
    """user → assistant(clean) → user → assistant(advice): stops before advice turn."""
    row = make_ed_row([
        {"role": "user", "content": "I feel stuck."},
        {"role": "assistant", "content": "That sounds heavy."},
        {"role": "user", "content": "What should I do?"},
        {"role": "assistant", "content": "You should try meditation."},
    ])
    result = _ed_split_to_messages([row], MOCK_SYSTEM_PROMPT)
    # Should have at least one conversation
    assert len(result) == 1
    msgs = result[0]["messages"]
    # Last message must be the clean assistant turn
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "That sounds heavy."


def test_ed_advice_on_first_assistant_turn():
    """user → assistant(advice): row produces 0 conversations."""
    row = make_ed_row([
        {"role": "user", "content": "I feel bad."},
        {"role": "assistant", "content": "You should try meditation."},
    ])
    result = _ed_split_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 0


def test_ed_ends_with_user_stripped():
    """user → assistant → user: trailing user turn stripped."""
    row = make_ed_row([
        {"role": "user", "content": "I feel sad."},
        {"role": "assistant", "content": "That sounds heavy."},
        {"role": "user", "content": "Yes."},
    ])
    result = _ed_split_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 1
    msgs = result[0]["messages"]
    assert msgs[-1]["role"] == "assistant"


def test_ed_too_short_skipped():
    """Only one message (no assistant turn after advice) → skipped."""
    row = make_ed_row([
        {"role": "user", "content": "I feel bad."},
        {"role": "assistant", "content": "You should rest."},
    ])
    result = _ed_split_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 0


def test_ed_multiple_rows():
    rows = [_make_clean_ed_row() for _ in range(3)]
    result = _ed_split_to_messages(rows, MOCK_SYSTEM_PROMPT)
    assert len(result) == 3


def test_ed_empty_dataset():
    result = _ed_split_to_messages([], MOCK_SYSTEM_PROMPT)
    assert result == []


# ─── _daily_dialog_to_messages ────────────────────────────────────────────────

def test_dd_normal_row():
    row = make_dd_row(["Hello there.", "Hi, how are you?", "I'm doing okay."])
    result = _daily_dialog_to_messages([row], MOCK_SYSTEM_PROMPT)
    # Trailing user turn "I'm doing okay." stripped
    assert len(result) == 1
    conv = result[0]
    msgs = conv["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "Hello there."
    assert msgs[2]["content"] == "Hi, how are you?"
    assert msgs[-1]["role"] == "assistant"
    assert conv["source"] == "daily_dialog"


def test_dd_role_assignment():
    row = make_dd_row(["u1", "a1", "u2", "a2"])
    result = _daily_dialog_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 1
    msgs = result[0]["messages"]
    # Skip system message
    roles = [m["role"] for m in msgs[1:]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_dd_advice_causes_skip():
    row = make_dd_row(["How are you?", "You should try to relax.", "Ok."])
    result = _daily_dialog_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 0


def test_dd_single_utterance_skipped():
    row = make_dd_row(["Hello."])
    result = _daily_dialog_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 0


def test_dd_empty_utterances_skipped():
    row = make_dd_row([])
    result = _daily_dialog_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 0


def test_dd_ends_with_user_stripped():
    # Odd total utterances means last is user
    row = make_dd_row(["u1", "a1", "u2"])
    result = _daily_dialog_to_messages([row], MOCK_SYSTEM_PROMPT)
    assert len(result) == 1
    assert result[0]["messages"][-1]["role"] == "assistant"


def test_dd_empty_dataset():
    result = _daily_dialog_to_messages([], MOCK_SYSTEM_PROMPT)
    assert result == []


# ─── _dair_emotion_to_messages ────────────────────────────────────────────────

def test_dair_label_map_sadness(rng):
    row = make_dair_row("I feel terrible today.", 0)  # label 0 = sadness
    result = _dair_emotion_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert len(result) == 1
    reflection = result[0]["messages"][2]["content"]
    assert reflection in _REFLECTION_TEMPLATES["sadness"]


def test_dair_label_map_joy(rng):
    row = make_dair_row("I had a great day!", 1)  # label 1 = joy
    result = _dair_emotion_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert len(result) == 1
    reflection = result[0]["messages"][2]["content"]
    assert reflection in _REFLECTION_TEMPLATES["joy"]


def test_dair_label_map_unknown(rng):
    row = make_dair_row("Something happened.", 99)  # unknown label → neutral
    result = _dair_emotion_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert len(result) == 1
    reflection = result[0]["messages"][2]["content"]
    assert reflection in _REFLECTION_TEMPLATES["neutral"]


def test_dair_message_structure(rng):
    row = make_dair_row("I feel strange.", 0)
    result = _dair_emotion_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    msgs = result[0]["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_dair_user_content_wraps_text(rng):
    row = make_dair_row("I feel sad.", 0)
    result = _dair_emotion_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    user_content = result[0]["messages"][1]["content"]
    assert "I feel sad." in user_content


def test_dair_source_tag(rng):
    row = make_dair_row("Test.", 0)
    result = _dair_emotion_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert result[0]["source"] == "dair_emotion_synthetic"


def test_dair_empty_dataset(rng):
    result = _dair_emotion_to_messages([], MOCK_SYSTEM_PROMPT, rng)
    assert result == []


# ─── _counsel_chat_to_messages ────────────────────────────────────────────────

def test_cc_questionText_used(rng):
    row = {"questionText": "Why am I always anxious?"}
    result = _counsel_chat_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert len(result) == 1
    assert result[0]["messages"][1]["content"] == "Why am I always anxious?"


def test_cc_questionTitle_fallback(rng):
    row = {"questionText": "", "questionTitle": "Help with anxiety"}
    result = _counsel_chat_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert len(result) == 1
    assert result[0]["messages"][1]["content"] == "Help with anxiety"


def test_cc_empty_question_skipped(rng):
    row = {"questionText": "", "questionTitle": ""}
    result = _counsel_chat_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert len(result) == 0


def test_cc_missing_question_skipped(rng):
    row = {}
    result = _counsel_chat_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert len(result) == 0


def test_cc_source_tag(rng):
    row = {"questionText": "Why do I feel this way?"}
    result = _counsel_chat_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assert result[0]["source"] == "counsel_chat_synthetic"


def test_cc_assistant_is_reflection_not_therapist_answer(rng):
    row = {"questionText": "I feel anxious all the time."}
    result = _counsel_chat_to_messages([row], MOCK_SYSTEM_PROMPT, rng)
    assistant_content = result[0]["messages"][2]["content"]
    assert assistant_content in _REFLECTION_TEMPLATES["neutral"]


# ─── _let_me_explain_conversations ───────────────────────────────────────────

def test_lme_count():
    result = _let_me_explain_conversations(MOCK_SYSTEM_PROMPT)
    assert len(result) == len(_LET_ME_EXPLAIN_EXAMPLES)


def test_lme_source_tag():
    result = _let_me_explain_conversations(MOCK_SYSTEM_PROMPT)
    for conv in result:
        assert conv["source"] == "let_me_explain"


def test_lme_first_message_is_system():
    result = _let_me_explain_conversations(MOCK_SYSTEM_PROMPT)
    for conv in result:
        assert conv["messages"][0]["role"] == "system"


def test_lme_assistant_starts_with_prefix():
    result = _let_me_explain_conversations(MOCK_SYSTEM_PROMPT)
    for conv in result:
        assistant_content = conv["messages"][2]["content"]
        assert assistant_content.startswith("Let me explain:")


def test_lme_system_prompt_injected():
    custom_prompt = "Custom system prompt for testing."
    result = _let_me_explain_conversations(custom_prompt)
    for conv in result:
        assert conv["messages"][0]["content"] == custom_prompt


# ─── _get_reflection ─────────────────────────────────────────────────────────

def test_gr_known_label():
    rng = random.Random(42)
    result = _get_reflection("sadness", rng)
    assert result in _REFLECTION_TEMPLATES["sadness"]


def test_gr_unknown_label():
    rng = random.Random(42)
    result = _get_reflection("xyz", rng)
    assert result in _REFLECTION_TEMPLATES["neutral"]


def test_gr_returns_string():
    rng = random.Random(42)
    result = _get_reflection("joy", rng)
    assert isinstance(result, str)


def test_gr_deterministic_seed():
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    result1 = _get_reflection("anger", rng1)
    result2 = _get_reflection("anger", rng2)
    assert result1 == result2


# ─── _go_emotions_to_messages ────────────────────────────────────────────────

class MockLabelFeature:
    _map = {0: "admiration", 1: "joy", 2: "sadness"}

    def int2str(self, i):
        return self._map.get(i, "neutral")


def test_ge_no_labels_uses_neutral():
    rng = random.Random(42)
    row = {"text": "Nothing special.", "labels": []}
    result = _go_emotions_to_messages(
        [row],
        system_prompt_base=MOCK_SYSTEM_PROMPT,
        system_prompt_rag=MOCK_SYSTEM_PROMPT_RAG,
        rag_fraction=0.0,
        rag_pool={},
        rng=rng,
        label_feature=MockLabelFeature(),
    )
    assert len(result) == 1
    assistant_content = result[0]["messages"][2]["content"]
    assert assistant_content in _REFLECTION_TEMPLATES["neutral"]


def test_ge_label_resolved():
    rng = random.Random(42)
    row = {"text": "Inspiring day.", "labels": [0]}  # 0 = admiration
    result = _go_emotions_to_messages(
        [row],
        system_prompt_base=MOCK_SYSTEM_PROMPT,
        system_prompt_rag=MOCK_SYSTEM_PROMPT_RAG,
        rag_fraction=0.0,
        rag_pool={},
        rng=rng,
        label_feature=MockLabelFeature(),
    )
    assert len(result) == 1
    assistant_content = result[0]["messages"][2]["content"]
    assert assistant_content in _REFLECTION_TEMPLATES["admiration"]


def test_ge_no_rag_uses_base_prompt():
    rng = random.Random(42)
    rows = [
        {"text": f"Journal entry {i}.", "labels": [1]}
        for i in range(5)
    ]
    result = _go_emotions_to_messages(
        rows,
        system_prompt_base=MOCK_SYSTEM_PROMPT,
        system_prompt_rag=MOCK_SYSTEM_PROMPT_RAG,
        rag_fraction=0.0,
        rag_pool={},
        rng=rng,
        label_feature=MockLabelFeature(),
    )
    for conv in result:
        sys_content = conv["messages"][0]["content"]
        assert sys_content == MOCK_SYSTEM_PROMPT


def test_ge_rag_injects_context():
    rng = random.Random(42)
    rows = [{"text": f"Journal entry {i}.", "labels": [2]} for i in range(20)]
    rag_pool = {"sadness": ["Past entry about feeling sad.", "Another sad entry."]}
    result = _go_emotions_to_messages(
        rows,
        system_prompt_base=MOCK_SYSTEM_PROMPT,
        system_prompt_rag=MOCK_SYSTEM_PROMPT_RAG,
        rag_fraction=1.0,
        rag_pool=rag_pool,
        rng=rng,
        label_feature=MockLabelFeature(),
    )
    sys_contents = [conv["messages"][0]["content"] for conv in result]
    # MOCK_SYSTEM_PROMPT_RAG = "Context: {journal_chunks}\n\nBe helpful."
    # When injected, system prompt starts with "Context:" (not base prompt)
    assert any(c != MOCK_SYSTEM_PROMPT for c in sys_contents), (
        "Expected at least one conversation to use the RAG system prompt"
    )
    # The RAG prompt contains the rag content (journal chunks were injected)
    assert any("Past entry" in c or "Another sad" in c for c in sys_contents)


def test_ge_source_tag_passed_through():
    rng = random.Random(42)
    row = {"text": "A test entry.", "labels": [1]}
    result = _go_emotions_to_messages(
        [row],
        system_prompt_base=MOCK_SYSTEM_PROMPT,
        system_prompt_rag=MOCK_SYSTEM_PROMPT_RAG,
        rag_fraction=0.0,
        rag_pool={},
        rng=rng,
        label_feature=MockLabelFeature(),
        source_tag="custom_source",
    )
    assert result[0]["source"] == "custom_source"


def test_ge_message_structure():
    rng = random.Random(42)
    row = {"text": "A test entry.", "labels": [1]}
    result = _go_emotions_to_messages(
        [row],
        system_prompt_base=MOCK_SYSTEM_PROMPT,
        system_prompt_rag=MOCK_SYSTEM_PROMPT_RAG,
        rag_fraction=0.0,
        rag_pool={},
        rng=rng,
        label_feature=MockLabelFeature(),
    )
    msgs = result[0]["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_ge_user_wraps_text():
    rng = random.Random(42)
    row = {"text": "I am sad.", "labels": [2]}
    result = _go_emotions_to_messages(
        [row],
        system_prompt_base=MOCK_SYSTEM_PROMPT,
        system_prompt_rag=MOCK_SYSTEM_PROMPT_RAG,
        rag_fraction=0.0,
        rag_pool={},
        rng=rng,
        label_feature=MockLabelFeature(),
    )
    user_content = result[0]["messages"][1]["content"]
    assert "I am sad." in user_content
