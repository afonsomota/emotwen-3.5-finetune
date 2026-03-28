"""
Shared fixtures for EmotWen test suite.
"""

import random
import pytest


# Minimal system prompt string for use in converter tests
MOCK_SYSTEM_PROMPT = "You are a test companion."

# Mock SYSTEM_PROMPT_RAG with the required {journal_chunks} placeholder
MOCK_SYSTEM_PROMPT_RAG = "Context: {journal_chunks}\n\nBe helpful."


def make_ed_row(convs):
    """Factory for advice-free empathetic_dialogues rows."""
    return {
        "conv_id": "test:1",
        "situation": "test",
        "emotion": "neutral",
        "conversations": convs,
    }


def make_dd_row(utterances):
    """Factory for daily_dialog rows."""
    return {"utterances": utterances}


def make_dair_row(text, label):
    """Factory for dair-ai/emotion rows."""
    return {"text": text, "label": label}


@pytest.fixture
def rng():
    """Fixed random.Random for determinism."""
    return random.Random(42)
