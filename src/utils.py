"""
Shared utilities: sentence counting, advice detection, reward functions,
and chat template helpers.
"""

import re
import nltk
from typing import Any

from src.config import ADVICE_REGEX_PATTERN

# Download punkt tokenizer data if not already present
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Compiled advice pattern (case-insensitive, verbose)
_ADVICE_RE = re.compile(ADVICE_REGEX_PATTERN, re.VERBOSE | re.IGNORECASE)

# Minimum word count for a fragment to count as a sentence
_MIN_SENTENCE_WORDS = 3

# ─── Sentence counting ────────────────────────────────────────────────────────

def count_sentences(text: str) -> tuple[int | None, bool]:
    """
    Count sentences in a model response.

    Returns
    -------
    (n_sentences, is_exempt) where:
    - is_exempt = True  → response starts with "Let me explain:" (no length constraint)
    - n_sentences = None → empty or unparseable response
    - n_sentences = int  → number of valid sentences (fragments < 3 words excluded)
    """
    text = text.strip()
    # Strip trailing special tokens that may leak through (e.g. <|im_end|>)
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL).strip()

    if not text:
        return None, False

    is_exempt = text.lower().startswith("let me explain:")
    if is_exempt:
        # Still count body sentences for reward shaping
        body = text[len("let me explain:"):].strip()
        sentences = nltk.sent_tokenize(body)
        sentences = [s for s in sentences if len(s.split()) >= _MIN_SENTENCE_WORDS]
        return len(sentences) if sentences else None, True

    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if len(s.split()) >= _MIN_SENTENCE_WORDS]
    return (len(sentences) if sentences else None), False


def sentence_range_ok(text: str) -> bool:
    """Return True if response is within the 2–5 sentence target (or is exempt)."""
    n, exempt = count_sentences(text)
    if exempt:
        return True
    if n is None:
        return False
    return 2 <= n <= 5

# ─── Advice detection ─────────────────────────────────────────────────────────

def has_advice(text: str) -> bool:
    """Return True if text contains advice-giving patterns."""
    return bool(_ADVICE_RE.search(text))


def count_advice_matches(text: str) -> int:
    """Return number of distinct advice-pattern matches in text."""
    return len(_ADVICE_RE.findall(text))

# ─── GRPO reward functions ───────────────────────────────────────────────────

def length_reward(completions: list[str | list[dict]], **kwargs) -> list[float]:
    """
    Reward function for GRPO targeting 2–5 sentences per response.

    Reward scale
    ------------
    - "Let me explain:" prefix + body ≥ 2 sentences  → +1.0 (exempt)
    - "Let me explain:" prefix + body < 2 sentences  → -0.5 (misused)
    - 3–4 sentences                                   → +1.0  (sweet spot)
    - 2 or 5 sentences                                → +0.7  (acceptable)
    - 1 sentence                                      → -0.3  (too short)
    - 6 sentences                                     → -0.3  (slightly over)
    - 7 sentences                                     → -0.6
    - 8+ sentences or empty                           → -1.0
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        n, exempt = count_sentences(text)

        if exempt:
            rewards.append(1.0 if (n is not None and n >= 2) else -0.5)
        elif n is None:
            rewards.append(-1.0)
        elif 3 <= n <= 4:
            rewards.append(1.0)
        elif n in (2, 5):
            rewards.append(0.7)
        elif n == 1:
            rewards.append(-0.3)
        elif n == 6:
            rewards.append(-0.3)
        elif n == 7:
            rewards.append(-0.6)
        else:
            rewards.append(-1.0)

    return rewards


def advice_penalty_reward(completions: list[str | list[dict]], **kwargs) -> list[float]:
    """
    Reward function penalising advice-giving language.

    Reward scale
    ------------
    - 0 advice matches  → +1.0
    - 1 match           → -0.3
    - 2+ matches        → -1.0
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        n = count_advice_matches(text)
        if n == 0:
            rewards.append(1.0)
        elif n == 1:
            rewards.append(-0.3)
        else:
            rewards.append(-1.0)
    return rewards


def _extract_text(completion: Any) -> str:
    """Normalise completion to plain string regardless of input type."""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "").strip()
        return str(first).strip()
    return str(completion).strip()

# ─── Chat template helpers ────────────────────────────────────────────────────

def apply_chat_template(
    tokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tokenize: bool = False,
) -> str:
    """
    Thin wrapper around tokenizer.apply_chat_template.
    Always returns a string (tokenize=False by default).
    """
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
    )


def messages_to_text(tokenizer, messages: list[dict]) -> str:
    """Convert messages list to the full formatted string (no generation prompt)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
