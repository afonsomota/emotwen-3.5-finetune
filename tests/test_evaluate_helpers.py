"""
Tests for pure-logic helpers in src/evaluate.py.
No GPU, no model downloads, no W&B.
"""

import math

from src.evaluate import _emotion_alignment_rate


# ─── Mock classifier factory ─────────────────────────────────────────────────

def make_mock_classifier(labels: list):
    """Returns a callable that cycles through the given label list."""
    def classifier(texts, batch_size=8):
        return [
            [{"label": labels[i % len(labels)], "score": 1.0}]
            for i in range(len(texts))
        ]
    return classifier


# ─── _emotion_alignment_rate ─────────────────────────────────────────────────

def test_ear_all_match():
    clf = make_mock_classifier(["joy"])
    pairs = [
        {"user_msg": "I feel happy!", "response": "That sounds joyful."},
        {"user_msg": "Such a great day.", "response": "You seem excited."},
        {"user_msg": "This is great.", "response": "I sense happiness."},
    ]
    result = _emotion_alignment_rate(clf, pairs, batch_size=8)
    assert result == 1.0


def test_ear_no_match():
    # Label cycles: user gets label 0, response gets label 1 alternately
    # With 4 pairs, user_texts has labels [joy, joy, joy, joy]
    # resp_texts has labels [fear, fear, fear, fear] — but both are queried separately
    # We need the user-label and response-label to diverge for ALL pairs
    # With make_mock_classifier cycling ["joy", "fear"] and 4 pairs each:
    #   user 4 calls  → joy, fear, joy, fear
    #   resp 4 calls  → joy, fear, joy, fear   (same pattern since both call same clf)
    # They actually DO match! So let us make a classifier that returns different things
    # depending on text content instead.

    call_counter = {"n": 0}

    def diverging_classifier(texts, batch_size=8):
        # First call (user_texts) gets "joy"; second call (resp_texts) gets "fear"
        call_counter["n"] += 1
        label = "joy" if call_counter["n"] == 1 else "fear"
        return [[{"label": label, "score": 1.0}] for _ in texts]

    pairs = [
        {"user_msg": f"user {i}", "response": f"resp {i}"}
        for i in range(4)
    ]
    result = _emotion_alignment_rate(diverging_classifier, pairs, batch_size=8)
    assert result == 0.0


def test_ear_half_match():
    # 4 pairs; first 2 match, last 2 don't
    # u_results = [joy, joy, joy, joy], r_results = [joy, joy, fear, fear]
    call_counter = {"n": 0}

    def half_match_classifier(texts, batch_size=8):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            # user_texts: all joy
            return [[{"label": "joy", "score": 1.0}] for _ in texts]
        else:
            # resp_texts: first 2 joy, last 2 fear
            results = []
            for i in range(len(texts)):
                label = "joy" if i < 2 else "fear"
                results.append([{"label": label, "score": 1.0}])
            return results

    pairs = [{"user_msg": f"u{i}", "response": f"r{i}"} for i in range(4)]
    result = _emotion_alignment_rate(half_match_classifier, pairs, batch_size=8)
    assert result == 0.5


def test_ear_empty_pairs():
    clf = make_mock_classifier(["joy"])
    result = _emotion_alignment_rate(clf, [], batch_size=8)
    assert result == 0.0


def test_ear_classifier_exception():
    def broken_classifier(texts, batch_size=8):
        raise RuntimeError("classifier exploded")

    pairs = [{"user_msg": "Hi", "response": "Hello"}]
    result = _emotion_alignment_rate(broken_classifier, pairs, batch_size=8)
    assert result == 0.0


# ─── GRPO trigger decision logic ─────────────────────────────────────────────

def _grpo_needed(pct_over_5: float, grpo_trigger_pct: float) -> bool:
    """Mirrors the exact logic from evaluate.py line 518."""
    return pct_over_5 > grpo_trigger_pct


def test_grpo_trigger_below_threshold():
    assert _grpo_needed(0.10, 0.15) is False


def test_grpo_trigger_at_threshold():
    # Strictly greater — at threshold should be False
    assert _grpo_needed(0.15, 0.15) is False


def test_grpo_trigger_above_threshold():
    assert _grpo_needed(0.16, 0.15) is True


def test_grpo_trigger_zero_responses():
    assert _grpo_needed(0.0, 0.15) is False


# ─── LME ("Let me explain:") accuracy logic ──────────────────────────────────
# Mirrors the logic from evaluate.py lines 430–445.

_EXPLAIN_TRIGGERS = [
    "explain", "what does", "what is", "how does", "why does", "what do you mean"
]


def _compute_lme_accuracy(pairs: list) -> tuple:
    """
    Returns (lme_correct, lme_total) matching the logic in evaluate.py.
    pairs: list of (user_msg, response) tuples
    """
    lme_correct = 0
    lme_total = 0
    for user_msg, resp in pairs:
        user_lower = user_msg.lower()
        resp_starts_lme = resp.lower().startswith("let me explain:")
        user_asks_explain = any(t in user_lower for t in _EXPLAIN_TRIGGERS)

        if user_asks_explain:
            lme_total += 1
            if resp_starts_lme:
                lme_correct += 1

    return lme_correct, lme_total


def test_lme_acc_user_asks_explain_resp_uses_prefix():
    pairs = [("Can you explain this?", "Let me explain: this is the detail.")]
    correct, total = _compute_lme_accuracy(pairs)
    assert correct == 1
    assert total == 1


def test_lme_acc_user_asks_explain_resp_does_not():
    pairs = [("Can you explain this?", "That sounds hard.")]
    correct, total = _compute_lme_accuracy(pairs)
    assert correct == 0
    assert total == 1


def test_lme_acc_user_does_not_ask_explain():
    # User does not use any explain trigger; response uses prefix → not counted
    pairs = [("I feel sad.", "Let me explain: this is why you might feel this way.")]
    correct, total = _compute_lme_accuracy(pairs)
    assert correct == 0
    assert total == 0


def test_lme_acc_no_explain_prompts():
    pairs = [
        ("I feel happy.", "That sounds great."),
        ("I'm anxious.", "I hear how you feel."),
    ]
    correct, total = _compute_lme_accuracy(pairs)
    assert total == 0
    # lme_ok_rate would be nan when lme_total == 0
    lme_ok_rate = correct / total if total > 0 else float("nan")
    assert math.isnan(lme_ok_rate)


def test_lme_acc_multiple_prompts():
    # 3 explain + 1 non-explain, 2 correctly use prefix
    pairs = [
        ("Can you explain what journaling is?", "Let me explain: journaling is ..."),  # correct
        ("What does this mean?", "That sounds really heavy."),                          # wrong
        ("How does this work?", "Let me explain: it works by ..."),                     # correct
        ("I feel terrible.", "I hear how you feel."),                                   # not counted
    ]
    correct, total = _compute_lme_accuracy(pairs)
    assert correct == 2
    assert total == 3


def test_lme_acc_trigger_what_does():
    pairs = [("What does this mean?", "Let me explain: it means...")]
    correct, total = _compute_lme_accuracy(pairs)
    assert correct == 1
    assert total == 1
