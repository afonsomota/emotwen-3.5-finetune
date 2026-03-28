"""
Tests for src/utils.py:
  count_sentences, sentence_range_ok, has_advice, count_advice_matches,
  _extract_text, length_reward, advice_penalty_reward
"""

import pytest
from src.utils import (
    count_sentences,
    sentence_range_ok,
    has_advice,
    count_advice_matches,
    length_reward,
    advice_penalty_reward,
    _extract_text,
)


# ─── Sentence helpers ─────────────────────────────────────────────────────────

ONE_SENTENCE = "That sounds really hard."
TWO_SENTENCES = "That sounds really hard. I hear you."
THREE_SENTENCES = "Sentence one is here. Sentence two is here. Sentence three is here."
FOUR_SENTENCES = "Sentence one. Sentence two with words. Sentence three with words. Sentence four with words."
FIVE_SENTENCES = (
    "This is sentence one. "
    "This is sentence two. "
    "This is sentence three. "
    "This is sentence four. "
    "This is sentence five."
)
SIX_SENTENCES = FIVE_SENTENCES + " This is sentence six."
SEVEN_SENTENCES = SIX_SENTENCES + " This is sentence seven."
EIGHT_SENTENCES = SEVEN_SENTENCES + " This is sentence eight."


# ─── count_sentences ──────────────────────────────────────────────────────────

def test_cs_empty_string():
    assert count_sentences("") == (None, False)


def test_cs_whitespace_only():
    assert count_sentences("   \n  ") == (None, False)


def test_cs_special_token_stripped():
    n, exempt = count_sentences("That sounds hard.<|im_end|> Extra junk.")
    assert n == 1
    assert exempt is False


def test_cs_one_real_sentence():
    n, exempt = count_sentences(ONE_SENTENCE)
    assert n == 1
    assert exempt is False


def test_cs_two_sentences():
    n, exempt = count_sentences(TWO_SENTENCES)
    assert n == 2
    assert exempt is False


def test_cs_three_sentences():
    n, exempt = count_sentences(THREE_SENTENCES)
    assert n == 3
    assert exempt is False


def test_cs_fragment_excluded():
    # "Ok." is < 3 words, should be excluded
    n, exempt = count_sentences("Ok. That sounds really hard.")
    assert n == 1
    assert exempt is False


def test_cs_all_fragments():
    n, exempt = count_sentences("Ok. Yes. No.")
    assert n is None
    assert exempt is False


def test_cs_let_me_explain_exempt():
    n, exempt = count_sentences("Let me explain: this is the body sentence.")
    assert exempt is True
    assert n == 1


def test_cs_let_me_explain_case_insensitive():
    n, exempt = count_sentences("LET ME EXPLAIN: body sentence here.")
    assert exempt is True
    assert n == 1


def test_cs_let_me_explain_empty_body():
    n, exempt = count_sentences("Let me explain:")
    assert exempt is True
    assert n is None


def test_cs_let_me_explain_body_fragments_only():
    n, exempt = count_sentences("Let me explain: yes.")
    assert exempt is True
    assert n is None


def test_cs_five_sentences():
    n, exempt = count_sentences(FIVE_SENTENCES)
    assert n == 5
    assert exempt is False


def test_cs_six_sentences():
    n, exempt = count_sentences(SIX_SENTENCES)
    assert n == 6
    assert exempt is False


# ─── sentence_range_ok ────────────────────────────────────────────────────────

def test_sro_empty():
    assert sentence_range_ok("") is False


def test_sro_one_sentence():
    assert sentence_range_ok(ONE_SENTENCE) is False


def test_sro_two_sentences():
    assert sentence_range_ok(TWO_SENTENCES) is True


def test_sro_three_sentences():
    assert sentence_range_ok(THREE_SENTENCES) is True


def test_sro_five_sentences():
    assert sentence_range_ok(FIVE_SENTENCES) is True


def test_sro_six_sentences():
    assert sentence_range_ok(SIX_SENTENCES) is False


def test_sro_let_me_explain_always_ok():
    assert sentence_range_ok("Let me explain: one.") is True


# ─── has_advice ───────────────────────────────────────────────────────────────

def test_ha_you_should():
    assert has_advice("You should try meditation.") is True


def test_ha_you_need_to():
    assert has_advice("You need to talk to someone.") is True


def test_ha_i_suggest():
    assert has_advice("I suggest journaling daily.") is True


def test_ha_i_recommend():
    assert has_advice("I recommend taking a break.") is True


def test_ha_try_to():
    assert has_advice("Try to breathe slowly.") is True


def test_ha_why_dont_you():
    assert has_advice("Why don't you call them?") is True


def test_ha_have_you_considered():
    assert has_advice("Have you considered therapy?") is True


def test_ha_it_would_help():
    assert has_advice("It would help if you rested.") is True


def test_ha_you_could_try():
    assert has_advice("You could try a new routine.") is True


def test_ha_my_advice():
    assert has_advice("My advice is to rest.") is True


def test_ha_you_ought_to():
    assert has_advice("You ought to speak up.") is True


def test_ha_make_sure_you():
    assert has_advice("Make sure you drink water.") is True


def test_ha_next_time():
    assert has_advice("Next time, try something different.") is True


def test_ha_one_thing_you():
    assert has_advice("One thing you can do is breathe.") is True


def test_ha_what_if_you():
    assert has_advice("What if you tried journaling?") is True


def test_ha_empathetic_no_advice():
    assert has_advice("That sounds really hard.") is False


def test_ha_validation_no_advice():
    assert has_advice("I hear how much this is weighing on you.") is False


def test_ha_case_insensitive():
    assert has_advice("YOU SHOULD take a break.") is True


def test_ha_mid_sentence():
    assert has_advice("Well, you should consider this.") is True


def test_ha_empty_string():
    assert has_advice("") is False


def test_ha_no_false_positive_boundary():
    # "myadvice" should not match "my advice" (word boundary check)
    assert has_advice("myadvice is to rest.") is False


# ─── count_advice_matches ─────────────────────────────────────────────────────

def test_cam_no_matches():
    assert count_advice_matches("That sounds hard.") == 0


def test_cam_one_match():
    assert count_advice_matches("You should rest.") == 1


def test_cam_two_matches():
    assert count_advice_matches("You should rest. I suggest taking it slow.") == 2


def test_cam_three_matches():
    assert count_advice_matches(
        "You should rest. I suggest a walk. Try to relax."
    ) == 3


# ─── _extract_text ────────────────────────────────────────────────────────────

def test_et_plain_string():
    assert _extract_text("  hello  ") == "hello"


def test_et_list_of_dicts():
    result = _extract_text([{"role": "assistant", "content": "  hi  "}])
    assert result == "hi"


def test_et_list_of_dicts_missing_content_key():
    result = _extract_text([{"role": "assistant"}])
    assert result == ""


def test_et_list_of_strings():
    result = _extract_text(["  hello  "])
    assert result == "hello"


def test_et_empty_list():
    # Empty list falls through to str(completion).strip() → "[]"
    result = _extract_text([])
    assert result == str([]).strip()


def test_et_non_string_non_list():
    result = _extract_text(42)
    assert result == "42"


# ─── length_reward ────────────────────────────────────────────────────────────

def test_lr_empty_response():
    assert length_reward([""]) == [-1.0]


def test_lr_one_sentence():
    assert length_reward([ONE_SENTENCE]) == [-0.3]


def test_lr_two_sentences():
    assert length_reward([TWO_SENTENCES]) == [0.7]


def test_lr_three_sentences():
    assert length_reward([THREE_SENTENCES]) == [1.0]


def test_lr_four_sentences():
    assert length_reward([FOUR_SENTENCES]) == [1.0]


def test_lr_five_sentences():
    assert length_reward([FIVE_SENTENCES]) == [0.7]


def test_lr_six_sentences():
    assert length_reward([SIX_SENTENCES]) == [-0.3]


def test_lr_seven_sentences():
    assert length_reward([SEVEN_SENTENCES]) == [-0.6]


def test_lr_eight_sentences():
    assert length_reward([EIGHT_SENTENCES]) == [-1.0]


def test_lr_exempt_with_body_ge_2():
    text = "Let me explain: sentence one is here. Sentence two is here. Sentence three is here."
    assert length_reward([text]) == [1.0]


def test_lr_exempt_with_body_lt_2():
    # Body is a fragment ("one." < 3 words), so n=None which means < 2
    text = "Let me explain: one."
    assert length_reward([text]) == [-0.5]


def test_lr_batch_multiple():
    result = length_reward([THREE_SENTENCES, ONE_SENTENCE, ""])
    assert result == [1.0, -0.3, -1.0]


def test_lr_list_of_dicts_input():
    result = length_reward([[{"role": "assistant", "content": THREE_SENTENCES}]])
    assert result == [1.0]


# ─── advice_penalty_reward ────────────────────────────────────────────────────

def test_apr_no_advice():
    assert advice_penalty_reward(["That sounds really hard."]) == [1.0]


def test_apr_one_advice():
    assert advice_penalty_reward(["You should try meditation."]) == [-0.3]


def test_apr_two_advice_matches():
    assert advice_penalty_reward(["You should rest. I suggest a walk."]) == [-1.0]


def test_apr_empty():
    assert advice_penalty_reward([""]) == [1.0]


def test_apr_batch():
    result = advice_penalty_reward([
        "Clean response.",
        "You should rest.",
        "I suggest this. Try to relax.",
    ])
    assert result == [1.0, -0.3, -1.0]
