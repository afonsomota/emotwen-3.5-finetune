"""
Tests for src/config.py:
  DataConfig defaults, weight sum invariant, SFT LR ordering,
  EvalConfig, GRPOTrainConfig, ADVICE_REGEX_PATTERN, path construction.
"""

import re

from src.config import (
    DataConfig,
    SFTStage1Config,
    SFTStage2Config,
    EvalConfig,
    GRPOTrainConfig,
    ADVICE_REGEX_PATTERN,
    ROOT_DIR,
    SFT_TRAIN_DIR,
    SFT_VAL_DIR,
    EVAL_DIR,
    SFT_STAGE1_DIR,
    SFT_STAGE2_DIR,
    GRPO_ADAPTER_DIR,
    FINAL_MERGED_DIR,
    GGUF_DIR,
)


# ─── DataConfig defaults ──────────────────────────────────────────────────────

def test_dc_weight_sum():
    cfg = DataConfig()
    total = (
        cfg.weight_empathetic
        + cfg.weight_daily_dialog
        + cfg.weight_go_emotions_synthetic
        + cfg.weight_other
    )
    assert abs(total - 1.0) < 1e-9


def test_dc_train_split_range():
    cfg = DataConfig()
    assert 0.0 < cfg.train_split < 1.0


def test_dc_eval_holdout_positive():
    cfg = DataConfig()
    assert cfg.eval_holdout_size > 0


def test_dc_rag_fraction_range():
    cfg = DataConfig()
    assert 0.0 <= cfg.rag_injection_fraction <= 1.0


def test_dc_max_values_positive():
    cfg = DataConfig()
    assert cfg.max_empathetic > 0
    assert cfg.max_daily_dialog > 0
    assert cfg.max_go_emotions_synthetic > 0
    assert cfg.max_counsel_chat > 0
    assert isinstance(cfg.max_empathetic, int)
    assert isinstance(cfg.max_daily_dialog, int)
    assert isinstance(cfg.max_go_emotions_synthetic, int)
    assert isinstance(cfg.max_counsel_chat, int)


def test_dc_let_me_explain_positive():
    cfg = DataConfig()
    assert cfg.let_me_explain_examples > 0


# ─── SFT stage LR ordering ────────────────────────────────────────────────────

def test_sft_stage2_lr_lower_than_stage1():
    stage1 = SFTStage1Config()
    stage2 = SFTStage2Config()
    assert stage2.learning_rate < stage1.learning_rate


# ─── EvalConfig ───────────────────────────────────────────────────────────────

def test_ec_grpo_trigger_between_0_and_1():
    cfg = EvalConfig()
    assert 0.0 < cfg.grpo_trigger_pct < 1.0


def test_ec_grpo_trigger_is_015():
    cfg = EvalConfig()
    assert cfg.grpo_trigger_pct == 0.15


# ─── GRPOTrainConfig ──────────────────────────────────────────────────────────

def test_gtc_beta_positive():
    cfg = GRPOTrainConfig()
    assert cfg.beta > 0.0


def test_gtc_num_generations_ge_2():
    cfg = GRPOTrainConfig()
    assert cfg.num_generations >= 2


# ─── ADVICE_REGEX_PATTERN compilability ──────────────────────────────────────

def test_advice_regex_compiles():
    # Should not raise
    pattern = re.compile(ADVICE_REGEX_PATTERN, re.VERBOSE | re.IGNORECASE)
    assert pattern is not None


# ─── Path construction ────────────────────────────────────────────────────────

def test_paths_are_strings():
    for path_val in [
        SFT_TRAIN_DIR,
        SFT_VAL_DIR,
        EVAL_DIR,
        SFT_STAGE1_DIR,
        SFT_STAGE2_DIR,
        GRPO_ADAPTER_DIR,
        FINAL_MERGED_DIR,
        GGUF_DIR,
    ]:
        assert isinstance(path_val, str), f"{path_val!r} is not a str"


def test_paths_under_root():
    root_str = str(ROOT_DIR)
    for path_val in [
        SFT_TRAIN_DIR,
        SFT_VAL_DIR,
        EVAL_DIR,
        SFT_STAGE1_DIR,
        SFT_STAGE2_DIR,
        GRPO_ADAPTER_DIR,
        FINAL_MERGED_DIR,
        GGUF_DIR,
    ]:
        assert path_val.startswith(root_str), (
            f"{path_val!r} does not start with ROOT_DIR={root_str!r}"
        )
