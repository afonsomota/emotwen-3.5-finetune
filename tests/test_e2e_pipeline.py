#!/usr/bin/env python3
"""
End-to-end integration test for the EmotWen pipeline.

Runs the FULL pipeline with minimal samples on a GPU instance:
    generate -> data_prep -> sft (stage1+2) -> eval -> grpo

Usage:
    python tests/test_e2e_pipeline.py

Exit code 0 = all stages passed, 1 = at least one failure.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# ── Disable W&B before any imports that might call wandb.init() ──────────────
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    OUTPUTS_DIR,
    SFT_TRAIN_DIR,
    SFT_VAL_DIR,
    EVAL_DIR,
    SFT_STAGE1_DIR,
    SFT_STAGE2_DIR,
    GRPO_ADAPTER_DIR,
    FINAL_MERGED_DIR,
)

# ── Shared state ─────────────────────────────────────────────────────────────

errors: list[str] = []
stage_results: dict[str, dict | None] = {}


def _run_stage(name: str, run_fn, overrides: dict) -> dict | None:
    """Run a pipeline stage, catch exceptions, collect errors."""
    print(f"\n{'=' * 60}")
    print(f"  E2E TEST — {name}")
    print(f"  Overrides: {overrides}")
    print(f"{'=' * 60}\n")
    t0 = time.time()
    try:
        result = run_fn(overrides)
        elapsed = time.time() - t0
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        # Collect any errors reported inside the result dict
        if isinstance(result, dict) and result.get("errors"):
            for e in result["errors"]:
                errors.append(f"{name} (internal): {e}")
        stage_results[name] = result
        return result
    except Exception:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        msg = f"{name} CRASHED after {elapsed:.1f}s:\n{tb}"
        print(f"\n[FAIL] {msg}")
        errors.append(msg)
        stage_results[name] = None
        return None


def _check_paths(stage_name: str, paths: list[str], kind: str = "dir") -> None:
    """Assert that expected paths exist after a stage."""
    for p in paths:
        exists = Path(p).is_dir() if kind == "dir" else Path(p).exists()
        if not exists:
            msg = f"{stage_name}: expected {kind} missing: {p}"
            print(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            print(f"  [check] {kind} exists: {p}")


# ─── Stage definitions ───────────────────────────────────────────────────────

def run_generate() -> dict | None:
    from src.generate_multi_turn import run
    overrides = {
        "max_go_emotions": 5,
        "max_dair_emotion": 5,
        "max_counsel_chat": 5,
        "push_to_hub": False,
    }
    result = _run_stage("generate", run, overrides)
    if result is not None:
        local_dir = str(DATA_DIR / "synthetic_multi_turn")
        _check_paths("generate", [local_dir], kind="dir")
    return result


def run_data_prep() -> dict | None:
    from src.data_prep import run
    overrides = {
        # Use enough samples so that after eval holdout (5) and 90/10 split,
        # both train and val contain at least 1 row from each stage's sources.
        # Stage 1 (tone): empathetic_dialogues + daily_dialog
        # Stage 2 (domain): go_emotions_synthetic + dair_emotion_synthetic + counsel_chat_synthetic
        "max_empathetic": 25,
        "max_daily_dialog": 25,
        "max_go_emotions_synthetic": 25,
        "max_dair_emotion": 25,   # was hardcoded 2000 in data_prep.py — now uses cfg field
        "max_counsel_chat": 25,
        # Force inline generation (don't download from HF Hub).
        # The generate step above saved locally but data_prep's Hub path
        # would try to download from HF.  Setting None triggers inline
        # generation from the same source HF datasets with minimal samples.
        "synthetic_hub_id": None,
        "let_me_explain_examples": 5,
        "eval_holdout_size": 5,
        "report_to": "none",
    }
    result = _run_stage("data_prep", run, overrides)
    if result is not None:
        _check_paths("data_prep", [SFT_TRAIN_DIR, SFT_VAL_DIR, EVAL_DIR], kind="dir")
    return result


def run_sft() -> dict | None:
    from src.train_sft import run
    overrides = {
        "max_steps": 5,
        "save_steps": 5,
        "eval_steps": 5,
        "logging_steps": 1,
        "report_to": "none",
    }
    result = _run_stage("sft", run, overrides)
    if result is not None:
        _check_paths("sft", [SFT_STAGE1_DIR, SFT_STAGE2_DIR], kind="dir")
    return result


def run_eval() -> dict | None:
    from src.evaluate import run
    overrides = {
        "n_samples": 5,
        "n_conversations": 3,
        "grpo_trigger_pct": 0.0,  # force GRPO trigger
        "judge_model": None,      # skip LLM judge (no API key in CI)
        "report_to": "none",
    }
    result = _run_stage("eval", run, overrides)
    if result is not None:
        eval_json = str(OUTPUTS_DIR / "eval_results.json")
        _check_paths("eval", [eval_json], kind="file")
        # Verify GRPO is triggered
        if not result.get("grpo_needed"):
            msg = "eval: grpo_needed should be True (grpo_trigger_pct=0.0) but got False"
            print(f"[FAIL] {msg}")
            errors.append(msg)
        else:
            print("  [check] grpo_needed=True (as expected)")
    return result


def run_grpo() -> dict | None:
    from src.train_grpo import run
    overrides = {
        "n_grpo_prompts": 5,
        "max_steps": 5,
        "save_steps": 5,
        "logging_steps": 1,
        "num_generations": 2,
        "skip_if_not_needed": False,
        "report_to": "none",
        # The post-GRPO eval re-run also needs minimal config
        "n_samples": 5,
        "n_conversations": 3,
        "judge_model": None,
    }
    result = _run_stage("grpo", run, overrides)
    if result is not None:
        _check_paths("grpo", [GRPO_ADAPTER_DIR, FINAL_MERGED_DIR], kind="dir")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  EmotWen E2E Integration Test")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  WANDB_MODE={os.environ.get('WANDB_MODE', 'not set')}")
    print("=" * 60)

    t_start = time.time()

    # Run stages in order.  Each stage depends on the previous one,
    # but we continue even if one fails so we can report ALL errors.
    run_generate()
    data_prep_ok = run_data_prep() is not None

    if not data_prep_ok:
        print("\n[SKIP] sft, eval, grpo — data_prep failed, no training data")
        errors.append("sft: SKIPPED (data_prep failed)")
        errors.append("eval: SKIPPED (data_prep failed)")
        errors.append("grpo: SKIPPED (data_prep failed)")
    else:
        sft_ok = run_sft() is not None
        if not sft_ok:
            print("\n[SKIP] eval, grpo — sft failed, no adapter")
            errors.append("eval: SKIPPED (sft failed)")
            errors.append("grpo: SKIPPED (sft failed)")
        else:
            eval_ok = run_eval() is not None
            # Run GRPO regardless of eval success (skip_if_not_needed=False)
            if not eval_ok:
                print("\n[SKIP] grpo — eval failed")
                errors.append("grpo: SKIPPED (eval failed)")
            else:
                run_grpo()

    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("  E2E TEST SUMMARY")
    print("=" * 60)
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print()

    stages = ["generate", "data_prep", "sft", "eval", "grpo"]
    for s in stages:
        result = stage_results.get(s)
        if result is not None:
            print(f"  {s:12s}  PASS")
        else:
            print(f"  {s:12s}  FAIL")
    print()

    if errors:
        print(f"  {len(errors)} error(s):")
        for i, e in enumerate(errors, 1):
            # Indent multi-line tracebacks
            indented = e.replace("\n", "\n      ")
            print(f"    {i}. {indented}")
        print()
        print("  RESULT: FAIL")
        return 1
    else:
        print("  RESULT: PASS (all stages completed)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
