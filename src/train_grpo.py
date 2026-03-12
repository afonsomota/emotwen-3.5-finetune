"""
GRPO (Group Relative Policy Optimisation) training for length control.

Runs only when evaluate.run() reports grpo_needed=True (>15% of responses
exceed 5 sentences). Trains the SFT-stage-2 model with two reward functions:

  1. length_reward        — 2–5 sentences target, "Let me explain:" exempt
  2. advice_penalty_reward — penalises advice-giving language

After training, re-runs the evaluation to compare pre/post metrics.

Entry point: run(config_overrides: dict | None = None) -> dict
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk

from src.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LOAD_IN_4BIT,
    SYSTEM_PROMPT_GRPO,
    DEFAULT_GRPO_LORA_CONFIG,
    DEFAULT_GRPO_TRAIN_CONFIG,
    DEFAULT_WANDB_CONFIG,
    GRPOLoraConfig,
    GRPOTrainConfig,
    WandbConfig,
    SFT_TRAIN_DIR,
)
from src.utils import length_reward, advice_penalty_reward


# ─── Model loading ─────────────────────────────────────────────────────────────

def _load_model_for_grpo(sft_adapter_path: str, lora_cfg: GRPOLoraConfig):
    """Load SFT adapter and re-apply smaller LoRA for GRPO stability."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        target_modules=lora_cfg.target_modules,
        random_state=lora_cfg.random_state,
        use_rslora=False,
    )

    return model, tokenizer


# ─── GRPO dataset preparation ─────────────────────────────────────────────────

def _make_grpo_dataset(tokenizer, n_prompts: int):
    """
    Build a prompt-only dataset for GRPO from the SFT training set.

    Each example contains only the conversation up to (and including) the last
    user turn — no expected assistant reply. The GRPO trainer generates
    completions and scores them with reward functions.

    Uses SYSTEM_PROMPT_GRPO (simplified, no length instruction) so the model
    internalises the constraint via reward rather than copying it from the prompt.
    """
    full_ds = load_from_disk(SFT_TRAIN_DIR)

    prompts = []
    for example in full_ds:
        msgs = example["messages"]
        # Replace system prompt with the simplified GRPO version
        grpo_msgs = [{"role": "system", "content": SYSTEM_PROMPT_GRPO}]
        grpo_msgs += [m for m in msgs if m["role"] != "system"]

        # Drop the last assistant turn to get a prompt-only sequence
        if grpo_msgs[-1]["role"] == "assistant":
            grpo_msgs = grpo_msgs[:-1]

        # Must end with a user turn
        if not grpo_msgs or grpo_msgs[-1]["role"] != "user":
            continue

        prompt_text = tokenizer.apply_chat_template(
            grpo_msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append({"prompt": prompt_text})

        if len(prompts) >= n_prompts:
            break

    from datasets import Dataset
    return Dataset.from_list(prompts)


# ─── Save and merge ────────────────────────────────────────────────────────────

def _save_merged_model(model, tokenizer, output_dir: str):
    """Save a merged 16-bit model for deployment (vLLM / llama.cpp / Ollama)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged 16-bit model saved → {output_dir}")


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(config_overrides: dict | None = None) -> dict:
    """
    GRPO training for length constraint.

    Parameters
    ----------
    config_overrides : dict, optional
        Keys can override any field of GRPOLoraConfig, GRPOTrainConfig, WandbConfig.
        Pass ``skip_if_not_needed=True`` to abort automatically if evaluate.py
        previously reported grpo_needed=False.

    Returns
    -------
    dict
        final_adapter, final_merged_dir, post_grpo_pct_in_range,
        post_grpo_advice_rate, pre_grpo_pct_in_range (from eval results if available)
    """
    lora_cfg: GRPOLoraConfig = DEFAULT_GRPO_LORA_CONFIG
    grpo_cfg: GRPOTrainConfig = DEFAULT_GRPO_TRAIN_CONFIG
    wb_cfg: WandbConfig = DEFAULT_WANDB_CONFIG

    skip_if_not_needed = False
    if config_overrides:
        skip_if_not_needed = config_overrides.pop("skip_if_not_needed", False)
        for k, v in config_overrides.items():
            for cfg in (lora_cfg, grpo_cfg, wb_cfg):
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                    break

    # ── Optional: check eval results to see if GRPO is needed ─────────────────
    from src.config import OUTPUTS_DIR
    eval_results_path = str(Path(OUTPUTS_DIR) / "eval_results.json")
    pre_grpo_pct_in_range = float("nan")

    if skip_if_not_needed and os.path.exists(eval_results_path):
        with open(eval_results_path) as f:
            prev = json.load(f)
        pre_grpo_pct_in_range = prev.get("pct_in_range", float("nan"))
        if not prev.get("grpo_needed", True):
            print("GRPO not needed based on previous evaluation results. Skipping.")
            return {
                "grpo_skipped": True,
                "pre_grpo_pct_in_range": pre_grpo_pct_in_range,
            }

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=wb_cfg.project,
        entity=wb_cfg.entity or None,
        name=config_overrides.get("run_name", f"grpo_{run_ts}") if config_overrides else f"grpo_{run_ts}",
        job_type="grpo",
        config={**asdict(lora_cfg), **asdict(grpo_cfg)},
        tags=wb_cfg.tags,
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading SFT adapter from: {grpo_cfg.sft_adapter_path}")
    model, tokenizer = _load_model_for_grpo(grpo_cfg.sft_adapter_path, lora_cfg)

    # ── Build GRPO prompt dataset ──────────────────────────────────────────────
    print(f"Building GRPO prompt dataset ({grpo_cfg.n_grpo_prompts} prompts) …")
    grpo_ds = _make_grpo_dataset(tokenizer, grpo_cfg.n_grpo_prompts)
    print(f"GRPO dataset size: {len(grpo_ds)}")

    # ── GRPOTrainer ────────────────────────────────────────────────────────────
    from trl import GRPOTrainer, GRPOConfig

    grpo_args = GRPOConfig(
        num_generations=grpo_cfg.num_generations,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
        per_device_train_batch_size=grpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_cfg.gradient_accumulation_steps,
        max_steps=grpo_cfg.max_steps,
        learning_rate=grpo_cfg.learning_rate,
        beta=grpo_cfg.beta,
        optim=grpo_cfg.optim,
        lr_scheduler_type=grpo_cfg.lr_scheduler_type,
        warmup_steps=grpo_cfg.warmup_steps,
        logging_steps=grpo_cfg.logging_steps,
        save_steps=grpo_cfg.save_steps,
        output_dir=grpo_cfg.output_dir,
        report_to=grpo_cfg.report_to,
        run_name=f"grpo_{run_ts}",
        seed=grpo_cfg.seed,
        use_vllm=False,  # vLLM not available on T4 free tier
        # Sequence length caps for GRPO
        max_prompt_length=512,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_args,
        train_dataset=grpo_ds,
        reward_funcs=[length_reward, advice_penalty_reward],
    )

    print("Starting GRPO training …")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU: {torch.cuda.get_device_properties(0).name}, {mem:.1f} GB total")

    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    Path(grpo_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(grpo_cfg.output_dir)
    tokenizer.save_pretrained(grpo_cfg.output_dir)
    print(f"GRPO adapter saved → {grpo_cfg.output_dir}")

    # ── Save merged model for deployment ──────────────────────────────────────
    _save_merged_model(model, tokenizer, grpo_cfg.final_merged_dir)

    wandb.finish()

    # ── Re-run evaluation after GRPO ──────────────────────────────────────────
    print("\nRe-running evaluation after GRPO …")
    del model
    torch.cuda.empty_cache()

    from src.evaluate import run as eval_run
    post_eval = eval_run({
        "adapter_path": grpo_cfg.output_dir,
        "run_name": f"eval_post_grpo_{run_ts}",
        "results_save_path": str(Path(grpo_cfg.output_dir) / "post_grpo_eval.json"),
    })

    results = {
        "final_adapter": grpo_cfg.output_dir,
        "final_merged_dir": grpo_cfg.final_merged_dir,
        "pre_grpo_pct_in_range": pre_grpo_pct_in_range,
        "post_grpo_pct_in_range": post_eval.get("pct_in_range", float("nan")),
        "post_grpo_advice_rate": post_eval.get("advice_rate", float("nan")),
        "post_grpo_grpo_still_needed": post_eval.get("grpo_needed", False),
    }

    print("\nGRPO training complete.")
    print(f"  pct_in_range:  {results['pre_grpo_pct_in_range']:.1%} → {results['post_grpo_pct_in_range']:.1%}")
    print(f"  advice_rate:   {results['post_grpo_advice_rate']:.1%}")
    return results


if __name__ == "__main__":
    results = run()
