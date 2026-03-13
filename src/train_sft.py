"""
Two-stage supervised fine-tuning for the EmotWen journal companion.

Stage 1 (tone):   empathetic_dialogues + daily_dialog filtered
Stage 2 (domain): synthetic journal conversations from go_emotions / dair-ai

Entry point: run(config_overrides: dict | None = None) -> dict
"""

from __future__ import annotations

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
    SFT_TRAIN_DIR,
    SFT_VAL_DIR,
    DEFAULT_LORA_CONFIG,
    DEFAULT_SFT_STAGE1_CONFIG,
    DEFAULT_SFT_STAGE2_CONFIG,
    DEFAULT_WANDB_CONFIG,
    LoraConfig,
    SFTStage1Config,
    SFTStage2Config,
    WandbConfig,
)


def _load_model_and_tokenizer(lora_cfg: LoraConfig, from_path: str | None = None):
    """Load Qwen3.5-0.8B with FastLanguageModel and apply LoRA."""
    from unsloth import FastLanguageModel

    model_name = from_path if from_path else MODEL_NAME

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=lora_cfg.load_in_4bit,
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
        use_rslora=lora_cfg.use_rslora,
    )

    return model, tokenizer


def _format_dataset(dataset, tokenizer):
    """
    Apply chat template to messages list → flat 'text' string.
    SFTTrainer reads the 'text' column.
    """
    def _fmt(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(_fmt, remove_columns=["messages", "source"])


def _build_trainer(model, tokenizer, train_ds, val_ds, sft_cfg, stage_name: str):
    """Build SFTTrainer with response-only loss masking."""
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    trainer_args = SFTConfig(
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        warmup_steps=sft_cfg.warmup_steps,
        max_steps=sft_cfg.max_steps,
        learning_rate=sft_cfg.learning_rate,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        optim=sft_cfg.optim,
        weight_decay=sft_cfg.weight_decay,
        logging_steps=sft_cfg.logging_steps,
        save_steps=sft_cfg.save_steps,
        eval_strategy="steps",
        eval_steps=sft_cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=sft_cfg.seed,
        output_dir=sft_cfg.output_dir,
        report_to=sft_cfg.report_to,
        run_name=stage_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=sft_cfg.packing,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=trainer_args,
    )

    # Mask loss on system/user tokens — train only on assistant turns
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    return trainer


def _print_gpu_stats(label: str):
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[{label}] GPU memory: {mem:.2f} GB / {total:.2f} GB ({100*mem/total:.1f}%)")


def _run_inference_demo(model, tokenizer, n: int = 3):
    """Generate a few sample responses to spot-check quality."""
    from unsloth import FastLanguageModel
    from src.config import SYSTEM_PROMPT_BASE

    FastLanguageModel.for_inference(model)

    demo_inputs = [
        "I've been feeling really disconnected from everything lately. Like I'm watching my life happen to someone else.",
        "I wrote this today: I don't know why I'm so angry. My coworker said something small and I couldn't stop thinking about it all day.",
        "Something good happened today and I feel almost guilty for feeling happy about it.",
    ]

    print("\n─── Inference demo ───────────────────────────────────────")
    for user_msg in demo_inputs[:n]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user", "content": user_msg},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # For Qwen VL tokenizers, the first positional argument can be treated as `images`.
        # Passing the chat template as a keyword `text=` ensures it is handled as pure text.
        inputs = tokenizer(
            text=input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                use_cache=True,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        print(f"\nUser: {user_msg}")
        print(f"Model: {response}")

    print("──────────────────────────────────────────────────────────\n")
    FastLanguageModel.for_training(model)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(config_overrides: dict | None = None) -> dict:
    """
    Two-stage SFT training.

    Parameters
    ----------
    config_overrides : dict, optional
        Keys may override any field of LoraConfig, SFTStage1Config, SFTStage2Config,
        or WandbConfig. Special key ``stage`` can be "1", "2", or "both" (default).

    Returns
    -------
    dict
        stage1_adapter, stage2_adapter, stage1_eval_loss, stage2_eval_loss
    """
    lora_cfg = DEFAULT_LORA_CONFIG
    s1_cfg = DEFAULT_SFT_STAGE1_CONFIG
    s2_cfg = DEFAULT_SFT_STAGE2_CONFIG
    wb_cfg = DEFAULT_WANDB_CONFIG

    stage = "both"
    if config_overrides:
        stage = config_overrides.pop("stage", "both")
        for k, v in config_overrides.items():
            for cfg in (lora_cfg, s1_cfg, s2_cfg, wb_cfg):
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                    break

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: dict = {}
    from unsloth import unsloth_train  # Fixed gradient accumulation (https://unsloth.ai/blog/gradient)

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("Loading training datasets from disk …")
    full_train_ds = load_from_disk(SFT_TRAIN_DIR)
    full_val_ds = load_from_disk(SFT_VAL_DIR)

    # Stage 1 uses tone data; Stage 2 uses journal/domain data
    tone_sources = {"empathetic_dialogues", "daily_dialog"}
    domain_sources = {
        "go_emotions_synthetic", "dair_emotion_synthetic",
        "counsel_chat_synthetic", "let_me_explain",
    }

    def filter_by_source(ds, sources):
        return ds.filter(lambda x: x["source"] in sources)

    # ── STAGE 1: Tone ─────────────────────────────────────────────────────────
    if stage in ("1", "both"):
        print("\n═══ Stage 1: Tone SFT ══════════════════════════════════════")

        wandb.init(
            project=wb_cfg.project,
            entity=wb_cfg.entity or None,
            name=config_overrides.get("run_name_s1", f"sft_stage1_{run_ts}") if config_overrides else f"sft_stage1_{run_ts}",
            job_type="sft_stage1",
            config={**asdict(lora_cfg), **asdict(s1_cfg)},
            tags=wb_cfg.tags,
        )

        model, tokenizer = _load_model_and_tokenizer(lora_cfg)

        s1_train = filter_by_source(full_train_ds, tone_sources)
        s1_val = filter_by_source(full_val_ds, tone_sources)
        print(f"Stage 1 train: {len(s1_train)}, val: {len(s1_val)}")

        s1_train_fmt = _format_dataset(s1_train, tokenizer)
        s1_val_fmt = _format_dataset(s1_val, tokenizer)

        trainer1 = _build_trainer(model, tokenizer, s1_train_fmt, s1_val_fmt, s1_cfg, f"sft_stage1_{run_ts}")
        _print_gpu_stats("before stage 1 training")
        unsloth_train(trainer1)
        _print_gpu_stats("after stage 1 training")

        s1_eval_loss = trainer1.state.best_metric or float("nan")
        print(f"Stage 1 best eval loss: {s1_eval_loss:.4f}")

        Path(s1_cfg.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(s1_cfg.output_dir)
        tokenizer.save_pretrained(s1_cfg.output_dir)
        print(f"Stage 1 adapter saved → {s1_cfg.output_dir}")

        _run_inference_demo(model, tokenizer)

        wandb.log({"stage1_eval_loss": s1_eval_loss})
        wandb.finish()

        results["stage1_adapter"] = s1_cfg.output_dir
        results["stage1_eval_loss"] = s1_eval_loss

        # Free memory before stage 2
        del model
        torch.cuda.empty_cache()

    # ── STAGE 2: Journal domain ───────────────────────────────────────────────
    if stage in ("2", "both"):
        print("\n═══ Stage 2: Journal Domain SFT ════════════════════════════")

        wandb.init(
            project=wb_cfg.project,
            entity=wb_cfg.entity or None,
            name=config_overrides.get("run_name_s2", f"sft_stage2_{run_ts}") if config_overrides else f"sft_stage2_{run_ts}",
            job_type="sft_stage2",
            config={**asdict(lora_cfg), **asdict(s2_cfg)},
            tags=wb_cfg.tags,
        )

        # Load from stage 1 adapter (or fresh base if stage 1 was skipped)
        s1_path = results.get("stage1_adapter", s1_cfg.output_dir)
        model, tokenizer = _load_model_and_tokenizer(lora_cfg, from_path=s1_path)

        s2_train = filter_by_source(full_train_ds, domain_sources)
        s2_val = filter_by_source(full_val_ds, domain_sources)
        # If val is too small, fall back to full val
        if len(s2_val) < 50:
            s2_val = full_val_ds
        print(f"Stage 2 train: {len(s2_train)}, val: {len(s2_val)}")

        s2_train_fmt = _format_dataset(s2_train, tokenizer)
        s2_val_fmt = _format_dataset(s2_val, tokenizer)

        trainer2 = _build_trainer(model, tokenizer, s2_train_fmt, s2_val_fmt, s2_cfg, f"sft_stage2_{run_ts}")
        _print_gpu_stats("before stage 2 training")
        unsloth_train(trainer2)
        _print_gpu_stats("after stage 2 training")

        s2_eval_loss = trainer2.state.best_metric or float("nan")
        print(f"Stage 2 best eval loss: {s2_eval_loss:.4f}")

        Path(s2_cfg.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(s2_cfg.output_dir)
        tokenizer.save_pretrained(s2_cfg.output_dir)
        print(f"Stage 2 adapter saved → {s2_cfg.output_dir}")

        _run_inference_demo(model, tokenizer)

        wandb.log({"stage2_eval_loss": s2_eval_loss})
        wandb.finish()

        results["stage2_adapter"] = s2_cfg.output_dir
        results["stage2_eval_loss"] = s2_eval_loss

    print("\nSFT training complete.")
    print(results)
    return results


if __name__ == "__main__":
    results = run()
