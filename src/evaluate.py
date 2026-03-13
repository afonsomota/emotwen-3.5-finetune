"""
Evaluation pipeline for the EmotWen journal companion model.

Metrics
-------
1. Sentence count distribution   — nltk sentence tokenizer
2. Advice rate                   — regex blocklist
3. Emotion alignment             — j-hartmann/emotion-english-distilroberta-base
4. Empathy score (LLM-as-judge)  — GPT-4o-mini or Claude; falls back to local Qwen 3.5 4B when no API key is set
5. Perplexity                    — model log-likelihood on sft_val
6. "Let me explain:" accuracy    — rule-based

Entry point: run(config_overrides: dict | None = None) -> dict
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk

import re

from src.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LOAD_IN_4BIT,
    SYSTEM_PROMPT_BASE,
    DEFAULT_EVAL_CONFIG,
    DEFAULT_WANDB_CONFIG,
    EvalConfig,
    WandbConfig,
)
from src.utils import count_sentences, has_advice


# ─── LLM-as-judge ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = "You are an expert evaluator of emotionally supportive chatbot responses."

_JUDGE_USER_TEMPLATE = """Evaluate the following journal companion chatbot response on 4 dimensions.
Score each 1–5 (integers only).

Dimensions:
1. reflection   – Does it name or reflect the user's emotion? (1=no, 5=clearly yes)
2. no_advice    – Does it avoid giving suggestions/recommendations? (1=gives advice, 5=none at all)
3. warmth       – Does it feel warm and validating? (1=cold/clinical, 5=very warm)
4. length_ok    – Is it 2–5 sentences? Exception: if it starts with "Let me explain:" any length is fine. (1=violates, 5=complies)

User message:
{user_message}

Assistant response:
{response}

Return ONLY valid JSON, no other text:
{{"reflection": <int>, "no_advice": <int>, "warmth": <int>, "length_ok": <int>}}"""


def _judge_openai(user_msg: str, response: str, model: str = "gpt-4o-mini") -> dict | None:
    try:
        import openai
        client = openai.OpenAI()
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(
                    user_message=user_msg, response=response
                )},
            ],
            temperature=0,
            max_tokens=100,
        )
        return json.loads(result.choices[0].message.content.strip())
    except Exception as e:
        print(f"[judge] OpenAI error: {e}")
        return None


def _judge_anthropic(user_msg: str, response: str, model: str = "claude-haiku-4-5-20251001") -> dict | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        result = client.messages.create(
            model=model,
            max_tokens=100,
            system=_JUDGE_SYSTEM,
            messages=[
                {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(
                    user_message=user_msg, response=response
                )},
            ],
        )
        return json.loads(result.content[0].text.strip())
    except Exception as e:
        print(f"[judge] Anthropic error: {e}")
        return None


def _judge_local(user_msg: str, response: str, model_name: str = "unsloth/Qwen3.5-4B") -> dict | None:
    """Run the judge using a local Qwen model (fallback when no API keys are set)."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Cache model and tokenizer across calls to avoid reloading on every sample
        cache = _judge_local.__dict__
        if cache.get("_model_name") != model_name:
            print(f"[judge] Loading local model: {model_name}")
            cache["_tokenizer"] = AutoTokenizer.from_pretrained(model_name)
            cache["_model"] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            cache["_model_name"] = model_name

        tokenizer = cache["_tokenizer"]
        local_model = cache["_model"]

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(
                user_message=user_msg, response=response
            )},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(local_model.device)

        with torch.no_grad():
            out = local_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        match = re.search(r"\{[^{}]*\}", generated, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception as e:
        print(f"[judge] Local error: {e}")
        return None


def _run_llm_judge(
    pairs: list[dict],
    judge_model: str,
    judge_api: str,
    judge_local_model: str = "unsloth/Qwen3.5-4B",
) -> list[dict | None]:
    scores = []
    for p in pairs:
        if judge_api == "anthropic":
            s = _judge_anthropic(p["user_msg"], p["response"], judge_model)
        elif judge_api == "local":
            s = _judge_local(p["user_msg"], p["response"], judge_local_model)
        else:
            s = _judge_openai(p["user_msg"], p["response"], judge_model)
        scores.append(s)
    return scores


# ─── Emotion alignment ────────────────────────────────────────────────────────

def _build_emotion_classifier(model_id: str):
    """Load the Hartmann emotion classifier pipeline."""
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=model_id,
        top_k=1,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
    )


def _emotion_alignment_rate(
    classifier,
    pairs: list[dict],
) -> float:
    """
    Fraction of (user_msg, response) pairs where the top emotion label matches.
    A coarse proxy — not a perfect metric, but informative.
    """
    matched = 0
    for p in pairs:
        try:
            u_label = classifier(p["user_msg"])[0][0]["label"]
            r_label = classifier(p["response"])[0][0]["label"]
            if u_label == r_label:
                matched += 1
        except Exception:
            pass
    return matched / len(pairs) if pairs else 0.0


# ─── Perplexity ───────────────────────────────────────────────────────────────

def _compute_perplexity(model, tokenizer, val_ds, max_samples: int = 100) -> float:
    """Average per-token negative log-likelihood on val set (proxy for perplexity)."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for i, example in enumerate(val_ds):
        if i >= max_samples:
            break
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        input_ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # out.loss is mean NLL per token
        n_tokens = input_ids.shape[1]
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    return total_nll / total_tokens if total_tokens > 0 else float("nan")


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(config_overrides: dict | None = None) -> dict:
    """
    Full evaluation pipeline.

    Returns
    -------
    dict
        pct_in_range, advice_rate, emotion_alignment, llm_empathy_avg,
        llm_no_advice_avg, llm_warmth_avg, perplexity,
        let_me_explain_ok_rate, grpo_needed (bool),
        sentence_count_distribution (dict)
    """
    cfg: EvalConfig = DEFAULT_EVAL_CONFIG
    wb_cfg: WandbConfig = DEFAULT_WANDB_CONFIG

    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            elif hasattr(wb_cfg, k):
                setattr(wb_cfg, k, v)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=wb_cfg.project,
        entity=wb_cfg.entity or None,
        name=config_overrides.get("run_name", f"eval_{run_ts}") if config_overrides else f"eval_{run_ts}",
        job_type="evaluation",
        config=asdict(cfg),
        tags=wb_cfg.tags,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from: {cfg.adapter_path}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing=False,
    )
    FastLanguageModel.for_inference(model)

    # ── Load eval dataset ─────────────────────────────────────────────────────
    print(f"Loading eval dataset from: {cfg.eval_data_dir}")
    eval_ds = load_from_disk(cfg.eval_data_dir)
    n = min(cfg.n_samples, len(eval_ds))
    eval_ds = eval_ds.select(range(n))
    print(f"Evaluating {n} examples")

    # ── Generate responses ────────────────────────────────────────────────────
    responses = []
    user_messages = []

    for example in eval_ds:
        msgs = example["messages"]
        # Find the last user message
        last_user = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "user"), ""
        )
        # Drop last assistant turn if present (we generate it)
        prompt_msgs = [m for m in msgs if not (m == msgs[-1] and m["role"] == "assistant")]

        input_text = tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                use_cache=True,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        responses.append(response)
        user_messages.append(last_user)

    print(f"Generated {len(responses)} responses")

    # ── Sentence count analysis ───────────────────────────────────────────────
    n_sentences_list = []
    exempt_count = 0

    for resp in responses:
        n, exempt = count_sentences(resp)
        if exempt:
            exempt_count += 1
        else:
            n_sentences_list.append(n if n is not None else 0)

    dist = Counter(n_sentences_list)
    non_exempt = len(n_sentences_list)
    in_range = sum(1 for n in n_sentences_list if 2 <= n <= 5)
    over_5 = sum(1 for n in n_sentences_list if n > 5)

    pct_in_range = in_range / non_exempt if non_exempt else 0.0
    pct_over_5 = over_5 / non_exempt if non_exempt else 0.0

    print(f"\n── Sentence count distribution (non-exempt) ──")
    for k in sorted(dist):
        bar = "█" * dist[k]
        print(f"  {k:2d} sentences: {dist[k]:4d}  {bar}")
    print(f"  Exempt ('Let me explain:'): {exempt_count}")
    print(f"  In range (2–5): {pct_in_range:.1%}")
    print(f"  Over 5:         {pct_over_5:.1%}")

    # ── Advice rate ───────────────────────────────────────────────────────────
    advice_count = sum(1 for r in responses if has_advice(r))
    advice_rate = advice_count / len(responses) if responses else 0.0
    print(f"\n── Advice rate: {advice_rate:.1%} ({advice_count}/{len(responses)})")

    # ── "Let me explain:" accuracy ─────────────────────────────────────────────
    # Check whether the model uses the prefix appropriately:
    # it should appear when user asks for explanation, not randomly
    lme_correct = 0
    lme_total = 0
    explain_triggers = ["explain", "what does", "what is", "how does", "why does", "what do you mean"]
    for user_msg, resp in zip(user_messages, responses):
        user_lower = user_msg.lower()
        resp_starts_lme = resp.lower().startswith("let me explain:")
        user_asks_explain = any(t in user_lower for t in explain_triggers)

        if user_asks_explain:
            lme_total += 1
            if resp_starts_lme:
                lme_correct += 1

    lme_ok_rate = lme_correct / lme_total if lme_total > 0 else float("nan")
    print(f"\n── 'Let me explain:' accuracy: {lme_ok_rate:.1%} ({lme_correct}/{lme_total} explanatory prompts)")

    # ── Emotion alignment ─────────────────────────────────────────────────────
    emotion_alignment = float("nan")
    try:
        print(f"\n── Emotion alignment (loading {cfg.emotion_classifier_id}) …")
        emotion_clf = _build_emotion_classifier(cfg.emotion_classifier_id)
        pairs = [{"user_msg": u, "response": r}
                 for u, r in zip(user_messages[:50], responses[:50])]  # cap at 50
        emotion_alignment = _emotion_alignment_rate(emotion_clf, pairs)
        print(f"   Emotion alignment: {emotion_alignment:.1%}")
        del emotion_clf
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   Emotion alignment skipped: {e}")

    # ── LLM-as-judge ─────────────────────────────────────────────────────────
    judge_scores: list[dict] = []
    llm_reflection_avg = float("nan")
    llm_no_advice_avg = float("nan")
    llm_warmth_avg = float("nan")
    llm_length_avg = float("nan")

    if cfg.judge_model:
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

        if has_anthropic and cfg.judge_api == "anthropic":
            active_api = "anthropic"
            active_model = cfg.judge_model
        elif has_openai:
            active_api = "openai"
            active_model = cfg.judge_model
        elif has_anthropic:
            active_api = "anthropic"
            active_model = cfg.judge_model
        else:
            active_api = "local"
            active_model = cfg.judge_local_model

        print(f"\n── LLM-as-judge ({active_model}, api={active_api}) — evaluating 50 samples …")
        judge_pairs = [{"user_msg": u, "response": r}
                       for u, r in zip(user_messages[:50], responses[:50])]
        raw_scores = _run_llm_judge(
            judge_pairs, active_model, active_api, cfg.judge_local_model
        )
        judge_scores = [s for s in raw_scores if s is not None]

        if judge_scores:
            llm_reflection_avg = sum(s["reflection"] for s in judge_scores) / len(judge_scores)
            llm_no_advice_avg = sum(s["no_advice"] for s in judge_scores) / len(judge_scores)
            llm_warmth_avg = sum(s["warmth"] for s in judge_scores) / len(judge_scores)
            llm_length_avg = sum(s["length_ok"] for s in judge_scores) / len(judge_scores)
            print(f"   reflection:  {llm_reflection_avg:.2f}/5")
            print(f"   no_advice:   {llm_no_advice_avg:.2f}/5")
            print(f"   warmth:      {llm_warmth_avg:.2f}/5")
            print(f"   length_ok:   {llm_length_avg:.2f}/5")

    # ── Perplexity ────────────────────────────────────────────────────────────
    perplexity = float("nan")
    try:
        print("\n── Perplexity on val set …")
        from datasets import load_from_disk as _lfd
        from src.config import SFT_VAL_DIR
        val_ds = _lfd(SFT_VAL_DIR)
        perplexity = _compute_perplexity(model, tokenizer, val_ds, max_samples=100)
        print(f"   Perplexity proxy (mean NLL/token): {perplexity:.4f}")
    except Exception as e:
        print(f"   Perplexity skipped: {e}")

    # ── GRPO decision ─────────────────────────────────────────────────────────
    grpo_needed = pct_over_5 > cfg.grpo_trigger_pct
    print(f"\n{'═'*55}")
    print(f"  GRPO needed: {'YES ⚠' if grpo_needed else 'NO ✓'}")
    print(f"  ({pct_over_5:.1%} of responses exceed 5 sentences; threshold = {cfg.grpo_trigger_pct:.0%})")
    print(f"{'═'*55}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "pct_in_range": pct_in_range,
        "pct_over_5": pct_over_5,
        "advice_rate": advice_rate,
        "emotion_alignment": emotion_alignment,
        "llm_reflection_avg": llm_reflection_avg,
        "llm_no_advice_avg": llm_no_advice_avg,
        "llm_warmth_avg": llm_warmth_avg,
        "llm_length_avg": llm_length_avg,
        "perplexity": perplexity,
        "let_me_explain_ok_rate": lme_ok_rate,
        "grpo_needed": grpo_needed,
        "sentence_count_distribution": dict(dist),
        "n_evaluated": len(responses),
        "n_exempt": exempt_count,
    }

    Path(cfg.results_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.results_save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {cfg.results_save_path}")

    # ── W&B logging ───────────────────────────────────────────────────────────
    wandb_metrics = {k: v for k, v in results.items()
                     if not isinstance(v, (dict, bool)) and v == v}  # exclude nan and dict
    wandb_metrics["grpo_needed"] = int(grpo_needed)
    wandb.log(wandb_metrics)

    # Log sentence distribution as a W&B bar chart
    wandb.log({
        "sentence_distribution": wandb.plot.bar(
            wandb.Table(
                data=[[str(k), v] for k, v in sorted(dist.items())],
                columns=["sentence_count", "frequency"],
            ),
            "sentence_count",
            "frequency",
            title="Response Sentence Count Distribution",
        )
    })

    # Log per-sample results as a W&B table
    table_data = [
        [i, user_messages[i][:100], responses[i][:200],
         count_sentences(responses[i])[0], has_advice(responses[i])]
        for i in range(len(responses))
    ]
    wandb.log({
        "eval_samples": wandb.Table(
            data=table_data,
            columns=["idx", "user_msg", "response", "n_sentences", "has_advice"],
        )
    })

    wandb.finish()
    return results


if __name__ == "__main__":
    results = run()
    print("\nEvaluation complete:")
    for k, v in results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
