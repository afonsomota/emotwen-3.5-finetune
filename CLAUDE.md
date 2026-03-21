# CLAUDE.md — EmotWen 3.5 Fine-tune

## Project Overview

Fine-tunes **Qwen 3.5 (0.8B)** into an empathetic journal companion chatbot (EmotWen). The model reflects and validates user emotions without giving advice. Training uses two-stage SFT followed by optional GRPO for length enforcement.

## Architecture

```
Seed Datasets (HuggingFace Hub)
    ↓ src/generate_multi_turn.py   (run once, or when regenerating)
HF Hub: brianist/emotwen-3.5-synthetic
    ↓ src/data_prep.py             (loads synthetic from Hub + real datasets)
data/sft_train, sft_val, eval_200
    ↓ src/train_sft.py  (Stage 1: tone, Stage 2: journal domain)
outputs/sft_stage2/
    ↓ src/evaluate.py   (includes multi-turn eval)
    → if >15% responses exceed 5 sentences → trigger GRPO
    ↓ src/train_grpo.py  (optional)
outputs/final_merged/  (16-bit merged, ready for deployment)
```

Each pipeline stage is runnable via the corresponding notebook in `nb/`.

## Running the Pipeline

All stages are Colab notebooks. Run them in order:

| Notebook | Stage | GPU needed |
|---|---|---|
| `nb/00_generate_multi_turn.ipynb` | Synthetic multi-turn generation → HF Hub | No (CPU) |
| `nb/01_data_prep.ipynb` | Data preparation (loads synthetic from Hub) | No (CPU) |
| `nb/02_sft_train.ipynb` | SFT Stage 1 + 2 | Yes (T4+) |
| `nb/03_eval.ipynb` | Evaluation + multi-turn eval + GRPO decision | Yes |
| `nb/04_grpo.ipynb` | GRPO (if triggered) | Yes |

> **Note:** Step 00 only needs to be re-run when you change templates, seed counts, or generation strategy. The published HF dataset is reused across data prep runs.

## Source Layout

```
src/
  config.py              # All hyperparams, system prompts, dataset IDs — edit here first
  generate_multi_turn.py # Standalone synthetic multi-turn generator → HF Hub
  data_prep.py           # Dataset loading, filtering, mixing (loads synthetic from Hub)
  train_sft.py           # Two-stage supervised fine-tuning
  train_grpo.py          # GRPO reinforcement learning
  evaluate.py            # Multi-metric evaluation + multi-turn eval + LLM judge
  utils.py               # Shared: sentence counter, advice detector, GRPO rewards, self-BLEU
```

## Configuration

Everything is in `src/config.py`. Key dataclasses:

- `GenerateMultiTurnConfig` — HF Hub repo, seed dataset sizes, extension fraction
- `DataConfig` — dataset IDs, max samples per source, RAG fraction, `synthetic_hub_id`
- `SFTStage1Config` / `SFTStage2Config` — LR, steps, batch size per stage
- `EvalConfig` — temperature, LLM judge model
- `MultiTurnEvalConfig` — turns, conversations, self-BLEU/relevance thresholds
- `GRPOTrainConfig` — KL penalty (beta), generation count, reward weights
- `WandbConfig` — project/run name, entity

Override at runtime by passing a dict to `run()`:

```python
from src.data_prep import run
run({"max_daily_dialog": 1000, "rag_injection_fraction": 0.2})
```

## Data Sources

| Dataset | HF ID | Role | Mix % |
|---|---|---|---|
| Empathetic Dialogues | `brianist/empathetic_dialogues` | Real empathetic conversations | ~55% |
| Daily Dialog | `brianist/roskoN_dailydialog_noscript` | Conversational tone | ~20% |
| GoEmotions | `google-research-datasets/go_emotions` (simplified) | Synthetic journal pairs | ~15% |
| DAIR Emotion | `dair-ai/emotion` | Synthetic journal pairs | ~5% |
| Counsel Chat | `nbertagnolli/counsel-chat` | Client prompts only | ~5% |

> **Note:** `roskoN_dailydialog_noscript` has `utterances` as a flat `list[str]` — not dicts with a `"text"` key.

## Key Invariants

- **No advice:** Training data is filtered with `has_advice()` (regex blocklist in `config.py`). Advice-giving examples are dropped.
- **Length target:** 2–5 sentences per assistant turn. GRPO reward penalises responses outside this range.
- **"Let me explain:" exemption:** When the user asks for an explanation, the model may prefix with `"Let me explain:"` and is exempt from the sentence-count constraint.
- **Response-only loss:** SFT trains only on assistant turns (user and system tokens are masked).
- **Two-stage SFT:** Stage 1 = tone (conversational data); Stage 2 = domain (journal/synthetic data at a lower LR to preserve tone).

## Outputs

| Path | Contents |
|---|---|
| `data/sft_train/` | HuggingFace Dataset, training split |
| `data/sft_val/` | HuggingFace Dataset, validation split |
| `data/eval_200/` | 200 held-out evaluation examples (never seen during training) |
| `outputs/sft_stage1/` | LoRA adapter after Stage 1 |
| `outputs/sft_stage2/` | LoRA adapter after Stage 2 (primary model) |
| `outputs/grpo_adapter/` | GRPO-refined adapter (if triggered) |
| `outputs/final_merged/` | 16-bit merged model for deployment |
| `outputs/eval_results.json` | Evaluation metrics |

## Experiment Tracking

All runs log to **Weights & Biases** (`WandbConfig` in `config.py`). Logged:
- Dataset composition stats (per-source counts, advice-filter drop counts)
- Training loss curves
- Eval metrics: sentence distribution, advice rate, emotion alignment, perplexity
- LLM-judge scores (reflection, no_advice, warmth, length_ok) — optional, requires OpenAI or Anthropic API key

## Common Tasks

**Re-run data prep with fewer samples (faster iteration):**
```python
run({"max_empathetic": 2000, "max_daily_dialog": 500, "max_go_emotions_synthetic": 500})
```

**Skip GRPO even if eval triggers it:**
```python
# In 04_grpo.ipynb, just don't run the cell
```

**Use Claude as LLM judge instead of GPT-4o-mini:**
```python
run({"judge_provider": "anthropic", "judge_model": "claude-haiku-4-5-20251001"})
```

**Check advice-filter regex:**
```python
from src.utils import has_advice
has_advice("You should try meditation.")  # True
has_advice("That sounds really hard.")    # False
```

## Blog Notes

Two files track the project story for future blog posts:

- **FINETUNE_BLOG_NOTES.md** — fine-tuning journey (data, training, bugs, experiments)
- **VASTAI_BLOG_NOTES.md** — Vast.ai infrastructure setup

When a user request results in a meaningful change (new feature, bug fix, architecture decision, or lesson learned), append a concise entry to the relevant blog notes file. Skip trivial renames or formatting-only changes.

## Dependencies

Install in Colab (done automatically by notebooks):
```bash
pip install datasets transformers nltk wandb trl peft unsloth openai anthropic
```

Required NLTK data (auto-downloaded):
```python
import nltk; nltk.download("punkt_tab")
```
