# EmotWen 3.5 Fine-tune — Blog Notes

Story of fine-tuning Qwen 3.5 (0.8B) into an empathetic journal companion.

---

## 1. The idea

Fine-tune a small LLM (Qwen 3.5 0.8B) to be a journal companion that reflects and validates emotions — never gives advice. The model should feel warm and present, like a good listener.

## 2. Initial pipeline (PR #1)

Built the full pipeline in one shot with Claude Code:
- **src/config.py** — single source of truth for all hyperparams, system prompts, dataset IDs
- **src/data_prep.py** — loads 5 datasets from HF Hub, filters advice, synthesizes journal pairs
- **src/train_sft.py** — two-stage SFT: Stage 1 (tone from conversational data), Stage 2 (domain from journal/synthetic data at lower LR)
- **src/evaluate.py** — sentence distribution, advice rate, emotion alignment, LLM-as-judge, perplexity
- **src/train_grpo.py** — conditional GRPO with length + advice penalty rewards
- **src/utils.py** — sentence counting, advice detection (regex blocklist), reward functions
- Colab notebooks wrapping each stage

Key design decisions:
- **Response-only loss** — mask user/system tokens during SFT
- **No-advice invariant** — `has_advice()` regex blocklist drops training examples that give advice
- **Length target** — 2-5 sentences per assistant turn, enforced by GRPO if >15% exceed 5 sentences
- **Two-stage SFT** — Stage 1 learns natural tone, Stage 2 specializes to journal domain at lower LR to preserve tone

## 3. The DailyDialog saga (PRs #2–#6)

Getting DailyDialog to load was surprisingly painful — went through 4 dataset variants:
1. `daily_dialog` (standard HF) — deprecated/broken
2. `ConvLab/dailydialog` — schema mismatch (`row["turns"][i]["utterance"]` vs `row["dialog"][i]`), plus `ontology.json` causing `DatasetGenerationCastError`
3. `roskoN/dailydialog` — utterances are dicts with `"text"` key
4. `brianist/roskoN_dailydialog_noscript` — utterances are plain strings, finally worked

Also fixed GoEmotions `AttributeError` — after converting to a list, `.features` attribute was gone. Had to extract `label_feature` in the caller.

## 4. Switching to Empathetic Dialogues fork (commit c99046b)

Changed the empathetic dialogues dataset ID to `brianist/empathetic_dialogues` — a cleaned fork.

## 5. QLoRA + Unsloth (commits c15d073, 38d895b)

Switched to 4-bit QLoRA and applied the Unsloth trainer for faster training. This was needed to fit on T4 GPUs in Colab.

## 6. NaN loss bug (PR #8)

Stage 2 was crashing with NaN loss. Root cause: when loading a saved LoRA adapter from Stage 1, `FastLanguageModel.from_pretrained` already returns a `PeftModel`. Calling `get_peft_model()` again raised `RuntimeError: You already added LoRA adapters!`. Fix: detect `adapter_config.json` and skip the redundant call.

## 7. LLM-as-judge fallback (PRs #9–#10)

Added a local Qwen 3.5 4B fallback for LLM-as-judge when no API keys are set — so evaluation works offline.

Hit a sneaky bug: Unsloth patches the tokenizer's `__call__` with signature `(self, images, text, videos, ...)`, so passing text as a positional arg mapped it to `images`, triggering the image processor. Fix: use `text=` keyword argument explicitly.

## 8. Local running + inference improvements (commits e9b35cc–ca36f08)

- Added local running script and requirements.txt
- Added tqdm to evaluation
- Fixed tokenizer calls (again)
- Fixed torchvision version mismatch on Vast.ai
- Merged all notebooks into one (`emotwen_pipeline.ipynb`)

## 9. Multi-turn repetition problem (commit aceea7c)

Observed two issues in practice:
- **Word-for-word repetition** after 3+ conversation turns
- **Out-of-context replies** (model ignoring what user just said)

Root causes identified:
- Context truncation at 1024 `max_seq_length`
- Attention leakage from packing
- Insufficient multi-turn training data
- No repetition penalty at inference

Planned 7 experiments: multi-turn eval harness, inference repetition penalty, context extension, multi-turn data augmentation, LoRA rank scaling, GRPO repetition reward, sliding-window attention.

## 10. Multi-turn eval + data augmentation (commit cae7127)

Implemented Experiments 0 and 3:
- **Multi-turn eval harness** — simulates 5-turn conversations, measures self-BLEU degradation, exact-substring repetition, contextual relevance (sentence-transformers). Logs per-turn W&B table + degradation curves.
- **Multi-turn training data** — 48 emotion-specific continuation templates (12 emotion categories, 2-5 variants each). `_extend_to_multi_turn()` converts 50% of synthetic single-turn examples to 2-turn conversations.
- New utility functions: `self_bleu()`, `pairwise_self_bleu()`, `longest_common_substring_tokens()`, `exact_repeat_check()`

## 11. Decoupling synthetic data generation (commit ba30e56)

Separated synthetic data generation into its own standalone step:
- `src/generate_multi_turn.py` — generates single + multi-turn synthetic conversations, pushes to HF Hub as `brianist/emotwen-3.5-synthetic`
- `src/data_prep.py` — now loads pre-generated synthetic from HF Hub by default, falls back to inline generation
- New `GenerateMultiTurnConfig` in config
- New notebook `nb/00_generate_multi_turn.ipynb`

Motivation: generation is CPU-only and slow, shouldn't re-run every time we iterate on training.

## 12. CLI pipeline runner (commits 38bc2eb, 47e31cf)

Added `main.py` as a CLI entry point for running stages:
- Individual stages: `python main.py sft`, `python main.py eval`, etc.
- `full_train` — data_prep → sft → eval → grpo (reuses existing synthetic data)
- `full_train_with_gen` — generate → data_prep → sft → eval → grpo

---

## 13. LLM-powered data generation strategies

Added two new generation strategies to `generate_multi_turn.py` to massively scale multi-turn training data:

**Self-chat via local Qwen 3.5 4B:**
- Model plays both user and assistant roles in 3-5 turn conversations
- Emotion-seeded: ~40 seed prompts across 9 emotion categories
- User role gets a separate system prompt instructing it to write as a journal author
- Conversations filtered for advice and validated for minimum turn count
- BF16 on RTX 4090 — no quantization needed for 4B model (~8GB VRAM)
- Target: 3000 conversations (configurable)

**Conversation augmentation (extending empathetic_dialogues):**
- Takes real empathetic_dialogues conversations and adds 1-2 extra user-assistant turns
- Supports local model (Qwen 3.5 4B) or API (OpenAI/Anthropic) backends
- Filters out advice in generated assistant turns
- Target: 2000 augmented conversations

Both are opt-in via `enable_self_chat=True` and `enable_augmentation=True` in config. Template-based generation (CPU) still runs by default.

Key design decisions:
- Self-chat and augmentation share the same model loading code (BF16 Unsloth → HF fallback)
- Both use the standard `has_advice()` filter on generated assistant turns
- Config keys are prefixed (`sc_`, `aug_`) to avoid collisions
- Model freed from GPU memory after each strategy finishes

**Batching for GPU throughput:**
Initial implementation was sequential (one conversation at a time) — completely underutilising the 4090. Switched to turn-synchronous batching: at each turn step, all active conversations are grouped into batches and generated together. Left-padding aligns generation start positions across variable-length prompts. Single-item fast path avoids padding overhead when batch shrinks to 1.

**Why BF16 over 4-bit for generation?**
For a 4B model on 24GB VRAM, 4-bit quantization would save memory we don't need (~3GB vs ~8GB). BF16 gives measurably better generation quality — quantization artifacts compound when the model is generating both sides of a conversation. The quality of synthetic training data directly impacts downstream fine-tune quality, so it's worth spending the extra VRAM.

## What's next

- Run the multi-turn repetition experiments on Vast.ai
- Evaluate whether GRPO repetition reward helps
- Try context extension beyond 1024 tokens
- Consider inference-time repetition penalty as a quick win
- Benchmark self-chat quality vs template quality
- Tune self-chat generation temperature for diversity vs coherence
