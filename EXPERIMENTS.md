# Repetition & Context-Drift Experiment Plan

## Problem Statement

1. **Multi-turn repetition** — After 3+ turns the model generates word-for-word repeats.
2. **Out-of-context replies** — Responses sometimes ignore what the user actually said.

---

## Root-Cause Analysis

| Symptom | Likely Causes |
|---------|---------------|
| Repetition after 3 turns | (a) 1024-token context fills up → earlier turns truncated → model "forgets" what it already said. (b) No repetition penalty at inference. (c) Eval is single-turn only so this was never caught. (d) Packing=True can leak attention across unrelated sequences. |
| Out-of-context replies | (a) Same truncation issue — context lost. (b) ~25% of training data is single-turn synthetic pairs, so the model never practices tracking prior turns. (c) 0.8B model has limited working memory; LoRA rank 8 may be too small to encode contextual tracking. |

---

## Experiments

Run these **in order**; each builds on the previous. Measure every change against the current baseline using the new multi-turn eval (Experiment 0).

### Experiment 0 — Multi-Turn Evaluation Harness (prerequisite)

**Goal:** Create a reliable way to detect both problems before and after fixes.

**Changes to `src/evaluate.py`:**
- Add `eval_multi_turn()`: feed 5-turn conversations from the eval set, generate a response at each turn, and measure:
  - **Self-BLEU** (4-gram) between the new response and all prior assistant responses. Flag if >0.6.
  - **Contextual-recall score**: embed (with the emotion classifier or a small sentence-transformer) the last user turn and the response; report cosine similarity. Flag if <0.3.
  - **Exact-substring repetition rate**: % of responses that contain a ≥10-token substring already present in a prior assistant turn.
- Log results to W&B as a new table `multi_turn_eval`.

**Why first:** Without this, every subsequent experiment is guesswork.

---

### Experiment 1 — Inference-Time Repetition Penalty

**Goal:** Quick win — suppress repetition without retraining.

| Parameter | Value to try |
|-----------|-------------|
| `repetition_penalty` | 1.1, 1.2, 1.3 |
| `no_repeat_ngram_size` | 3, 4 |
| `encoder_repetition_penalty` | 1.0 (off), 1.15 |

**Where:** Add these to `EvalConfig` and pass them to `model.generate()` in `evaluate.py` and any inference path.

**Measure:** Run Experiment 0 harness at each setting. Pick the combo that minimises self-BLEU without hurting judge scores.

**Expected outcome:** Reduces but does not eliminate the problem (treats the symptom, not the cause).

---

### Experiment 2 — Extend Context & Fix Packing

**Goal:** Give the model room to "see" prior turns and stop attention leaks.

| Change | Detail |
|--------|--------|
| `max_seq_length` | 1024 → **2048** (Qwen 2.5 0.5B natively supports 32 k; 2048 is safe on T4 with QLoRA) |
| `packing` | `True` → **False** for both SFT stages |
| Padding | Add `DataCollatorForSeq2Seq` with right-padding; mask pad tokens in loss |

**Why packing=False:** With packing, two unrelated conversations are concatenated into one 1024-token block. The model sees cross-conversation attention, which teaches it that "any prior text is valid context" — making it more likely to produce generic/repeated phrases. Disabling packing ensures each training example is one coherent conversation.

**Trade-off:** Training is slower (~1.5×). Compensate by increasing `max_steps` proportionally (Stage 1: 200→300, Stage 2: 150→225).

**Measure:** Experiment 0 harness + existing eval suite.

---

### Experiment 3 — Multi-Turn Training Data Augmentation

**Goal:** Teach the model to track context across turns.

**Changes to `src/data_prep.py`:**

1. **Synthetic multi-turn extension:**
   For each single-turn synthetic example (go_emotions, dair, counsel_chat), generate a 3-turn continuation:
   - Turn 1: existing user message → existing assistant response
   - Turn 2: template follow-up user message (e.g., "That reminds me of something else…", "Actually, I think it's more about…") drawn from a pool of ~30 hand-written follow-ups
   - Turn 3: new assistant response generated from reflection templates that **references Turn 1 content** (e.g., "Earlier you mentioned {emotion_word} — and now it sounds like…")

   This teaches the model that assistant turns should reference prior context.

2. **Conversation-history deduplication reward signal:**
   In `data_prep.py`, drop any training example where two assistant turns are >80% token-overlap (self-BLEU > 0.8). This removes examples that accidentally teach repetition.

**Volume target:** At least 30% of final training set should be ≥3 turns.

---

### Experiment 4 — LoRA Rank & Capacity

**Goal:** Give the adapter enough capacity to encode multi-turn tracking.

| Setting | Current | Try |
|---------|---------|-----|
| `lora_r` | 8 | **16**, 32 |
| `lora_alpha` | 16 | **32**, 64 (keep alpha = 2×r) |

**Why:** Rank-8 is 0.5% of parameters. For a 0.8B model that needs to learn contextual tracking (not just tone), rank 16-32 may be necessary.

**Constraint:** Must still fit on T4 16 GB. Rank 32 + 2048 context + QLoRA should be ~12 GB peak.

**Measure:** Experiment 0 harness. Compare self-BLEU and contextual-recall at rank 8 vs 16 vs 32.

---

### Experiment 5 — GRPO Repetition Reward

**Goal:** Add an explicit anti-repetition signal to RL.

**Add to `src/utils.py`:**

```python
def repetition_reward(completions, prompts=None, **kwargs) -> list[float]:
    """
    Penalise responses that repeat n-grams already present in the prompt
    (which includes prior assistant turns via chat template).

    Score:
      No 4-gram overlap with prior assistant turns  → +0.5
      1-3 repeated 4-grams                          → 0.0
      4+ repeated 4-grams                           → -1.0
    """
```

**Add to `GRPOTrainConfig.reward_functions`** as a third reward alongside length and advice penalty.

**Measure:** Post-GRPO Experiment 0 harness vs pre-GRPO.

---

### Experiment 6 — Attention-Sink / Sliding-Window at Inference

**Goal:** If repetition persists at long context, explore inference-time mitigations.

- Try **sliding-window attention** with window = 1536 tokens (keep recent turns sharp).
- Try **attention-sink** (keep first 4 + last N tokens) — cheap approximation that some small models respond well to.

This is a fallback if Experiments 2–5 don't fully resolve the issue.

---

## Experiment Priority & Dependencies

```
                    Exp 0 (eval harness)
                    ├── Exp 1 (inference penalty)      ← quick, no retrain
                    ├── Exp 2 (context + packing)      ← retrain needed
                    │   └── Exp 3 (multi-turn data)    ← retrain needed
                    │       └── Exp 5 (GRPO reward)    ← GRPO retrain
                    ├── Exp 4 (LoRA rank)              ← retrain needed
                    └── Exp 6 (sliding window)         ← inference only
```

**Recommended order:** 0 → 1 → 2 → 3 → 4 → 5 → 6

**Minimum viable fix (fastest):** Experiments 0 + 1 + 2 (eval harness + repetition penalty + longer context without packing). This can be done in one training run.

---

## Success Criteria

| Metric | Current (est.) | Target |
|--------|---------------|--------|
| Self-BLEU between consecutive assistant turns | >0.6 (high) | <0.3 |
| Exact-substring repetition rate (≥10 tokens) | ~30-40% at turn 4+ | <5% |
| Contextual-recall cosine similarity | unknown | >0.5 |
| Existing eval: sentence range compliance | passing | no regression |
| Existing eval: advice rate | passing | no regression |
| LLM judge mean scores | passing | no regression |

---

## Implementation Status

### Experiment 0 — IMPLEMENTED

**Files changed:**

| File | What was added |
|------|---------------|
| `src/config.py` | `MultiTurnEvalConfig` dataclass with thresholds and embedding model |
| `src/utils.py` | `self_bleu()`, `pairwise_self_bleu()`, `longest_common_substring_tokens()`, `exact_repeat_check()` |
| `src/evaluate.py` | `eval_multi_turn()` — full conversation driver with follow-up pool, W&B table + degradation curve logging |

**How to run:**

```python
from src.evaluate import run
# Multi-turn eval runs automatically as part of the standard eval pipeline.
# To configure:
results = run({
    "n_turns": 5,              # assistant turns per conversation
    "n_conversations": 50,     # conversations to simulate
    "self_bleu_threshold": 0.6,
    "lcs_token_threshold": 10,
    "relevance_threshold": 0.3,
})
```

**New W&B outputs:**
- `multi_turn_eval` table — per-turn rows with self-BLEU, LCS, relevance, etc.
- `mt_bleu_degradation` line chart — self-BLEU by turn position
- `mt_relevance_curve` line chart — contextual relevance by turn position
- Scalar metrics: `mt_repetition_rate`, `mt_exact_repeat_rate`, `mt_mean_relevance`, `mt_off_topic_rate`

**New dependency:** `pip install sentence-transformers` (for contextual relevance; ~80MB). Skipped gracefully if not installed.

### Experiment 3 (partial) — Multi-Turn Data Augmentation IMPLEMENTED

**Files changed:**

| File | What was added |
|------|---------------|
| `src/config.py` | `DataConfig.multi_turn_extension_fraction` (default: 0.50) |
| `src/data_prep.py` | `_CONTINUATION_TEMPLATES` (48 emotion-specific follow-up pairs), `_CONTINUATION_LABEL_MAP`, `_extend_to_multi_turn()` |
| `src/train_sft.py` | Added `*_multi_turn` source tags to Stage 2 domain filter |

**How it works:**
- 50% of go_emotions + dair_emotion synthetic examples are extended from 1 turn to 2 assistant turns
- Each continuation includes a user follow-up and an assistant reflection that references prior context
- Covers 12 emotion categories with 2-5 template variants each
- Extended examples get source tag `*_multi_turn` and are routed to SFT Stage 2

**To adjust fraction:**
```python
from src.data_prep import run
run({"multi_turn_extension_fraction": 0.70})  # extend 70% instead of 50%
```
