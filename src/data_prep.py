"""
Dataset preparation pipeline for the EmotWen journal companion fine-tuning.

Loads and filters:
  - facebook/empathetic_dialogues  (tone)
  - daily_dialog                   (tone)
  - google-research-datasets/go_emotions  (synthetic journal pairs)
  - dair-ai/emotion                (synthetic journal pairs)
  - nbertagnolli/counsel-chat      (client-side prompts only)

Outputs HuggingFace Dataset objects saved to disk:
  data/sft_train/
  data/sft_val/
  data/eval_200/

Entry point: run(config_overrides: dict | None = None) -> dict
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb
from datasets import Dataset, concatenate_datasets, load_dataset

from src.config import (
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_RAG,
    DataConfig,
    WandbConfig,
    DEFAULT_DATA_CONFIG,
    DEFAULT_WANDB_CONFIG,
)
from src.utils import has_advice

# ─── Emotion reflection templates ─────────────────────────────────────────────
# Used to synthesize assistant turns for go_emotions/dair-ai examples.
# 3+ variants per category so training data has natural variation.

_REFLECTION_TEMPLATES: dict[str, list[str]] = {
    "sadness": [
        "There's real sadness in what you've shared, and it makes sense that you're feeling it. "
        "That kind of heaviness doesn't just lift on its own. "
        "What does the sadness feel like it's trying to tell you right now?",
        "It sounds like you're carrying a lot of grief. "
        "I hear how much this is weighing on you. "
        "What part of this feels hardest to sit with?",
        "That sadness sounds very present and real. "
        "I'm glad you wrote about it. "
        "Is there a particular moment where it feels strongest?",
    ],
    "anger": [
        "I can hear how much frustration and anger is there. "
        "That feeling is completely valid given what you've described. "
        "What does the anger feel like it needs from you right now?",
        "There's a lot of intensity in what you're expressing — real anger. "
        "It sounds like something important to you was crossed. "
        "What feels most raw about it?",
        "The anger you're describing makes a lot of sense. "
        "Something clearly mattered deeply here. "
        "Where in your body do you feel it sitting?",
    ],
    "joy": [
        "There's such warmth in what you've written — it sounds like genuine joy. "
        "I love that you captured this moment. "
        "What made it feel so meaningful?",
        "Reading this, I can feel the lightness you're describing. "
        "That kind of happiness is worth pausing on. "
        "What's it like to hold this feeling right now?",
        "That sounds like a really bright moment. "
        "Joy like that can be surprisingly easy to brush past. "
        "What do you want to remember most about it?",
    ],
    "fear": [
        "I can hear the anxiety and fear underneath what you've written. "
        "That sense of uncertainty sounds really unsettling. "
        "What feels most unknown or scary from where you're standing?",
        "There's a lot of worry in these words. "
        "Fear like this often points to something that matters a great deal. "
        "What feels most at stake for you right now?",
        "That kind of fear can feel very isolating. "
        "It sounds like you're holding a lot of uncertainty. "
        "What would it mean if things went differently than you fear?",
    ],
    "surprise": [
        "That sounds like it caught you completely off guard. "
        "Sometimes the unexpected shakes us in ways we're still processing. "
        "How are you sitting with it now?",
        "It sounds like this hit you in a way you didn't anticipate. "
        "Those moments can leave us needing time to absorb. "
        "What's the feeling that sits underneath the surprise?",
    ],
    "disgust": [
        "What you're describing sounds deeply uncomfortable — like something felt really wrong. "
        "That reaction makes sense. "
        "What is it about the situation that feels most difficult to accept?",
        "There's a strong sense of discomfort and rejection in what you've written. "
        "Those feelings are telling you something important. "
        "What's the thing you most wish had been different?",
    ],
    "neutral": [
        "Thank you for sharing this. "
        "It sounds like you're observing your inner world carefully. "
        "What feels most alive or present for you as you reflect on it?",
        "There's something thoughtful in the way you've described this. "
        "I'm curious what it feels like to look back at it now. "
        "Is there anything about it that still has an emotional charge?",
        "I'm sitting with what you've shared. "
        "Sometimes the quieter entries hold the most. "
        "What drew you to write about this today?",
    ],
    # go_emotions fine-grained labels mapped to one of the above groups
    "admiration": [
        "There's such appreciation and warmth in what you've described. "
        "It sounds like this person or thing genuinely moved you. "
        "What is it about them that you most want to hold onto?",
    ],
    "amusement": [
        "I can feel the lightness and amusement in what you've written. "
        "Those moments of genuine laughter are worth noticing. "
        "What made it especially funny to you?",
    ],
    "caring": [
        "There's a lot of love and care in what you've expressed. "
        "It sounds like this relationship means a great deal to you. "
        "What does it feel like to care this much?",
    ],
    "confusion": [
        "It sounds like you're in the middle of something that doesn't quite make sense yet. "
        "That kind of confusion can be disorienting. "
        "What piece of it feels most unclear?",
    ],
    "curiosity": [
        "There's a real sense of wonder in what you've described. "
        "That curiosity sounds alive and energising. "
        "What draws you most toward this?",
    ],
    "disappointment": [
        "That sounds genuinely disappointing — like something you hoped for didn't land. "
        "Those feelings can be surprisingly heavy. "
        "What were you most hoping for?",
    ],
    "disapproval": [
        "I hear how strongly you feel that something wasn't right. "
        "That discomfort points to something you value. "
        "What felt most out of alignment with what matters to you?",
    ],
    "embarrassment": [
        "That sounds like it was a really uncomfortable moment. "
        "Embarrassment can linger in a way that feels bigger than the event. "
        "What's it like to look back at it now?",
    ],
    "excitement": [
        "There's so much energy and excitement in what you've written! "
        "That anticipation sounds genuinely alive. "
        "What feels most electric about it?",
    ],
    "gratitude": [
        "There's such warmth and appreciation in what you've described. "
        "Gratitude like this is worth sitting with. "
        "What made this feel especially meaningful to acknowledge?",
    ],
    "grief": [
        "That sounds like a deep and real grief. "
        "Losing something or someone that mattered is one of the hardest things to carry. "
        "What do you find yourself missing most?",
    ],
    "love": [
        "There's so much warmth and love in what you've shared. "
        "Those connections are really precious. "
        "What does loving this feel like from the inside?",
    ],
    "nervousness": [
        "I can hear the nervousness underneath your words. "
        "That kind of anticipatory anxiety is very real. "
        "What's the thing you're most uncertain about?",
    ],
    "optimism": [
        "There's a real sense of hope in what you've described. "
        "That forward-looking feeling sounds grounding. "
        "What's making you feel like things could go well?",
    ],
    "pride": [
        "It sounds like you're feeling genuinely proud, and that makes a lot of sense. "
        "Recognising your own effort matters. "
        "What does this mean to you personally?",
    ],
    "relief": [
        "That sounds like such a release — genuine relief after tension. "
        "Those moments of exhaling are important to notice. "
        "What are you most relieved about?",
    ],
    "remorse": [
        "There's a real sense of regret in what you've written. "
        "That kind of remorse shows how much you care about doing right. "
        "What's weighing on you most as you sit with it?",
    ],
}

# Fallback template for any unmapped label
_DEFAULT_TEMPLATE = _REFLECTION_TEMPLATES["neutral"]

# ─── "Let me explain:" exemplar conversations ────────────────────────────────

_LET_ME_EXPLAIN_EXAMPLES: list[dict] = [
    {
        "question": "Can you explain what journaling does for mental health?",
        "answer": "Let me explain: journaling creates a dedicated space for externalising your inner world — by writing thoughts and feelings down, you shift them from the busy, reactive part of the brain into something more reflective and observational. Research suggests this can reduce emotional intensity, improve clarity about what you're actually feeling, and over time help you notice patterns in your emotional life. It's less about finding answers and more about giving your experiences somewhere to land. Does that resonate with what you're curious about?",
    },
    {
        "question": "I don't really understand what 'sitting with an emotion' means. What does that actually look like?",
        "answer": "Let me explain: sitting with an emotion means allowing yourself to notice and stay present with a feeling without immediately trying to fix it, change it, or push it away. In practice, it might look like pausing and asking yourself: where do I feel this in my body? What's its texture or weight? You're not trying to analyse or solve it — you're more like a curious observer watching weather pass through. It can feel uncomfortable at first, especially if you're used to keeping busy or distracting yourself. Does that help clarify it?",
    },
    {
        "question": "What's the difference between validating an emotion and wallowing in it?",
        "answer": "Let me explain: validation means acknowledging that what you feel is real and makes sense given your experience — it's an act of self-recognition rather than self-indulgence. Wallowing typically implies getting stuck in an emotion without movement, often amplifying it through rumination. The key distinction is whether you're with the emotion or whether the emotion has taken over the whole narrative. Validation tends to create a little space between you and the feeling, which paradoxically makes it easier to move through. Does that distinction feel useful?",
    },
    {
        "question": "Why do you never give advice? What's the point if you're not helping me solve anything?",
        "answer": "Let me explain: the premise here is that you already hold more insight into your own life than any outside perspective can provide. Advice — even well-intentioned — often shortcuts the process of arriving at your own understanding, and it can subtly communicate that you need to be told what to do. What this space is designed for is helping you hear yourself more clearly. Often when people feel truly listened to and reflected back, something shifts on its own — not because they were given a solution, but because clarity emerged. That said, I hear the frustration in your question, and it's a fair one to raise.",
    },
    {
        "question": "What exactly is a journal companion? How is that different from a therapist?",
        "answer": "Let me explain: a journal companion is a reflective presence — something to write to or talk to as a way of processing your inner world, without the clinical or diagnostic frame that therapy involves. A therapist works within a professional relationship that includes assessment, treatment planning, and evidence-based interventions for mental health. This is much simpler: it's a space to be heard, to explore what you're feeling, and to reflect on your own experience without agenda. It's closer to journaling itself than to therapy — the 'companion' part just means you're not doing it completely alone.",
    },
]


# ─── Core conversion functions ────────────────────────────────────────────────

def _ed_split_to_messages(
    dataset,
    system_prompt: str,
) -> list[dict]:
    """Convert empathetic_dialogues split to list of messages dicts."""
    # Group by conv_id
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for row in dataset:
        groups[row["conv_id"]].append(row)

    conversations = []
    for conv_id, turns in groups.items():
        turns.sort(key=lambda x: x["utterance_idx"])
        messages = [{"role": "system", "content": system_prompt}]
        has_advice_turn = False
        for turn in turns:
            role = "user" if turn["speaker_idx"] == 0 else "assistant"
            text = turn["utterance"].strip()
            if role == "assistant" and has_advice(text):
                has_advice_turn = True
                break
            messages.append({"role": role, "content": text})

        if has_advice_turn:
            continue
        # Need at least system + 1 user + 1 assistant
        if len(messages) < 3:
            continue
        # Ensure the last message is from assistant
        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
        if len(messages) < 3:
            continue

        conversations.append({"messages": messages, "source": "empathetic_dialogues"})

    return conversations


def _daily_dialog_to_messages(
    dataset,
    system_prompt: str,
) -> list[dict]:
    """Convert daily_dialog examples to messages dicts."""
    conversations = []
    for row in dataset:
        dialog = row["dialog"]
        if len(dialog) < 2:
            continue

        messages = [{"role": "system", "content": system_prompt}]
        has_advice_turn = False
        for i, utt in enumerate(dialog):
            role = "user" if i % 2 == 0 else "assistant"
            text = utt.strip()
            if role == "assistant" and has_advice(text):
                has_advice_turn = True
                break
            messages.append({"role": role, "content": text})

        if has_advice_turn:
            continue
        if len(messages) < 3:
            continue
        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
        if len(messages) < 3:
            continue

        conversations.append({"messages": messages, "source": "daily_dialog"})

    return conversations


def _get_reflection(label_name: str, rng: random.Random) -> str:
    templates = _REFLECTION_TEMPLATES.get(label_name, _DEFAULT_TEMPLATE)
    return rng.choice(templates)


def _go_emotions_to_messages(
    dataset,
    system_prompt_base: str,
    system_prompt_rag: str,
    rag_fraction: float,
    rag_pool: dict[str, list[str]],
    rng: random.Random,
    source_tag: str = "go_emotions_synthetic",
) -> list[dict]:
    """Create synthetic single-turn journal conversations from go_emotions."""
    label_feature = dataset.features["labels"].feature
    conversations = []

    for row in dataset:
        text = row["text"].strip()
        label_ids = row["labels"]
        if not label_ids:
            label_name = "neutral"
        else:
            label_name = label_feature.int2str(label_ids[0])

        reflection = _get_reflection(label_name, rng)

        # Decide whether to inject RAG context
        use_rag = rng.random() < rag_fraction
        if use_rag and rag_pool.get(label_name):
            similar = rng.sample(rag_pool[label_name], min(2, len(rag_pool[label_name])))
            chunks = "\n\n---\n\n".join(similar)
            sys_prompt = system_prompt_rag.format(journal_chunks=chunks)
        else:
            sys_prompt = system_prompt_base

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"I wrote this in my journal today:\n\n\"{text}\""},
            {"role": "assistant", "content": reflection},
        ]
        conversations.append({"messages": messages, "source": source_tag})

    return conversations


def _dair_emotion_to_messages(
    dataset,
    system_prompt: str,
    rng: random.Random,
) -> list[dict]:
    """Create synthetic conversations from dair-ai/emotion dataset."""
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    conversations = []
    for row in dataset:
        text = row["text"].strip()
        label_name = label_map.get(row["label"], "neutral")
        reflection = _get_reflection(label_name, rng)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I wrote this in my journal:\n\n\"{text}\""},
            {"role": "assistant", "content": reflection},
        ]
        conversations.append({"messages": messages, "source": "dair_emotion_synthetic"})
    return conversations


def _counsel_chat_to_messages(
    dataset,
    system_prompt: str,
    rng: random.Random,
) -> list[dict]:
    """
    Use only the client/question side of counsel-chat.
    Discard therapist answers (contain advice). Generate a fresh reflection.
    """
    conversations = []
    for row in dataset:
        question = (row.get("questionText") or row.get("questionTitle") or "").strip()
        if not question:
            continue
        reflection = _get_reflection("neutral", rng)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": reflection},
        ]
        conversations.append({"messages": messages, "source": "counsel_chat_synthetic"})
    return conversations


def _let_me_explain_conversations(system_prompt: str) -> list[dict]:
    """Generate the fixed 'Let me explain:' exemplar conversations."""
    conversations = []
    for ex in _LET_ME_EXPLAIN_EXAMPLES:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
        conversations.append({"messages": messages, "source": "let_me_explain"})
    return conversations


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(config_overrides: dict | None = None) -> dict:
    """
    Full data preparation pipeline.

    Returns
    -------
    dict with keys:
      train_size, val_size, eval_size,
      advice_filtered_ed, advice_filtered_dd,
      sources_breakdown (dict),
      grpo_needed (always False at this stage — set by evaluate.py)
    """
    cfg: DataConfig = DEFAULT_DATA_CONFIG
    wb_cfg: WandbConfig = DEFAULT_WANDB_CONFIG

    # Apply overrides
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            elif hasattr(wb_cfg, k):
                setattr(wb_cfg, k, v)

    wandb.init(
        project=wb_cfg.project,
        entity=wb_cfg.entity or None,
        name=config_overrides.get("run_name", f"data_prep_{datetime.now():%Y%m%d_%H%M%S}") if config_overrides else f"data_prep_{datetime.now():%Y%m%d_%H%M%S}",
        job_type="data_prep",
        config=asdict(cfg),
        tags=wb_cfg.tags,
    )

    rng = random.Random(cfg.random_seed)
    Path(cfg.train_save_dir).parent.mkdir(parents=True, exist_ok=True)

    print("Loading datasets …")

    # ── empathetic_dialogues ──────────────────────────────────────────────────
    print(f"  {cfg.empathetic_dialogues_id}")
    ed_ds = load_dataset(cfg.empathetic_dialogues_id)
    ed_train = list(ed_ds["train"])
    rng.shuffle(ed_train)
    if cfg.max_empathetic:
        ed_train = ed_train[: cfg.max_empathetic]
    ed_convs_raw = _ed_split_to_messages(ed_train, SYSTEM_PROMPT_BASE)
    advice_filtered_ed = len(ed_train) // 10 - len(ed_convs_raw)  # rough proxy
    print(f"    empathetic_dialogues: {len(ed_convs_raw)} conversations kept")

    # ── daily_dialog ──────────────────────────────────────────────────────────
    print("  daily_dialog")
    dd_ds = load_dataset(cfg.daily_dialog_id)
    dd_train = list(dd_ds["train"])
    rng.shuffle(dd_train)
    if cfg.max_daily_dialog:
        dd_train = dd_train[: cfg.max_daily_dialog]
    dd_convs_raw = _daily_dialog_to_messages(dd_train, SYSTEM_PROMPT_BASE)
    advice_filtered_dd = len(dd_train) - len(dd_convs_raw)
    print(f"    daily_dialog: {len(dd_convs_raw)} conversations kept")

    # ── go_emotions (build RAG pool first, then synthetic pairs) ─────────────
    print("  go_emotions")
    ge_ds = load_dataset(cfg.go_emotions_id, cfg.go_emotions_config)
    ge_train = list(ge_ds["train"])
    label_feature = ge_ds["train"].features["labels"].feature

    # Build RAG pool: label_name → list of texts
    rag_pool: dict[str, list[str]] = {}
    for row in ge_train:
        for lid in row["labels"]:
            lname = label_feature.int2str(lid)
            rag_pool.setdefault(lname, []).append(row["text"])

    rng.shuffle(ge_train)
    if cfg.max_go_emotions_synthetic:
        ge_train = ge_train[: cfg.max_go_emotions_synthetic]
    ge_convs = _go_emotions_to_messages(
        ge_train, SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_RAG,
        cfg.rag_injection_fraction, rag_pool, rng,
    )
    print(f"    go_emotions synthetic: {len(ge_convs)} conversations")

    # ── dair-ai/emotion ───────────────────────────────────────────────────────
    print("  dair-ai/emotion")
    em_ds = load_dataset(cfg.dair_emotion_id)
    em_train = list(em_ds["train"])
    rng.shuffle(em_train)
    em_convs = _dair_emotion_to_messages(em_train[:2000], SYSTEM_PROMPT_BASE, rng)
    print(f"    dair emotion synthetic: {len(em_convs)} conversations")

    # ── counsel-chat ──────────────────────────────────────────────────────────
    print("  counsel-chat")
    cc_ds = load_dataset(cfg.counsel_chat_id)
    cc_train = list(cc_ds["train"])
    rng.shuffle(cc_train)
    if cfg.max_counsel_chat:
        cc_train = cc_train[: cfg.max_counsel_chat]
    cc_convs = _counsel_chat_to_messages(cc_train, SYSTEM_PROMPT_BASE, rng)
    print(f"    counsel-chat synthetic: {len(cc_convs)} conversations")

    # ── Let me explain exemplars ──────────────────────────────────────────────
    lme_convs = _let_me_explain_conversations(SYSTEM_PROMPT_BASE)
    print(f"    'Let me explain:' exemplars: {len(lme_convs)}")

    # ── Combine all ───────────────────────────────────────────────────────────
    all_convs = ed_convs_raw + dd_convs_raw + ge_convs + em_convs + cc_convs + lme_convs
    rng.shuffle(all_convs)

    sources = {}
    for c in all_convs:
        s = c.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print(f"\nTotal conversations: {len(all_convs)}")
    for s, n in sorted(sources.items()):
        print(f"  {s}: {n}")

    # ── Split: eval_200 first (held-out, never trained on) ────────────────────
    rng.shuffle(all_convs)
    eval_convs = all_convs[: cfg.eval_holdout_size]
    remaining = all_convs[cfg.eval_holdout_size :]

    # ── Train / val split ─────────────────────────────────────────────────────
    n_train = int(len(remaining) * cfg.train_split)
    train_convs = remaining[:n_train]
    val_convs = remaining[n_train:]

    print(f"\nSplit → train: {len(train_convs)}, val: {len(val_convs)}, eval: {len(eval_convs)}")

    # ── Save as HuggingFace Datasets ──────────────────────────────────────────
    def to_hf_dataset(convs: list[dict]) -> Dataset:
        return Dataset.from_list([{"messages": c["messages"], "source": c.get("source", "")} for c in convs])

    train_ds = to_hf_dataset(train_convs)
    val_ds = to_hf_dataset(val_convs)
    eval_ds = to_hf_dataset(eval_convs)

    Path(cfg.train_save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.val_save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.eval_save_dir).mkdir(parents=True, exist_ok=True)

    train_ds.save_to_disk(cfg.train_save_dir)
    val_ds.save_to_disk(cfg.val_save_dir)
    eval_ds.save_to_disk(cfg.eval_save_dir)

    print(f"\nSaved to disk:")
    print(f"  train → {cfg.train_save_dir}")
    print(f"  val   → {cfg.val_save_dir}")
    print(f"  eval  → {cfg.eval_save_dir}")

    # ── W&B logging ───────────────────────────────────────────────────────────
    stats = {
        "train_size": len(train_convs),
        "val_size": len(val_convs),
        "eval_size": len(eval_convs),
        "advice_filtered_empathetic_dialogues": advice_filtered_ed,
        "advice_filtered_daily_dialog": advice_filtered_dd,
        **{f"source_{k}": v for k, v in sources.items()},
    }
    wandb.log(stats)

    artifact = wandb.Artifact("journal_chat_dataset", type="dataset")
    artifact.add_dir(str(Path(cfg.train_save_dir).parent))
    wandb.log_artifact(artifact)

    wandb.finish()

    return {
        "train_size": len(train_convs),
        "val_size": len(val_convs),
        "eval_size": len(eval_convs),
        "advice_filtered_ed": advice_filtered_ed,
        "advice_filtered_dd": advice_filtered_dd,
        "sources_breakdown": sources,
    }


if __name__ == "__main__":
    results = run()
    print("\nData prep complete:", results)
