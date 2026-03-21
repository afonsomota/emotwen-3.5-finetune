"""
Standalone multi-turn synthetic conversation generator.

Generates multi-turn empathetic journal conversations from seed datasets
(go_emotions, dair-ai/emotion, counsel-chat) and publishes the result
as a versioned HuggingFace dataset.

This runs *before* data_prep.py.  data_prep.py then loads the published
dataset and mixes it into the training set, so generation only needs to
happen once (or when you want to regenerate with different settings).

Entry point: run(config_overrides: dict | None = None) -> dict
"""

from __future__ import annotations

import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from src.config import (
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_RAG,
    GenerateMultiTurnConfig,
    DEFAULT_GENERATE_MT_CONFIG,
)
from src.data_prep import (
    _REFLECTION_TEMPLATES,
    _DEFAULT_TEMPLATE,
    _CONTINUATION_TEMPLATES,
    _CONTINUATION_LABEL_MAP,
    _get_reflection,
    _extend_to_multi_turn,
)
from src.utils import has_advice


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_rag_pool(ge_train: list[dict], label_feature) -> dict[str, list[str]]:
    """Build emotion_label → list of texts lookup for RAG injection."""
    pool: dict[str, list[str]] = {}
    for row in ge_train:
        for lid in row["labels"]:
            lname = label_feature.int2str(lid)
            pool.setdefault(lname, []).append(row["text"])
    return pool


def _make_single_turn_go_emotions(
    dataset: list[dict],
    label_feature,
    system_prompt_base: str,
    system_prompt_rag: str,
    rag_fraction: float,
    rag_pool: dict[str, list[str]],
    rng: random.Random,
) -> list[dict]:
    """Create single-turn synthetic journal conversations from go_emotions."""
    conversations = []
    for row in dataset:
        text = row["text"].strip()
        label_ids = row["labels"]
        label_name = label_feature.int2str(label_ids[0]) if label_ids else "neutral"
        reflection = _get_reflection(label_name, rng)

        use_rag = rng.random() < rag_fraction
        if use_rag and rag_pool.get(label_name):
            similar = rng.sample(rag_pool[label_name], min(2, len(rag_pool[label_name])))
            chunks = "\n\n---\n\n".join(similar)
            sys_prompt = system_prompt_rag.format(journal_chunks=chunks)
        else:
            sys_prompt = system_prompt_base

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f'I wrote this in my journal today:\n\n"{text}"'},
            {"role": "assistant", "content": reflection},
        ]
        conversations.append({
            "messages": messages,
            "source": "go_emotions_synthetic",
            "emotion_label": label_name,
            "n_turns": 1,
        })
    return conversations


def _make_single_turn_dair(
    dataset: list[dict],
    system_prompt: str,
    rng: random.Random,
) -> list[dict]:
    """Create single-turn synthetic conversations from dair-ai/emotion."""
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    conversations = []
    for row in dataset:
        text = row["text"].strip()
        label_name = label_map.get(row["label"], "neutral")
        reflection = _get_reflection(label_name, rng)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'I wrote this in my journal:\n\n"{text}"'},
            {"role": "assistant", "content": reflection},
        ]
        conversations.append({
            "messages": messages,
            "source": "dair_emotion_synthetic",
            "emotion_label": label_name,
            "n_turns": 1,
        })
    return conversations


def _make_single_turn_counsel(
    dataset: list[dict],
    system_prompt: str,
    rng: random.Random,
) -> list[dict]:
    """Create single-turn synthetic conversations from counsel-chat (client side only)."""
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
        conversations.append({
            "messages": messages,
            "source": "counsel_chat_synthetic",
            "emotion_label": "neutral",
            "n_turns": 1,
        })
    return conversations


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run(config_overrides: dict | None = None) -> dict:
    """
    Generate multi-turn synthetic conversations and publish to HF Hub.

    Steps
    -----
    1. Load seed datasets from HF Hub
    2. Generate single-turn synthetic conversations (same logic as data_prep.py)
    3. Extend a fraction to multi-turn via templates
    4. Save locally + push to HF Hub

    Returns
    -------
    dict with generation stats
    """
    cfg = DEFAULT_GENERATE_MT_CONFIG
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    rng = random.Random(cfg.random_seed)
    print(f"Multi-turn generation config: {asdict(cfg)}\n")

    # ── Load source datasets ──────────────────────────────────────────────────
    print("Loading seed datasets …")

    # go_emotions
    ge_ds = load_dataset(cfg.go_emotions_id, cfg.go_emotions_config)
    ge_train_full = list(ge_ds["train"])
    label_feature = ge_ds["train"].features["labels"].feature
    rag_pool = _build_rag_pool(ge_train_full, label_feature)

    rng.shuffle(ge_train_full)
    ge_train = ge_train_full[: cfg.max_go_emotions]
    print(f"  go_emotions: {len(ge_train)} seed examples")

    # dair-ai/emotion
    em_ds = load_dataset(cfg.dair_emotion_id)
    em_train = list(em_ds["train"])
    rng.shuffle(em_train)
    em_train = em_train[: cfg.max_dair_emotion]
    print(f"  dair emotion: {len(em_train)} seed examples")

    # counsel-chat
    cc_ds = load_dataset(cfg.counsel_chat_id)
    cc_train = list(cc_ds["train"])
    rng.shuffle(cc_train)
    cc_train = cc_train[: cfg.max_counsel_chat]
    print(f"  counsel-chat: {len(cc_train)} seed examples")

    # ── Generate single-turn conversations ────────────────────────────────────
    print("\nGenerating single-turn conversations …")
    ge_convs = _make_single_turn_go_emotions(
        ge_train, label_feature, SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_RAG,
        cfg.rag_injection_fraction, rag_pool, rng,
    )
    em_convs = _make_single_turn_dair(em_train, SYSTEM_PROMPT_BASE, rng)
    cc_convs = _make_single_turn_counsel(cc_train, SYSTEM_PROMPT_BASE, rng)

    single_turn = ge_convs + em_convs + cc_convs
    print(f"  Total single-turn: {len(single_turn)}")

    # ── Extend to multi-turn ──────────────────────────────────────────────────
    print("\nExtending to multi-turn via templates …")
    extendable = [c for c in single_turn if c["source"] in
                  {"go_emotions_synthetic", "dair_emotion_synthetic"}]
    rng.shuffle(extendable)
    n_to_extend = int(len(extendable) * cfg.template_extension_fraction)

    multi_turn = []
    for conv in extendable[:n_to_extend]:
        extended = _extend_to_multi_turn(conv, conv["emotion_label"], rng)
        if extended is not None:
            extended["n_turns"] = 2
            multi_turn.append(extended)

    print(f"  Extended: {len(multi_turn)} / {n_to_extend} candidates")

    # ── Combine all ───────────────────────────────────────────────────────────
    all_convs = single_turn + multi_turn
    rng.shuffle(all_convs)

    # Stats
    sources = {}
    turn_counts = {}
    for c in all_convs:
        s = c.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
        nt = c.get("n_turns", 1)
        turn_counts[nt] = turn_counts.get(nt, 0) + 1

    print(f"\nTotal conversations: {len(all_convs)}")
    print("  By source:")
    for s, n in sorted(sources.items()):
        print(f"    {s}: {n}")
    print("  By turn count:")
    for t, n in sorted(turn_counts.items()):
        print(f"    {t} turn(s): {n}")

    # ── Build HF Dataset ──────────────────────────────────────────────────────
    # Flatten: keep messages, source, emotion_label, n_turns
    records = []
    for c in all_convs:
        records.append({
            "messages": c["messages"],
            "source": c.get("source", ""),
            "emotion_label": c.get("emotion_label", ""),
            "n_turns": c.get("n_turns", 1),
        })

    ds = Dataset.from_list(records)
    ds_dict = DatasetDict({"train": ds})

    # ── Save locally ──────────────────────────────────────────────────────────
    local_path = Path(cfg.local_save_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(local_path))
    print(f"\nSaved locally → {local_path}")

    # ── Push to HF Hub ────────────────────────────────────────────────────────
    if cfg.push_to_hub:
        print(f"\nPushing to HF Hub → {cfg.hub_repo_id}")
        ds_dict.push_to_hub(
            cfg.hub_repo_id,
            commit_message=(
                f"Generated {len(all_convs)} synthetic conversations "
                f"({len(multi_turn)} multi-turn) — "
                f"{datetime.now():%Y-%m-%d %H:%M}"
            ),
            private=False,
        )
        print(f"  Done! https://huggingface.co/datasets/{cfg.hub_repo_id}")
    else:
        print("\nSkipping HF Hub push (push_to_hub=False)")

    stats = {
        "total": len(all_convs),
        "single_turn": len(single_turn),
        "multi_turn": len(multi_turn),
        "sources": sources,
        "turn_counts": turn_counts,
        "hub_repo_id": cfg.hub_repo_id if cfg.push_to_hub else None,
        "local_path": str(local_path),
    }
    return stats


if __name__ == "__main__":
    stats = run()
    print("\nGeneration complete:", stats)
