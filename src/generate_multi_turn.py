"""
Standalone multi-turn synthetic conversation generator.

Generates multi-turn empathetic journal conversations from seed datasets
(go_emotions, dair-ai/emotion, counsel-chat) and publishes the result
as a versioned HuggingFace dataset.

Generation strategies:
1. **Template-based** — single-turn seeds extended via handwritten templates (CPU)
2. **Self-chat** — local Qwen 3.5 4B generates both user and assistant turns (GPU)
3. **Conversation augmentation** — extends real empathetic_dialogues with extra
   LLM-generated turns (GPU or API)

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
    SelfChatConfig,
    ConversationAugmentConfig,
    DEFAULT_GENERATE_MT_CONFIG,
    DEFAULT_SELF_CHAT_CONFIG,
    DEFAULT_CONVERSATION_AUGMENT_CONFIG,
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


# ─── Emotion seeds for self-chat ─────────────────────────────────────────────
# Short journal-like prompts per emotion to seed self-chat conversations.

_SELF_CHAT_SEEDS: dict[str, list[str]] = {
    "sadness": [
        "I've been feeling really down lately. Nothing seems to bring me joy anymore.",
        "I lost someone close to me recently and I'm struggling to process it.",
        "I feel like I'm drifting through life without any real connections.",
        "Today I cried for no reason. I just feel so overwhelmed by everything.",
        "I keep replaying a painful memory and I can't seem to let it go.",
    ],
    "anger": [
        "I'm so frustrated with how things went at work today.",
        "Someone I trusted completely betrayed me and I can't stop fuming about it.",
        "I feel like nobody listens to me no matter how clearly I try to communicate.",
        "I keep getting angry at small things and I don't understand why.",
        "I'm furious about an unfair situation and I feel powerless to change it.",
    ],
    "joy": [
        "Something wonderful happened today and I just need to share it.",
        "I accomplished something I've been working toward for months.",
        "I had the most beautiful moment with my family today.",
        "I woke up feeling genuinely grateful and happy for the first time in a while.",
        "A stranger did the kindest thing for me today.",
    ],
    "fear": [
        "I have a big decision to make and I'm terrified of choosing wrong.",
        "I keep having this nagging anxiety about the future.",
        "I'm starting something new and the uncertainty is paralyzing me.",
        "I'm afraid of losing what I have and it's keeping me up at night.",
        "There's a conversation I need to have and I'm dreading it.",
    ],
    "surprise": [
        "Something completely unexpected happened and I'm still processing it.",
        "I found out news today that changed everything I thought I knew.",
        "Someone did something I never would have predicted and I'm reeling.",
    ],
    "disgust": [
        "I witnessed something that made me deeply uncomfortable today.",
        "I'm disgusted with myself for how I handled a situation.",
        "Something happened that violated my values and I feel sick about it.",
    ],
    "neutral": [
        "I've been thinking a lot about where I am in life right now.",
        "Today was an ordinary day but something about it felt different.",
        "I've been reflecting on my relationships and what they mean to me.",
        "I noticed something about myself today that I hadn't seen before.",
        "I'm not sure how I feel right now. I just wanted to write.",
    ],
    "gratitude": [
        "I want to write about how thankful I am for the people in my life.",
        "Something small happened today that reminded me how lucky I am.",
        "I've been taking so much for granted and I just realized it.",
    ],
    "disappointment": [
        "Things didn't turn out the way I hoped and I'm trying to accept it.",
        "I let myself get my hopes up and now I feel foolish.",
        "Someone I looked up to really let me down.",
    ],
}

# System prompt for the self-chat "user" role — instructs the model to
# write as a journal author continuing a conversation.
_SELF_CHAT_USER_PROMPT = """You are a person writing in their journal and talking to a journal companion. Continue the conversation naturally from your perspective as the journal writer.

Rules:
- Write 1-3 sentences as the journal writer
- Be emotionally authentic — share feelings, not just facts
- Reference what the companion just said when relevant
- Don't ask the companion for advice
- Stay in the emotional tone of the conversation"""


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


# ─── Self-chat generation ────────────────────────────────────────────────────

def _resolve_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }.get(dtype_str, torch.bfloat16)


def _load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool,
    dtype: str = "bfloat16",
):
    """Load a model + tokenizer for generation (Unsloth if available, else HF).

    For RTX 4090 (24GB), BF16 is optimal for a 4B model — no quantization
    needed, best generation quality. Use load_in_4bit=True for smaller GPUs.
    """
    torch_dtype = _resolve_dtype(dtype)
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=1024,
            load_in_4bit=load_in_4bit,
            dtype=torch_dtype,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype or _resolve_dtype("bfloat16"),
            device_map="auto",
        )
        model.eval()
        return model, tokenizer


def _generate_batch(
    model,
    tokenizer,
    batch_messages: list[list[dict]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate responses for a batch of message histories.

    Pads from the left so generation starts at the same position for all
    sequences in the batch. Returns one decoded string per input.
    """
    import torch

    if not batch_messages:
        return []

    # Single-item fast path (no padding overhead)
    if len(batch_messages) == 1:
        prompt = tokenizer.apply_chat_template(
            batch_messages[0], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return [tokenizer.decode(new_tokens, skip_special_tokens=True).strip()]

    # Batch path: left-pad, generate, decode per-sequence
    prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        for msgs in batch_messages
    ]

    # Left-pad for batched generation
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        text=prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=1024,
    ).to(model.device)

    prompt_lengths = [
        (row != tokenizer.pad_token_id).sum().item()
        for row in inputs["input_ids"]
    ]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokenizer.padding_side = orig_side

    results = []
    for idx, plen in enumerate(prompt_lengths):
        new_tokens = output_ids[idx][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)

    return results


def _generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate a single response given a message history."""
    return _generate_batch(
        model, tokenizer, [messages], max_new_tokens, temperature, top_p,
    )[0]


def generate_self_chat(
    cfg: SelfChatConfig,
    rag_pool: dict[str, list[str]] | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Generate multi-turn conversations via model self-chat.

    The model plays both roles:
    - **Assistant**: uses the EmotWen system prompt (reflect, validate, no advice)
    - **User**: uses a separate prompt that instructs it to write as a journal author

    Parameters
    ----------
    cfg : SelfChatConfig
    rag_pool : optional emotion→texts pool for RAG injection
    rng : random generator

    Returns
    -------
    list of conversation dicts with keys: messages, source, emotion_label, n_turns
    """
    if rng is None:
        rng = random.Random(cfg.random_seed)

    print(f"\n── Self-chat generation ──────────────────────────────────────────")
    print(f"  Model: {cfg.model_id}")
    print(f"  Target conversations: {cfg.n_conversations}")
    print(f"  Turn range: {cfg.min_turns}-{cfg.max_turns}")

    model, tokenizer = _load_model_and_tokenizer(
        cfg.model_id, cfg.load_in_4bit, cfg.dtype,
    )
    print("  Model loaded.")

    # Build seed list: (emotion_label, seed_text) pairs
    seed_list = []
    for emotion, seeds in _SELF_CHAT_SEEDS.items():
        for seed in seeds:
            seed_list.append((emotion, seed))

    # ── Initialise all conversations with seeds ──────────────────────────────
    # Each "active" entry tracks messages, target turns, emotion, and status.
    active = []
    for _ in range(cfg.n_conversations):
        emotion_label, seed_text = rng.choice(seed_list)
        n_turns = rng.randint(cfg.min_turns, cfg.max_turns)

        use_rag = rng.random() < cfg.rag_injection_fraction
        if use_rag and rag_pool and rag_pool.get(emotion_label):
            similar = rng.sample(
                rag_pool[emotion_label],
                min(2, len(rag_pool[emotion_label])),
            )
            chunks = "\n\n---\n\n".join(similar)
            sys_prompt = SYSTEM_PROMPT_RAG.format(journal_chunks=chunks)
        else:
            sys_prompt = SYSTEM_PROMPT_BASE

        active.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": seed_text},
            ],
            "emotion_label": emotion_label,
            "target_turns": n_turns,
            "actual_turns": 0,
            "failed": False,
        })

    batch_size = cfg.batch_size
    max_turn = cfg.max_turns

    # ── Turn-synchronous batched generation ──────────────────────────────────
    # At each turn step, batch all still-active conversations together.
    for turn_idx in range(max_turn):
        # Filter to conversations that still need this turn
        pending = [a for a in active if not a["failed"]
                   and a["actual_turns"] < a["target_turns"]]
        if not pending:
            break

        # --- Assistant turn (batched) ---
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            batch_msgs = [a["messages"] for a in batch]
            responses = _generate_batch(
                model, tokenizer, batch_msgs,
                cfg.max_new_tokens, cfg.temperature, cfg.top_p,
            )
            for a, text in zip(batch, responses):
                if not text or has_advice(text):
                    a["failed"] = True
                else:
                    a["messages"].append({"role": "assistant", "content": text})
                    a["actual_turns"] += 1

        # --- User follow-up turn (batched, for conversations that continue) ---
        need_user = [a for a in active if not a["failed"]
                     and a["actual_turns"] < a["target_turns"]]
        if not need_user:
            continue

        for batch_start in range(0, len(need_user), batch_size):
            batch = need_user[batch_start:batch_start + batch_size]
            # Swap system prompt to user-writer persona
            batch_msgs = [
                [{"role": "system", "content": _SELF_CHAT_USER_PROMPT}]
                + a["messages"][1:]  # skip original system prompt
                for a in batch
            ]
            responses = _generate_batch(
                model, tokenizer, batch_msgs,
                100, cfg.temperature, cfg.top_p,
            )
            for a, text in zip(batch, responses):
                if not text:
                    a["failed"] = True
                else:
                    a["messages"].append({"role": "user", "content": text})

        n_ok = sum(1 for a in active if not a["failed"])
        print(f"  Turn {turn_idx + 1}: {n_ok}/{len(active)} active "
              f"({len(active) - n_ok} failed)")

    # ── Collect results ──────────────────────────────────────────────────────
    conversations = []
    n_failed = 0
    for a in active:
        if not a["failed"] and a["actual_turns"] >= cfg.min_turns:
            msgs = a["messages"]
            if msgs[-1]["role"] != "assistant":
                msgs = msgs[:-1]
            conversations.append({
                "messages": msgs,
                "source": "self_chat",
                "emotion_label": a["emotion_label"],
                "n_turns": a["actual_turns"],
            })
        else:
            n_failed += 1

    print(f"  Self-chat complete: {len(conversations)} conversations "
          f"({n_failed} failed / filtered)")

    # Free GPU memory
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return conversations


# ─── Conversation augmentation ───────────────────────────────────────────────

def _generate_via_api(
    messages: list[dict],
    backend: str,
    api_model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate a response via OpenAI or Anthropic API."""
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=api_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    elif backend == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        # Extract system message
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_messages.append(m)
        response = client.messages.create(
            model=api_model,
            system=system_msg,
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content[0].text.strip()
    else:
        raise ValueError(f"Unknown API backend: {backend}")


def augment_conversations(
    cfg: ConversationAugmentConfig,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Augment existing empathetic_dialogues conversations with extra turns.

    Loads real conversations, then uses an LLM to generate 1-2 additional
    user-assistant turn pairs, extending them into longer multi-turn dialogues.

    Parameters
    ----------
    cfg : ConversationAugmentConfig
    rng : random generator

    Returns
    -------
    list of augmented conversation dicts
    """
    if rng is None:
        rng = random.Random(cfg.random_seed)

    print(f"\n── Conversation augmentation ─────────────────────────────────────")
    print(f"  Source: {cfg.source_dataset_id}")
    print(f"  Backend: {cfg.backend}")
    print(f"  Target conversations: {cfg.n_conversations}")
    print(f"  Extra turns: {cfg.min_extra_turns}-{cfg.max_extra_turns}")

    # Load source conversations
    ed_ds = load_dataset(cfg.source_dataset_id)
    source_convs = []
    for row in ed_ds["train"]:
        conv = row["conversations"]
        # Need at least 2 turns (1 user + 1 assistant) to augment
        if len(conv) >= 2 and conv[-1]["role"] == "assistant":
            # Check no advice in existing assistant turns
            if not any(has_advice(t["content"]) for t in conv if t["role"] == "assistant"):
                source_convs.append({
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT_BASE}] + conv,
                    "emotion": row.get("emotion", "neutral"),
                })

    rng.shuffle(source_convs)
    to_augment = source_convs[:cfg.n_conversations]
    print(f"  Eligible conversations: {len(source_convs)}, augmenting: {len(to_augment)}")

    # Load model if using local backend
    model, tokenizer = None, None
    if cfg.backend == "local":
        model, tokenizer = _load_model_and_tokenizer(
            cfg.local_model_id, cfg.load_in_4bit, cfg.dtype,
        )
        print("  Model loaded.")

    # Prepare active state for each conversation
    active = []
    for conv_data in to_augment:
        active.append({
            "messages": list(conv_data["messages"]),
            "emotion": conv_data["emotion"],
            "target_extra": rng.randint(cfg.min_extra_turns, cfg.max_extra_turns),
            "turns_added": 0,
            "failed": False,
        })

    batch_size = cfg.batch_size
    max_extra = cfg.max_extra_turns

    # ── Turn-synchronous batched augmentation ────────────────────────────────
    for extra_idx in range(max_extra):
        pending = [a for a in active if not a["failed"]
                   and a["turns_added"] < a["target_extra"]]
        if not pending:
            break

        # --- User follow-up (batched for local, sequential for API) ---
        if cfg.backend == "local":
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start:batch_start + batch_size]
                batch_msgs = [
                    [{"role": "system", "content": _SELF_CHAT_USER_PROMPT}]
                    + a["messages"][1:]
                    for a in batch
                ]
                responses = _generate_batch(
                    model, tokenizer, batch_msgs,
                    100, cfg.temperature, cfg.top_p,
                )
                for a, text in zip(batch, responses):
                    if not text:
                        a["failed"] = True
                    else:
                        a["messages"].append({"role": "user", "content": text})
        else:
            for a in pending:
                user_msgs = (
                    [{"role": "system", "content": _SELF_CHAT_USER_PROMPT}]
                    + a["messages"][1:]
                )
                text = _generate_via_api(
                    user_msgs, cfg.backend, cfg.api_model,
                    100, cfg.temperature,
                )
                if not text:
                    a["failed"] = True
                else:
                    a["messages"].append({"role": "user", "content": text})

        # --- Assistant response (batched for local, sequential for API) ---
        need_assistant = [a for a in pending if not a["failed"]]
        if cfg.backend == "local":
            for batch_start in range(0, len(need_assistant), batch_size):
                batch = need_assistant[batch_start:batch_start + batch_size]
                batch_msgs = [a["messages"] for a in batch]
                responses = _generate_batch(
                    model, tokenizer, batch_msgs,
                    cfg.max_new_tokens, cfg.temperature, cfg.top_p,
                )
                for a, text in zip(batch, responses):
                    if not text or has_advice(text):
                        a["messages"].pop()  # remove the user turn
                        a["failed"] = True
                    else:
                        a["messages"].append({"role": "assistant", "content": text})
                        a["turns_added"] += 1
        else:
            for a in need_assistant:
                text = _generate_via_api(
                    a["messages"], cfg.backend, cfg.api_model,
                    cfg.max_new_tokens, cfg.temperature,
                )
                if not text or has_advice(text):
                    a["messages"].pop()
                    a["failed"] = True
                else:
                    a["messages"].append({"role": "assistant", "content": text})
                    a["turns_added"] += 1

        n_ok = sum(1 for a in active if not a["failed"])
        print(f"  Extra turn {extra_idx + 1}: {n_ok}/{len(active)} active")

    # ── Collect results ──────────────────────────────────────────────────────
    augmented = []
    n_failed = 0
    for a in active:
        if a["turns_added"] > 0:
            n_assistant = sum(1 for m in a["messages"] if m["role"] == "assistant")
            augmented.append({
                "messages": a["messages"],
                "source": "augmented_empathetic_dialogues",
                "emotion_label": a["emotion"],
                "n_turns": n_assistant,
            })
        else:
            n_failed += 1

    print(f"  Augmentation complete: {len(augmented)} conversations "
          f"({n_failed} failed / filtered)")

    # Free GPU memory
    if model is not None:
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return augmented


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run(config_overrides: dict | None = None) -> dict:
    """
    Generate multi-turn synthetic conversations and publish to HF Hub.

    Steps
    -----
    1. Load seed datasets from HF Hub
    2. Generate single-turn synthetic conversations (same logic as data_prep.py)
    3. Extend a fraction to multi-turn via templates
    4. (Optional) Self-chat: generate thousands of 3-5 turn conversations via local model
    5. (Optional) Conversation augmentation: extend real empathetic_dialogues with extra turns
    6. Save locally + push to HF Hub

    Config keys
    -----------
    Pass ``enable_self_chat=True`` and/or ``enable_augmentation=True`` in
    config_overrides to activate the LLM-based generation strategies.
    Self-chat and augmentation config keys are prefixed with ``sc_`` and
    ``aug_`` respectively (e.g. ``sc_n_conversations=5000``).

    Returns
    -------
    dict with generation stats
    """
    cfg = DEFAULT_GENERATE_MT_CONFIG
    sc_cfg = SelfChatConfig()
    aug_cfg = ConversationAugmentConfig()

    # Feature flags (off by default — template generation is CPU-only)
    enable_self_chat = False
    enable_augmentation = False

    if config_overrides:
        for k, v in config_overrides.items():
            if k == "enable_self_chat":
                enable_self_chat = v
            elif k == "enable_augmentation":
                enable_augmentation = v
            elif k.startswith("sc_"):
                attr = k[3:]  # strip sc_ prefix
                if hasattr(sc_cfg, attr):
                    setattr(sc_cfg, attr, v)
            elif k.startswith("aug_"):
                attr = k[4:]  # strip aug_ prefix
                if hasattr(aug_cfg, attr):
                    setattr(aug_cfg, attr, v)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)

    rng = random.Random(cfg.random_seed)
    print(f"Multi-turn generation config: {asdict(cfg)}")
    if enable_self_chat:
        print(f"Self-chat config: {asdict(sc_cfg)}")
    if enable_augmentation:
        print(f"Augmentation config: {asdict(aug_cfg)}")
    print()

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

    # ── Extend to multi-turn via templates ────────────────────────────────────
    print("\nExtending to multi-turn via templates …")
    extendable = [c for c in single_turn if c["source"] in
                  {"go_emotions_synthetic", "dair_emotion_synthetic"}]
    rng.shuffle(extendable)
    n_to_extend = int(len(extendable) * cfg.template_extension_fraction)

    template_multi_turn = []
    for conv in extendable[:n_to_extend]:
        extended = _extend_to_multi_turn(conv, conv["emotion_label"], rng)
        if extended is not None:
            extended["n_turns"] = 2
            template_multi_turn.append(extended)

    print(f"  Extended: {len(template_multi_turn)} / {n_to_extend} candidates")

    # ── Self-chat generation (GPU) ────────────────────────────────────────────
    self_chat_convs = []
    if enable_self_chat:
        self_chat_convs = generate_self_chat(sc_cfg, rag_pool=rag_pool, rng=rng)

    # ── Conversation augmentation (GPU or API) ────────────────────────────────
    augmented_convs = []
    if enable_augmentation:
        augmented_convs = augment_conversations(aug_cfg, rng=rng)

    # ── Combine all ───────────────────────────────────────────────────────────
    all_convs = single_turn + template_multi_turn + self_chat_convs + augmented_convs
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
    strategy_parts = ["template"]
    if enable_self_chat:
        strategy_parts.append(f"self-chat({len(self_chat_convs)})")
    if enable_augmentation:
        strategy_parts.append(f"augmented({len(augmented_convs)})")
    strategy_str = " + ".join(strategy_parts)

    if cfg.push_to_hub:
        print(f"\nPushing to HF Hub → {cfg.hub_repo_id}")
        ds_dict.push_to_hub(
            cfg.hub_repo_id,
            commit_message=(
                f"Generated {len(all_convs)} conversations [{strategy_str}] — "
                f"{datetime.now():%Y-%m-%d %H:%M}"
            ),
            private=False,
        )
        print(f"  Done! https://huggingface.co/datasets/{cfg.hub_repo_id}")
    else:
        print("\nSkipping HF Hub push (push_to_hub=False)")

    n_multi = len(template_multi_turn) + len(self_chat_convs) + len(augmented_convs)
    stats = {
        "total": len(all_convs),
        "single_turn": len(single_turn),
        "multi_turn": n_multi,
        "template_multi_turn": len(template_multi_turn),
        "self_chat": len(self_chat_convs),
        "augmented": len(augmented_convs),
        "sources": sources,
        "turn_counts": turn_counts,
        "hub_repo_id": cfg.hub_repo_id if cfg.push_to_hub else None,
        "local_path": str(local_path),
    }
    return stats


if __name__ == "__main__":
    stats = run()
    print("\nGeneration complete:", stats)
