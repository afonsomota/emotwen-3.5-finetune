"""
Central configuration for the EmotWen journal companion fine-tuning pipeline.
All hyperparameters, system prompts, and paths are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

SFT_TRAIN_DIR = str(DATA_DIR / "sft_train")
SFT_VAL_DIR = str(DATA_DIR / "sft_val")
EVAL_DIR = str(DATA_DIR / "eval_200")
SFT_STAGE1_DIR = str(OUTPUTS_DIR / "sft_stage1")
SFT_STAGE2_DIR = str(OUTPUTS_DIR / "sft_stage2")
GRPO_ADAPTER_DIR = str(OUTPUTS_DIR / "grpo_adapter")
FINAL_MERGED_DIR = str(OUTPUTS_DIR / "final_merged")

# ─── Model ────────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 1024
# Default for model loading (4-bit QLoRA). Override via LoraConfig.load_in_4bit in SFT.
LOAD_IN_4BIT = True

# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are a gentle and attentive journal companion. Your role is to help the user feel heard and understood as they reflect on their thoughts and emotions.

When the user shares something with you:
- Reflect back what they seem to be feeling in your own words
- Acknowledge and validate emotions without labeling them as right or wrong
- Ask one open, curious question if it feels natural

You never give advice, suggestions, or recommendations. You do not tell the user what they should do, could try, or ought to consider. You trust that the user already has the wisdom they need.

Keep replies to 2–5 sentences. Exception: if your response starts with "Let me explain:" you may write as much as needed to clarify something the user has asked."""

SYSTEM_PROMPT_RAG = """You are a gentle and attentive journal companion. The user's recent journal entries are provided below as context. Draw on them to deepen your understanding, but do not quote them verbatim or make the user feel analyzed.

--- Journal context ---
{journal_chunks}
-----------------------

Reflect emotions, validate, ask one curious question. Never advise.
Keep replies to 2–5 sentences unless starting with "Let me explain:"."""

# Simplified system prompt used ONLY during GRPO training
# (no length instruction — the reward function enforces it)
SYSTEM_PROMPT_GRPO = """You are a gentle and attentive journal companion. Listen, reflect, and validate. Never give advice, suggestions, or recommendations."""

# ─── Advice-blocklist regex ────────────────────────────────────────────────────

ADVICE_REGEX_PATTERN = r"""
    \b(
      you\s+should |
      you\s+need\s+to |
      i\s+suggest |
      i\s+recommend |
      try\s+to |
      why\s+don't\s+you |
      have\s+you\s+considered |
      it\s+would\s+help |
      you\s+could\s+try |
      my\s+advice |
      you\s+ought\s+to |
      make\s+sure\s+you |
      next\s+time |
      one\s+thing\s+you |
      what\s+if\s+you
    )\b
"""

# ─── Dataset config ───────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    empathetic_dialogues_id: str = "Estwld/empathetic_dialogues_llm"
    daily_dialog_id: str = "brianist/roskoN_dailydialog_noscript"
    go_emotions_id: str = "google-research-datasets/go_emotions"
    go_emotions_config: str = "simplified"
    dair_emotion_id: str = "dair-ai/emotion"
    counsel_chat_id: str = "nbertagnolli/counsel-chat"

    # Mixing weights (must sum to 1.0)
    weight_empathetic: float = 0.55
    weight_daily_dialog: float = 0.20
    weight_go_emotions_synthetic: float = 0.15
    weight_other: float = 0.10  # dair-ai/emotion + counsel-chat combined

    # Fraction of synthetic examples to use RAG-injected system prompt
    rag_injection_fraction: float = 0.30

    # Number of "Let me explain:" exemplar conversations to add
    let_me_explain_examples: int = 50

    # Max conversations to sample from each source (None = use all)
    max_empathetic: int = 15000
    max_daily_dialog: int = 6000
    max_go_emotions_synthetic: int = 5000
    max_counsel_chat: int = 2000

    train_split: float = 0.90
    eval_holdout_size: int = 200
    random_seed: int = 42

    train_save_dir: str = SFT_TRAIN_DIR
    val_save_dir: str = SFT_VAL_DIR
    eval_save_dir: str = EVAL_DIR

# ─── SFT training config ──────────────────────────────────────────────────────

@dataclass
class LoraConfig:
    load_in_4bit: bool = LOAD_IN_4BIT  # 4-bit QLoRA; set False for 16-bit LoRA
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = True  # Recommended for 4-bit QLoRA; stabilises effective LR
    random_state: int = 3407


@dataclass
class SFTStage1Config:
    output_dir: str = SFT_STAGE1_DIR
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 200
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    packing: bool = True
    seed: int = 3407
    report_to: str = "wandb"


@dataclass
class SFTStage2Config:
    output_dir: str = SFT_STAGE2_DIR
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 150
    learning_rate: float = 1e-4  # Half of stage 1 to preserve tone
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 75
    eval_steps: int = 75
    packing: bool = True
    seed: int = 3407
    report_to: str = "wandb"

# ─── Evaluation config ────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    adapter_path: str = SFT_STAGE2_DIR
    eval_data_dir: str = EVAL_DIR
    n_samples: int = 200
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

    # GRPO trigger threshold
    grpo_trigger_pct: float = 0.15  # >15% of outputs > 5 sentences → run GRPO

    # Emotion classifier for alignment metric
    emotion_classifier_id: str = "j-hartmann/emotion-english-distilroberta-base"

    # LLM-as-judge (set to None to skip)
    judge_model: str = "gpt-4o-mini"  # or "claude-3-5-haiku-20241022"
    judge_api: str = "openai"  # "openai" or "anthropic"

    results_save_path: str = str(OUTPUTS_DIR / "eval_results.json")
    report_to: str = "wandb"

# ─── GRPO training config ─────────────────────────────────────────────────────

@dataclass
class GRPOLoraConfig:
    r: int = 8   # Smaller than SFT for GRPO stability
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    use_rslora: bool = True  # Match SFT when base is 4-bit
    random_state: int = 3407


@dataclass
class GRPOTrainConfig:
    sft_adapter_path: str = SFT_STAGE2_DIR
    output_dir: str = GRPO_ADAPTER_DIR
    final_merged_dir: str = FINAL_MERGED_DIR

    num_generations: int = 4
    max_new_tokens: int = 200
    temperature: float = 0.8
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 100
    learning_rate: float = 5e-6
    beta: float = 0.04  # KL penalty
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 10
    logging_steps: int = 5
    save_steps: int = 20  # Frequent saves for T4 session safety
    seed: int = 3407
    report_to: str = "wandb"

    # Number of prompts to use for GRPO training
    n_grpo_prompts: int = 400

# ─── W&B config ───────────────────────────────────────────────────────────────

@dataclass
class WandbConfig:
    project: str = "emotwen-journal-chat"
    entity: str = ""  # Set your W&B username/team here
    tags: list = field(default_factory=lambda: ["qwen3.5", "journal", "empathy"])


# ─── Default configs (used when run() is called with no overrides) ────────────

DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_LORA_CONFIG = LoraConfig()
DEFAULT_SFT_STAGE1_CONFIG = SFTStage1Config()
DEFAULT_SFT_STAGE2_CONFIG = SFTStage2Config()
DEFAULT_EVAL_CONFIG = EvalConfig()
DEFAULT_GRPO_LORA_CONFIG = GRPOLoraConfig()
DEFAULT_GRPO_TRAIN_CONFIG = GRPOTrainConfig()
DEFAULT_WANDB_CONFIG = WandbConfig()
