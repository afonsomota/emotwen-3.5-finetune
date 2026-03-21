#!/bin/bash
# EmotWen 3.5 Fine-tune — vast.ai Provisioning Script
#
# Works with the "PyTorch Development Environment" template on vast.ai.
# PyTorch, uv, conda, Jupyter, and the vast CLI are already installed.
#
# ── Setup ──────────────────────────────────────────────────────────────────
#
#   1. On vast.ai, select the "PyTorch" template → Edit → Save As your own
#   2. Add environment variables:
#        PROVISIONING_SCRIPT=https://raw.githubusercontent.com/afonsomota/emotwen-3.5-finetune/main/docker/provisioning.sh
#        WANDB_API_KEY=...
#        HF_TOKEN=...          (optional)
#        OPENAI_API_KEY=...    (optional, for LLM judge)
#        ANTHROPIC_API_KEY=... (optional, for LLM judge)
#
#   For headless "run + terminate" mode, also set:
#        EMOTWEN_HEADLESS=true
#        EMOTWEN_STAGE=full_train       (or: full_train_with_gen, generate, data_prep, sft, eval, grpo)
#        EMOTWEN_OVERRIDES="key=val"    (optional, space-separated)
#
#   3. Launch mode: Jupyter + SSH (recommended for interactive use)
#
# ── What this script does ──────────────────────────────────────────────────
#
#   - Installs Unsloth, TRL, and other deps not in the PyTorch template
#   - Clones the repo into /workspace/emotwen-3.5-finetune
#   - Optionally runs the full pipeline and self-destructs (headless mode)
#
# This runs once on first boot (vast.ai touches /.provisioning_complete).

set -eo pipefail

REPO_URL="${REPO_URL:-https://github.com/afonsomota/emotwen-3.5-finetune.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_DIR="/workspace/emotwen-3.5-finetune"

echo "════════════════════════════════════════════════════════════"
echo "  EmotWen 3.5 — Provisioning"
echo "════════════════════════════════════════════════════════════"

# ── Activate the PyTorch template venv ───────────────────────────────────────
# The PyTorch template provides /venv/main with torch pre-installed.
. /venv/main/bin/activate
echo "[provisioning] venv: $(which python) — torch $(python -c 'import torch; print(torch.__version__)')"

# ── Unsloth (must come after torch) ─────────────────────────────────────────
echo "[provisioning] Installing Unsloth..."
uv pip install \
    'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo' \
    'unsloth[base] @ git+https://github.com/unslothai/unsloth'

# ── Pin TRL + transformers versions ─────────────────────────────────────────
echo "[provisioning] Pinning TRL / transformers..."
uv pip install --upgrade --no-deps tokenizers 'trl==0.22.2' unsloth unsloth_zoo
uv pip install 'transformers==5.2.0'

# ── Flash attention extensions ──────────────────────────────────────────────
echo "[provisioning] Building flash-linear-attention + causal_conv1d (~10 min)..."
uv pip install --no-build-isolation flash-linear-attention 'causal_conv1d==1.6.0'

# ── Data, tracking, optional packages ───────────────────────────────────────
echo "[provisioning] Installing data/tracking packages..."
uv pip install \
    wandb datasets nltk huggingface_hub tqdm \
    openai anthropic sentence-transformers bitsandbytes

# ── NLTK data ───────────────────────────────────────────────────────────────
echo "[provisioning] Downloading NLTK punkt_tab..."
python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data')"
echo "NLTK_DATA=/opt/nltk_data" >> /etc/environment
export NLTK_DATA=/opt/nltk_data

# ── Clone the repo ──────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[provisioning] Cloning $REPO_URL (branch: $REPO_BRANCH)..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "[provisioning] Repo exists, pulling latest..."
    git -C "$REPO_DIR" pull origin "$REPO_BRANCH" || true
fi

# ── HuggingFace login ───────────────────────────────────────────────────────
if [ -n "$HF_TOKEN" ]; then
    echo "[provisioning] Logging in to HuggingFace Hub..."
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Provisioning complete!"
echo "  Repo:   $REPO_DIR"
echo "  Usage:  cd $REPO_DIR && python main.py <stage>"
echo "  Stages: generate | data_prep | sft | eval | grpo | full_train | full_train_with_gen"
echo "════════════════════════════════════════════════════════════"

# ── Headless mode: run pipeline + self-destruct ─────────────────────────────
if [ "${EMOTWEN_HEADLESS:-false}" = "true" ]; then
    STAGE="${EMOTWEN_STAGE:-full_train}"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Headless mode: running stage '$STAGE'"
    echo "════════════════════════════════════════════════════════════"

    cd "$REPO_DIR"
    # shellcheck disable=SC2086
    python main.py "$STAGE" ${EMOTWEN_OVERRIDES:-}

    echo ""
    echo "[headless] Pipeline complete. Destroying instance..."

    # Give W&B a moment to finish uploading
    sleep 10

    # Self-destruct — CONTAINER_ID and CONTAINER_API_KEY are set by vast.ai
    if [ -n "$CONTAINER_ID" ] && [ -n "$CONTAINER_API_KEY" ]; then
        vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
    else
        echo "[headless] WARNING: CONTAINER_ID or CONTAINER_API_KEY not set."
        echo "[headless] Cannot self-destruct. Stopping instead..."
        vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>/dev/null || true
    fi
fi
