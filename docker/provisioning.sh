#!/bin/bash
# EmotWen 3.5 Fine-tune — vast.ai Provisioning Script
#
# Use this with any vast.ai base image (vastai/base-image or vastai/pytorch).
# No custom Docker build needed — just set this as PROVISIONING_SCRIPT.
#
# Template setup:
#   1. Create a new vast.ai template
#   2. Image: vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310
#   3. Add environment variable:
#        PROVISIONING_SCRIPT=https://raw.githubusercontent.com/afonsomota/emotwen-3.5-finetune/main/docker/provisioning.sh
#   4. Add your keys as env vars:
#        WANDB_API_KEY=...
#        HF_TOKEN=...          (optional)
#        OPENAI_API_KEY=...    (optional, for LLM judge)
#        ANTHROPIC_API_KEY=... (optional, for LLM judge)
#   5. Launch mode: Jupyter + SSH (recommended)
#
# This script runs once on first instance start. It installs all Python
# dependencies, clones the repo, and downloads NLTK data.

set -eo pipefail

REPO_URL="${REPO_URL:-https://github.com/afonsomota/emotwen-3.5-finetune.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_DIR="/workspace/emotwen-3.5-finetune"

echo "════════════════════════════════════════════════════════════"
echo "  EmotWen 3.5 — Provisioning"
echo "════════════════════════════════════════════════════════════"

# ── Activate the base-image venv ─────────────────────────────────────────────
# vastai/base-image provides /venv/main; plain images may not have it.
if [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate
    echo "[provisioning] Using /venv/main"
else
    echo "[provisioning] No venv found, installing to system Python"
fi

# ── Ensure uv is available ───────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    pip install --quiet uv
fi

# ── Core torch stack ─────────────────────────────────────────────────────────
echo "[provisioning] Installing torch stack..."
pip uninstall -y torchvision torchaudio 2>/dev/null || true
uv pip install \
    'torch==2.8.0' 'triton>=3.3.0' numpy pillow \
    bitsandbytes 'xformers==0.0.32.post2'

# ── Unsloth ──────────────────────────────────────────────────────────────────
echo "[provisioning] Installing Unsloth..."
uv pip install \
    'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo' \
    'unsloth[base] @ git+https://github.com/unslothai/unsloth'

# ── Pin versions ─────────────────────────────────────────────────────────────
echo "[provisioning] Pinning TRL / transformers versions..."
uv pip install --upgrade --no-deps tokenizers 'trl==0.22.2' unsloth unsloth_zoo
uv pip install 'transformers==5.2.0'

# ── Flash attention extensions ───────────────────────────────────────────────
echo "[provisioning] Building flash-linear-attention + causal_conv1d (this takes ~10 min)..."
uv pip install --no-build-isolation flash-linear-attention 'causal_conv1d==1.6.0'

# ── Data, tracking, optional packages ────────────────────────────────────────
echo "[provisioning] Installing data/tracking packages..."
uv pip install \
    wandb datasets nltk huggingface_hub tqdm \
    openai anthropic sentence-transformers

# ── NLTK data ────────────────────────────────────────────────────────────────
echo "[provisioning] Downloading NLTK punkt_tab..."
python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data')"

# ── Persist NLTK_DATA for all sessions ───────────────────────────────────────
echo "NLTK_DATA=/opt/nltk_data" >> /etc/environment

# ── Clone the repo ───────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[provisioning] Cloning $REPO_URL (branch: $REPO_BRANCH)..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "[provisioning] Repo exists, pulling latest..."
    git -C "$REPO_DIR" pull origin "$REPO_BRANCH" || true
fi

# ── HuggingFace login ────────────────────────────────────────────────────────
if [ -n "$HF_TOKEN" ]; then
    echo "[provisioning] Logging in to HuggingFace Hub..."
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Provisioning complete!"
echo ""
echo "  Repo:     $REPO_DIR"
echo "  Usage:    cd $REPO_DIR && python main.py <stage>"
echo "  Stages:   data_prep | sft | eval | grpo | all"
echo "════════════════════════════════════════════════════════════"
