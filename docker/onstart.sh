#!/bin/bash
# EmotWen 3.5 Fine-tune — vast.ai On-Start Script
#
# Paste this into the "On-start Script" field of your vast.ai template.
# It runs every time the instance (re)starts — use it for lightweight setup
# that should happen on each boot (repo update, env vars, login).
#
# NOTE: Heavy installs belong in provisioning.sh (PROVISIONING_SCRIPT)
# or a derived Dockerfile — not here. This script should be fast.

set -eo pipefail

REPO_URL="${REPO_URL:-https://github.com/afonsomota/emotwen-3.5-finetune.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_DIR="/workspace/emotwen-3.5-finetune"

# ── Persist env vars across SSH sessions and reboots ─────────────────────────
env >> /etc/environment

# ── Activate venv if available ───────────────────────────────────────────────
if [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate
fi

# ── Clone or update repo ────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull origin "$REPO_BRANCH" || true
fi

# ── HuggingFace login ────────────────────────────────────────────────────────
if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true
fi

echo "[onstart] EmotWen ready at $REPO_DIR"
echo "[onstart] Run: cd $REPO_DIR && python main.py <stage>"
