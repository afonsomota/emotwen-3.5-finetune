#!/bin/bash
set -e

REPO_URL="${REPO_URL:-https://github.com/afonsomota/emotwen-3.5-finetune.git}"
REPO_DIR="/workspace/emotwen-3.5-finetune"
REPO_BRANCH="${REPO_BRANCH:-main}"

# ── Clone or update the repo ─────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[entrypoint] Cloning $REPO_URL (branch: $REPO_BRANCH)..."
    git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
else
    echo "[entrypoint] Updating existing repo..."
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH" && \
    git -C "$REPO_DIR" reset --hard "origin/$REPO_BRANCH" || \
    echo "[entrypoint] Warning: git update failed, using existing checkout."
fi

cd "$REPO_DIR"

# ── HuggingFace login (if token provided) ────────────────────────────────────
if [ -n "$HF_TOKEN" ]; then
    echo "[entrypoint] Logging in to HuggingFace Hub..."
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
fi

# ── W&B status ───────────────────────────────────────────────────────────────
if [ -n "$WANDB_API_KEY" ]; then
    echo "[entrypoint] WANDB_API_KEY detected (will auto-login on import)."
fi

# ── Dispatch ─────────────────────────────────────────────────────────────────
start_jupyter() {
    echo "[entrypoint] Starting JupyterLab on port 8888..."
    exec jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --ServerApp.token="${JUPYTER_TOKEN:-}" \
        --notebook-dir="$REPO_DIR"
}

case "${1:-jupyter}" in
    jupyter|lab)
        start_jupyter
        ;;
    bash|shell)
        echo "[entrypoint] Starting interactive shell in $REPO_DIR"
        exec /bin/bash
        ;;
    data_prep|sft|eval|grpo|all)
        echo "[entrypoint] Running: python main.py $*"
        exec python main.py "$@"
        ;;
    *)
        echo "[entrypoint] Unknown command: $1"
        echo "Usage: <jupyter|bash|data_prep|sft|eval|grpo|all> [overrides...]"
        exit 1
        ;;
esac
