# EmotWen 3.5 Fine-tune — vast.ai Docker Template
#
# Build:  docker build -t emotwen:latest .
# Run:    docker run --gpus all -p 8888:8888 -e WANDB_API_KEY=... emotwen:latest
#
# Modes:
#   (default)         → JupyterLab on :8888
#   jupyter           → JupyterLab on :8888
#   bash              → interactive shell
#   all / sft / eval  → python main.py <stage> [overrides...]

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/opt/nltk_data

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        git curl wget ca-certificates && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ── uv (fast pip replacement) ────────────────────────────────────────────────
RUN pip install --no-cache-dir uv

# ── Layer: Core torch stack (heaviest layer, ~8GB, rarely changes) ───────────
RUN pip uninstall -y torchvision torchaudio 2>/dev/null || true && \
    uv pip install --system --no-cache \
        'torch==2.8.0' 'triton>=3.3.0' numpy pillow \
        bitsandbytes 'xformers==0.0.32.post2'

# ── Layer: Unsloth from git (must come after torch) ─────────────────────────
RUN uv pip install --system --no-cache \
        'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo' \
        'unsloth[base] @ git+https://github.com/unslothai/unsloth'

# ── Layer: Pin versions (--no-deps avoids overwriting torch) ─────────────────
RUN uv pip install --system --no-cache --upgrade --no-deps \
        tokenizers 'trl==0.22.2' unsloth unsloth_zoo && \
    uv pip install --system --no-cache 'transformers==5.2.0'

# ── Layer: Flash attention extensions (compile CUDA kernels) ─────────────────
RUN uv pip install --system --no-cache --no-build-isolation \
        flash-linear-attention 'causal_conv1d==1.6.0'

# ── Layer: Data, tracking, optional packages ─────────────────────────────────
RUN uv pip install --system --no-cache \
        wandb datasets nltk huggingface_hub tqdm \
        openai anthropic sentence-transformers

# ── Layer: NLTK data + JupyterLab ────────────────────────────────────────────
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data')" && \
    uv pip install --system --no-cache jupyterlab

# ── Working directory ────────────────────────────────────────────────────────
RUN mkdir -p /workspace
WORKDIR /workspace

# ── Entrypoint ───────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/entrypoint.sh"]
