# EmotWen 3.5 Fine-tune — Derived vast.ai Image (optional)
#
# Extends the vast.ai base image with all dependencies pre-baked.
# Because it shares base layers with vastai/base-image (already cached on
# vast.ai hosts), only the added dependency layers need downloading — giving
# much faster instance startup than installing at runtime.
#
# Build & push:
#   docker build -t ghcr.io/<you>/emotwen:latest .
#   docker push ghcr.io/<you>/emotwen:latest
#
# Then set this as the Image in your vast.ai template.
# If you don't want to build a custom image, use the PROVISIONING_SCRIPT
# approach instead — see docker/provisioning.sh.

FROM vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310

# ── Install into the base-image venv ─────────────────────────────────────────
# vastai/base-image provides /venv/main with Python 3.10 and uv pre-installed.

# Core torch stack (heaviest layer, ~8GB, rarely changes)
RUN . /venv/main/bin/activate && \
    pip uninstall -y torchvision torchaudio 2>/dev/null || true && \
    uv pip install --no-cache \
        'torch==2.8.0' 'triton>=3.3.0' numpy pillow \
        bitsandbytes 'xformers==0.0.32.post2'

# Unsloth from git (must come after torch)
RUN . /venv/main/bin/activate && \
    uv pip install --no-cache \
        'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo' \
        'unsloth[base] @ git+https://github.com/unslothai/unsloth'

# Pin versions (--no-deps avoids overwriting torch)
RUN . /venv/main/bin/activate && \
    uv pip install --no-cache --upgrade --no-deps \
        tokenizers 'trl==0.22.2' unsloth unsloth_zoo && \
    uv pip install --no-cache 'transformers==5.2.0'

# Flash attention extensions (compile CUDA kernels, ~10 min)
RUN . /venv/main/bin/activate && \
    uv pip install --no-cache --no-build-isolation \
        flash-linear-attention 'causal_conv1d==1.6.0'

# Data, tracking, optional packages
RUN . /venv/main/bin/activate && \
    uv pip install --no-cache \
        wandb datasets nltk huggingface_hub tqdm \
        openai anthropic sentence-transformers

# NLTK data
RUN . /venv/main/bin/activate && \
    python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data')"

ENV NLTK_DATA=/opt/nltk_data
