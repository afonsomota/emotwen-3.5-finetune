# EmotWen 3.5 Fine-tune — Pre-baked vast.ai Image (optional)
#
# Only use this if you want faster instance startup by pre-baking deps.
# For most users, the PROVISIONING_SCRIPT approach is simpler — see
# docker/provisioning.sh.
#
# Build & push:
#   docker build -t ghcr.io/<you>/emotwen:latest .
#   docker push ghcr.io/<you>/emotwen:latest
#
# Then use this image in your vast.ai template instead of the default
# PyTorch template. Base layers are cached on vast.ai hosts.

FROM vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310

# PyTorch (the PyTorch template has this, but base-image doesn't)
RUN . /venv/main/bin/activate && \
    uv pip install --no-cache \
        'torch==2.8.0' 'triton>=3.3.0' numpy pillow \
        bitsandbytes 'xformers==0.0.32.post2'

# Unsloth
RUN . /venv/main/bin/activate && \
    uv pip install --no-cache \
        'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo' \
        'unsloth[base] @ git+https://github.com/unslothai/unsloth'

# Pin TRL + transformers
RUN . /venv/main/bin/activate && \
    uv pip install --no-cache --upgrade --no-deps \
        tokenizers 'trl==0.22.2' unsloth unsloth_zoo && \
    uv pip install --no-cache 'transformers==5.2.0'

# Flash attention extensions (~10 min compile)
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
