# Vast.ai Setup for EmotWen — Blog Notes

Story of setting up a reproducible GPU training environment on Vast.ai.

---

## 1. Why Vast.ai

Colab T4s work for small experiments but are slow and unreliable for full training runs. Vast.ai gives access to better GPUs (A100, H100) at spot prices, with persistent storage and SSH access.

## 2. First attempt: custom Docker image (commit 8b8bc00)

Built a custom Dockerfile on `nvidia/cuda:12.8` that pre-baked all dependencies (torch 2.8.0, unsloth, flash-linear-attention, TRL, etc.). The idea was zero install time on boot.

Problems:
- Large image (~15 GB), slow to pull on Vast.ai hosts
- Vast.ai overwrites ENTRYPOINT in Jupyter/SSH mode, so our custom `entrypoint.sh` was ignored
- Had to maintain CUDA version parity with the host

## 3. Switching to vast.ai base image (commit 5d67437)

Rewrote the Docker setup to use `vastai/base-image` — layers are cached on Vast.ai hosts for fast startup. Split into three scripts:
- `Dockerfile` — derives FROM vastai/base-image, minimal additions
- `docker/provisioning.sh` — installs all deps on first boot (set as `PROVISIONING_SCRIPT` env var)
- `docker/onstart.sh` — lightweight on-start script for repo clone/update

## 4. Simplifying to PyTorch template (commit 8c2f945)

Realized Vast.ai's "PyTorch Development Environment" template already has torch, uv, conda, Jupyter, and the vast CLI installed. No need for a custom Docker image at all.

Final approach:
- Use the PyTorch template as-is
- `docker/provisioning.sh` does everything: installs Unsloth, TRL, flash-linear-attention, data packages, NLTK data, clones the repo
- Removed `onstart.sh` (redundant — provisioning handles it, vast.ai base persists env vars)
- Dockerfile kept as optional, only for pre-baked deps if you want faster cold starts

## 5. Headless mode

Added a "run and self-destruct" workflow for unattended training:
- Set `EMOTWEN_HEADLESS=true` + `EMOTWEN_STAGE=full_train` as env vars
- Provisioning script installs deps, runs the pipeline, then calls `vastai destroy instance` using built-in `CONTAINER_ID` + `CONTAINER_API_KEY` env vars
- No SSH needed — fire and forget

## 6. Setup instructions

1. On Vast.ai, select "PyTorch" template → Edit → Save As your own
2. Add environment variables:
   - `PROVISIONING_SCRIPT=https://raw.githubusercontent.com/afonsomota/emotwen-3.5-finetune/main/docker/provisioning.sh`
   - `WANDB_API_KEY=...`
   - `HF_TOKEN=...` (optional)
   - `OPENAI_API_KEY=...` (optional, for LLM judge)
   - `ANTHROPIC_API_KEY=...` (optional, for LLM judge)
3. Launch with Jupyter + SSH mode

For headless: also set `EMOTWEN_HEADLESS=true`, `EMOTWEN_STAGE=full_train`, optionally `EMOTWEN_OVERRIDES="key=val"`.

---

## What's next

<!-- Fill in as we go -->
