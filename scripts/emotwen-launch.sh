#!/usr/bin/env bash
#
# EmotWen — Launch a training run on vast.ai.
#
# Wraps vastai-launch.sh with EmotWen-specific defaults.
# Reads API keys from environment or an .env file.
#
# Usage:
#   emotwen-launch.sh [options]
#
# Options:
#   --stage STAGE          Pipeline stage (default: full_train)
#                          One of: generate, data_prep, sft, eval, grpo,
#                                  full_train, full_train_with_gen
#   --overrides ARGS       Config overrides, space-separated key=val
#   --interactive          Launch in Jupyter+SSH mode (no auto-run)
#   --branch BRANCH        Git branch to clone (default: main)
#   --gpu QUERY            GPU search query (default: 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95')
#   --max-price PRICE      Max $/hr (default: 2.0)
#   --disk DISK            Disk space in GB (default: 50)
#   --cloud-sync CONN:PATH Cloud sync — connection_id:remote_path (e.g. 52:/emotwen)
#                          Provider is auto-detected from your vast.ai connection.
#                          Syncs outputs/ and data/ before and after the job.
#   --env-file FILE        Extra env vars file (default: .env if it exists)
#   --dry-run              Show what would be launched without creating
#
# Environment variables (set these or put them in .env):
#   WANDB_API_KEY          Required — Weights & Biases tracking
#   HF_TOKEN               Optional — HuggingFace Hub (private datasets/push)
#   OPENAI_API_KEY         Optional — LLM judge (GPT-4o-mini)
#   ANTHROPIC_API_KEY      Optional — LLM judge (Claude)
#   VAST_API_KEY           Required for --cloud-sync (full vast.ai API key)
#
# Examples:
#   # Quick headless training run
#   emotwen-launch.sh --stage full_train
#
#   # Interactive session on A100
#   emotwen-launch.sh --interactive --gpu 'gpu_name=A100_SXM4 num_gpus=1' --max-price 3.0
#
#   # Headless with cloud sync and config overrides
#   emotwen-launch.sh --stage full_train --cloud-sync 52:/emotwen \
#     --overrides "max_empathetic=5000 stage1_max_steps=500"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENERIC_LAUNCHER="$SCRIPT_DIR/vastai-launch.sh"

# ── Defaults ─────────────────────────────────────────────────────────────────
STAGE="full_train"
OVERRIDES=""
INTERACTIVE=false
BRANCH="main"
GPU_QUERY='gpu_name=RTX_4090 num_gpus=1 reliability>0.95'
MAX_PRICE="2.0"
DISK="50"
CLOUD_SYNC=""      # connection_id:remote_path
ENV_FILE=""
DRY_RUN=false
PROVISIONING_URL="https://raw.githubusercontent.com/afonsomota/emotwen-3.5-finetune/$BRANCH/docker/provisioning.sh"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)        STAGE="$2";       shift 2 ;;
        --overrides)    OVERRIDES="$2";   shift 2 ;;
        --interactive)  INTERACTIVE=true; shift ;;
        --branch)       BRANCH="$2";      shift 2 ;;
        --gpu)          GPU_QUERY="$2";   shift 2 ;;
        --max-price)    MAX_PRICE="$2";   shift 2 ;;
        --disk)         DISK="$2";        shift 2 ;;
        --cloud-sync)   CLOUD_SYNC="$2";  shift 2 ;;
        --env-file)     ENV_FILE="$2";    shift 2 ;;
        --dry-run)      DRY_RUN=true;     shift ;;
        -h|--help)
            head -42 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate stage ───────────────────────────────────────────────────────────
VALID_STAGES="generate data_prep sft eval grpo full_train full_train_with_gen"
if ! echo "$VALID_STAGES" | grep -qw "$STAGE"; then
    echo "Invalid stage: $STAGE" >&2
    echo "Valid stages: $VALID_STAGES" >&2
    exit 1
fi

# ── Re-derive provisioning URL if branch changed ────────────────────────────
PROVISIONING_URL="https://raw.githubusercontent.com/afonsomota/emotwen-3.5-finetune/$BRANCH/docker/provisioning.sh"

# ── Build env var list ───────────────────────────────────────────────────────
declare -a LAUNCH_ARGS=()
LAUNCH_ARGS+=(--query "$GPU_QUERY")
LAUNCH_ARGS+=(--max-price "$MAX_PRICE")
LAUNCH_ARGS+=(--disk "$DISK")
LAUNCH_ARGS+=(--label "emotwen-$STAGE")

# Provisioning script
LAUNCH_ARGS+=(--env "PROVISIONING_SCRIPT=$PROVISIONING_URL")
LAUNCH_ARGS+=(--env "REPO_BRANCH=$BRANCH")

# Headless config
if ! $INTERACTIVE; then
    LAUNCH_ARGS+=(--env "EMOTWEN_HEADLESS=true")
    LAUNCH_ARGS+=(--env "EMOTWEN_STAGE=$STAGE")
    [[ -n "$OVERRIDES" ]] && LAUNCH_ARGS+=(--env "EMOTWEN_OVERRIDES=$OVERRIDES")
fi

if $INTERACTIVE; then
    LAUNCH_ARGS+=(--ssh --direct)
fi

# Cloud sync
if [[ -n "$CLOUD_SYNC" ]]; then
    SYNC_CONNECTION="${CLOUD_SYNC%%:*}"
    SYNC_PATH="${CLOUD_SYNC#*:}"
    if [[ "$SYNC_CONNECTION" == "$CLOUD_SYNC" ]]; then
        echo "Error: --cloud-sync must be CONNECTION_ID:PATH (e.g. 52:/emotwen)" >&2
        exit 1
    fi
    LAUNCH_ARGS+=(--env "CLOUD_SYNC_CONNECTION=$SYNC_CONNECTION")
    LAUNCH_ARGS+=(--env "CLOUD_SYNC_PATH=$SYNC_PATH")

    # Cloud sync needs the full API key
    if [[ -n "${VAST_API_KEY:-}" ]]; then
        LAUNCH_ARGS+=(--env "VAST_API_KEY=$VAST_API_KEY")
    else
        echo "Warning: VAST_API_KEY not set. Cloud sync may fail without the full API key." >&2
    fi
fi

# Pass through API keys from environment
[[ -n "${WANDB_API_KEY:-}" ]]     && LAUNCH_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
[[ -n "${HF_TOKEN:-}" ]]          && LAUNCH_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")
[[ -n "${OPENAI_API_KEY:-}" ]]    && LAUNCH_ARGS+=(--env "OPENAI_API_KEY=$OPENAI_API_KEY")
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && LAUNCH_ARGS+=(--env "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY")

# Extra env file
if [[ -n "$ENV_FILE" ]]; then
    LAUNCH_ARGS+=(--env-file "$ENV_FILE")
elif [[ -f "$SCRIPT_DIR/../.env" ]]; then
    LAUNCH_ARGS+=(--env-file "$SCRIPT_DIR/../.env")
fi

$DRY_RUN && LAUNCH_ARGS+=(--dry-run)

# ── Launch ───────────────────────────────────────────────────────────────────
echo "EmotWen — Launching $STAGE on vast.ai"
echo "  Branch: $BRANCH"
echo "  Mode:   $($INTERACTIVE && echo "interactive" || echo "headless")"
[[ -n "$CLOUD_SYNC" ]] && echo "  Sync:   connection=$SYNC_CONNECTION path=$SYNC_PATH"
[[ -n "$OVERRIDES" ]]  && echo "  Config: $OVERRIDES"
echo ""

exec "$GENERIC_LAUNCHER" "${LAUNCH_ARGS[@]}"
