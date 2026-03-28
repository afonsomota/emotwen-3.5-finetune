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
#   --branch BRANCH        Git branch to clone (default: current branch)
#   --gpu QUERY            GPU search query (default: 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95')
#   --max-price PRICE      Max $/hr (default: 2.0)
#   --disk DISK            Disk space in GB (default: 50)
#   --cloud-sync CONN:PATH Cloud sync — connection_id:remote_path (e.g. 52:/emotwen/{run_id})
#                          Provider is auto-detected from your vast.ai connection.
#                          Syncs outputs/ and data/ before and after the job.
#                          Use {run_id} in the path to get a unique timestamp per run.
#   --env-file FILE        Extra env vars file (default: .env if it exists)
#   --insecure             Allow unverified GPU providers (removes verified=true filter)
#   --no-token-push        Don't push API keys to the instance (for interactive auth)
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
#   emotwen-launch.sh --stage full_train --cloud-sync 38826:/emotwen/{run_id} \
#     --overrides "max_empathetic=5000 stage1_max_steps=500"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENERIC_LAUNCHER="$SCRIPT_DIR/vastai-launch.sh"

# ── Defaults ─────────────────────────────────────────────────────────────────
STAGE="full_train"
OVERRIDES=""
INTERACTIVE=false
BRANCH="$(git -C "$(dirname "$0")" rev-parse --abbrev-ref HEAD)"
GPU_QUERY='gpu_name=RTX_4090 num_gpus=1 reliability>0.90 verified=true geolocation!=CN cpu_ram>=32'
MAX_PRICE="0.5"
DISK="100"
CLOUD_SYNC="38826:/emotwen/{run_id}"  # BlackblazeMain — connection_id:remote_path
ENV_FILE=""
DRY_RUN=false
INSECURE=false
NO_TOKEN_PUSH=false
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
        --dry-run)      DRY_RUN=true;       shift ;;
        --insecure)     INSECURE=true;      shift ;;
        --no-token-push) NO_TOKEN_PUSH=true; shift ;;
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

# ── Apply --insecure: allow unverified providers ─────────────────────────────
if $INSECURE; then
    GPU_QUERY="${GPU_QUERY// verified=true/}"
fi

# ── Warn if there are uncommitted changes ────────────────────────────────────
if ! git -C "$SCRIPT_DIR" diff --quiet HEAD 2>/dev/null; then
    echo "⚠️  WARNING: You have uncommitted changes. The remote instance will clone" >&2
    echo "   branch '$BRANCH' from GitHub, which may not include your local changes." >&2
fi

# ── Re-derive provisioning URL if branch changed ────────────────────────────
PROVISIONING_URL="https://raw.githubusercontent.com/afonsomota/emotwen-3.5-finetune/$BRANCH/docker/provisioning.sh"

# ── Build env var list ───────────────────────────────────────────────────────
declare -a LAUNCH_ARGS=()
LAUNCH_ARGS+=(--query "$GPU_QUERY")
LAUNCH_ARGS+=(--max-price "$MAX_PRICE")
LAUNCH_ARGS+=(--disk "$DISK")
LAUNCH_ARGS+=(--label "emotwen-$STAGE")

# Provisioning script — download and execute on instance start
LAUNCH_ARGS+=(--env "PROVISIONING_SCRIPT=$PROVISIONING_URL")
LAUNCH_ARGS+=(--env "REPO_BRANCH=$BRANCH")
LAUNCH_ARGS+=(--onstart-cmd "bash -c 'curl -fsSL \$PROVISIONING_SCRIPT | bash'")

# Headless config
if ! $INTERACTIVE; then
    LAUNCH_ARGS+=(--env "EMOTWEN_HEADLESS=true")
    LAUNCH_ARGS+=(--env "EMOTWEN_STAGE=$STAGE")
    [[ -n "$OVERRIDES" ]] && LAUNCH_ARGS+=(--env "EMOTWEN_OVERRIDES=$OVERRIDES")
fi

if $INTERACTIVE; then
    LAUNCH_ARGS+=(--jupyter --jupyter-lab --ssh --direct)
    LAUNCH_ARGS+=(--env "-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384")
    LAUNCH_ARGS+=(--env "OPEN_BUTTON_PORT=1111")
    LAUNCH_ARGS+=(--env "OPEN_BUTTON_TOKEN=1")
    LAUNCH_ARGS+=(--env "JUPYTER_DIR=/")
    LAUNCH_ARGS+=(--env "DATA_DIRECTORY=/workspace/")
    LAUNCH_ARGS+=(--env 'PORTAL_CONFIG=localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard')
fi

# Cloud sync (skip if --no-token-push — cloud sync requires API key)
if [[ -n "$CLOUD_SYNC" ]] && ! $NO_TOKEN_PUSH; then
    SYNC_CONNECTION="${CLOUD_SYNC%%:*}"
    SYNC_PATH="${CLOUD_SYNC#*:}"
    # Replace {run_id} placeholder with a unique timestamp-based ID
    RUN_ID="$(date +%Y%m%d-%H%M%S)"
    SYNC_PATH="${SYNC_PATH//\{run_id\}/$RUN_ID}"
    if [[ "$SYNC_CONNECTION" == "$CLOUD_SYNC" ]]; then
        echo "Error: --cloud-sync must be CONNECTION_ID:PATH (e.g. 52:/emotwen)" >&2
        exit 1
    fi
    LAUNCH_ARGS+=(--env "CLOUD_SYNC_CONNECTION=$SYNC_CONNECTION")
    LAUNCH_ARGS+=(--env "CLOUD_SYNC_PATH=$SYNC_PATH")

    # Cloud sync needs the full API key — fall back to vastai CLI cache
    if [[ -z "${VAST_API_KEY:-}" ]] && [[ -f "$HOME/.config/vastai/vast_api_key" ]]; then
        VAST_API_KEY="$(cat "$HOME/.config/vastai/vast_api_key")"
    fi
    if [[ -n "${VAST_API_KEY:-}" ]]; then
        LAUNCH_ARGS+=(--env "VAST_API_KEY=$VAST_API_KEY")
    else
        echo "Warning: VAST_API_KEY not set. Cloud sync may fail without the full API key." >&2
    fi
elif [[ -n "$CLOUD_SYNC" ]] && $NO_TOKEN_PUSH; then
    echo "Note: Cloud sync skipped (--no-token-push prevents sending API key to instance)." >&2
fi

# Pass through API keys (unless --no-token-push)
if ! $NO_TOKEN_PUSH; then
    [[ -n "${WANDB_API_KEY:-}" ]]     && LAUNCH_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
    # Fall back to the token cached by `huggingface-cli login`
    if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.cache/huggingface/token" ]]; then
        HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    fi
    [[ -n "${HF_TOKEN:-}" ]]          && LAUNCH_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")
    [[ -n "${OPENAI_API_KEY:-}" ]]    && LAUNCH_ARGS+=(--env "OPENAI_API_KEY=$OPENAI_API_KEY")
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] && LAUNCH_ARGS+=(--env "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY")
fi

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
[[ -n "${SYNC_CONNECTION:-}" ]] && echo "  Sync:   connection=$SYNC_CONNECTION path=$SYNC_PATH"
[[ -n "$OVERRIDES" ]]  && echo "  Config: $OVERRIDES"
echo ""

exec "$GENERIC_LAUNCHER" "${LAUNCH_ARGS[@]}"
