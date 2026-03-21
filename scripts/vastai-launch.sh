#!/usr/bin/env bash
#
# Generic vast.ai instance launcher.
#
# Usage:
#   vastai-launch.sh [options]
#
# Options:
#   --query QUERY          Vast.ai search query (default: 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95')
#   --max-price PRICE      Max $/hr (default: 2.0)
#   --image IMAGE          Docker image (default: vastai/pytorch — the PyTorch template)
#   --disk DISK            Disk space in GB (default: 50)
#   --env KEY=VAL          Environment variable (repeatable)
#   --env-file FILE        Load KEY=VAL lines from file (skips blanks and #comments)
#   --onstart-cmd CMD      Command to run on instance start
#   --label LABEL          Instance label
#   --ssh                  Launch in SSH mode (default: jupyter)
#   --direct               Use direct (non-proxy) connections
#   --dry-run              Print the create command without executing
#
# Requires: vastai CLI configured with `vastai set api-key <KEY>`
#
# Examples:
#   # Launch with defaults (cheapest RTX 4090, Jupyter mode)
#   ./vastai-launch.sh --env WANDB_API_KEY=xxx
#
#   # Custom GPU, SSH mode, provisioning script
#   ./vastai-launch.sh \
#     --query 'gpu_name=A100_SXM4 num_gpus=1' \
#     --max-price 3.0 --disk 100 --ssh --direct \
#     --env PROVISIONING_SCRIPT=https://example.com/setup.sh \
#     --env WANDB_API_KEY=xxx

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
QUERY='gpu_name=RTX_4090 num_gpus=1 reliability>0.95'
MAX_PRICE=2.0
IMAGE="vastai/pytorch"
DISK=50
LABEL=""
ONSTART_CMD=""
MODE="--jupyter"
DIRECT=""
DRY_RUN=false
declare -a ENV_PAIRS=()

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --query)        QUERY="$2";        shift 2 ;;
        --max-price)    MAX_PRICE="$2";    shift 2 ;;
        --image)        IMAGE="$2";        shift 2 ;;
        --disk)         DISK="$2";         shift 2 ;;
        --label)        LABEL="$2";        shift 2 ;;
        --onstart-cmd)  ONSTART_CMD="$2";  shift 2 ;;
        --ssh)          MODE="--ssh";      shift ;;
        --direct)       DIRECT="--direct"; shift ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        --env)
            ENV_PAIRS+=("$2")
            shift 2 ;;
        --env-file)
            while IFS= read -r line; do
                [[ -z "$line" || "$line" == \#* ]] && continue
                ENV_PAIRS+=("$line")
            done < "$2"
            shift 2 ;;
        -h|--help)
            head -35 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Build env string ─────────────────────────────────────────────────────────
ENV_STR=""
for pair in "${ENV_PAIRS[@]}"; do
    ENV_STR+=" -e $pair"
done

# ── Search for offers ────────────────────────────────────────────────────────
echo "Searching: $QUERY  (max \$$MAX_PRICE/hr) ..."
OFFERS=$(vastai search offers "$QUERY dph<=$MAX_PRICE rentable=true" -o 'dph' --raw 2>/dev/null) || {
    echo "Error: vastai search failed. Is the CLI configured?" >&2
    exit 1
}

OFFER_ID=$(echo "$OFFERS" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
if not offers:
    sys.exit(1)
print(offers[0]['id'])
" 2>/dev/null) || {
    echo "No offers found matching query. Try relaxing --query or --max-price." >&2
    exit 1
}

OFFER_PRICE=$(echo "$OFFERS" | python3 -c "
import json, sys
offers = json.load(sys.stdin)
o = offers[0]
print(f\"{o.get('gpu_name','?')} x{o.get('num_gpus',1)} — \${o.get('dph_total', o.get('dph', '?')):.3f}/hr\")
" 2>/dev/null)

echo "Best offer: #$OFFER_ID  ($OFFER_PRICE)"

# ── Build create command ─────────────────────────────────────────────────────
CMD=(vastai create instance "$OFFER_ID"
     --image "$IMAGE"
     --disk "$DISK"
     $MODE
)
[[ -n "$DIRECT" ]]      && CMD+=($DIRECT)
[[ -n "$ENV_STR" ]]      && CMD+=(--env "$ENV_STR")
[[ -n "$LABEL" ]]        && CMD+=(--label "$LABEL")
[[ -n "$ONSTART_CMD" ]]  && CMD+=(--onstart-cmd "$ONSTART_CMD")

if $DRY_RUN; then
    echo ""
    echo "[dry-run] Would execute:"
    echo "  ${CMD[*]}"
    exit 0
fi

# ── Create instance ──────────────────────────────────────────────────────────
echo "Creating instance..."
CREATE_OUTPUT=$("${CMD[@]}" 2>&1) || {
    echo "Error creating instance: $CREATE_OUTPUT" >&2
    exit 1
}

echo "$CREATE_OUTPUT"

INSTANCE_ID=$(echo "$CREATE_OUTPUT" | grep -oP 'new contract\s+\K\d+' || echo "$CREATE_OUTPUT" | grep -oP '\d+' | head -1)

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Instance created: #$INSTANCE_ID"
echo "  Monitor:  vastai show instance $INSTANCE_ID"
echo "  Logs:     vastai logs $INSTANCE_ID"
echo "  SSH:      vastai ssh-url $INSTANCE_ID"
echo "  Destroy:  vastai destroy instance $INSTANCE_ID"
echo "════════════════════════════════════════════════════════════"
