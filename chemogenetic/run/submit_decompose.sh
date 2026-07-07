#!/bin/bash
# submit_decompose.sh
# ===================
# Submit decompose job for one config (one comparison group).
# Processes all fish in all_fish sequentially with skip logic —
# already-computed fish are skipped automatically.
#
# Usage (from repo root):
#     bash chemogenetic/submit_decompose.sh --config config_hcrt_trpv1_csn_120min

# ── Parse --config argument ───────────────────────────────────────────────────
CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required."
    echo "Usage: bash chemogenetic/submit_decompose.sh --config config_hcrt_trpv1_csn_120min"
    exit 1
fi

# ── Bootstrap PYTHON_BIN from registration config ─────────────────────────────
PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'chemogenetic/config')
from ${CONFIG} import PYTHON_BIN
print(PYTHON_BIN)
")

LOG_DIR="logs/decompose"
mkdir -p "$LOG_DIR"

echo "Submitting decompose job for config: $CONFIG"

sbatch \
    --job-name="decompose_${CONFIG}" \
    --cpus-per-task=32 \
    --mem=128G \
    --output="${LOG_DIR}/${CONFIG}.log" \
    --wrap="$PYTHON chemogenetic/run/run_decompose.py --config $CONFIG"

echo "Submitted."
echo "Monitor: squeue -u \$USER"
echo "Log:     ${LOG_DIR}/${CONFIG}.log"
