#!/bin/bash
# submit_medoids.sh
# =================
# Submit medoid computation + transform job for one config.
# Runs one fish only when TEST_MODE=True (default) for QC,
# or all fish when TEST_MODE=False.
#
# Usage (from repo root):
#     bash chemogenetic/submit/submit_medoids.sh --config config_hcrt_trpv1_csn_120min

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
    echo "Usage: bash chemogenetic/submit/submit_medoids.sh --config config_hcrt_trpv1_csn_120min"
    exit 1
fi

# ── Bootstrap PYTHON_BIN from config ─────────────────────────────────────────
PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'chemogenetic/config')
from ${CONFIG} import PYTHON_BIN
print(PYTHON_BIN)
")

LOG_DIR="logs/medoids"
mkdir -p "$LOG_DIR"

echo "Submitting medoids job for config: $CONFIG"

sbatch \
    --job-name="medoids_${CONFIG}" \
    --cpus-per-task=8 \
    --mem=64G \
    --output="${LOG_DIR}/${CONFIG}.log" \
    --wrap="$PYTHON chemogenetic/run/run_medoids.py --config $CONFIG"

echo "Submitted."
echo "Monitor: squeue -u \$USER"
echo "Log:     ${LOG_DIR}/${CONFIG}.log"
