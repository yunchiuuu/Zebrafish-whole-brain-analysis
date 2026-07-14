#!/bin/bash
# submit_glm.sh
# =============
# Submit run_dprime for one config.
#
# Usage (from repo root):
#     bash chemogenetic/submit/submit_dprime.sh --config config_hcrt_trpv1_csn_120min

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required."
    echo "Usage: bash chemogenetic/submit/submit_dprime.sh --config config_hcrt_trpv1_csn_120min"
    exit 1
fi

PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'chemogenetic/config')
from ${CONFIG} import PYTHON_BIN
print(PYTHON_BIN)
")

LOG_DIR="logs/dprime"
mkdir -p "$LOG_DIR"

echo "Submitting dprime job for config: $CONFIG"

sbatch \
    --job-name="dprime_${CONFIG}" \
    --cpus-per-task=32 \
    --mem=256G \
    --output="${LOG_DIR}/${CONFIG}.log" \
    --wrap="$PYTHON chemogenetic/run/run_dprime.py --config $CONFIG"

echo "Submitted."
echo "Monitor: squeue -u \$USER"
echo "Log:     ${LOG_DIR}/${CONFIG}.log"
