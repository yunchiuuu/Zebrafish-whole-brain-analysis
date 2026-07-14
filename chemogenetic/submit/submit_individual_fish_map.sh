#!/bin/bash
# submit_individual_fish_map.sh
# ==============================
# Submit individual fish brain map job.
#
# Usage (from repo root):
#     bash chemogenetic/submit/submit_individual_fish_map.sh --config config_hcrt_trpv1_pooled_csn_120min

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required."
    exit 1
fi

PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'chemogenetic/config')
from ${CONFIG} import PYTHON_BIN
print(PYTHON_BIN)
")

LOG_DIR="logs/individual_fish_map"
mkdir -p "$LOG_DIR"

echo "Submitting individual fish map for config: $CONFIG"

sbatch \
    --job-name="indiv_${CONFIG}" \
    --cpus-per-task=8 \
    --mem=64G \
    --output="${LOG_DIR}/${CONFIG}.log" \
    --wrap="$PYTHON chemogenetic/run/run_individual_fish_map.py --config $CONFIG"

echo "Submitted."
echo "Monitor: squeue -u \$USER"
echo "Log:     ${LOG_DIR}/${CONFIG}.log"
