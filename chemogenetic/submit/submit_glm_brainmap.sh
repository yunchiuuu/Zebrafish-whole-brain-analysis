#!/bin/bash
# submit_glm_brainmap.sh
# Usage (from repo root):
#     bash chemogenetic/submit/submit_glm_brainmap.sh --config config_hcrt_trpv1_csn_120min

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required."; exit 1
fi

PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'chemogenetic/config')
from ${CONFIG} import PYTHON_BIN
print(PYTHON_BIN)
")

LOG_DIR="logs/glm_brainmap"
mkdir -p "$LOG_DIR"

echo "Submitting GLM brain maps for config: $CONFIG"

sbatch \
    --job-name="glmbm_${CONFIG}" \
    --cpus-per-task=4 \
    --mem=64G \
    --output="${LOG_DIR}/${CONFIG}.log" \
    --wrap="$PYTHON chemogenetic/run/run_glm_brainmap.py --config $CONFIG"

echo "Submitted."
echo "Log: ${LOG_DIR}/${CONFIG}.log"
