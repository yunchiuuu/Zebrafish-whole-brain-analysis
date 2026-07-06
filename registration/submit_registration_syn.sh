#!/bin/bash
# submit_registration_syn.sh
# ==========================
# Submit one sbatch job per fish for SyN registration to the shared mean brain.
# Fish list and Python path are read directly from config_registration.py.
# Run AFTER submit_mean_brain.sh has completed.
#
# Usage (from repo root):
#     bash registration/submit_registration_syn.sh

SCRIPT="registration/run_registration_syn.py"
CONFIG="registration/config/config_registration.py"
CPUS=16
MEM="64G"
LOG_DIR="logs/registration"

mkdir -p "$LOG_DIR"

# Bootstrap: use system python3 to read PYTHON_BIN from config
PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'registration/config')
from config_registration import PYTHON_BIN
print(PYTHON_BIN)
")

# Read all expt_IDs from config
EXPT_IDS=$($PYTHON -c "
import sys
sys.path.insert(0, 'registration/config')
from config_registration import all_fish
for proj_id, expt_id in all_fish:
    print(expt_id)
")

if [ -z "$EXPT_IDS" ]; then
    echo "ERROR: No fish found in config. Check config_registration.py."
    exit 1
fi

N=$(echo "$EXPT_IDS" | wc -l)
echo "Submitting $N registration jobs..."

while IFS= read -r EXPT_ID; do
    JOB_NAME="reg_${EXPT_ID}"
    LOG="${LOG_DIR}/${EXPT_ID}.log"

    sbatch \
        --job-name="$JOB_NAME" \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --output="$LOG" \
        --wrap="$PYTHON $SCRIPT --config $CONFIG --expt_ID $EXPT_ID"

    echo "  Submitted: $EXPT_ID"
done <<< "$EXPT_IDS"

echo ""
echo "All $N jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      $LOG_DIR/"
