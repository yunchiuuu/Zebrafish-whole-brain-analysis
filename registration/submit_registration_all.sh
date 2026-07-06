#!/bin/bash
# submit_registration_all.sh
# ==========================
# Submit mean brain generation, then SyN registration for all fish.
# SyN jobs are held until mean brain completes successfully.
#
# Usage (from repo root):
#     bash registration/submit_registration_all.sh

# Bootstrap: read PYTHON_BIN from config
PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'registration/config')
from config_registration import PYTHON_BIN
print(PYTHON_BIN)
")

SCRIPT="registration/run_registration_syn.py"
CONFIG="registration/config/config_registration.py"
CPUS=16
MEM="64G"
LOG_DIR="logs/registration"

mkdir -p "$LOG_DIR"

# ── Step 1: Submit mean brain job ─────────────────────────────────────────────
MEAN_JOB=$(sbatch \
    --job-name=mean_brain \
    --cpus-per-task=16 \
    --mem=128G \
    --output="$LOG_DIR/mean_brain.log" \
    --parsable \
    --wrap="$PYTHON registration/run_registration_mean_brain_batch.py")

echo "Mean brain job submitted: $MEAN_JOB"
echo "Log: $LOG_DIR/mean_brain.log"

# ── Step 2: Read fish list from config ────────────────────────────────────────
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

# ── Step 3: Submit SyN jobs — held until mean brain completes ─────────────────
N=$(echo "$EXPT_IDS" | wc -l)
echo ""
echo "Submitting $N SyN jobs (pending mean brain job $MEAN_JOB)..."

while IFS= read -r EXPT_ID; do
    JOB_NAME="reg_${EXPT_ID}"
    LOG="${LOG_DIR}/${EXPT_ID}.log"

    sbatch \
        --job-name="$JOB_NAME" \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --output="$LOG" \
        --dependency=afterok:$MEAN_JOB \
        --wrap="$PYTHON $SCRIPT --config $CONFIG --expt_ID $EXPT_ID"

    echo "  Queued: $EXPT_ID"
done <<< "$EXPT_IDS"

echo ""
echo "All jobs submitted."
echo "SyN jobs will start automatically once job $MEAN_JOB completes."
echo "Monitor with: squeue -u \$USER"
