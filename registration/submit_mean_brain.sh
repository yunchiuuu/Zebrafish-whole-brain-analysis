#!/bin/bash
# submit_mean_brain.sh
# ====================
# Submit mean brain generation as a single sbatch job.
# Run from repo root after completing interactive QC.
#
# Usage:
#     bash registration/submit_mean_brain.sh

# Bootstrap: use system python3 to read PYTHON_BIN from config
PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'registration/config')
from config_registration import PYTHON_BIN
print(PYTHON_BIN)
")

LOG_DIR="logs/registration"
mkdir -p "$LOG_DIR"

sbatch \
    --job-name=mean_brain \
    --cpus-per-task=16 \
    --mem=128G \
    --output="$LOG_DIR/mean_brain.log" \
    --wrap="$PYTHON registration/run_registration_mean_brain_batch.py"

echo "Mean brain job submitted."
echo "Monitor: squeue -u \$USER"
echo "Log:     $LOG_DIR/mean_brain.log"
