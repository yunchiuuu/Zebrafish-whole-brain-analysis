#!/bin/bash
# submit_registration_syn.sh
# ==========================
# Submit one sbatch job per fish for SyN registration to the shared template.
# Fish list and Python path are read from the config passed with --config.
# The template (generate_template.py) must already exist; its versioned
# filename comes from the same config (MEAN_BRAIN_PATH).
#
# --config is a MODULE NAME under registration/config/ (no folder, no .py),
# the same convention as the chemogenetic submit scripts.
#
# Usage (from repo root):
#     bash registration/submit_registration_syn.sh --config config_registration

SCRIPT="registration/run_registration_syn.py"
CPUS=32
MEM="24G"
LOG_DIR="logs/registration"

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
    echo "Usage: bash registration/submit_registration_syn.sh --config config_registration"
    exit 1
fi

if [[ "$CONFIG" == *.py || "$CONFIG" == */* ]]; then
    echo "ERROR: --config must be a module name (e.g. config_registration), not a path: $CONFIG"
    exit 1
fi

mkdir -p "$LOG_DIR"

# Bootstrap: use system python3 to read PYTHON_BIN from config
PYTHON=$(python3 -c "
import sys
sys.path.insert(0, 'registration/config')
from ${CONFIG} import PYTHON_BIN
print(PYTHON_BIN)
") || { echo "ERROR: could not import ${CONFIG} from registration/config/"; exit 1; }

# Read all expt_IDs from config
EXPT_IDS=$($PYTHON -c "
import sys
sys.path.insert(0, 'registration/config')
from ${CONFIG} import all_fish
for proj_id, expt_id in all_fish:
    print(expt_id)
")

if [ -z "$EXPT_IDS" ]; then
    echo "ERROR: No fish found in ${CONFIG}. Check that fish are uncommented in all_fish."
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
