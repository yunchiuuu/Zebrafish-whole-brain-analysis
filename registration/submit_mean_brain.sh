#!/bin/bash
# submit_mean_brain.sh
# ====================
# Submit mean brain generation as a single sbatch job.
# Run from repo root after completing interactive QC.
#
# Usage:
#     bash registration/submit_mean_brain.sh

mkdir -p logs/registration

sbatch \
    --job-name=mean_brain \
    --cpus-per-task=16 \
    --mem=128G \
    --output=logs/registration/mean_brain.log \
    --wrap="python registration/run_registration_mean_brain_batch.py"

echo "Mean brain job submitted."
echo "Monitor: squeue -u \$USER"
echo "Log:     logs/registration/mean_brain.log"
