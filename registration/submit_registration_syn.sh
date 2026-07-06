#!/bin/bash
# submit_registration_syn.sh
# ==========================
# Submit one sbatch job per fish for SyN registration to the shared mean brain.
# Run AFTER run_registration_mean_brain.py has completed and saved MEAN_BRAIN_FNAME.
#
# Usage:
#     bash registration/submit_registration_syn.sh
#
# From repo root on HPC.

CONFIG="registration/config_registration.py"
SCRIPT="registration/run_registration_syn.py"
CPUS=16
MEM="64G"
LOG_DIR="logs/registration"

mkdir -p "$LOG_DIR"

# ── CSN ctrl fish (N=7) ───────────────────────────────────────────────────────
CTRL_CSN=(
    "251021_huc-h2b-g8m_csn_10uM_fish1"
    "251118_huc-h2b-g8m_csn_10uM_fish1"
    "251118_huc-h2b-g8m_csn_10uM_fish2"
    "251118_huc-h2b-g8m_csn_10uM_fish3"
    "251118_huc-h2b-g8m_csn_10uM_fish4"
    "251118_huc-h2b-g8m_csn_10uM_fish5"
    "251126_huc-h2b-g8m_csn_10uM_fish1"
)

# ── CSN expt fish — main cohort (N=11) ───────────────────────────────────────
EXPT_CSN=(
    "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"
    "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3"
    "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4"
    "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"
    "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"
    "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3"
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"
    "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"
    "260515_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"
)

# ── CSN expt fish — transient hcrt-h2b-g8m injection (N=3) ──────────────────
EXPT_CSN_INJ=(
    "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish1"
    "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish2"
    "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish3"
)

# ── YNT185 fish (N=6 confirmed; add fish3 from 260414 once verified) ─────────
YNT=(
    "260413_huc-h2b-g8m_ynt_10uM_fish1"
    "260413_huc-h2b-g8m_ynt_10uM_fish2"
    "260413_huc-h2b-g8m_ynt_10uM_fish3"
    "260413_huc-h2b-g8m_ynt_10uM_fish4"
    "260414_huc-h2b-g8m_ynt_10uM_fish1"
    "260414_huc-h2b-g8m_ynt_10uM_fish2"
)

# ── Submit all ────────────────────────────────────────────────────────────────
ALL_FISH=(
    "${CTRL_CSN[@]}"
    "${EXPT_CSN[@]}"
    "${EXPT_CSN_INJ[@]}"
    "${YNT[@]}"
)

echo "Submitting ${#ALL_FISH[@]} registration jobs..."

for EXPT_ID in "${ALL_FISH[@]}"; do
    JOB_NAME="reg_${EXPT_ID}"
    LOG="${LOG_DIR}/${EXPT_ID}.log"

    sbatch \
        --job-name="$JOB_NAME" \
        --cpus-per-task=$CPUS \
        --mem=$MEM \
        --output="$LOG" \
        --wrap="python $SCRIPT --config $CONFIG --expt_ID $EXPT_ID"

    echo "  Submitted: $EXPT_ID"
done

echo ""
echo "All ${#ALL_FISH[@]} jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      $LOG_DIR/"
