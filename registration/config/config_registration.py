"""
registration/config_registration.py
=====================================
Shared registration config for ALL chemogenetic experiments.

The mean brain is built ONCE from all fish across all experiment groups
and used as a common reference for SyN registration. This config is
experiment-agnostic — not tied to any single drug/condition.

To add a new experiment batch:
    1. Add its fish tuples to the relevant list below
    2. Re-run run_registration_mean_brain.py to regenerate the mean brain
    3. Re-run run_registration_syn.py for ALL fish (mean brain changed)

IMPORTANT — proj_ID change from old config
------------------------------------------
Ctrl fish previously lived under hcrt-trpv1_huc-h2b-g8m_csn_120min
(old design: ctrl copied into expt folder). They now live under their
own proj_ID: huc-h2b-g8m_csn_120min. If registration was already run
for these fish, move or re-run their outputs before proceeding:
    dir_registration / hcrt-trpv1_huc-h2b-g8m_csn_120min / <expt_ID> /
    → dir_registration / huc-h2b-g8m_csn_120min / <expt_ID> /

Location:
    ~/Zebrafish-whole-brain-analysis/registration/config_registration.py
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PYTHON_BIN       = "/resnick/home/ychiu/miniconda3/envs/voluseg/bin/python"
dir_voluseg      = Path("/resnick/groups/Proberlab/yun/lightsheet/")
dir_registration = Path("/resnick/groups/Proberlab/yun/lightsheet/analysis_output/registration/")

# ============================================================
# MEAN BRAIN
# ============================================================
# Renamed from mean_brain_HCRT_TRPV1_ALL.nii.gz to reflect that YNT
# fish are now included. Rebuild required whenever fish are added.
MEAN_BRAIN_FNAME  = "template_260514_hcrt-trpv1_csn_fish1.nii.gz"
MEAN_BRAIN_PATH   = str(dir_registration / MEAN_BRAIN_FNAME)

# Template fish used as fixed reference for all SyN registrations.
# TEMPLATE_EXPT_ID is used by run_registration_syn.py to auto-gitgenerate
# the template NIfTI if it doesn't exist on disk yet.
TEMPLATE_IDX      = 15   # index into all_fish_for_mean_brain
TEMPLATE_EXPT_ID  = "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"

# ============================================================
# IMAGING SPECS
# (shared across all experiments — update here if acquisition changes)
# ============================================================
n_slices       = 40
depth          = 250            # µm
binning        = 1
res_x          = 1.52 * binning # µm/pixel (lateral)
res_y          = 1.52 * binning # µm/pixel (lateral)
res_z          = depth / n_slices  # 6.25 µm/slice (axial)
rotation_k     = 2              # 180° rotation for MapZebrain alignment
target_spacing = (3.0, 3.0, 3.0)  # ANTs SyN resampling target (isotropic µm)

# ============================================================
# FISH LISTS — each entry is a (proj_ID, expt_ID) tuple
# ============================================================

# ── CSN 120min — CTRL (huc-h2b-g8m, no HCRT:TRPV1) ─────────────────────────
# NOTE: proj_ID changed from old config (was hcrt-trpv1_huc-h2b-g8m_csn_120min)
_CTRL_PROJ_CSN = "huc-h2b-g8m_csn_120min"

ctrl_fish_csn = [
    (_CTRL_PROJ_CSN, "251021_huc-h2b-g8m_csn_10uM_fish1"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish1"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish2"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish3"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish4"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish5"),
    (_CTRL_PROJ_CSN, "251126_huc-h2b-g8m_csn_10uM_fish1"),
]  # N = 7

# ── CSN 120min — EXPT (hcrt-trpv1_huc-h2b-g8m) — main cohort ───────────────
_EXPT_PROJ_CSN = "hcrt-trpv1_huc-h2b-g8m_csn_120min"

expt_fish_csn = [
    (_EXPT_PROJ_CSN, "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (_EXPT_PROJ_CSN, "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3"),
    (_EXPT_PROJ_CSN, "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4"),
    (_EXPT_PROJ_CSN, "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (_EXPT_PROJ_CSN, "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (_EXPT_PROJ_CSN, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (_EXPT_PROJ_CSN, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (_EXPT_PROJ_CSN, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3"),
    (_EXPT_PROJ_CSN, "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (_EXPT_PROJ_CSN, "260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (_EXPT_PROJ_CSN, "260515_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
]  # N = 11

# ── CSN 120min — EXPT with transient hcrt-h2b-g8m injection ─────────────────
# Analyzed separately from main cohort (see hcrt_trpv1_csn_inj_vs_ctrl.py)
# but included here for registration — brains are valid for mean brain.
expt_fish_csn_inj = [
    (_EXPT_PROJ_CSN, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish1"),
    (_EXPT_PROJ_CSN, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish2"),
    (_EXPT_PROJ_CSN, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish3"),
]  # N = 3

# ── YNT185 120min — huc-h2b-g8m + HCRTR agonist ────────────────────────────
# NOTE: these fish were processed with old MATLAB voluseg code. Registration
# is unaffected, but trace format must be verified before running analysis.
_PROJ_YNT = "huc-h2b-g8m_ynt185_120min"   # confirmed folder name on disk

ynt_fish = [
    (_PROJ_YNT, "260413_huc-h2b-g8m_ynt_10uM_fish1"),
    (_PROJ_YNT, "260413_huc-h2b-g8m_ynt_10uM_fish2"),
    (_PROJ_YNT, "260413_huc-h2b-g8m_ynt_10uM_fish3"),
    (_PROJ_YNT, "260413_huc-h2b-g8m_ynt_10uM_fish4"),
    (_PROJ_YNT, "260414_huc-h2b-g8m_ynt_10uM_fish1"),
    (_PROJ_YNT, "260414_huc-h2b-g8m_ynt_10uM_fish2"),
    # (_PROJ_YNT, "260414_huc-h2b-g8m_ynt_10uM_fish3"),  # TODO: confirm exists on disk
]  # N = 6 confirmed; uncomment fish3 once verified

# --- Add future experiment groups below ---
# _PROJ_CNQX = "hcrt-trpv1_huc-h2b-g8m_cnqx-csn_120min"
# expt_fish_cnqx = [...]

# ============================================================
# COMBINED FISH LIST FOR MEAN BRAIN
# All fish whose brains contribute to the shared reference.
# Rebuild mean brain whenever this list changes.
# ============================================================
all_fish_for_mean_brain = (
    ctrl_fish_csn       # N = 7
    + expt_fish_csn     # N = 13
    + expt_fish_csn_inj # N = 3
    + ynt_fish          # N = 6 confirmed (fish3 from 260414 TBC)
)                       # Total = 29 confirmed

# Alias used by run_registration_syn.py — registers any fish in this list
all_fish = all_fish_for_mean_brain
