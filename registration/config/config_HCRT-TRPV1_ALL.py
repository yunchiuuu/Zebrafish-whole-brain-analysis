"""
config_registration.py
======================
Shared registration config for all chemogenetic experiments.

The mean brain is built ONCE from all fish across all experiment groups
and used as a common reference for SyN registration. This config is
experiment-agnostic — it is not tied to any single drug/condition.

To add a new experiment batch:
    1. Add its fish tuples to the relevant list below (ctrl or expt)
    2. Re-run run_registration_mean_brain.py to regenerate the mean brain
    3. Re-run run_registration_syn.py for any new fish

Location:
    ~/Zebrafish-whole-brain-analysis/registration/config_registration.py
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
dir_registration = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/registration/"

# ============================================================
# MEAN BRAIN
# ============================================================
MEAN_BRAIN_FNAME = "mean_brain_HCRT_TRPV1_ALL.nii.gz"
MEAN_BRAIN_PATH  = str(Path(dir_registration) / MEAN_BRAIN_FNAME)

# ============================================================
# IMAGING SPECS
# (shared across all experiments — update here if acquisition changes)
# ============================================================
n_slices   = 40
depth      = 250        # microns
binning    = 1
res_x      = 1.52 * binning
res_y      = 1.52 * binning
res_z      = depth / n_slices
rotation_k = 2          # 180° rotation for mapzebrain alignment

# ANTs resampling target for SyN (isotropic µm)
target_spacing = (3.0, 3.0, 3.0)

# ============================================================
# FISH LISTS — each fish is a (proj_ID, expt_ID) tuple
# Add new experiment batches here as they are collected.
# ============================================================

# --- HCRT-TRPV1 CSN 120min ---
_CTRL_PROJ_CSN = "hcrt-trpv1_huc-h2b-g8m_csn_120min"
_EXPT_PROJ_CSN = "hcrt-trpv1_huc-h2b-g8m_csn_120min"

ctrl_fish_csn = [
    (_CTRL_PROJ_CSN, "251021_huc-h2b-g8m_csn_10uM_fish1"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish1"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish2"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish3"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish4"),
    (_CTRL_PROJ_CSN, "251118_huc-h2b-g8m_csn_10uM_fish5"),
    (_CTRL_PROJ_CSN, "251126_huc-h2b-g8m_csn_10uM_fish1"),
]

expt_fish_csn = [
    (_EXPT_PROJ_CSN, "251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4"),
    (_EXPT_PROJ_CSN, "251102_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (_EXPT_PROJ_CSN, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"),
    (_EXPT_PROJ_CSN, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish2"),
    (_EXPT_PROJ_CSN, "251210_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish3"),
    (_EXPT_PROJ_CSN, "260113_hcrt-trpv1_huc-h2b-g8m_csn_5uM_fish1"),   # NOTE: 5 uM
    (_EXPT_PROJ_CSN, "260113_hcrt-trpv1_huc-h2b-g8m_csn_5uM_fish2"),   # NOTE: 5 uM
]

# --- Add future experiment groups below ---
# _CTRL_PROJ_YNT = "hcrt-trpv1_huc-h2b-g8m_ynt185_120min"
# ctrl_fish_ynt  = [...]
# expt_fish_ynt  = [...]

# ============================================================
# COMBINED FISH LIST FOR MEAN BRAIN
# Add new groups to all_fish_for_mean_brain as they become available.
# ============================================================
all_fish_for_mean_brain = (
    ctrl_fish_csn +
    expt_fish_csn
    # + ctrl_fish_ynt
    # + expt_fish_ynt
)
