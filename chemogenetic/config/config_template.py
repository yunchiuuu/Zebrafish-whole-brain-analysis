"""
EXPERIMENT_NAME.py
==================
Config for [DRUG] [DURATION]-minute chemogenetic experiment.

Copy this file, rename it to match the experiment
(e.g. ynt185_120min.py, cnqx_csn_90min.py), and fill in every
field marked TODO. Fields marked with a default value are safe to
leave unless this experiment differs.

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/config/EXPERIMENT_NAME.py
"""

# ============================================================
# BASE PATHS  (shared across all chemogenetic experiments)
# ============================================================
dir_voluseg      = "/resnick/groups/Proberlab/yun/lightsheet/"
dir_registration = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/registration/"
dir_analysis     = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/chemogenetic/"


# ============================================================
# PROJECT FOLDERS  — must match directory names on disk exactly
# ============================================================
CTRL_PROJ = "TODO: e.g. huc-h2b-g8m_csn_120min"
EXPT_PROJ = "TODO: e.g. huc-h2b-g8m_ynt185_120min"


# ============================================================
# FISH LISTS  — each fish is a (proj_ID, expt_ID) tuple
# ============================================================
ctrl_fish = [
    # (CTRL_PROJ, "YYMMDD_expt_ID_fishN"),
]

expt_fish = [
    # (EXPT_PROJ, "YYMMDD_expt_ID_fishN"),
]

all_fish = ctrl_fish + expt_fish


# ============================================================
# IMAGING SPECS  (defaults match light-sheet setup; change if needed)
# ============================================================
sec_per_volume = 1
volume_per_sec = 1
sampling_rate_hz = volume_per_sec

n_slices = 40
depth    = 250                         # microns
binning  = 1
res_x    = 1.52 * binning
res_y    = 1.52 * binning
res_z    = depth / n_slices
rotation_k = 2                         # 180° rotation for mapzebrain alignment


# ============================================================
# DRUG PERFUSION  (CSTR model: dC/dt = (Q/V)(Cin - C))
# ============================================================
drug_uM  = None    # TODO: e.g. 10.0
V_ml     = 15.0
Q_ml_min = 4.5

# NOTE: if fish within this experiment were run at different concentrations,
# add a per-fish lookup dict here:
#   drug_uM_per_fish = {
#       (EXPT_PROJ, "260113_..._fish1"): 5.0,
#       ...
#   }
# and use it in run_spearman.py / run_glm.py instead of drug_uM.

# Epoch windows in FRAMES (minutes * 60 * volume_per_sec)
# TODO: adjust for your experiment duration
baseline_start = 0  * 60 * volume_per_sec
baseline_end   = 45 * 60 * volume_per_sec

drug_start = 46 * 60 * volume_per_sec
drug_end   = 90 * 60 * volume_per_sec   # TODO: 90 or 120?

wash_start = 91  * 60 * volume_per_sec
wash_end   = 120 * 60 * volume_per_sec  # TODO: match total duration


# ============================================================
# DECOMPOSITION  (F → F_tonic + F_phasic)
# ============================================================
df_f_percentile     = 20               # default
f_tonic_window_size = 600              # seconds; default
f_tonic_percentile  = 20               # default


# ============================================================
# PERMUTATION TEST + BH-FDR
# ============================================================
p_thresh_permutation   = 0.005         # default
n_resample_permutation = 500           # default
BH_Q = 0.05                            # default


# ============================================================
# TONIC GLM
# ============================================================
input_tag    = "C"                     # "C" or "dC" — TODO: confirm for this drug
K_global     = 600                     # kernel length (frames); default
drift_global = 1                       # polynomial drift order; default
lam_global   = 0.5                     # ridge penalty; default
lag_global   = 0                       # causal lag (frames); default

INCLUDED_BASELINE     = 15.0           # minutes of baseline in GLM fit window
CLIP_ABS_DZ           = 50.0           # |ΔZ| clipping threshold
NULL_TAG              = "iaaft"        # "iaaft" or "shift"
RESPONDER_NULL_THRESH = 95             # null percentile for responder cutoff
L_MIN                 = 20.0           # plateau duration (minutes) for plateau-ΔZ

# Derived: subfolder name for all GLM outputs (do not edit manually)
param_folder_name = (
    f"in{input_tag}_K{K_global}_drift{drift_global}_lam{lam_global}_lag{lag_global}"
)


# ============================================================
# GROUP TAGS + PLOT META
# ============================================================
EXPT_TAG = "TODO: e.g. YNT185"
CTRL_TAG = "CTRL"

PLOT_META = {
    EXPT_TAG: {"label": EXPT_TAG, "color": "TODO: e.g. steelblue", "alpha": 0.7},
    CTRL_TAG: {"label": CTRL_TAG, "color": "grey",                  "alpha": 0.7},
}

GROUP_OF_FISH = {fish: CTRL_TAG for fish in ctrl_fish}
GROUP_OF_FISH.update({fish: EXPT_TAG for fish in expt_fish})


def get_group_meta(fish):
    """Return (group_tag, plot_meta_dict) for a fish tuple."""
    tag = GROUP_OF_FISH[fish]
    return tag, PLOT_META[tag]
