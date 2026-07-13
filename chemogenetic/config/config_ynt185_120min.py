"""
config_ynt185_120min.py
=======================
Config for the YNT185 HCRTR-agonist 120-minute experiment.

Fish: pan-neuronal huc-h2b-g8m only (no HCRT:TRPV1).
Drug: YNT185 — direct pharmacological HCRT receptor agonist (perfused).
N = 6 fish.

Comparison note:
    No within-config ctrl group. For EXPT vs CTRL comparisons, import
    ctrl_fish_csn from config_hcrt_trpv1_csn_120min.py as the ctrl group.

Location:
    ~/zwba/chemogenetic/config/config_ynt185_120min.py
"""

# ============================================================
# BASE PATHS
# ============================================================
dir_voluseg      = "/resnick/groups/Proberlab/yun/lightsheet/"
dir_registration = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/registration/"
dir_analysis     = "/resnick/groups/Proberlab/yun/lightsheet/analysis_output/chemogenetic/"

PYTHON_BIN = "/resnick/home/ychiu/miniconda3/envs/voluseg/bin/python"

# ============================================================
# PROJECT FOLDER
# ============================================================
YNT_PROJ = "huc-h2b-g8m_ynt185_120min"

# ============================================================
# FISH LISTS
# ============================================================
ynt_fish = [
    (YNT_PROJ, "260413_huc-h2b-g8m_ynt_10uM_fish1"),
    (YNT_PROJ, "260413_huc-h2b-g8m_ynt_10uM_fish2"),
    (YNT_PROJ, "260413_huc-h2b-g8m_ynt_10uM_fish3"),
    (YNT_PROJ, "260413_huc-h2b-g8m_ynt_10uM_fish4"),
    (YNT_PROJ, "260414_huc-h2b-g8m_ynt_10uM_fish1"),
    (YNT_PROJ, "260414_huc-h2b-g8m_ynt_10uM_fish2"),
]  # N = 6

all_fish = ynt_fish


# ============================================================
# IMAGING SPECS
# ============================================================
sec_per_volume   = 1
volume_per_sec   = 1
sampling_rate_hz = volume_per_sec

n_slices  = 40
depth     = 250             # µm
binning   = 1
res_x     = 1.52 * binning
res_y     = 1.52 * binning
res_z     = depth / n_slices
rotation_k = 2              # 180° rotation for MapZebrain alignment


# ============================================================
# DRUG PERFUSION  (YNT185)
# ============================================================
drug_uM   = 10.0
V_ml      = 15.0
Q_ml_min  = 4

# Epoch windows in FRAMES (minutes * 60 * volume_per_sec)
baseline_start = 0  * 60 * volume_per_sec
baseline_end   = 45 * 60 * volume_per_sec

drug_start = 45 * 60 * volume_per_sec
drug_end   = 90 * 60 * volume_per_sec

wash_start = 90  * 60 * volume_per_sec
wash_end   = 120 * 60 * volume_per_sec


# ============================================================
# DECOMPOSITION
# ============================================================
df_f_percentile     = 20
f_tonic_window_size = 600   # seconds
f_tonic_percentile  = 20


# ============================================================
# PERMUTATION TEST + BH-FDR
# ============================================================
p_thresh_permutation   = 0.005
n_resample_permutation = 500
BH_Q = 0.05


# ============================================================
# TONIC GLM
# ============================================================
input_tag    = "C"
K_global     = 600
drift_global = 1
lam_global   = 0.5
lag_global   = 0

INCLUDED_BASELINE     = 15.0
CLIP_ABS_DZ           = 50.0
NULL_TAG              = "iaaft"
RESPONDER_NULL_THRESH = 95
L_MIN                 = 20.0

param_folder_name = (
    f"in{input_tag}_K{K_global}_drift{drift_global}_lam{lam_global}_lag{lag_global}"
)


# ============================================================
# GROUP TAGS + PLOT META
# ============================================================
EXPT_TAG = "YNT185"
CTRL_TAG = "CTRL"     # refers to ctrl_fish_csn from config_hcrt_trpv1_csn_120min

PLOT_META = {
    EXPT_TAG: {"label": EXPT_TAG, "color": "mediumpurple", "alpha": 0.7},
    CTRL_TAG: {"label": CTRL_TAG, "color": "grey",         "alpha": 0.7},
}

GROUP_OF_FISH = {fish: EXPT_TAG for fish in ynt_fish}


def get_group_meta(fish):
    tag = GROUP_OF_FISH.get(fish, CTRL_TAG)
    return tag, PLOT_META[tag]


# ============================================================
# COMPARISON TAG + FIGURE PATH
# ============================================================
COMPARISON_TAG = f"{EXPT_TAG}_vs_{CTRL_TAG}"   # "YNT185_vs_CTRL"


def comparison_fig_dir(dir_analysis, comparison_tag=None):
    from pathlib import Path
    tag = comparison_tag if comparison_tag is not None else COMPARISON_TAG
    p = Path(dir_analysis) / "comparisons" / tag / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
