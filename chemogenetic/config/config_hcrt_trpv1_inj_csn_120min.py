"""
config_hcrt_trpv1_inj_csn_120min.py
========================
Config for the hcrt-trpv1_huc-h2b-g8m + injected hcrt-h2b-g8m
capsaicin 120-min experiment.

Fish: hcrt-trpv1_huc-h2b-g8m (same genotype as main expt cohort)
      + transient hcrt-h2b-g8m injection for HCRT-cell-specific GCaMP.
Drug: Capsaicin 10 µM (perfused).
N = 3 fish.

Purpose:
    REGRESSOR DERIVATION ONLY.
    The injected hcrt-h2b-g8m allows extraction of the empirical HCRT
    population activity regressor (hcrt_regressor.csv) from identified
    HCRT cells. These fish are NOT pooled with the main expt cohort
    (config_hcrt_trpv1_csn_120min) for whole-brain comparisons, but
    they are functionally HCRT-TRPV1 expt fish.

Location:
    ~/zwba/chemogenetic/config/config_hcrt_trpv1_inj_csn_120min.py
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
INJ_PROJ = "hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_120min"

# ============================================================
# FISH LISTS
# ============================================================
expt_fish_csn_inj = [
    (INJ_PROJ, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish1"),
    (INJ_PROJ, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish2"),
    (INJ_PROJ, "260525_hcrt-trpv1_huc-h2b-g8m_inj_hcrt-h2b-g8m_csn_10uM_fish3"),
]  # N = 3  — regressor derivation cohort

# alias expected by run_temporal_intensity_map.py
expt_fish = expt_fish_csn_inj

# ctrl group — same huc-h2b-g8m fish used across all CSN comparisons
CTRL_PROJ = "huc-h2b-g8m_csn_120min"
ctrl_fish = [
    (CTRL_PROJ, "251021_huc-h2b-g8m_csn_10uM_fish1"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish1"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish2"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish3"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish4"),
    (CTRL_PROJ, "251118_huc-h2b-g8m_csn_10uM_fish5"),
    (CTRL_PROJ, "251126_huc-h2b-g8m_csn_10uM_fish1"),
]  # N = 7

all_fish = expt_fish_csn_inj


# ============================================================
# IMAGING SPECS
# ============================================================
sec_per_volume   = 1
volume_per_sec   = 1
sampling_rate_hz = volume_per_sec

n_slices   = 40
depth      = 250            # µm
binning    = 1
res_x      = 1.52 * binning
res_y      = 1.52 * binning
res_z      = depth / n_slices
rotation_k = 2              # 180° rotation for MapZebrain alignment


# ============================================================
# DRUG PERFUSION  (capsaicin, same as main cohort)
# ============================================================
drug_uM  = 10.0
V_ml     = 15.0
Q_ml_min = 4

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
# Same tags as main expt config — these are HCRT-TRPV1 fish.
# Ctrl group (ctrl_fish_csn) lives in config_hcrt_trpv1_csn_120min.
EXPT_TAG = "HCRT-TRPV1-INJ"
CTRL_TAG = "CTRL"

PLOT_META = {
    EXPT_TAG: {"label": EXPT_TAG, "color": "indianred", "alpha": 0.7},
    CTRL_TAG: {"label": CTRL_TAG, "color": "grey",      "alpha": 0.7},
}

GROUP_OF_FISH = {fish: EXPT_TAG for fish in expt_fish_csn_inj}


def get_group_meta(fish):
    tag = GROUP_OF_FISH[fish]
    return tag, PLOT_META[tag]


# ============================================================
# COMPARISON TAG + FIGURE PATH
# ============================================================
COMPARISON_TAG = f"{EXPT_TAG}_vs_{CTRL_TAG}"   # "HCRT-TRPV1-INJ_vs_CTRL"


def comparison_fig_dir(dir_analysis, comparison_tag=None):
    from pathlib import Path
    tag = comparison_tag if comparison_tag is not None else COMPARISON_TAG
    p = Path(dir_analysis) / "comparisons" / tag / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
