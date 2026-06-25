"""
run_figures_spearman.py
=======================
Figure runner for Spearman tonic correlation results.

Calls all four spearman_viz plotting functions in order:
    A. plot_rho_tail_fraction_by_lag    — 2 x N_lags grid per lag regressor
    B. plot_rho_tail_fraction_maxlag    — 1 x 2 summary at best lag
    C. plot_lag_preference_histogram    — lag preference histogram (top p99 cells)
    D. plot_lag_binned_mean_traces      — mean F_tonic traces binned by preferred lag

This script is the sbatch target. Modules do nothing when imported.

Usage:
    python run/run_figures_spearman.py
    sbatch --wrap="python run/run_figures_spearman.py"

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_figures_spearman.py
"""

import sys
from pathlib import Path

# make repo root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chemogenetic.config.hcrt_trpv1_csn_120min import (
    CTRL_TAG,
    EXPT_TAG,
    PLOT_META,
    ctrl_fish,
    expt_fish,
    dir_analysis,
    sampling_rate_hz,
    drug_start,
    wash_end,
    lag_max_sec,
    lag_step_sec,
    COMPARISON_TAG,
)
from chemogenetic.spearman_viz import (
    plot_rho_tail_fraction_by_lag,
    plot_rho_tail_fraction_maxlag,
    plot_lag_preference_histogram,
    plot_lag_binned_mean_traces,
)

# ============================================================
# CONFIG
# ============================================================

RHO_POS_THRESH = 0.9
RHO_NEG_THRESH = -0.9
POS_THRESH_TRACES = 0.6      # looser threshold for trace averaging (more cells)
NEG_THRESH_TRACES = -0.6
TOP_Q = 0.99                 # top percentile for lag histogram
LAG_BINS_MIN = (0, 5, 10, 15, 20)

SHOW_PLOTS = False

# Group-level figure output directory
fig_dir = Path(dir_analysis) / "comparisons" / COMPARISON_TAG / "figures" / "spearman"
fig_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# A. 2 x N_lags grid: pos/neg tail fractions per lag
# ============================================================
print("=== A. Rho tail fractions per lag ===")
plot_rho_tail_fraction_by_lag(
    ctrl_fish=ctrl_fish,
    expt_fish=expt_fish,
    ctrl_tag=CTRL_TAG,
    expt_tag=EXPT_TAG,
    plot_meta=PLOT_META,
    dir_analysis=dir_analysis,
    fig_dir=fig_dir,
    rho_pos_thresh=RHO_POS_THRESH,
    rho_neg_thresh=RHO_NEG_THRESH,
    fig_name_tag="allLags",
    save=True,
    show=SHOW_PLOTS,
)

# ============================================================
# B. 1 x 2 summary at best lag
# ============================================================
print("=== B. Rho tail fractions at best lag ===")
plot_rho_tail_fraction_maxlag(
    ctrl_fish=ctrl_fish,
    expt_fish=expt_fish,
    ctrl_tag=CTRL_TAG,
    expt_tag=EXPT_TAG,
    plot_meta=PLOT_META,
    dir_analysis=dir_analysis,
    fig_dir=fig_dir,
    rho_pos_thresh=RHO_POS_THRESH,
    rho_neg_thresh=RHO_NEG_THRESH,
    save=True,
    show=SHOW_PLOTS,
)

# ============================================================
# C. Lag preference histogram (top p99 cells)
# ============================================================
print("=== C. Lag preference histogram ===")
plot_lag_preference_histogram(
    ctrl_fish=ctrl_fish,
    expt_fish=expt_fish,
    ctrl_tag=CTRL_TAG,
    expt_tag=EXPT_TAG,
    plot_meta=PLOT_META,
    dir_analysis=dir_analysis,
    fig_dir=fig_dir,
    lag_max_sec=lag_max_sec,
    lag_step_sec=lag_step_sec,
    top_q=TOP_Q,
    use_abs_rho=True,
    save=True,
    show=SHOW_PLOTS,
)

# ============================================================
# D. Lag-binned mean F_tonic traces (pos and neg separately)
# ============================================================
print("=== D. Lag-binned mean traces ===")
plot_lag_binned_mean_traces(
    ctrl_fish=ctrl_fish,
    expt_fish=expt_fish,
    ctrl_tag=CTRL_TAG,
    expt_tag=EXPT_TAG,
    plot_meta=PLOT_META,
    dir_analysis=dir_analysis,
    fig_dir=fig_dir,
    sampling_rate=sampling_rate_hz,
    drug_start_frame=drug_start,
    wash_end_frame=wash_end,
    pos_thresh=POS_THRESH_TRACES,
    neg_thresh=NEG_THRESH_TRACES,
    lag_bins_min=LAG_BINS_MIN,
    zscore_cells=True,
    save=True,
    show=SHOW_PLOTS,
)

print("=== run_figures_spearman.py complete ===")
