"""
run_figures.py
==============
Orchestrate all group-comparison figures for one config.

This is the only run code that compares ctrl vs expt fish — it reads
already-computed results from dir_analysis and passes them to the
visualization modules. No computation happens here beyond loading.

Stages (all togglable):
    A. Per-fish trace plots      — tonic/phasic z-score per fish
    B. Responder fraction        — fraction pos/neg cells per group
    C. Fixed-window ΔZ boxplot   — mean ΔZ per fish, ctrl vs expt
    D. Plateau ΔZ boxplot        — mean plateau ΔZ per fish
    E. Phasic d′ boxplot         — mean d′ per fish by responder sign
    F. Tonic ΔZ vs phasic d′     — cross-modal scatter

Usage
-----
    python run_figures.py
    # (not typically sbatch'd — fast, no heavy compute)

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_figures.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chemogenetic.config.hcrt_trpv1_csn_120min import (
    all_fish,
    COMPARISON_TAG,
    comparison_fig_dir,
    baseline_end,
    baseline_start,
    CLIP_ABS_DZ,
    ctrl_fish,
    CTRL_TAG,
    dir_analysis,
    drug_end,
    drug_start,
    expt_fish,
    EXPT_TAG,
    L_MIN,
    NULL_TAG,
    param_folder_name,
    PLOT_META,
    RESPONDER_NULL_THRESH,
    sampling_rate_hz,
)
from chemogenetic.responders_effectsize import (
    frames_to_min_pair,
    _dz_output_paths,
    _load_responder_idx,
)
from utils.data_io import fish_dir

from chemogenetic.visualization.traces     import plot_tonic_phasic_zscore
from chemogenetic.visualization.population import (
    plot_responder_fractions,
    plot_dz_boxplot,
    plot_plateau_dz_boxplot,
    plot_dprime_boxplot,
)
from chemogenetic.visualization.scatter import plot_tonic_phasic_scatter

# ============================================================
# STAGE TOGGLES
# ============================================================
RUN_TRACES      = True
RUN_FRACTIONS   = True
RUN_FIXED_DZ    = True
RUN_PLATEAU_DZ  = True
RUN_DPRIME      = True
RUN_SCATTER     = True

SHOW_PLOTS      = False   # True for interactive; False for headless sbatch
SAVE_PLOTS      = True
AMPLITUDE_MODE  = "raw"

# ============================================================
# SHARED SETUP
# ============================================================
BASELINE_MIN = frames_to_min_pair(baseline_start, baseline_end, sampling_rate_hz)
DRUG_MIN     = frames_to_min_pair(drug_start,     drug_end,     sampling_rate_hz)

ctrl_meta = PLOT_META[CTRL_TAG]
expt_meta = PLOT_META[EXPT_TAG]

# Shared figure output dir (project-level)
FIG_DIR = comparison_fig_dir(dir_analysis, COMPARISON_TAG)


# ============================================================
# LOADERS
# ============================================================

def _load_responder_fractions(fish_list):
    """Return (pos_fracs, neg_fracs) as arrays of per-fish fractions."""
    pos_fracs, neg_fracs = [], []
    for fish in fish_list:
        proj_ID, expt_ID = fish
        base_dir = fish_dir(dir_analysis, fish)
        Ft_path  = base_dir / "data_array_f_tonic.npy"
        if not Ft_path.exists():
            continue
        Ft = np.load(str(Ft_path), mmap_mode="r")
        n_cells = Ft.shape[0]
        try:
            pos_idx, neg_idx, _, _ = _load_responder_idx(
                base_dir, NULL_TAG, RESPONDER_NULL_THRESH,
            )
            pos_fracs.append(pos_idx.size / n_cells)
            neg_fracs.append(neg_idx.size / n_cells)
        except FileNotFoundError:
            pass
    return np.array(pos_fracs, dtype=float), np.array(neg_fracs, dtype=float)


def _load_fixed_dz(fish_list):
    """Return (pos_means, neg_means) — one per-fish mean ΔZ per group."""
    pos_means, neg_means = [], []
    for fish in fish_list:
        proj_ID, expt_ID = fish
        cache_root = Path(dir_analysis) / proj_ID / "results_dz_vectors"
        fish_cache = cache_root / expt_ID
        pos_p, neg_p = _dz_output_paths(
            fish_cache, NULL_TAG, RESPONDER_NULL_THRESH,
            BASELINE_MIN, DRUG_MIN, CLIP_ABS_DZ,
        )
        if pos_p.exists() and neg_p.exists():
            dz_pos = np.load(str(pos_p))
            dz_neg = np.load(str(neg_p))
            if dz_pos.size > 0:
                pos_means.append(float(np.nanmean(dz_pos)))
            if dz_neg.size > 0:
                neg_means.append(float(np.nanmean(dz_neg)))
    return np.array(pos_means, dtype=float), np.array(neg_means, dtype=float)


def _load_plateau_dz(fish_list):
    """Return (pos_means, neg_means) from saved plateau ΔZ files."""
    pos_means, neg_means = [], []
    Ltag = f"{int(round(L_MIN))}min"
    nt   = f"_{NULL_TAG}"
    ptag = int(RESPONDER_NULL_THRESH)
    for fish in fish_list:
        base_dir = fish_dir(dir_analysis, fish)
        pos_p = base_dir / f"tonic_pos_plateauDz{nt}_L{Ltag}_nullp{ptag}.npy"
        neg_p = base_dir / f"tonic_neg_plateauDz{nt}_L{Ltag}_nullp{ptag}.npy"
        if pos_p.exists() and neg_p.exists():
            dz_pos = np.load(str(pos_p))
            dz_neg = np.load(str(neg_p))
            if dz_pos.size > 0:
                pos_means.append(float(np.nanmean(dz_pos)))
            if dz_neg.size > 0:
                neg_means.append(float(np.nanmean(dz_neg)))
    return np.array(pos_means, dtype=float), np.array(neg_means, dtype=float)


def _load_dprime_by_sign(fish_list):
    """
    Return (pos_means, neg_means): per-fish mean d′ restricted to
    tonic pos / neg responder cells.
    """
    pos_means, neg_means = [], []
    ptag = int(RESPONDER_NULL_THRESH)
    for fish in fish_list:
        base_dir = fish_dir(dir_analysis, fish)
        dp_path  = base_dir / f"phasic_dprime_cells_{AMPLITUDE_MODE}.npy"
        if not dp_path.exists():
            continue
        try:
            pos_idx, neg_idx, _, _ = _load_responder_idx(
                base_dir, NULL_TAG, RESPONDER_NULL_THRESH,
            )
        except FileNotFoundError:
            continue
        dp = np.load(str(dp_path))
        if pos_idx.size > 0:
            pos_means.append(float(np.nanmean(dp[pos_idx])))
        if neg_idx.size > 0:
            neg_means.append(float(np.nanmean(dp[neg_idx])))
    return np.array(pos_means, dtype=float), np.array(neg_means, dtype=float)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"run_figures | null={NULL_TAG} p{RESPONDER_NULL_THRESH} | fig_dir={FIG_DIR}")

    # ── A: Per-fish traces ────────────────────────────────────
    if RUN_TRACES:
        print("\n── A: Per-fish tonic/phasic traces ─────────────────────")
        for fish in all_fish:
            plot_tonic_phasic_zscore(
                fish=fish,
                dir_analysis=dir_analysis,
                drug_start=drug_start,
                drug_end=drug_end,
                drug_label="Drug",
                save=SAVE_PLOTS,
                show=SHOW_PLOTS,
            )

    # ── B: Responder fractions ────────────────────────────────
    if RUN_FRACTIONS:
        print("\n── B: Responder fractions ───────────────────────────────")
        ctrl_pos_f, ctrl_neg_f = _load_responder_fractions(ctrl_fish)
        expt_pos_f, expt_neg_f = _load_responder_fractions(expt_fish)
        plot_responder_fractions(
            ctrl_pos_f, ctrl_neg_f, expt_pos_f, expt_neg_f,
            ctrl_meta, expt_meta,
            null_tag=NULL_TAG, null_percentile=RESPONDER_NULL_THRESH,
            fig_dir=FIG_DIR, save=SAVE_PLOTS, show=SHOW_PLOTS,
        )

    # ── C: Fixed-window ΔZ ───────────────────────────────────
    if RUN_FIXED_DZ:
        print("\n── C: Fixed-window ΔZ boxplot ───────────────────────────")
        ctrl_pos_dz, ctrl_neg_dz = _load_fixed_dz(ctrl_fish)
        expt_pos_dz, expt_neg_dz = _load_fixed_dz(expt_fish)
        plot_dz_boxplot(
            ctrl_pos_dz, ctrl_neg_dz, expt_pos_dz, expt_neg_dz,
            ctrl_meta, expt_meta,
            null_tag=NULL_TAG, null_percentile=RESPONDER_NULL_THRESH,
            baseline_min_pair=BASELINE_MIN, drug_min_pair=DRUG_MIN,
            fig_dir=FIG_DIR, save=SAVE_PLOTS, show=SHOW_PLOTS,
        )

    # ── D: Plateau ΔZ ────────────────────────────────────────
    if RUN_PLATEAU_DZ:
        print("\n── D: Plateau ΔZ boxplot ────────────────────────────────")
        ctrl_pos_pdz, ctrl_neg_pdz = _load_plateau_dz(ctrl_fish)
        expt_pos_pdz, expt_neg_pdz = _load_plateau_dz(expt_fish)
        plot_plateau_dz_boxplot(
            ctrl_pos_pdz, ctrl_neg_pdz, expt_pos_pdz, expt_neg_pdz,
            ctrl_meta, expt_meta,
            null_tag=NULL_TAG, null_percentile=RESPONDER_NULL_THRESH,
            L_min=L_MIN,
            fig_dir=FIG_DIR, save=SAVE_PLOTS, show=SHOW_PLOTS,
        )

    # ── E: Phasic d′ ─────────────────────────────────────────
    if RUN_DPRIME:
        print("\n── E: Phasic d′ boxplot ─────────────────────────────────")
        ctrl_pos_dp, ctrl_neg_dp = _load_dprime_by_sign(ctrl_fish)
        expt_pos_dp, expt_neg_dp = _load_dprime_by_sign(expt_fish)
        plot_dprime_boxplot(
            ctrl_pos_dp, ctrl_neg_dp, expt_pos_dp, expt_neg_dp,
            ctrl_meta, expt_meta,
            amplitude_mode=AMPLITUDE_MODE,
            fig_dir=FIG_DIR, save=SAVE_PLOTS, show=SHOW_PLOTS,
        )

    # ── F: Tonic ΔZ vs phasic d′ scatter ────────────────────
    if RUN_SCATTER:
        print("\n── F: Tonic ΔZ vs phasic d′ scatter ────────────────────")
        ctrl_pos_dz, ctrl_neg_dz   = _load_fixed_dz(ctrl_fish)
        expt_pos_dz, expt_neg_dz   = _load_fixed_dz(expt_fish)
        ctrl_pos_dp, ctrl_neg_dp   = _load_dprime_by_sign(ctrl_fish)
        expt_pos_dp, expt_neg_dp   = _load_dprime_by_sign(expt_fish)
        plot_tonic_phasic_scatter(
            ctrl_pos_dz, ctrl_pos_dp,
            ctrl_neg_dz, ctrl_neg_dp,
            expt_pos_dz, expt_pos_dp,
            expt_neg_dz, expt_neg_dp,
            ctrl_meta, expt_meta,
            null_tag=NULL_TAG, null_percentile=RESPONDER_NULL_THRESH,
            amplitude_mode=AMPLITUDE_MODE,
            fig_dir=FIG_DIR, save=SAVE_PLOTS, show=SHOW_PLOTS,
        )

    print(f"\n✅ run_figures complete. Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
