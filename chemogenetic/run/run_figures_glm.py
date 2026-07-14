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

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True,
                    help="Config module under chemogenetic/config/")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

all_fish              = cfg.all_fish
COMPARISON_TAG        = cfg.COMPARISON_TAG
comparison_fig_dir    = cfg.comparison_fig_dir
baseline_end          = cfg.baseline_end
baseline_start        = cfg.baseline_start
CLIP_ABS_DZ           = cfg.CLIP_ABS_DZ
ctrl_fish             = cfg.ctrl_fish
CTRL_TAG              = cfg.CTRL_TAG
dir_analysis          = cfg.dir_analysis
drug_end              = cfg.drug_end
drug_start            = cfg.drug_start
expt_fish             = cfg.expt_fish
EXPT_TAG              = cfg.EXPT_TAG
L_MIN                 = cfg.L_MIN
NULL_TAG              = cfg.NULL_TAG
param_folder_name     = cfg.param_folder_name
PLOT_META             = cfg.PLOT_META
RESPONDER_NULL_THRESH = cfg.RESPONDER_NULL_THRESH
sampling_rate_hz      = cfg.sampling_rate_hz
from chemogenetic.responders_effectsize import (
    frames_to_min_pair,
    _dz_output_paths,
    _load_responder_idx,
)
from utils.data_io import fish_dir

from chemogenetic.population import (
    plot_responder_fractions,
    plot_dz_boxplot,
    plot_plateau_dz_boxplot,
    plot_dprime_boxplot,
)
from chemogenetic.scatter import plot_tonic_phasic_scatter

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
# STAGE A: RESPONDER vs NON-RESPONDER MEAN TRACES
# ============================================================

BL_START_VOL = 1200   # vol 20 min — skip habituation transient

def _plot_responder_mean_traces(fish, group_tag, fig_dir, save=True, show=False):
    """
    Plot mean z-scored F_tonic and F_phasic traces split by GLM responder category:
        - Positive responders (activated by HCRT)   — red
        - Negative responders (suppressed by HCRT)  — blue
        - Non-responders                             — gray

    Z-score per cell uses baseline vols BL_START_VOL:baseline_end.
    Saves one PNG per fish.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj_ID, expt_ID = fish
    base_dir = fish_dir(dir_analysis, fish)

    ft_path = base_dir / "f_tonic.npy"
    fp_path = base_dir / "f_phasic.npy"
    if not ft_path.exists() or not fp_path.exists():
        print(f"  ⚠️  {expt_ID}: f_tonic/f_phasic missing — skipping trace plot")
        return

    try:
        pos_idx, neg_idx, _, _ = _load_responder_idx(
            base_dir, NULL_TAG, RESPONDER_NULL_THRESH,
        )
    except FileNotFoundError:
        print(f"  ⚠️  {expt_ID}: responder idx missing — skipping trace plot")
        return

    Ft = np.load(str(ft_path), mmap_mode="r").astype(np.float32)
    Fp = np.load(str(fp_path), mmap_mode="r").astype(np.float32)
    n_cells, T = Ft.shape
    t_min = np.arange(T) / (60 * sampling_rate_hz)

    # non-responders = all cells minus pos and neg
    resp_all = np.union1d(pos_idx, neg_idx)
    all_idx  = np.arange(n_cells)
    non_idx  = np.setdiff1d(all_idx, resp_all)

    def _zscore_group(arr, idx):
        """Z-score each cell on baseline then take mean ± SEM."""
        if idx.size == 0:
            return None, None
        sub = arr[idx]
        mu  = sub[:, BL_START_VOL:int(baseline_end)].mean(axis=1, keepdims=True)
        sg  = sub[:, BL_START_VOL:int(baseline_end)].std(axis=1, keepdims=True).clip(1e-6)
        z   = (sub - mu) / sg
        mean = z.mean(axis=0)
        sem  = z.std(axis=0) / np.sqrt(idx.size)
        return mean, sem

    groups = {
        f"Pos responders (N={pos_idx.size:,})": (pos_idx,  "#c0392b"),
        f"Neg responders (N={neg_idx.size:,})": (neg_idx,  "#2980b9"),
        f"Non-responders  (N={non_idx.size:,})": (non_idx, "#7f8c8d"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
    fig.suptitle(f"{expt_ID}  [{group_tag}] — Responder mean traces",
                 fontsize=11, fontweight="bold")

    for ax, arr, ylabel, title in zip(
        axes,
        [Ft, Fp],
        ["F_tonic (z-score)", "F_phasic (z-score)"],
        ["F_tonic", "F_phasic"],
    ):
        for label, (idx, color) in groups.items():
            mean, sem = _zscore_group(arr, idx)
            if mean is None:
                continue
            ax.plot(t_min, mean, color=color, lw=1.2, label=label, zorder=5)
            ax.fill_between(t_min, mean - sem, mean + sem,
                            color=color, alpha=0.2, zorder=4)

        ax.axvline(drug_start / (60 * sampling_rate_hz),
                   color="tomato",    lw=1.0, ls="--", alpha=0.8, label="drug start")
        ax.axvline(drug_end   / (60 * sampling_rate_hz),
                   color="steelblue", lw=1.0, ls="--", alpha=0.8, label="drug end")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(0, t_min[-1])
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    fish_fig_dir = base_dir / "figures"
    fish_fig_dir.mkdir(parents=True, exist_ok=True)
    out = fish_fig_dir / "responder_mean_traces.png"
    if save:
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  ✅ {expt_ID} → {out.name}")
    if show:
        plt.show()
    plt.close(fig)

    del Ft, Fp

def _load_responder_fractions(fish_list):
    """Return (pos_fracs, neg_fracs) as arrays of per-fish fractions."""
    pos_fracs, neg_fracs = [], []
    for fish in fish_list:
        proj_ID, expt_ID = fish
        base_dir = fish_dir(dir_analysis, fish)
        Ft_path  = base_dir / "f_tonic.npy"
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

    # ── A: Responder vs non-responder mean traces ─────────────
    if RUN_TRACES:
        print("\n── A: Responder vs non-responder mean traces ────────────")
        for fish in ctrl_fish:
            _plot_responder_mean_traces(fish, CTRL_TAG,
                                        FIG_DIR, save=SAVE_PLOTS, show=SHOW_PLOTS)
        for fish in expt_fish:
            _plot_responder_mean_traces(fish, EXPT_TAG,
                                        FIG_DIR, save=SAVE_PLOTS, show=SHOW_PLOTS)

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
