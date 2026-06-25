"""
spearman_viz.py
===============
Visualization functions for tonic Spearman correlation results.

Three plots:
    plot_rho_tail_fraction_by_lag   : 2 x N_lags grid — pos/neg tail fractions
                                      per lag regressor, ctrl vs expt boxplots
    plot_rho_tail_fraction_maxlag   : 1 x 2 summary boxplot using best-lag rho only
    plot_lag_preference_histogram   : side-by-side bar histogram of preferred lag
                                      times for top-p99 cells, pooled across fish

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/spearman_viz.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from utils.data_io import fish_dir


# ============================================================
# SHARED HELPERS
# ============================================================

def _mw_test_and_stars(x, y):
    """Mann-Whitney U (two-sided) + star annotation string."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan, "n/a"
    _, p = mannwhitneyu(x, y, alternative="two-sided")
    stars = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
    return float(p), stars


def _add_scatter(ax, data, x_pos, color, jitter=0.06, alpha=0.85):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    x = x_pos + np.random.uniform(-jitter, jitter, size=len(data))
    ax.scatter(x, data, s=35, color=color, edgecolor="black",
               linewidth=0.5, alpha=alpha, zorder=3)


def _get_style(geno, plot_meta):
    if geno not in plot_meta:
        raise KeyError(f"{geno} not in PLOT_META. Keys: {list(plot_meta.keys())}")
    m = plot_meta[geno]
    return {"label": m.get("label", geno), "color": m.get("color", "gray"),
            "alpha": m.get("alpha", 0.7)}


def _load_lag_grid(dir_analysis, fish_list):
    """Find and return rho_tonic_lag_grid_sec.npy from the first fish that has it."""
    for fish in fish_list:
        p = fish_dir(dir_analysis, fish) / "rho_tonic_lag_grid_sec.npy"
        if p.exists():
            g = np.load(str(p)).astype(np.float32).ravel()
            if g.size > 0:
                return g
    return None


def _zscore_rows(X, eps=1e-6):
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True) + eps
    return (X - mu) / sd


def _sem(X, axis=0):
    return np.nanstd(X, axis=axis) / np.sqrt(np.sum(np.isfinite(X), axis=axis))


# ============================================================
# PLOT A — 2 x N_lags grid: pos/neg tail fractions per lag
# ============================================================

def _compute_tail_fractions_per_lag(
    fish_list,
    geno_map,
    dir_analysis,
    rho_pos_thresh=0.9,
    rho_neg_thresh=-0.9,
):
    """
    Load rho_tonic_all_lags.npy for each fish and compute fraction of cells
    above/below threshold for each lag.

    Returns a DataFrame with columns:
        expt_ID, genotype, lag_idx, frac_pos_strong, frac_neg_strong, n_cells, n_lags
    """
    rows = []
    for fish in fish_list:
        proj_ID, expt_ID = fish
        p = fish_dir(dir_analysis, fish) / "rho_tonic_all_lags.npy"
        if not p.exists():
            print(f"  ⚠️ missing rho_tonic_all_lags.npy: {expt_ID}")
            continue

        rho_all = np.load(str(p)).astype(np.float32)
        if rho_all.ndim != 2:
            print(f"  ⚠️ bad rho_all shape for {expt_ID}: {rho_all.shape}")
            continue

        n_cells, n_lags = rho_all.shape
        geno = geno_map.get(fish)
        if geno is None:
            print(f"  ⚠️ fish not in geno_map: {fish}")
            continue

        for li in range(n_lags):
            r = rho_all[:, li]
            r = r[np.isfinite(r)]
            frac_pos = float(np.mean(r > rho_pos_thresh)) if r.size else np.nan
            frac_neg = float(np.mean(r < rho_neg_thresh)) if r.size else np.nan
            rows.append({
                "expt_ID": expt_ID,
                "genotype": geno,
                "lag_idx": int(li),
                "n_cells": int(n_cells),
                "n_lags": int(n_lags),
                "frac_pos_strong": frac_pos,
                "frac_neg_strong": frac_neg,
            })

    return pd.DataFrame(rows)


def plot_rho_tail_fraction_by_lag(
    ctrl_fish,
    expt_fish,
    ctrl_tag,
    expt_tag,
    plot_meta,
    dir_analysis,
    fig_dir,
    rho_pos_thresh=0.9,
    rho_neg_thresh=-0.9,
    fig_name_tag="allLags",
    save=True,
    show=True,
    seed=0,
):
    """
    2 x N_lags grid of boxplots showing fraction of cells with strong positive
    (top row) or negative (bottom row) Spearman correlation per lag regressor.

    Each column is one lag (0, 5, 10, ... lag_max min).
    Each box compares ctrl vs expt fish.

    Parameters
    ----------
    ctrl_fish, expt_fish : list of (proj_ID, expt_ID)
    ctrl_tag, expt_tag : str
        Keys into plot_meta.
    plot_meta : dict
        From config.PLOT_META.
    dir_analysis : str or Path
    fig_dir : str or Path
        Where to save the figure.
    rho_pos_thresh, rho_neg_thresh : float
    fig_name_tag : str
    save, show : bool
    seed : int
        RNG seed for jitter.
    """
    np.random.seed(seed)

    all_fish = ctrl_fish + expt_fish
    geno_map = {f: ctrl_tag for f in ctrl_fish}
    geno_map.update({f: expt_tag for f in expt_fish})

    lag_grid = _load_lag_grid(dir_analysis, all_fish)
    if lag_grid is None:
        raise FileNotFoundError("Could not find rho_tonic_lag_grid_sec.npy in any fish folder.")

    df = _compute_tail_fractions_per_lag(
        all_fish, geno_map, dir_analysis, rho_pos_thresh, rho_neg_thresh
    )
    if df.empty:
        raise RuntimeError("No fish had rho_tonic_all_lags.npy or all were invalid.")

    groups = [ctrl_tag, expt_tag]
    styles = {g: _get_style(g, plot_meta) for g in groups}
    lag_grid_min = lag_grid / 60.0
    n_lags = int(lag_grid_min.size)

    fig, axs = plt.subplots(2, n_lags, figsize=(3.2 * n_lags, 7.0),
                             sharey=False, constrained_layout=True)
    if n_lags == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    pos_vals = df["frac_pos_strong"].to_numpy(dtype=float)
    neg_vals = df["frac_neg_strong"].to_numpy(dtype=float)
    pos_ylim = max(0.002, float(np.nanmax(pos_vals[np.isfinite(pos_vals)])) * 1.35) \
        if np.any(np.isfinite(pos_vals)) else 0.05
    neg_ylim = max(0.002, float(np.nanmax(neg_vals[np.isfinite(neg_vals)])) * 1.25) \
        if np.any(np.isfinite(neg_vals)) else 0.05

    for li in range(n_lags):
        df_l = df[df["lag_idx"] == li]
        labels = [styles[g]["label"] for g in groups]

        for row, (metric, ylim) in enumerate(
            [("frac_pos_strong", pos_ylim), ("frac_neg_strong", neg_ylim)]
        ):
            ax = axs[row, li]
            data = [df_l.loc[df_l["genotype"] == g, metric].values for g in groups]

            bp = ax.boxplot(data, labels=labels, widths=0.5, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5),
                            showfliers=False)
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(styles[g]["color"])
                patch.set_alpha(styles[g]["alpha"])

            for i, g in enumerate(groups, start=1):
                _add_scatter(ax, df_l.loc[df_l["genotype"] == g, metric].values,
                             x_pos=i, color=styles[g]["color"],
                             alpha=max(styles[g]["alpha"], 0.85))

            p_val, stars = _mw_test_and_stars(data[0], data[1])
            y = ylim * 0.92
            h = ylim * 0.025
            ax.plot([1, 1, 2, 2], [y - h, y, y, y - h], color="black", lw=1)
            ax.text(1.5, y + ylim * 0.01,
                    f"{stars}\n(p={p_val:.2e})" if np.isfinite(p_val) else "n/a",
                    ha="center", va="bottom", fontsize=9)

            if li == 0:
                ax.set_title(f"lag {lag_grid_min[li]:g} min", pad=14)
                thresh_str = f"rho > {rho_pos_thresh}" if row == 0 else f"rho < {rho_neg_thresh}"
                ax.text(-0.55, 0.5, f"{'Positive' if row == 0 else 'Negative'} tail\n({thresh_str})",
                        transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=12, fontweight="bold")
                ax.set_ylabel("Fraction per fish")
            else:
                ax.set_title(f"lag {lag_grid_min[li]:g} min", pad=14)

            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(0, ylim)

    fig.suptitle("Fraction of Cells with Strong Tonic Spearman Correlation (per lag regressor)",
                 y=1.03, fontsize=13)

    if save:
        out = Path(fig_dir) / f"rho_tail_pos{rho_pos_thresh}neg{rho_neg_thresh}_boxplot_2x{n_lags}_{fig_name_tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  ✅ Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# PLOT B — 1 x 2 summary: pos/neg tail fractions at best lag
# ============================================================

def _compute_tail_fractions_maxlag(
    fish_list,
    geno_map,
    dir_analysis,
    rho_pos_thresh=0.9,
    rho_neg_thresh=-0.9,
):
    """
    Load rho_tonic_lagmax.npy for each fish and compute tail fractions.

    Returns a DataFrame with columns:
        expt_ID, genotype, frac_pos_strong, frac_neg_strong, n_cells
    """
    rows = []
    for fish in fish_list:
        proj_ID, expt_ID = fish
        p = fish_dir(dir_analysis, fish) / "rho_tonic_lagmax.npy"
        if not p.exists():
            print(f"  ⚠️ missing rho_tonic_lagmax.npy: {expt_ID}")
            continue

        rho = np.load(str(p)).astype(np.float32)
        rho = rho[np.isfinite(rho)]
        if rho.size == 0:
            continue

        geno = geno_map.get(fish)
        if geno is None:
            continue

        rows.append({
            "expt_ID": expt_ID,
            "genotype": geno,
            "n_cells": int(rho.size),
            "frac_pos_strong": float(np.mean(rho > rho_pos_thresh)),
            "frac_neg_strong": float(np.mean(rho < rho_neg_thresh)),
        })

    return pd.DataFrame(rows)


def plot_rho_tail_fraction_maxlag(
    ctrl_fish,
    expt_fish,
    ctrl_tag,
    expt_tag,
    plot_meta,
    dir_analysis,
    fig_dir,
    rho_pos_thresh=0.9,
    rho_neg_thresh=-0.9,
    save=True,
    show=True,
    seed=0,
):
    """
    1 x 2 boxplot summary of fraction of cells with strong Spearman correlation
    at the best lag, comparing ctrl vs expt.

    Left panel: positive tail (rho > rho_pos_thresh)
    Right panel: negative tail (rho < rho_neg_thresh)

    Parameters
    ----------
    ctrl_fish, expt_fish : list of (proj_ID, expt_ID)
    ctrl_tag, expt_tag : str
    plot_meta : dict
    dir_analysis : str or Path
    fig_dir : str or Path
    rho_pos_thresh, rho_neg_thresh : float
    save, show : bool
    seed : int
    """
    np.random.seed(seed)

    all_fish = ctrl_fish + expt_fish
    geno_map = {f: ctrl_tag for f in ctrl_fish}
    geno_map.update({f: expt_tag for f in expt_fish})

    df = _compute_tail_fractions_maxlag(
        all_fish, geno_map, dir_analysis, rho_pos_thresh, rho_neg_thresh
    )
    if df.empty:
        raise RuntimeError("No fish had rho_tonic_lagmax.npy or all were empty.")

    groups = [ctrl_tag, expt_tag]
    styles = {g: _get_style(g, plot_meta) for g in groups}
    labels = [styles[g]["label"] for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, metric, title_str, thresh in [
        (axes[0], "frac_pos_strong",
         f"Positive correlated cells\n(ranked corr. > {rho_pos_thresh})", rho_pos_thresh),
        (axes[1], "frac_neg_strong",
         f"Negative correlated cells\n(ranked corr. < {rho_neg_thresh})", rho_neg_thresh),
    ]:
        data = [df.loc[df["genotype"] == g, metric].values for g in groups]

        bp = ax.boxplot(data, labels=labels, widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        showfliers=False)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(styles[g]["color"])
            patch.set_alpha(styles[g]["alpha"])

        for i, g in enumerate(groups, start=1):
            _add_scatter(ax, df.loc[df["genotype"] == g, metric].values,
                         x_pos=i, color=styles[g]["color"],
                         alpha=max(styles[g]["alpha"], 0.85))

        p_val, stars = _mw_test_and_stars(data[0], data[1])
        all_vals = np.concatenate([d[np.isfinite(d)] for d in data])
        y_max = float(np.nanmax(all_vals)) if all_vals.size else 0.01
        y = y_max * 1.12 if y_max > 0 else 0.01
        ax.plot([1, 1, 2, 2], [y * 0.98, y, y, y * 0.98], color="black", lw=1)
        ax.text(1.5, y * 1.02,
                f"{stars}\n(p={p_val:.2e})" if np.isfinite(p_val) else "n/a",
                ha="center", va="bottom")

        ax.set_title(title_str)
        ax.set_ylabel("Fraction of cells per fish")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, y * 1.15)

    fig.suptitle("Fraction of Cells with Strong Tonic Correlation with Drug Regressor",
                 y=1.03, fontsize=13)
    fig.tight_layout()

    if save:
        out = Path(fig_dir) / f"rho_tail_pos{rho_pos_thresh}neg{rho_neg_thresh}_boxplot_maxlag.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  ✅ Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# PLOT C — Lag preference histogram (top-p99 cells)
# ============================================================

def _pooled_lag_pref(fish_list, dir_analysis, top_q=0.99, use_abs_rho=True):
    """
    Pool preferred lag times (in seconds) from the top-p99 cells across fish.
    Returns array of lag_sec values.
    """
    pooled = []
    for fish in fish_list:
        p_lag = fish_dir(dir_analysis, fish) / "rho_tonic_lag_argmax_sec.npy"
        p_rho = fish_dir(dir_analysis, fish) / "rho_tonic_lagmax.npy"
        if not p_lag.exists() or not p_rho.exists():
            print(f"  ⚠️ missing lag/rho files: {fish[1]}")
            continue

        lag = np.load(str(p_lag)).astype(np.float32)
        rho = np.load(str(p_rho)).astype(np.float32)
        n = min(lag.size, rho.size)
        lag, rho = lag[:n], rho[:n]

        good = np.isfinite(lag) & np.isfinite(rho)
        lag, rho = lag[good], rho[good]
        if use_abs_rho:
            rho = np.abs(rho)

        if lag.size == 0:
            continue

        thr = np.quantile(rho, top_q)
        pooled.append(lag[rho >= thr])

    return np.concatenate(pooled) if pooled else np.array([])


def plot_lag_preference_histogram(
    ctrl_fish,
    expt_fish,
    ctrl_tag,
    expt_tag,
    plot_meta,
    dir_analysis,
    fig_dir,
    lag_max_sec,
    lag_step_sec,
    top_q=0.99,
    use_abs_rho=True,
    save=True,
    show=True,
):
    """
    Side-by-side bar histogram of preferred lag times for top-p99 cells,
    pooled across fish separately for ctrl and expt.

    X-axis: preferred lag in minutes (discrete bins matching lag scan grid).
    Y-axis: fraction of selected cells.

    Parameters
    ----------
    ctrl_fish, expt_fish : list of (proj_ID, expt_ID)
    ctrl_tag, expt_tag : str
    plot_meta : dict
    dir_analysis : str or Path
    fig_dir : str or Path
    lag_max_sec : float
        Should match config value used during Spearman computation.
    lag_step_sec : float
        Should match config value used during Spearman computation.
    top_q : float
        Quantile threshold for selecting high-rho cells (default 0.99 = top 1%).
    use_abs_rho : bool
        If True, rank by |rho| regardless of sign.
    save, show : bool
    """
    ctrl_lag_sec = _pooled_lag_pref(ctrl_fish, dir_analysis, top_q, use_abs_rho)
    expt_lag_sec = _pooled_lag_pref(expt_fish, dir_analysis, top_q, use_abs_rho)

    lag_vals_sec = np.arange(0, lag_max_sec + 1, lag_step_sec)
    lag_vals_min = lag_vals_sec / 60.0
    step_min = lag_step_sec / 60.0
    edges_min = np.r_[
        lag_vals_min[0] - step_min / 2,
        (lag_vals_min[:-1] + lag_vals_min[1:]) / 2,
        lag_vals_min[-1] + step_min / 2,
    ]

    def _to_hist(lag_sec):
        if lag_sec.size == 0:
            return np.zeros(len(lag_vals_min))
        h, _ = np.histogram(lag_sec / 60.0, bins=edges_min)
        return h / max(h.sum(), 1)

    hist_ctrl = _to_hist(ctrl_lag_sec)
    hist_expt = _to_hist(expt_lag_sec)

    bar_w = 0.35 * step_min
    x = lag_vals_min

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bar_w / 2, hist_ctrl, width=bar_w,
           color=plot_meta[ctrl_tag]["color"], alpha=plot_meta[ctrl_tag]["alpha"],
           label=f'{plot_meta[ctrl_tag]["label"]} (top p{int(top_q*100)} cells)')
    ax.bar(x + bar_w / 2, hist_expt, width=bar_w,
           color=plot_meta[expt_tag]["color"], alpha=plot_meta[expt_tag]["alpha"],
           label=f'{plot_meta[expt_tag]["label"]} (top p{int(top_q*100)} cells)')

    ax.set_title(f"Tonic lag preference (top p{int(top_q*100)} by |rho|): {ctrl_tag} vs {expt_tag}")
    ax.set_xlabel("Preferred lag (minutes)")
    ax.set_ylabel("Fraction of selected cells")
    ax.set_xticks(x)
    ax.set_xlim(edges_min[0], edges_min[-1])
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        out = Path(fig_dir) / f"lag_pref_hist_top_p{int(top_q*100)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
        print(f"  ✅ Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# PLOT D — Lag-binned mean F_tonic traces
# ============================================================

def _aligned_regressor_segment(C, win_start, win_end, lag_sec, sampling_rate):
    T = C.shape[0]
    lag_frames = int(round(lag_sec * sampling_rate))
    t0 = max(win_start, lag_frames)
    t1 = min(win_end, T)
    if t1 <= t0:
        return None, None
    Cseg = C[t0 - lag_frames: t1 - lag_frames].astype(np.float32)
    Cseg = (Cseg - np.nanmean(Cseg)) / (np.nanstd(Cseg) + 1e-6)
    x_min = (np.arange(t0, t1) - win_start) / sampling_rate / 60.0
    return x_min, Cseg


def _select_cells_by_rho(rho, mode, pos_thresh, neg_thresh):
    rho = np.asarray(rho, dtype=np.float32)
    if mode == "pos":
        return np.where(rho > pos_thresh)[0]
    elif mode == "neg":
        return np.where(rho < neg_thresh)[0]
    raise ValueError("mode must be 'pos' or 'neg'")


def _bin_lags(lag_sec, lag_bins_min):
    lag_min = lag_sec / 60.0
    lag_bins_min = np.asarray(lag_bins_min, dtype=float)
    idx = np.nanargmin(np.abs(lag_min[:, None] - lag_bins_min[None, :]), axis=1)
    return lag_bins_min[idx].astype(int)


def _collect_traces_by_lag(
    fish_list,
    dir_analysis,
    sampling_rate,
    drug_start_frame,
    wash_end_frame,
    mode,
    pos_thresh,
    neg_thresh,
    lag_bins_min,
    zscore_cells=True,
    min_cells_per_bin=3,
):
    """
    Per-fish: select high-rho cells, bin by preferred lag, compute mean F_tonic trace.

    Returns
    -------
    fish_traces : dict  {lag_bin_min -> list of per-fish mean traces}
    C_ref : np.ndarray or None
    """
    ws = int(round(drug_start_frame))
    we = int(round(wash_end_frame))

    per_fish = []
    C_ref = None
    valid_lengths = []

    for fish in fish_list:
        proj_ID, expt_ID = fish
        d = fish_dir(dir_analysis, fish)
        paths = {k: d / v for k, v in [
            ("Ft",  "data_array_f_tonic.npy"),
            ("rho", "rho_tonic_lagmax.npy"),
            ("lag", "rho_tonic_lag_argmax_sec.npy"),
            ("C",   "C_capsaicin.npy"),
        ]}
        if not all(p.exists() for p in paths.values()):
            print(f"  ⚠️ missing files for {expt_ID}, skipping")
            continue

        Ft  = np.load(str(paths["Ft"]), mmap_mode="r")
        rho = np.load(str(paths["rho"]))
        lag = np.load(str(paths["lag"]))
        C   = np.load(str(paths["C"]))

        if C_ref is None:
            C_ref = np.asarray(C).copy()

        n_cells, T = Ft.shape
        ws_clamped = max(0, ws)
        we_clamped = min(we, T)
        if we_clamped <= ws_clamped:
            print(f"  ⚠️ invalid window for {expt_ID}, skipping")
            continue

        idx_sel = _select_cells_by_rho(rho, mode, pos_thresh, neg_thresh)
        if idx_sel.size == 0:
            continue

        lag_bins = _bin_lags(lag[idx_sel], lag_bins_min)
        X = np.asarray(Ft[idx_sel, ws_clamped:we_clamped])
        if zscore_cells:
            X = _zscore_rows(X)
        if X.shape[1] == 0:
            continue

        per_bin_means = {}
        for lb in np.unique(lag_bins):
            m = lag_bins == lb
            if np.sum(m) < min_cells_per_bin:
                continue
            per_bin_means[int(lb)] = np.nanmean(X[m], axis=0).astype(np.float32)

        if per_bin_means:
            per_fish.append({"expt_ID": expt_ID, "L": X.shape[1],
                              "per_bin_means": per_bin_means})
            valid_lengths.append(X.shape[1])

    if not per_fish:
        return {}, C_ref

    common_L = min(valid_lengths)
    fish_traces = {}
    for rec in per_fish:
        for lb, tr in rec["per_bin_means"].items():
            fish_traces.setdefault(int(lb), []).append(
                np.asarray(tr[:common_L], dtype=np.float32)
            )
    return fish_traces, C_ref


def plot_lag_binned_mean_traces(
    ctrl_fish,
    expt_fish,
    ctrl_tag,
    expt_tag,
    plot_meta,
    dir_analysis,
    fig_dir,
    sampling_rate,
    drug_start_frame,
    wash_end_frame,
    pos_thresh=0.6,
    neg_thresh=-0.6,
    lag_bins_min=(0, 5, 10, 15, 20),
    zscore_cells=True,
    save=True,
    show=True,
):
    """
    For each mode (pos / neg), plot a 1 x N_lags grid of mean F_tonic traces,
    one subplot per preferred-lag bin.

    Each subplot shows ctrl and expt mean ± SEM, with the lag-shifted drug
    regressor C(t) overlaid as a dashed line.

    Two figures are saved: one for positive-rho cells, one for negative-rho cells.

    Parameters
    ----------
    ctrl_fish, expt_fish : list of (proj_ID, expt_ID)
    ctrl_tag, expt_tag : str
    plot_meta : dict
    dir_analysis : str or Path
    fig_dir : str or Path
    sampling_rate : float
    drug_start_frame : int
    wash_end_frame : int
    pos_thresh : float
    neg_thresh : float
    lag_bins_min : tuple of int
        Lag bin centers in minutes.
    zscore_cells : bool
    save, show : bool
    """
    ctrl_color = plot_meta[ctrl_tag].get("color", "tab:blue")
    expt_color = plot_meta[expt_tag].get("color", "tab:orange")

    for mode, thresh, tag in [
        ("pos", pos_thresh, f"pos_rho_gt_{pos_thresh:.3f}"),
        ("neg", neg_thresh, f"neg_rho_lt_{neg_thresh:.3f}"),
    ]:
        ctrl_by_lag, C_ctrl = _collect_traces_by_lag(
            ctrl_fish, dir_analysis, sampling_rate, drug_start_frame, wash_end_frame,
            mode=mode, pos_thresh=pos_thresh, neg_thresh=neg_thresh,
            lag_bins_min=lag_bins_min, zscore_cells=zscore_cells,
        )
        expt_by_lag, C_expt = _collect_traces_by_lag(
            expt_fish, dir_analysis, sampling_rate, drug_start_frame, wash_end_frame,
            mode=mode, pos_thresh=pos_thresh, neg_thresh=neg_thresh,
            lag_bins_min=lag_bins_min, zscore_cells=zscore_cells,
        )
        C_ref = C_ctrl if C_ctrl is not None else C_expt
        if C_ref is None:
            print(f"  ⚠️ No C_capsaicin.npy found, skipping mode={mode}")
            continue

        lag_keys = sorted(set(ctrl_by_lag.keys()) | set(expt_by_lag.keys()))
        if not lag_keys:
            print(f"  ⚠️ No lag bins had enough cells for mode={mode}, skipping")
            continue

        n = len(lag_keys)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True)
        if n == 1:
            axes = [axes]

        all_lens = []
        for lb in lag_keys:
            for d in [ctrl_by_lag, expt_by_lag]:
                if lb in d:
                    all_lens.extend([len(x) for x in d[lb]])
        if not all_lens:
            plt.close(fig)
            continue
        common_L = min(all_lens)
        x_trace = np.arange(common_L) / sampling_rate / 60.0

        for ax, lb in zip(axes, lag_keys):
            for traces, color, label in [
                (ctrl_by_lag.get(lb, []), ctrl_color, ctrl_tag),
                (expt_by_lag.get(lb, []), expt_color, expt_tag),
            ]:
                if traces:
                    A = np.vstack([np.asarray(x[:common_L]) for x in traces])
                    m = np.nanmean(A, axis=0)
                    s = _sem(A, axis=0)
                    ax.plot(x_trace, m, lw=2.5, color=color,
                            label=f"{label} (n={A.shape[0]})")
                    ax.fill_between(x_trace, m - s, m + s,
                                    color=color, alpha=0.20, linewidth=0)

            xC, Cseg = _aligned_regressor_segment(
                C=C_ref,
                win_start=int(round(drug_start_frame)),
                win_end=int(round(drug_start_frame)) + common_L,
                lag_sec=float(lb) * 60.0,
                sampling_rate=sampling_rate,
            )
            if xC is not None:
                ax.plot(xC, Cseg, lw=1.5, ls="--", color="blue",
                        alpha=0.8, label=f"C aligned (+{lb} min)")

            ax.axvline(0, color="k", lw=1, ls=":")
            ax.set_title(f"Preferred lag = {lb} min")
            ax.set_xlabel("Time from drug start (min)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("F_tonic (Z-score)")
        axes[0].legend(loc="best", frameon=False)

        thresh_str = (f"rho > {pos_thresh}" if mode == "pos"
                      else f"rho < {neg_thresh}")
        fig.suptitle(
            f"Cells with {thresh_str}, grouped by preferred lag (drug+wash window)",
            y=1.02)
        fig.tight_layout()

        if save:
            out = Path(fig_dir) / f"spearman_mean_traces_{tag}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=300, bbox_inches="tight")
            print(f"  ✅ Saved: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)
