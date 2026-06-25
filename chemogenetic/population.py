"""
population.py
=============
Group-level population plots comparing ctrl vs experimental fish:
    - plot_responder_fractions : boxplot of pos/neg responder fraction per fish
    - plot_dz_boxplot          : fixed-window ΔZ boxplot per group
    - plot_plateau_dz_boxplot  : plateau ΔZ boxplot per group
    - plot_dprime_boxplot      : phasic d′ boxplot per group

All functions take fish lists (tuples) + preloaded per-fish arrays,
so they can be called from run_figures.py after loading.
They never load data themselves — data loading is the run code's job.

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/population.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.stats import mw_test_and_stars, add_sig_bar


# ============================================================
# SHARED HELPERS
# ============================================================

def _boxplot_two_groups(
    ax,
    ctrl_pos, ctrl_neg,
    expt_pos, expt_neg,
    ctrl_meta, expt_meta,
    ylabel, title,
    positions=(0, 1, 2, 3),
    seed=0,
):
    """
    Four-box layout: CTRL(+), EXPT(+), CTRL(-), EXPT(-)
    with jittered scatter overlay and MWU significance bars.
    """
    data   = [ctrl_pos, expt_pos, ctrl_neg, expt_neg]
    groups = [ctrl_meta, expt_meta, ctrl_meta, expt_meta]

    bp = ax.boxplot(
        data,
        positions=list(positions),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )

    for patch, meta in zip(bp["boxes"], groups):
        patch.set_facecolor(meta["color"])
        patch.set_alpha(meta["alpha"])
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(seed)
    for x0, yvals, meta in zip(positions, data, groups):
        if len(yvals) == 0:
            continue
        x = x0 + rng.uniform(-0.12, 0.12, size=len(yvals))
        ax.scatter(x, yvals, s=38, color=meta["color"], alpha=meta["alpha"],
                   edgecolor="black", linewidth=0.5, zorder=3)

    ax.axhline(0, ls="--", lw=1.2, color="black", alpha=0.4, zorder=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # significance bars
    nonempty = [d for d in data if len(d) > 0]
    if not nonempty:
        return

    y_all  = np.concatenate(nonempty)
    ymax   = float(np.nanmax(y_all))
    ymin   = float(np.nanmin(y_all))
    yrng   = (ymax - ymin) if ymax > ymin else 1.0

    p_pos, stars_pos = mw_test_and_stars(ctrl_pos, expt_pos)
    p_neg, stars_neg = mw_test_and_stars(ctrl_neg, expt_neg)

    y1 = ymax + 0.10 * yrng
    y2 = ymax + 0.28 * yrng

    add_sig_bar(ax, positions[0], positions[1], y1,
                f"{stars_pos} (p={p_pos:.1e})" if np.isfinite(p_pos) else "n/a")
    add_sig_bar(ax, positions[2], positions[3], y2,
                f"{stars_neg} (p={p_neg:.1e})" if np.isfinite(p_neg) else "n/a")

    ax.set_ylim(ymin - 0.10 * yrng, ymax + 0.50 * yrng)


# ============================================================
# RESPONDER FRACTION
# ============================================================

def plot_responder_fractions(
    ctrl_pos_fracs, ctrl_neg_fracs,
    expt_pos_fracs, expt_neg_fracs,
    ctrl_meta, expt_meta,
    null_tag, null_percentile,
    fig_dir=None,
    figsize=(7, 5),
    save=True,
    show=True,
):
    """
    Boxplot of responder fractions (pos/neg) per fish for ctrl and expt.

    Parameters
    ----------
    ctrl_pos_fracs, ctrl_neg_fracs : array-like
        Fraction of pos/neg responder cells per ctrl fish.
    expt_pos_fracs, expt_neg_fracs : array-like
        Same for experimental fish.
    ctrl_meta, expt_meta : dict
        From config.PLOT_META — must have keys 'label', 'color', 'alpha'.
    null_tag : str
    null_percentile : int
    fig_dir : Path or None
        If None and save=True, figure is not saved.
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [
        f"{ctrl_meta['label']} (+)",
        f"{expt_meta['label']} (+)",
        f"{ctrl_meta['label']} (−)",
        f"{expt_meta['label']} (−)",
    ]

    _boxplot_two_groups(
        ax,
        np.asarray(ctrl_pos_fracs, dtype=float),
        np.asarray(ctrl_neg_fracs, dtype=float),
        np.asarray(expt_pos_fracs, dtype=float),
        np.asarray(expt_neg_fracs, dtype=float),
        ctrl_meta, expt_meta,
        ylabel="Fraction of cells",
        title=f"Tonic responder fraction ({null_tag} null p{null_percentile})",
    )

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels)
    plt.tight_layout()

    if save and fig_dir is not None:
        out = Path(fig_dir) / f"responder_fraction_{null_tag}_p{null_percentile}.pdf"
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# FIXED-WINDOW ΔZ BOXPLOT
# ============================================================

def plot_dz_boxplot(
    ctrl_pos_dz, ctrl_neg_dz,
    expt_pos_dz, expt_neg_dz,
    ctrl_meta, expt_meta,
    null_tag, null_percentile,
    baseline_min_pair, drug_min_pair,
    fig_dir=None,
    figsize=(7, 5),
    save=True,
    show=True,
):
    """
    Boxplot of per-fish mean fixed-window ΔZ for pos/neg responders.

    Each element of the input arrays is one fish's mean ΔZ
    (mean over that fish's responder cells).
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [
        f"{ctrl_meta['label']} (+)",
        f"{expt_meta['label']} (+)",
        f"{ctrl_meta['label']} (−)",
        f"{expt_meta['label']} (−)",
    ]

    b0, b1 = baseline_min_pair
    d0, d1 = drug_min_pair

    _boxplot_two_groups(
        ax,
        np.asarray(ctrl_pos_dz, dtype=float),
        np.asarray(ctrl_neg_dz, dtype=float),
        np.asarray(expt_pos_dz, dtype=float),
        np.asarray(expt_neg_dz, dtype=float),
        ctrl_meta, expt_meta,
        ylabel="Fish mean ΔZ over responder cells",
        title=(
            f"Tonic ΔZ ({null_tag} null p{null_percentile})\n"
            f"base {b0:g}–{b1:g} min | drug {d0:g}–{d1:g} min"
        ),
    )

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels)
    plt.tight_layout()

    if save and fig_dir is not None:
        out = Path(fig_dir) / (
            f"dz_fixed_window_{null_tag}_p{null_percentile}"
            f"_b{b0:g}-{b1:g}_d{d0:g}-{d1:g}.pdf"
        )
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# PLATEAU ΔZ BOXPLOT
# ============================================================

def plot_plateau_dz_boxplot(
    ctrl_pos_dz, ctrl_neg_dz,
    expt_pos_dz, expt_neg_dz,
    ctrl_meta, expt_meta,
    null_tag, null_percentile,
    L_min,
    fig_dir=None,
    figsize=(7, 5),
    save=True,
    show=True,
):
    """
    Boxplot of per-fish mean plateau ΔZ for pos/neg responders.
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [
        f"{ctrl_meta['label']} (+)",
        f"{expt_meta['label']} (+)",
        f"{ctrl_meta['label']} (−)",
        f"{expt_meta['label']} (−)",
    ]

    _boxplot_two_groups(
        ax,
        np.asarray(ctrl_pos_dz, dtype=float),
        np.asarray(ctrl_neg_dz, dtype=float),
        np.asarray(expt_pos_dz, dtype=float),
        np.asarray(expt_neg_dz, dtype=float),
        ctrl_meta, expt_meta,
        ylabel=f"Plateau ΔZ (L={L_min:g} min)",
        title=f"Tonic plateau ΔZ ({null_tag} null p{null_percentile})",
    )

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels)
    plt.tight_layout()

    if save and fig_dir is not None:
        out = Path(fig_dir) / \
              f"dz_plateau_L{int(L_min)}min_{null_tag}_p{null_percentile}.pdf"
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# D′ BOXPLOT
# ============================================================

def plot_dprime_boxplot(
    ctrl_pos_dp, ctrl_neg_dp,
    expt_pos_dp, expt_neg_dp,
    ctrl_meta, expt_meta,
    amplitude_mode,
    fig_dir=None,
    figsize=(7, 5),
    save=True,
    show=True,
):
    """
    Boxplot of per-fish mean phasic d′ split by tonic responder sign.

    Each element is one fish's mean d′ over its pos (or neg) responder cells.
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [
        f"{ctrl_meta['label']} (+)",
        f"{expt_meta['label']} (+)",
        f"{ctrl_meta['label']} (−)",
        f"{expt_meta['label']} (−)",
    ]

    _boxplot_two_groups(
        ax,
        np.asarray(ctrl_pos_dp, dtype=float),
        np.asarray(ctrl_neg_dp, dtype=float),
        np.asarray(expt_pos_dp, dtype=float),
        np.asarray(expt_neg_dp, dtype=float),
        ctrl_meta, expt_meta,
        ylabel=f"Fish mean d′ ({amplitude_mode})",
        title=f"Phasic d′ by tonic responder sign (amplitude_mode={amplitude_mode!r})",
    )

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(labels)
    plt.tight_layout()

    if save and fig_dir is not None:
        out = Path(fig_dir) / f"dprime_{amplitude_mode}_by_tonic_sign.pdf"
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)
