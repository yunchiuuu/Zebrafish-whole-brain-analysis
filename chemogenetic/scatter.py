"""
scatter.py
==========
Cross-modal scatter plots for tonic vs phasic comparison:
    - plot_tonic_phasic_scatter : per-fish mean tonic ΔZ vs phasic d′
                                  for pos and neg responders separately

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/visualization/scatter.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ============================================================
# TONIC ΔZ vs PHASIC D′ SCATTER
# ============================================================

def plot_tonic_phasic_scatter(
    ctrl_tonic_pos,  ctrl_dprime_pos,
    ctrl_tonic_neg,  ctrl_dprime_neg,
    expt_tonic_pos,  expt_dprime_pos,
    expt_tonic_neg,  expt_dprime_neg,
    ctrl_meta, expt_meta,
    null_tag, null_percentile,
    amplitude_mode="raw",
    fig_dir=None,
    figsize=(6, 6),
    save=True,
    show=True,
):
    """
    Scatter plot of per-fish mean tonic ΔZ (x-axis) vs phasic d′ (y-axis),
    separately for pos and neg responders.

    Each point is one fish. Ctrl and expt fish are differentiated by marker
    style (open vs filled) and color.

    Parameters
    ----------
    ctrl_tonic_pos : array-like, shape (n_ctrl_fish,)
        Per-fish mean fixed-window or plateau ΔZ over pos responder cells.
    ctrl_dprime_pos : array-like, shape (n_ctrl_fish,)
        Per-fish mean d′ over pos responder cells.
    ... same pattern for neg, expt groups.
    ctrl_meta, expt_meta : dict
        From config.PLOT_META.
    null_tag : str
    null_percentile : int
    amplitude_mode : str
        Label for the d′ y-axis.
    fig_dir : Path or None
    save, show : bool
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, sign, \
        (c_x, c_y), (e_x, e_y) in zip(
            axes,
            ["Positive responders", "Negative responders"],
            [(ctrl_tonic_pos, ctrl_dprime_pos),
             (ctrl_tonic_neg, ctrl_dprime_neg)],
            [(expt_tonic_pos, expt_dprime_pos),
             (expt_tonic_neg, expt_dprime_neg)],
        ):

        c_x = np.asarray(c_x, dtype=float)
        c_y = np.asarray(c_y, dtype=float)
        e_x = np.asarray(e_x, dtype=float)
        e_y = np.asarray(e_y, dtype=float)

        # ctrl: open markers
        ok_c = np.isfinite(c_x) & np.isfinite(c_y)
        ax.scatter(
            c_x[ok_c], c_y[ok_c],
            s=60,
            facecolors="none",
            edgecolors=ctrl_meta["color"],
            linewidths=1.8,
            alpha=ctrl_meta["alpha"],
            label=ctrl_meta["label"],
            zorder=3,
        )

        # expt: filled markers
        ok_e = np.isfinite(e_x) & np.isfinite(e_y)
        ax.scatter(
            e_x[ok_e], e_y[ok_e],
            s=60,
            color=expt_meta["color"],
            alpha=expt_meta["alpha"],
            label=expt_meta["label"],
            zorder=3,
        )

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)

        ax.set_xlabel("Tonic ΔZ (mean over responder cells)")
        ax.set_ylabel(f"Phasic d′ ({amplitude_mode})")
        ax.set_title(sign)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Tonic ΔZ vs Phasic d′ | {null_tag} null p{null_percentile}",
        fontsize=12,
    )
    plt.tight_layout()

    if save and fig_dir is not None:
        out = Path(fig_dir) / \
              f"scatter_tonic_dz_vs_phasic_dprime_{null_tag}_p{null_percentile}.pdf"
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)
