"""
kernel_map.py
=============
Compute per-cell kernel shape metrics from GLM outputs and generate
spatial brain maps showing where different response types are located.

Five metrics computed per responder cell:
    tau_star     : lag of peak |h_i| (0–K vols)
    is_early     : τ* in 0–K/2 (0–5 min for K=600)
    is_middle    : τ* in K/2–K (5–10 min for K=600)
    is_transient : clear single peak + kernel returns toward zero at late lags
    is_biphasic  : kernel changes sign between early and late halves

Brain maps generated:
    1. Lag map             — mean τ* per voxel (sequential colormap)
    2. Early density map   — fraction of early responders per voxel
    3. Middle density map  — fraction of middle responders per voxel
    4. Transient density   — fraction of transient responders per voxel
    5. Biphasic density    — fraction of biphasic responders per voxel

Location:
    ~/zwba/chemogenetic/kernel_map.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from chemogenetic.brainmap import (
    _coarse_shape,
    _voxelize_cell_values,
    _proj,
)


# ── kernel metric computation ──────────────────────────────────────────────────

def compute_kernel_metrics(
    h: np.ndarray,
    K_early: int | None = None,
    transient_ratio_thresh: float = 0.65,
    biphasic_abs_thresh: float = 0.05,
) -> dict[str, np.ndarray]:
    """
    Compute per-cell kernel shape metrics from the FIR kernel matrix.

    Parameters
    ----------
    h : np.ndarray, shape (n_cells, K)
        FIR kernel weights from GLM refit (kernel_h_hat.npy).
        Only call this on GLM responder cells (already thresholded by ΔR²).
    K_early : int or None
        Lag index separating "early" from "middle/late".
        Default = K // 2 (splits kernel in half).
    transient_ratio_thresh : float
        Fraction of absolute kernel weight in early half that defines transient.
        Cell is transient if |early_mean| / (|early_mean| + |late_mean|) > thresh.
    biphasic_abs_thresh : float
        Minimum absolute mean weight in BOTH halves required to call a cell biphasic.
        Guards against noise-driven sign flips in weak kernels.

    Returns
    -------
    dict with keys:
        tau_star       : (n_cells,) int    — lag of peak |h_i|
        early_weight   : (n_cells,) float  — mean signed weight in early half
        late_weight    : (n_cells,) float  — mean signed weight in late half
        is_early       : (n_cells,) bool   — τ* in early half
        is_middle      : (n_cells,) bool   — τ* in late half
        is_transient   : (n_cells,) bool
        is_biphasic    : (n_cells,) bool
    """
    h        = np.asarray(h, dtype=np.float32)
    n_cells, K = h.shape
    K_e      = K_early if K_early is not None else K // 2

    tau_star     = np.argmax(np.abs(h), axis=1).astype(np.int32)
    early_weight = h[:, :K_e].mean(axis=1)
    late_weight  = h[:, K_e:].mean(axis=1)

    abs_early = np.abs(early_weight)
    abs_late  = np.abs(late_weight)
    total_abs = abs_early + abs_late + 1e-9

    is_early  = tau_star < K_e
    is_middle = tau_star >= K_e

    # transient: dominated by early lags
    is_transient = (abs_early / total_abs) > transient_ratio_thresh

    # biphasic: sign flip with both halves above noise threshold
    is_biphasic = (
        (np.sign(early_weight) != np.sign(late_weight)) &
        (abs_early > biphasic_abs_thresh) &
        (abs_late  > biphasic_abs_thresh)
    )

    return {
        "tau_star":     tau_star,
        "early_weight": early_weight,
        "late_weight":  late_weight,
        "is_early":     is_early,
        "is_middle":    is_middle,
        "is_transient": is_transient,
        "is_biphasic":  is_biphasic,
    }


# ── voxelization helpers ───────────────────────────────────────────────────────

def voxelize_kernel_metrics(
    coords:  np.ndarray,
    metrics: dict,
    brain_shape: tuple,
    DS:      tuple,
) -> dict[str, np.ndarray | None]:
    """
    Voxelize kernel metrics into coarse brain grid.

    Returns dict with same keys as metrics, each mapped to a coarse volume.
    Boolean metrics (is_*) are converted to fraction of True cells per voxel.
    tau_star is converted to mean lag (in vols) per voxel.
    """
    from chemogenetic.brainmap import _voxelize_cell_values

    valid = np.isfinite(coords).all(axis=1)
    coords_v = coords[valid].astype(np.int32)

    result = {}
    for key, vals in metrics.items():
        vals_v = np.asarray(vals, dtype=np.float32)[valid]
        result[key] = _voxelize_cell_values(coords_v, vals_v, brain_shape, DS)

    return result


# ── plotting ───────────────────────────────────────────────────────────────────

_LAG_CMAP   = "plasma"    # sequential for lag time
_FRAC_CMAP  = "YlOrRd"   # sequential for density/fraction


def plot_kernel_maps(
    group_voxels:  dict[str, dict[str, np.ndarray | None]],
    window_label:  str,
    brain_shape:   tuple,
    DS:            tuple,
    K:             int,
    sampling_rate_hz: float = 1.0,
    fig_dir:       Path | None = None,
    filename:      str  = "kernel_maps.png",
    show:          bool = False,
) -> plt.Figure:
    """
    Plot 5 kernel metric maps side by side for each group.

    Layout: rows = groups, cols = [lag map | early | middle | transient | biphasic]

    Parameters
    ----------
    group_voxels : dict {group_tag: {metric_key: coarse_volume}}
        Output of voxelize_kernel_metrics() per group.
    window_label : str
        Label for the figure (e.g. "HCRT-TRPV1 (N=9)").
    K : int
        Kernel length in vols (used to set lag axis scale).
    """
    Xc, Yc, Zc = _coarse_shape(brain_shape, DS)
    K_mins      = K / sampling_rate_hz / 60   # kernel duration in minutes

    groups     = list(group_voxels.keys())
    n_groups   = len(groups)
    map_cols   = ["tau_star", "is_early", "is_middle", "is_transient", "is_biphasic"]
    col_labels = [
        f"Lag map\n(0–{K_mins:.0f} min)",
        "Early\n(0–5 min)",
        "Middle\n(5–10 min)",
        "Transient",
        "Biphasic",
    ]
    cmaps  = [_LAG_CMAP, _FRAC_CMAP, _FRAC_CMAP, _FRAC_CMAP, _FRAC_CMAP]
    vmaxes = [K,         1.0,        1.0,         1.0,         1.0       ]
    vmins  = [0,         0.0,        0.0,          0.0,         0.0      ]
    n_cols = len(map_cols)

    scale  = 1.4 / Xc
    fig_w  = scale * Xc * n_cols + 1.2
    fig_h  = scale * Yc * n_groups + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_groups, n_cols,
        hspace=0.06, wspace=0.04,
        top=0.90, bottom=0.06,
        left=0.12, right=0.88,
    )

    last_ims = [None] * n_cols

    for gi, gtag in enumerate(groups):
        vox = group_voxels[gtag]

        for ci, (mkey, clabel, cmap, vmin, vmax) in enumerate(
            zip(map_cols, col_labels, cmaps, vmins, vmaxes)
        ):
            ax  = fig.add_subplot(gs[gi, ci])
            vol = vox.get(mkey)

            if gi == 0:
                ax.set_title(clabel, fontsize=8, pad=3, fontweight="bold")

            if vol is not None and np.isfinite(vol).any():
                xy  = _proj(vol, axis=2, mode="mean")
                im  = ax.imshow(
                    xy.T,
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    origin="upper", aspect="equal",
                    interpolation="nearest",
                )
                last_ims[ci] = im

            ax.axis("off")

        # group label
        fig.text(
            gs[gi, 0].get_position(fig).x0 - 0.01,
            gs[gi, 0].get_position(fig).y0
            + gs[gi, 0].get_position(fig).height / 2,
            gtag, va="center", ha="right",
            fontsize=8, fontweight="bold",
        )

    # colorbars — one per column
    for ci, (im, mkey, vmin, vmax) in enumerate(
        zip(last_ims, map_cols, vmins, vmaxes)
    ):
        if im is None:
            continue
        # small colorbar below each column
        pos0 = gs[n_groups - 1, ci].get_position(fig)
        cbar_ax = fig.add_axes([
            pos0.x0, pos0.y0 - 0.035,
            pos0.width, 0.012,
        ])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=6)
        if mkey == "tau_star":
            cbar.set_label("lag (vols)", fontsize=6)
        else:
            cbar.set_label("fraction", fontsize=6)

    fig.suptitle(f"Kernel shape maps | {window_label}", fontsize=11,
                 fontweight="bold")

    if fig_dir is not None:
        out = Path(fig_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  ✅ Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
