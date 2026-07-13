"""
brainmap.py
===========
Brain-map helpers and plotting for temporal intensity maps.

Helper functions (_coarse_shape, _voxelize_cell_values, _proj, _rotate,
_coarse_shape) are lifted verbatim from the notebook Step 7 / ALL CELLS cell.

plot_group_comparison() is the modular equivalent of the notebook's
final plotting block: 2-row-per-window (XY + YZ), 2-column (CTRL vs EXPT)
GridSpec layout, bwr colormap.

Location:
    ~/zwba/chemogenetic/brainmap.py
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ── settings (match notebook defaults) ────────────────────────────────────
DEFAULT_DS      = (5, 5, 2)    # coarse-grid downsampling (X, Y, Z)
DEFAULT_CMAP    = "bwr"
DEFAULT_VMIN    = -5.0
DEFAULT_VMAX    =  5.0
DEFAULT_PROJ    = "mean"        # projection mode for signed ΔZ
ROTATE_XY_MINUS_90 = True      # notebook default


# ── helpers — lifted verbatim from notebook ────────────────────────────────

def _coarse_shape(brain_shape, ds):
    """Compute coarse-grid shape after downsampling by ds."""
    brain_shape = np.array(brain_shape, dtype=int)
    ds = np.array(ds, dtype=int)
    return tuple(((brain_shape + ds - 1) // ds).astype(int))


def _voxelize_cell_values(coords, vals, brain_shape, ds):
    """
    Bin per-cell scalar values onto a coarse voxel grid.
    Lifted verbatim from notebook ALL CELLS cell.

    Parameters
    ----------
    coords : array (n_cells, 3)
        Template-space voxel coordinates [i, j, k].  NaN = invalid.
    vals : array (n_cells,)
        Per-cell scalar values (ΔZ or d').
    brain_shape : tuple (X, Y, Z)
        Full-resolution template brain shape.
    ds : tuple (dx, dy, dz)
        Downsampling factors per axis.

    Returns
    -------
    out : np.ndarray shape (Xc, Yc, Zc), dtype float32, or None if empty.
    """
    coords = np.asarray(coords)
    vals   = np.asarray(vals)

    mask   = np.isfinite(vals) & (~np.isnan(coords).any(axis=1))
    coords = coords[mask]
    vals   = vals[mask]

    X, Y, Z = brain_shape
    inb = (
        (coords[:, 0] >= 0) & (coords[:, 0] < X) &
        (coords[:, 1] >= 0) & (coords[:, 1] < Y) &
        (coords[:, 2] >= 0) & (coords[:, 2] < Z)
    )
    coords = coords[inb]
    vals   = vals[inb]

    if coords.shape[0] == 0:
        return None

    ds_arr    = np.array(ds, dtype=int)
    Xc, Yc, Zc = _coarse_shape((X, Y, Z), ds_arr)

    cidx    = np.floor(coords / ds_arr).astype(int)
    cidx[:, 0] = np.clip(cidx[:, 0], 0, Xc - 1)
    cidx[:, 1] = np.clip(cidx[:, 1], 0, Yc - 1)
    cidx[:, 2] = np.clip(cidx[:, 2], 0, Zc - 1)

    lin  = (cidx[:, 0] * (Yc * Zc) + cidx[:, 1] * Zc + cidx[:, 2]).astype(np.int64)
    nvox = Xc * Yc * Zc

    sumv = np.zeros(nvox, dtype=np.float64)
    cntv = np.zeros(nvox, dtype=np.int64)
    np.add.at(sumv, lin, vals.astype(np.float64))
    np.add.at(cntv, lin, 1)

    out    = np.full(nvox, np.nan, dtype=np.float32)
    m      = cntv > 0
    out[m] = (sumv[m] / cntv[m]).astype(np.float32)
    return out.reshape((Xc, Yc, Zc))


def _proj(vol, axis, mode="mean"):
    """Projection over one axis. Lifted verbatim from notebook."""
    if mode == "mean":
        return np.nanmean(vol, axis=axis)
    if mode == "max":
        return np.nanmax(vol, axis=axis)
    if mode == "maxabs":
        a    = np.abs(vol)
        a    = np.where(np.isfinite(a), a, -np.inf)
        idx  = np.nanargmax(a, axis=axis)
        vol0 = np.where(np.isfinite(vol), vol, np.nan)
        take = np.take_along_axis(vol0, np.expand_dims(idx, axis=axis), axis=axis)
        return np.squeeze(take, axis=axis)
    raise ValueError(f"Unknown proj mode: {mode!r}")


def _rotate(arr):
    """XY rotation applied before display. Lifted verbatim from notebook."""
    return np.rot90(arr, k=3) if ROTATE_XY_MINUS_90 else arr


# ── main plotting function ─────────────────────────────────────────────────

def plot_group_comparison(
    ctrl_maps:      list,          # len = N_WINDOWS, each (Xc, Yc, Zc) or None
    expt_maps:      list,          # len = N_WINDOWS, each (Xc, Yc, Zc) or None
    window_labels:  list[str],
    brain_shape:    tuple,         # full-res template (X, Y, Z)
    DS:             tuple  = DEFAULT_DS,
    ctrl_tag:       str    = "CTRL",
    expt_tag:       str    = "EXPT",
    metric_label:   str    = "Tonic ΔZ (window mean z) vs baseline",
    title:          str    = "",
    cmap:           str    = DEFAULT_CMAP,
    vmin:           float  = DEFAULT_VMIN,
    vmax:           float  = DEFAULT_VMAX,
    proj_mode:      str    = DEFAULT_PROJ,
    fig_dir:        Path | None = None,
    filename:       str    = "temporal_intensity_map.png",
    show:           bool   = False,
) -> plt.Figure:
    """
    Plot temporal intensity maps in the notebook style:
        - 2 rows per window: top = XY projection (dorsal), bottom = YZ projection (lateral)
        - 2 columns: CTRL (left) vs EXPT (right)
        - Window labels as rotated text on the left spine
        - GridSpec with height_ratios proportional to coarse-grid dimensions

    Parameters
    ----------
    ctrl_maps, expt_maps : list of np.ndarray or None
        Per-window coarse-grid volumes from voxelize step.
        None for windows with no data.
    window_labels : list of str
        Label per window, e.g. ["0–15 min", "10–25 min", ...].
    brain_shape : tuple (X, Y, Z)
        Full-resolution template brain dimensions (used for axis sizing).
    DS : tuple (dx, dy, dz)
        Downsampling factors (same as used in voxelization).
    """
    n_windows = len(window_labels)
    titles    = [ctrl_tag, expt_tag]
    maps_list = [ctrl_maps, expt_maps]
    n_cols    = 2

    Xc, Yc, Zc = _coarse_shape(brain_shape, DS)

    # height_ratios: alternate Xc (XY row) and Zc (YZ row) for each window
    height_ratios = []
    for _ in range(n_windows):
        height_ratios.extend([Xc, Zc])

    fig = plt.figure(figsize=(5 * n_cols, 2.0 * n_windows))
    gs  = GridSpec(
        nrows         = 2 * n_windows,
        ncols         = n_cols,
        height_ratios = height_ratios,
        hspace        = 0.1,
        wspace        = 0.05,
        top           = 0.88,
        bottom        = 0.05,
    )

    axs = np.empty((2 * n_windows, n_cols), dtype=object)
    for r in range(2 * n_windows):
        for c in range(n_cols):
            axs[r, c] = fig.add_subplot(gs[r, c])

    last_im = None

    for wi in range(n_windows):
        row_xy = 2 * wi
        row_yz = 2 * wi + 1

        for j, (key, maps) in enumerate(zip(titles, maps_list)):
            vol = maps[wi]

            if vol is None:
                axs[row_xy, j].axis("off")
                axs[row_yz, j].axis("off")
                continue

            # ── XY: mean projection over Z (axis=2) ──────────────────────
            xy = _proj(vol, axis=2, mode=proj_mode)
            if not np.isfinite(xy).any():
                axs[row_xy, j].axis("off")
            else:
                xy      = _rotate(xy)
                xy_disp = np.rot90(xy, k=4).T
                axs[row_xy, j].imshow(
                    xy_disp,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    origin="upper", aspect="equal", interpolation="nearest",
                    extent=[0, Yc, Xc, 0],
                )
                axs[row_xy, j].axis("off")
                axs[row_xy, j].set_xlim(0, Yc)
                if wi == 0:
                    axs[row_xy, j].set_title(key, fontsize=14, pad=10)

            # ── YZ: mean projection over X (axis=0) ──────────────────────
            yz = _proj(vol, axis=0, mode=proj_mode)
            if not np.isfinite(yz).any():
                axs[row_yz, j].axis("off")
            else:
                yz      = yz.T
                yz      = np.rot90(yz, k=3).T
                yz_disp = np.flipud(yz)

                last_im = axs[row_yz, j].imshow(
                    yz_disp,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    origin="upper", aspect="equal", interpolation="nearest",
                    extent=[0, Yc, Zc, 0],
                )
                axs[row_yz, j].axis("off")
                axs[row_yz, j].invert_yaxis()
                axs[row_yz, j].set_xlim(0, Yc)

        # ── window label on left spine (between XY and YZ rows) ──────────
        pos_top  = axs[row_xy, 0].get_position()
        pos_bot  = axs[row_yz, 0].get_position()
        y_center = (pos_top.y1 + pos_bot.y0) / 2

        fig.text(
            0.02, y_center, window_labels[wi],
            va="center", ha="right", rotation=90,
            fontsize=12, fontweight="bold",
            transform=fig.transFigure,
        )

    # ── colorbar ─────────────────────────────────────────────────────────
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axs.ravel().tolist(),
            fraction=0.02,
            pad=0.02,
        )
        cbar.set_label(metric_label, fontsize=10)
    else:
        print("⚠️  No finite maps — colorbar skipped.")

    if title:
        fig.suptitle(title, y=0.96, fontsize=14)

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
