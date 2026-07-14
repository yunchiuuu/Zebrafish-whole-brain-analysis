"""
run_combined_fish_map.py
========================
Combine individual fish temporal ΔZ maps into:
    1. combined_temporal_dz.png  — all fish stacked, one row per fish
    2. combined_temporal_dz.pdf  — one fish per page (for detailed inspection)

Rows = fish, Cols = time windows.
Group boundaries (main vs inj cohort) marked with a horizontal separator.

Usage:
    python run_combined_fish_map.py --config config_hcrt_trpv1_pooled_csn_120min

Location:
    ~/zwba/chemogenetic/run/run_combined_fish_map.py
"""

import argparse
import gc
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_hcrt_trpv1_pooled_csn_120min")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

from utils.data_io import fish_dir
from chemogenetic.temporal_windows import (
    load_epoch_dz,
    load_epoch_dprime,
    N_WINDOWS,
    WINDOW_LABELS,
)
from chemogenetic.brainmap import (
    _coarse_shape,
    _voxelize_cell_values,
    _proj,
)

# ── settings ──────────────────────────────────────────────────────────────────
dir_analysis  = cfg.dir_analysis
DS            = (5, 5, 2)
BRAIN_SHAPE   = (288, 568, 40)
CLIP_ABS_DZ   = cfg.CLIP_ABS_DZ
VMIN_DZ, VMAX_DZ = -3.0,  3.0
VMIN_DP, VMAX_DP = -0.6,  0.6
CMAP          = "bwr"

METRICS = {
    "dz":     {"loader": load_epoch_dz,     "vmin": VMIN_DZ, "vmax": VMAX_DZ,
               "label": "Tonic ΔZ",   "suffix": "dz"},
    "dprime": {"loader": load_epoch_dprime, "vmin": VMIN_DP, "vmax": VMAX_DP,
               "label": "Phasic d′",  "suffix": "dprime"},
}

# build ordered fish list with group tags
# works for any config that has expt_fish + ctrl_fish, or just expt_fish
FISH_GROUPS = []
if hasattr(cfg, "expt_fish_csn") and hasattr(cfg, "expt_fish_csn_inj"):
    # pooled config: separate the two cohorts
    for fish in cfg.expt_fish_csn:
        FISH_GROUPS.append((fish, "HCRT-TRPV1 (main)"))
    for fish in cfg.expt_fish_csn_inj:
        FISH_GROUPS.append((fish, "HCRT-TRPV1 (inj)"))
else:
    for fish in cfg.expt_fish:
        FISH_GROUPS.append((fish, cfg.EXPT_TAG))

n_fish = len(FISH_GROUPS)
print(f"Combining {n_fish} fish | config: {args.config}")

# output directory: comparisons folder
from pathlib import Path as _P
out_dir = _P(dir_analysis) / "comparisons" / cfg.COMPARISON_TAG / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# ── pre-compute all voxelized maps ─────────────────────────────────────────────
Xc, Yc, Zc = _coarse_shape(BRAIN_SHAPE, DS)

all_fish_data = []   # list of (fish, group_tag, {metric: [vols]}, n_ok)

for fish, group_tag in FISH_GROUPS:
    proj_id, expt_id = fish
    f_dir = fish_dir(dir_analysis, fish)

    medoids_path = f_dir / "medoids_template_vox.npy"

    if not medoids_path.exists():
        print(f"  ⚠️  {expt_id}: medoids missing — inserting blank row")
        all_fish_data.append((fish, group_tag, {m: [None]*N_WINDOWS for m in METRICS}, 0))
        continue

    coords = np.load(str(medoids_path))
    valid  = np.isfinite(coords).all(axis=1)
    coords_v = coords[valid].astype(np.int32)

    metric_vols = {}
    for metric, mcfg in METRICS.items():
        arr_path = f_dir / f"epoch_{metric}.npy"
        if not arr_path.exists():
            print(f"  ⚠️  {expt_id}: epoch_{metric}.npy missing")
            metric_vols[metric] = [None] * N_WINDOWS
            continue
        arr   = mcfg["loader"](f_dir)
        arr_v = arr[valid]
        vols  = []
        for w in range(N_WINDOWS):
            vals = np.clip(arr_v[:, w], -CLIP_ABS_DZ, CLIP_ABS_DZ)
            vols.append(_voxelize_cell_values(coords_v, vals, BRAIN_SHAPE, DS))
        metric_vols[metric] = vols
        del arr, arr_v, vols

    all_fish_data.append((fish, group_tag, metric_vols, int(valid.sum())))
    print(f"  {expt_id}: {int(valid.sum()):,} cells")

    del coords
    gc.collect()

# ── helper: plot one row of brains ─────────────────────────────────────────────
def _plot_row(axs_row, vols, norm, last_im_ref):
    last_im = last_im_ref
    for wi, (ax, vol) in enumerate(zip(axs_row, vols)):
        ax.axis("off")
        if vol is None or not np.isfinite(vol).any():
            continue
        xy = _proj(vol, axis=2, mode="mean")
        if np.isfinite(xy).any():
            last_im = ax.imshow(
                xy.T, cmap=CMAP, norm=norm,
                origin="upper", aspect="equal",
                interpolation="nearest",
            )
    return last_im


# ── generate figures for each metric ──────────────────────────────────────────
for metric, mcfg in METRICS.items():
    vmin      = mcfg["vmin"]
    vmax      = mcfg["vmax"]
    m_label   = mcfg["label"]
    suffix    = mcfg["suffix"]
    norm      = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    n_fish    = len(all_fish_data)

    print(f"\n── {m_label} ──")

    # ── 1. COMBINED STACKED PNG ───────────────────────────────────────────────
    scale  = 1.0 / Xc
    row_h  = scale * Yc
    fig_w  = scale * Xc * N_WINDOWS + 1.8
    fig_h  = row_h * n_fish + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_fish, N_WINDOWS,
        hspace = 0.06, wspace = 0.04,
        top=0.95, bottom=0.02, left=0.30, right=0.90,
    )

    last_im    = None
    prev_group = None

    for ri, (fish, group_tag, metric_vols, n_ok) in enumerate(all_fish_data):
        proj_id, expt_id = fish

        if prev_group is not None and group_tag != prev_group:
            pos = gs[ri, 0].get_position(fig)
            fig.add_artist(plt.Line2D(
                [0.29, 0.91], [pos.y1 + 0.005, pos.y1 + 0.005],
                transform=fig.transFigure,
                color="black", lw=1.5, ls="--",
            ))

        prev_group = group_tag

        axs_row = [fig.add_subplot(gs[ri, wi]) for wi in range(N_WINDOWS)]
        last_im = _plot_row(axs_row, metric_vols[metric], norm, last_im)

        if ri == 0:
            for wi, ax in enumerate(axs_row):
                ax.set_title(WINDOW_LABELS[wi], fontsize=7, pad=2,
                             fontweight="bold")

        fig.text(
            0.29,
            gs[ri, 0].get_position(fig).y0
            + gs[ri, 0].get_position(fig).height / 2,
            f"{expt_id}\n({n_ok:,})",
            va="center", ha="right",
            fontsize=5.5, fontfamily="monospace",
            multialignment="center",
        )

    groups_seen = {}
    for ri, (fish, group_tag, _, _) in enumerate(all_fish_data):
        if group_tag not in groups_seen:
            groups_seen[group_tag] = ri
    for group_tag, first_ri in groups_seen.items():
        last_ri = max(i for i, (_, g, _, _) in enumerate(all_fish_data)
                      if g == group_tag)
        y_mid   = (gs[first_ri, 0].get_position(fig).y1
                   + gs[last_ri,  0].get_position(fig).y0) / 2
        fig.text(0.01, y_mid, group_tag, va="center", ha="left",
                 fontsize=8, fontweight="bold", rotation=90, color="dimgray")

    if last_im is not None:
        cbar_ax = fig.add_axes([0.91, 0.08, 0.015, 0.82])
        cbar    = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(m_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Individual fish {m_label} | {args.config} | DS={DS} | clip={CLIP_ABS_DZ}",
        fontsize=10, fontweight="bold",
    )

    png_out = out_dir / f"combined_temporal_{suffix}.png"
    fig.savefig(str(png_out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ PNG → {png_out}")

    # ── 2. PER-FISH PDF ───────────────────────────────────────────────────────
    pdf_out = out_dir / f"combined_temporal_{suffix}.pdf"
    with PdfPages(str(pdf_out)) as pdf:
        for fish, group_tag, metric_vols, n_ok in all_fish_data:
            proj_id, expt_id = fish

            fig2 = plt.figure(figsize=(N_WINDOWS * 1.8, 4.2))
            gs2  = GridSpec(1, N_WINDOWS,
                            hspace=0.02, wspace=0.04,
                            top=0.82, bottom=0.05,
                            left=0.02, right=0.90)

            axs_row  = [fig2.add_subplot(gs2[0, wi]) for wi in range(N_WINDOWS)]
            last_im2 = _plot_row(axs_row, metric_vols[metric], norm, None)

            for wi, ax in enumerate(axs_row):
                ax.set_title(WINDOW_LABELS[wi], fontsize=8, pad=2,
                             fontweight="bold")

            if last_im2 is not None:
                cbar_ax2 = fig2.add_axes([0.91, 0.10, 0.015, 0.65])
                cbar2    = fig2.colorbar(last_im2, cax=cbar_ax2)
                cbar2.set_label(m_label, fontsize=8)
                cbar2.ax.tick_params(labelsize=7)

            fig2.suptitle(
                f"{expt_id}  [{group_tag}]  ({n_ok:,} cells)",
                fontsize=10, fontweight="bold",
            )
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    print(f"  ✅ PDF → {pdf_out}")

print("\nDone.")
