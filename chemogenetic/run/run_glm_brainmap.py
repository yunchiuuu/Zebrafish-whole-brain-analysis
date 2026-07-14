"""
run_glm_brainmap.py
===================
Generate spatial brain maps from GLM responder outputs.

Three maps per group (CTRL | EXPT | EXPT−CTRL):
    1. ΔR² map         — mean ΔR² of responder cells per voxel (how strongly?)
    2. Sign map         — (frac_pos − frac_neg) per voxel (activated vs suppressed?)
    3. Density map      — fraction of all cells that are responders per voxel (where?)

Uses existing brainmap.py voxelization + plotting infrastructure.

Prerequisites:
    run_glm.py       → kernel_delta_r2_fit.npy, responder idx files
    run_medoids.py   → medoids_template_vox.npy

Usage:
    python run_glm_brainmap.py --config config_hcrt_trpv1_csn_120min

Location:
    ~/zwba/chemogenetic/run/run_glm_brainmap.py
"""

import argparse
import gc
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_hcrt_trpv1_csn_120min")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

from utils.data_io import fish_dir
from chemogenetic.brainmap import (
    _coarse_shape,
    _voxelize_cell_values,
    _proj,
)

# ── settings ──────────────────────────────────────────────────────────────────
dir_analysis  = cfg.dir_analysis
EXPT_TAG      = cfg.EXPT_TAG
CTRL_TAG      = cfg.CTRL_TAG
EXPT_FISH     = cfg.expt_fish
CTRL_FISH     = cfg.ctrl_fish

NULL_TAG              = cfg.NULL_TAG
RESPONDER_NULL_THRESH = cfg.RESPONDER_NULL_THRESH
K_GLOBAL              = cfg.K_global
input_tag             = cfg.input_tag

# Match run_glm.py: detect HCRT regressor
HCRT_BASE = "/resnick/home/ychiu/yun/lightsheet/hcrt-trpv1_hcrt-h2b-g8m_120min"
HCRT_FISH_PATHS = [
    f"{HCRT_BASE}/260426_hcrt-trpv1_hcrt-h2b-g8m_csn_10uM_fish{i}/hcrt_all.csv"
    for i in [1, 2, 3]
]
if all(Path(p).exists() for p in HCRT_FISH_PATHS):
    input_tag = "HCRT"
    print(f"  HCRT regressor detected → input_tag={input_tag}")

DS          = (5, 5, 2)
BRAIN_SHAPE = (288, 568, 40)

COMPARISON_FIG_DIR = (
    Path(dir_analysis) / "comparisons" / cfg.COMPARISON_TAG / "figures"
)
COMPARISON_FIG_DIR.mkdir(parents=True, exist_ok=True)

Xc, Yc, Zc = _coarse_shape(BRAIN_SHAPE, DS)

print(f"GLM brain maps | config: {args.config}")
print(f"  null={NULL_TAG} p{RESPONDER_NULL_THRESH}  input_tag={input_tag}")
print(f"  DS={DS}  BRAIN_SHAPE={BRAIN_SHAPE}")
print()


# ── per-fish loader ───────────────────────────────────────────────────────────

def _get_glm_run_dir(fish):
    """Construct GLM run directory path for a fish."""
    f_dir = fish_dir(dir_analysis, fish)
    param_tag = (f"in{input_tag}_K{K_GLOBAL}_drift{cfg.drift_global}"
                 f"_lam{cfg.lam_global}_lag{cfg.lag_global}")
    return f_dir, f_dir / "glm" / param_tag


def _load_fish_glm_data(fish):
    """
    Load GLM outputs for one fish.

    Returns dict with:
        coords     : (n_cells, 3) template voxel coords
        dR2        : (n_cells,) ΔR² values
        pos_idx    : indices of positive responders
        neg_idx    : indices of negative responders
        n_cells    : total cell count
    Or None if files missing.
    """
    f_dir, run_dir = _get_glm_run_dir(fish)
    proj_id, expt_id = fish

    coords_path = f_dir / "medoids_template_vox.npy"
    dR2_path    = run_dir / "kernel_delta_r2_fit.npy"
    ptag        = int(RESPONDER_NULL_THRESH)
    pos_path    = f_dir / f"tonic_pos_glm_{NULL_TAG}_nullp{ptag}_idxs.npy"
    neg_path    = f_dir / f"tonic_neg_glm_{NULL_TAG}_nullp{ptag}_idxs.npy"

    for p, name in [(coords_path, "medoids"), (dR2_path, "ΔR²"),
                    (pos_path, "pos_idx"), (neg_path, "neg_idx")]:
        if not p.exists():
            print(f"  ⚠️  {expt_id}: {name} missing — skipping")
            return None

    coords  = np.load(str(coords_path))                         # (n_cells, 3)
    dR2     = np.load(str(dR2_path)).astype(np.float32).ravel()  # (n_cells,)
    pos_idx = np.load(str(pos_path))
    neg_idx = np.load(str(neg_path))

    print(f"  {expt_id}: {pos_idx.size:,} pos + {neg_idx.size:,} neg "
          f"/ {dR2.size:,} total")

    return {
        "coords":  coords,
        "dR2":     dR2,
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "n_cells": dR2.size,
    }


# ── voxelize per fish ─────────────────────────────────────────────────────────

def _voxelize_fish(data):
    """
    Compute three voxelized maps for one fish:
        dR2_vol      — mean ΔR² of responders per voxel
        sign_vol     — (frac_pos - frac_neg) per voxel, range [-1, +1]
        density_vol  — fraction of all cells that are responders per voxel
    """
    coords  = data["coords"]
    dR2     = data["dR2"]
    pos_idx = data["pos_idx"]
    neg_idx = data["neg_idx"]
    n_cells = data["n_cells"]

    valid = np.isfinite(coords).all(axis=1)
    coords_v = coords[valid].astype(np.int32)

    # 1. ΔR² map — responders only, mean per voxel
    resp_idx = np.union1d(pos_idx, neg_idx)
    resp_mask = np.zeros(n_cells, dtype=bool)
    resp_mask[resp_idx] = True
    resp_and_valid = resp_mask & valid

    dR2_resp = np.full(n_cells, np.nan, dtype=np.float32)
    dR2_resp[resp_and_valid] = dR2[resp_and_valid]
    dR2_vol = _voxelize_cell_values(coords_v, dR2_resp[valid], BRAIN_SHAPE, DS)

    # 2. Sign map — +1 for pos, -1 for neg, 0 for non-responder
    sign_arr = np.zeros(n_cells, dtype=np.float32)
    sign_arr[pos_idx] = 1.0
    sign_arr[neg_idx] = -1.0
    sign_vol = _voxelize_cell_values(coords_v, sign_arr[valid], BRAIN_SHAPE, DS)

    # 3. Density map — 1 for responder, 0 for non-responder → mean = fraction
    density_arr = np.zeros(n_cells, dtype=np.float32)
    density_arr[resp_idx] = 1.0
    density_vol = _voxelize_cell_values(coords_v, density_arr[valid], BRAIN_SHAPE, DS)

    return dR2_vol, sign_vol, density_vol


# ── process groups ────────────────────────────────────────────────────────────

def _process_group(fish_list, group_tag):
    """Load, voxelize, and average across fish."""
    print(f"\n── {group_tag} ({len(fish_list)} fish) ──")

    dR2_list     = []
    sign_list    = []
    density_list = []

    for fish in fish_list:
        data = _load_fish_glm_data(fish)
        if data is None:
            continue

        dR2_v, sign_v, dens_v = _voxelize_fish(data)

        if dR2_v is not None:
            dR2_list.append(dR2_v)
        if sign_v is not None:
            sign_list.append(sign_v)
        if dens_v is not None:
            density_list.append(dens_v)

        del data
        gc.collect()

    def _avg(vols):
        if not vols:
            return None
        return np.nanmean(np.stack(vols, axis=0), axis=0).astype(np.float32)

    return {
        "dR2":     _avg(dR2_list),
        "sign":    _avg(sign_list),
        "density": _avg(density_list),
    }


# ── run ───────────────────────────────────────────────────────────────────────
n_expt = len(EXPT_FISH)
n_ctrl = len(CTRL_FISH)

expt_vols = _process_group(EXPT_FISH, EXPT_TAG)
ctrl_vols = _process_group(CTRL_FISH, CTRL_TAG)

# EXPT − CTRL difference
diff_vols = {}
for key in ("dR2", "sign", "density"):
    e, c = expt_vols[key], ctrl_vols[key]
    if e is not None and c is not None:
        diff_vols[key] = (e - c).astype(np.float32)
    else:
        diff_vols[key] = None


# ── plotting ──────────────────────────────────────────────────────────────────

MAP_CONFIGS = [
    {
        "key": "dR2",
        "label": "Mean ΔR² (responders)",
        "cmap": "hot",
        "vmin": 0.0,
        "vmax": None,   # auto from data
        "diverging": False,
    },
    {
        "key": "sign",
        "label": "Sign (pos − neg fraction)",
        "cmap": "bwr",
        "vmin": -0.5,
        "vmax": 0.5,
        "diverging": True,
    },
    {
        "key": "density",
        "label": "Responder density (fraction)",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": None,
        "diverging": False,
    },
]


def _auto_vmax(vol, percentile=99):
    if vol is None:
        return 0.1
    flat = vol[np.isfinite(vol)]
    return float(np.percentile(np.abs(flat), percentile)) if flat.size else 0.1


# One figure per metric: 3 columns [CTRL | EXPT | EXPT−CTRL] × 2 rows [XY | YZ]
for mcfg in MAP_CONFIGS:
    key   = mcfg["key"]
    label = mcfg["label"]
    cmap  = mcfg["cmap"]
    div   = mcfg["diverging"]

    vols = {
        f"{CTRL_TAG}\n(N={n_ctrl})": ctrl_vols[key],
        f"{EXPT_TAG}\n(N={n_expt})": expt_vols[key],
        "EXPT − CTRL": diff_vols[key],
    }

    # determine vmax
    if mcfg["vmax"] is not None:
        vmax = mcfg["vmax"]
    else:
        vmax = max(_auto_vmax(v) for v in vols.values())
    vmin = mcfg["vmin"] if not div else -vmax

    if div:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    n_cols = len(vols)
    GAP_W  = max(1, int(Xc * 0.08))
    col_widths = []
    for i in range(n_cols):
        col_widths.extend([Xc, Zc])
        if i < n_cols - 1:
            col_widths.append(GAP_W)

    scale = 2.0 / Xc
    fig_w = scale * sum(col_widths) + 1.6
    fig_h = scale * Yc + 1.2
    fig   = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        1, len(col_widths),
        width_ratios = col_widths,
        wspace = 0.02,
        left   = 0.10,
        right  = 0.88,
        top    = 0.85,
        bottom = 0.05,
    )

    last_im = None
    col_idx = 0
    for gi, (gtag, vol) in enumerate(vols.items()):
        ax_xy = fig.add_subplot(gs[0, col_idx])
        ax_yz = fig.add_subplot(gs[0, col_idx + 1])

        ax_xy.set_title(gtag, fontsize=9, fontweight="bold", pad=3)

        if vol is not None and np.isfinite(vol).any():
            xy = _proj(vol, axis=2, mode="mean")
            if np.isfinite(xy).any():
                last_im = ax_xy.imshow(
                    xy.T, cmap=cmap, norm=norm,
                    origin="upper", aspect="equal",
                    interpolation="nearest",
                )

            yz = _proj(vol, axis=0, mode="mean")
            if np.isfinite(yz).any():
                last_im = ax_yz.imshow(
                    yz, cmap=cmap, norm=norm,
                    origin="upper", aspect="equal",
                    interpolation="nearest",
                )

        ax_xy.axis("off")
        ax_yz.axis("off")

        # advance past XY + YZ + GAP
        col_idx += 3 if gi < n_cols - 1 else 2

    # colorbar
    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.10, 0.016, 0.70])
        cbar    = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"GLM {label} | {args.config} | null p{RESPONDER_NULL_THRESH}",
        fontsize=12, fontweight="bold",
    )

    fname = f"glm_{key}_brainmap.png"
    fig.savefig(str(COMPARISON_FIG_DIR / fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {fname}")

print(f"\nAll GLM brain maps saved → {COMPARISON_FIG_DIR}")
