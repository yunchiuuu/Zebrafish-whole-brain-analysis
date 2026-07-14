"""
run_kernel_map.py
=================
Generate kernel shape brain maps from GLM outputs.

For each group (CTRL, EXPT), loads GLM kernels for responder cells only,
computes per-cell kernel shape metrics, voxelizes to template space,
and plots 5 spatial maps:
    1. Lag map      — where in the brain are early vs late responders?
    2. Early map    — density of cells with peak response in 0–5 min
    3. Middle map   — density of cells with peak response in 5–10 min
    4. Transient    — density of transient (onset-only) responders
    5. Biphasic     — density of cells with sign flip in kernel

Prerequisites:
    run_glm.py       →  kernel_h_hat.npy per fish
    run_medoids.py   →  medoids_template_vox.npy per fish

Usage:
    python run_kernel_map.py --config config_hcrt_trpv1_csn_120min

Location:
    ~/zwba/chemogenetic/run/run_kernel_map.py
"""

import argparse
import gc
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_hcrt_trpv1_csn_120min")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

from utils.data_io import fish_dir
from chemogenetic.brainmap import _coarse_shape, _voxelize_cell_values
from chemogenetic.kernel_map import (
    compute_kernel_metrics,
    voxelize_kernel_metrics,
    plot_kernel_maps,
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

# Match run_glm.py: if HCRT csv files exist, GLM was run with input_tag="HCRT"
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

# kernel metric thresholds
K_EARLY                = K_GLOBAL // 2    # split at 5 min (300 vols for K=600)
TRANSIENT_RATIO_THRESH = 0.65
BIPHASIC_ABS_THRESH    = 0.05

COMPARISON_FIG_DIR = (
    Path(dir_analysis) / "comparisons" / cfg.COMPARISON_TAG / "figures"
)
COMPARISON_FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Kernel shape maps | config: {args.config}")
print(f"  K={K_GLOBAL}  K_early={K_EARLY}  DS={DS}")
print(f"  null={NULL_TAG} p{RESPONDER_NULL_THRESH}")
print()


def _load_responder_kernels(fish, n_pos_only=True):
    """
    Load kernel_h_hat.npy for GLM responder cells of one fish.

    Returns (h_responders, coords_responders) or (None, None) if missing.
    n_pos_only=True: only positive responders (activated by HCRT).
                    False: both pos and neg responders combined.
    """
    from chemogenetic.glm import make_param_tag, get_run_dir

    f_dir   = fish_dir(dir_analysis, fish)
    run_dir = f_dir / "glm" / make_param_tag(
        K_GLOBAL, cfg.drift_global, cfg.lam_global, input_tag, cfg.lag_global
    )

    h_path = run_dir / "kernel_h_hat.npy"
    if not h_path.exists():
        print(f"  ⚠️  {fish[1]}: kernel_h_hat.npy missing — skipping")
        return None, None

    # load responder indices
    ptag   = int(RESPONDER_NULL_THRESH)
    pos_p  = f_dir / f"tonic_pos_glm_{NULL_TAG}_nullp{ptag}_idxs.npy"
    neg_p  = f_dir / f"tonic_neg_glm_{NULL_TAG}_nullp{ptag}_idxs.npy"

    if not pos_p.exists():
        print(f"  ⚠️  {fish[1]}: responder idx missing — skipping")
        return None, None

    pos_idx = np.load(str(pos_p))
    if n_pos_only:
        resp_idx = pos_idx
    else:
        neg_idx  = np.load(str(neg_p)) if neg_p.exists() else np.array([], dtype=np.int64)
        resp_idx = np.unique(np.concatenate([pos_idx, neg_idx]))

    if resp_idx.size == 0:
        return None, None

    h_all = np.load(str(h_path), mmap_mode="r")   # (n_cells, K)
    h_resp = h_all[resp_idx].astype(np.float32)

    # load medoid coords for these cells
    coords_path = f_dir / "medoids_template_vox.npy"
    if not coords_path.exists():
        print(f"  ⚠️  {fish[1]}: medoids_template_vox.npy missing — skipping")
        return None, None

    coords_all  = np.load(str(coords_path))     # (n_cells, 3)
    coords_resp = coords_all[resp_idx]

    print(f"  {fish[1]}: {resp_idx.size:,} responders loaded")
    return h_resp, coords_resp


def _process_group(fish_list, group_tag) -> dict[str, np.ndarray | None]:
    """
    Process all fish in a group: compute metrics, voxelize, average.
    Returns dict {metric_key: averaged_coarse_volume}.
    """
    print(f"\n── {group_tag} ──")
    metric_keys = ["tau_star", "is_early", "is_middle", "is_transient", "is_biphasic"]
    fish_vols   = {k: [] for k in metric_keys}

    for fish in fish_list:
        h_resp, coords_resp = _load_responder_kernels(fish, n_pos_only=False)
        if h_resp is None:
            continue

        metrics = compute_kernel_metrics(
            h_resp,
            K_early               = K_EARLY,
            transient_ratio_thresh= TRANSIENT_RATIO_THRESH,
            biphasic_abs_thresh   = BIPHASIC_ABS_THRESH,
        )

        vox = voxelize_kernel_metrics(coords_resp, metrics, BRAIN_SHAPE, DS)

        for k in metric_keys:
            if vox.get(k) is not None:
                fish_vols[k].append(vox[k])

        del h_resp, coords_resp, metrics, vox
        gc.collect()

    # average across fish
    group_mean = {}
    for k in metric_keys:
        vlist = fish_vols[k]
        if vlist:
            group_mean[k] = np.nanmean(np.stack(vlist, axis=0), axis=0).astype(np.float32)
        else:
            group_mean[k] = None
        print(f"    {k}: {len(vlist)} fish contributed")

    return group_mean


# ── run both groups ────────────────────────────────────────────────────────────
group_voxels = {
    f"{EXPT_TAG} (N={len(EXPT_FISH)})": _process_group(EXPT_FISH, EXPT_TAG),
    f"{CTRL_TAG} (N={len(CTRL_FISH)})": _process_group(CTRL_FISH, CTRL_TAG),
}

# ── plot ───────────────────────────────────────────────────────────────────────
plot_kernel_maps(
    group_voxels  = group_voxels,
    window_label  = f"{args.config} | K={K_GLOBAL} | null p{RESPONDER_NULL_THRESH}",
    brain_shape   = BRAIN_SHAPE,
    DS            = DS,
    K             = K_GLOBAL,
    fig_dir       = COMPARISON_FIG_DIR,
    filename      = "kernel_shape_maps.png",
)

# ── also plot EXPT only with YZ side views ─────────────────────────────────────
# separate figure per metric for publication-quality detail
from chemogenetic.brainmap import _proj
import matplotlib.pyplot as plt

for metric, label, cmap in [
    ("tau_star",    f"Lag map (0–{K_GLOBAL//60} min)",     "plasma"),
    ("is_early",    "Early responders (0–5 min)",          "YlOrRd"),
    ("is_middle",   "Middle responders (5–10 min)",        "YlOrRd"),
    ("is_transient","Transient responders",                 "YlOrRd"),
    ("is_biphasic", "Biphasic responders",                  "YlOrRd"),
]:
    expt_key = f"{EXPT_TAG} (N={len(EXPT_FISH)})"
    vol = group_voxels[expt_key].get(metric)
    if vol is None:
        continue

    Xc, Yc, Zc = _coarse_shape(BRAIN_SHAPE, DS)
    scale  = 2.0 / Xc
    fig, axes = plt.subplots(1, 2, figsize=(scale * (Xc + Zc) + 0.8, scale * Yc + 0.5),
                              gridspec_kw={"width_ratios": [Xc, Zc], "wspace": 0.03})

    vmax = K_GLOBAL if metric == "tau_star" else 1.0

    xy = _proj(vol, axis=2, mode="mean")
    axes[0].imshow(xy.T, cmap=cmap, vmin=0, vmax=vmax,
                   origin="upper", aspect="equal", interpolation="nearest")
    axes[0].axis("off")
    axes[0].set_title("XY dorsal", fontsize=8)

    yz = _proj(vol, axis=0, mode="mean")
    im = axes[1].imshow(yz, cmap=cmap, vmin=0, vmax=vmax,
                        origin="upper", aspect="equal", interpolation="nearest")
    axes[1].axis("off")
    axes[1].set_title("YZ side", fontsize=8)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar    = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("lag (vols)" if metric == "tau_star" else "fraction", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{EXPT_TAG} | {label}", fontsize=11, fontweight="bold")

    fname = f"kernel_{metric}_{EXPT_TAG.replace(' ', '_')}.png"
    fig.savefig(str(COMPARISON_FIG_DIR / fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {fname}")

print(f"\nAll kernel maps saved → {COMPARISON_FIG_DIR}")
