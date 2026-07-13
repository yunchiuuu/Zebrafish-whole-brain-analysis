"""
run_temporal_intensity_map.py
==============================
Analysis 1: Temporal intensity maps of tonic ΔZ and phasic d'
across 7 temporal windows post-capsaicin onset.

Plotting structure lifted from notebook cell:
    "【ALL CELLS】 ALL CELLS Averaged Tonic Activity Change Effect Size
     (Averaged Intensity Map)"

    Layout: 2 rows per window (XY dorsal + YZ lateral), 2 columns (CTRL vs EXPT)
    Coarse voxelization: DS=(5,5,2)
    Colormap: bwr, vmin=-5, vmax=5

Prerequisite pipeline:
    run_decompose.py    →  f_tonic.npy, f_phasic.npy   (per fish)
    run_medoids.py      →  medoids_template_vox.npy    (per fish, one-time)

This script:
    Cell 1 — compute epoch_dz.npy + epoch_dprime.npy per fish (if not cached)
    Cell 2 — voxelize per fish (parallel), average across fish per group
    Cell 3 — plot and save figures

Output figures (written to comparisons/{COMPARISON_TAG}/figures/):
    {GROUP}_temporal_dz.png
    {GROUP}_temporal_dprime.png
    EXPT_minus_CTRL_temporal_dz.png
    EXPT_minus_CTRL_temporal_dprime.png

Location:
    ~/zwba/chemogenetic/run/run_temporal_intensity_map.py
"""

# %% ── Cell 0: imports + config ───────────────────────────────────────────

import gc
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ── set config name for interactive use; override via --config for sbatch ──
CONFIG_NAME = "config_hcrt_trpv1_csn_120min"   # ← edit as needed
cfg = importlib.import_module(f"chemogenetic.config.{CONFIG_NAME}")

from utils.data_io import fish_dir
from chemogenetic.temporal_windows import (
    compute_epoch_dz,
    compute_epoch_dprime,
    load_ftonic,
    load_fphasic,
    save_epoch_dz,
    save_epoch_dprime,
    load_epoch_dz,
    load_epoch_dprime,
    N_WINDOWS,
    WINDOW_LABELS,
)
from chemogenetic.brainmap import (
    _coarse_shape,
    _voxelize_cell_values,
    plot_group_comparison,
)

# ── directories + fish lists ──────────────────────────────────────────────
dir_analysis = cfg.dir_analysis
EXPT_FISH    = cfg.expt_fish_csn
CTRL_FISH    = cfg.ctrl_fish_csn
EXPT_TAG     = cfg.EXPT_TAG
CTRL_TAG     = cfg.CTRL_TAG

COMPARISON_FIG_DIR = (
    Path(dir_analysis) / "comparisons" / cfg.COMPARISON_TAG / "figures"
)
COMPARISON_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── USER SETTINGS (match notebook defaults) ───────────────────────────────
DS     = (5, 5, 2)       # coarse-grid downsampling (X, Y, Z)
N_JOBS = 25              # parallel workers for voxelization

VMIN_DZ = -5.0
VMAX_DZ =  5.0
VMIN_DP = -5.0
VMAX_DP =  5.0

# Template brain shape (X, Y, Z) in full-resolution template voxels.
# Verify with: ants.image_read(TEMPLATE_PATH).shape
BRAIN_SHAPE = (280, 544, 40)   # ← verify against your template

RECOMPUTE_EPOCHS = False   # True → recompute even if epoch arrays exist


# %% ── Cell 1: compute epoch ΔZ + d' for all fish ─────────────────────────

print("=" * 60)
print("Analysis 1 — Temporal intensity maps  (Cell 1: epoch metrics)")
print(f"  N_WINDOWS = {N_WINDOWS}  |  DS = {DS}")
print("=" * 60)

all_groups = {EXPT_TAG: EXPT_FISH, CTRL_TAG: CTRL_FISH}

for group_tag, fish_list in all_groups.items():
    print(f"\n── {group_tag} ({len(fish_list)} fish) ──")
    for fish in fish_list:
        proj_id, expt_id = fish
        f_dir = fish_dir(dir_analysis, fish)
        f_dir.mkdir(parents=True, exist_ok=True)

        dz_path = f_dir / "epoch_dz.npy"
        dp_path = f_dir / "epoch_dprime.npy"

        if dz_path.exists() and dp_path.exists() and not RECOMPUTE_EPOCHS:
            print(f"  ⏩ {expt_id}: epoch arrays cached")
            continue

        print(f"  {expt_id}: computing epochs...")
        f_tonic  = load_ftonic(f_dir)
        f_phasic = load_fphasic(f_dir)

        dz, _, _ = compute_epoch_dz(f_tonic)
        dprime   = compute_epoch_dprime(f_phasic)

        save_epoch_dz(dz, f_dir)
        save_epoch_dprime(dprime, f_dir)
        del f_tonic, f_phasic, dz, dprime
        gc.collect()

print("\nCell 1 complete.")


# %% ── Cell 2: voxelize per fish (parallel) + average across fish ─────────

print("=" * 60)
print("Analysis 1 — Cell 2: voxelize + group average")
print("=" * 60)


def _process_one_fish(fish, metric: str) -> list:
    """
    Load epoch metric + template medoids for one fish, voxelize per window.
    Returns list of N_WINDOWS coarse-grid volumes (or None per window).

    Parameters
    ----------
    fish : (proj_id, expt_id) tuple
    metric : "dz" or "dprime"
    """
    f_dir  = fish_dir(dir_analysis, fish)
    coords_path = f_dir / "medoids_template_vox.npy"

    if not coords_path.exists():
        print(f"  ⚠️  medoids_template_vox.npy missing for {fish[1]} — skipping")
        return [None] * N_WINDOWS

    coords = np.load(str(coords_path))   # (n_cells, 3), NaN = invalid

    if metric == "dz":
        arr = load_epoch_dz(f_dir)       # (n_cells, N_WINDOWS)
    else:
        arr = load_epoch_dprime(f_dir)   # (n_cells, N_WINDOWS)

    if arr.shape[0] != coords.shape[0]:
        print(f"  ⚠️  shape mismatch for {fish[1]}: "
              f"arr={arr.shape[0]} coords={coords.shape[0]} — skipping")
        return [None] * N_WINDOWS

    vols = []
    for w in range(N_WINDOWS):
        vol = _voxelize_cell_values(coords, arr[:, w], BRAIN_SHAPE, DS)
        vols.append(vol)

    del coords, arr
    gc.collect()
    return vols


def _run_group(fish_list, metric: str, group_tag: str) -> list:
    """
    Voxelize all fish in a group in parallel, then nanmean across fish.
    Returns list of N_WINDOWS averaged coarse volumes (or None).
    """
    print(f"  🚀 {group_tag} ({len(fish_list)} fish)...")
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_process_one_fish)(fish, metric)
        for fish in tqdm(fish_list, desc=f"{group_tag} {metric}")
    )

    avg_maps = []
    for w in range(N_WINDOWS):
        fish_vols = [res[w] for res in results if res[w] is not None]
        if not fish_vols:
            avg_maps.append(None)
        else:
            stack = np.stack(fish_vols, axis=0)
            mean  = np.nanmean(stack, axis=0).astype(np.float32)
            avg_maps.append(mean if np.isfinite(mean).any() else None)

    return avg_maps


# ── run for both metrics ──────────────────────────────────────────────────
group_maps = {}   # group_maps[metric][tag] = list of N_WINDOWS volumes

for metric in ("dz", "dprime"):
    group_maps[metric] = {}
    print(f"\n── metric: {metric} ──")
    for tag, fish_list in all_groups.items():
        group_maps[metric][tag] = _run_group(fish_list, metric, tag)

print("\nCell 2 complete.")


# %% ── Cell 3: plot and save figures ──────────────────────────────────────

print("=" * 60)
print("Analysis 1 — Cell 3: figures")
print("=" * 60)

n_expt = len(EXPT_FISH)
n_ctrl = len(CTRL_FISH)

# ── per-group tonic ΔZ ───────────────────────────────────────────────────
for tag, fish_list in all_groups.items():
    n = len(fish_list)
    plot_group_comparison(
        ctrl_maps    = group_maps["dz"][CTRL_TAG],
        expt_maps    = group_maps["dz"][EXPT_TAG],
        window_labels= WINDOW_LABELS,
        brain_shape  = BRAIN_SHAPE,
        DS           = DS,
        ctrl_tag     = f"{CTRL_TAG} (N={n_ctrl})",
        expt_tag     = f"{EXPT_TAG} (N={n_expt})",
        metric_label = "Tonic ΔZ (window mean z-score vs baseline)",
        title        = f"Tonic ΔZ | DS={DS} | 15min window 10min step",
        vmin         = VMIN_DZ,
        vmax         = VMAX_DZ,
        fig_dir      = COMPARISON_FIG_DIR,
        filename     = "temporal_dz_CTRL_vs_EXPT.png",
    )
    break   # one figure covers both groups (cols = CTRL vs EXPT)

# ── per-group phasic d' ───────────────────────────────────────────────────
plot_group_comparison(
    ctrl_maps    = group_maps["dprime"][CTRL_TAG],
    expt_maps    = group_maps["dprime"][EXPT_TAG],
    window_labels= WINDOW_LABELS,
    brain_shape  = BRAIN_SHAPE,
    DS           = DS,
    ctrl_tag     = f"{CTRL_TAG} (N={n_ctrl})",
    expt_tag     = f"{EXPT_TAG} (N={n_expt})",
    metric_label = "Phasic d′ (window mean z-score vs baseline)",
    title        = f"Phasic d′ | DS={DS} | 15min window 10min step",
    vmin         = VMIN_DP,
    vmax         = VMAX_DP,
    fig_dir      = COMPARISON_FIG_DIR,
    filename     = "temporal_dprime_CTRL_vs_EXPT.png",
)

# ── EXPT − CTRL difference maps ───────────────────────────────────────────
for metric, label, vmin, vmax, fname in [
    ("dz",     "Tonic ΔZ (EXPT − CTRL)",     VMIN_DZ, VMAX_DZ,
     "temporal_dz_EXPT_minus_CTRL.png"),
    ("dprime", "Phasic d′ (EXPT − CTRL)",    VMIN_DP, VMAX_DP,
     "temporal_dprime_EXPT_minus_CTRL.png"),
]:
    diff = []
    for w in range(N_WINDOWS):
        e = group_maps[metric][EXPT_TAG][w]
        c = group_maps[metric][CTRL_TAG][w]
        if e is not None and c is not None:
            diff.append((e - c).astype(np.float32))
        else:
            diff.append(None)

    # for the difference map, both columns show the same diff
    # (re-use plot_group_comparison with identical maps and retitle cols)
    plot_group_comparison(
        ctrl_maps    = diff,
        expt_maps    = diff,
        window_labels= WINDOW_LABELS,
        brain_shape  = BRAIN_SHAPE,
        DS           = DS,
        ctrl_tag     = "EXPT − CTRL",
        expt_tag     = "EXPT − CTRL",
        metric_label = label,
        title        = f"{label} | DS={DS}",
        vmin         = vmin,
        vmax         = vmax,
        fig_dir      = COMPARISON_FIG_DIR,
        filename     = fname,
    )

print(f"\nAll figures saved → {COMPARISON_FIG_DIR}")
