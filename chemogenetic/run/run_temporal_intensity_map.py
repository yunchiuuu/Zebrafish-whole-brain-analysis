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

import argparse
import gc
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ── config: --config arg for sbatch, fallback to hardcoded for interactive ──
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_hcrt_trpv1_csn_120min",
                    help="Config module under chemogenetic/config/")
args, _ = parser.parse_known_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

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
    TEMPORAL_WINDOWS,
)
from chemogenetic.brainmap import (
    _coarse_shape,
    _voxelize_cell_values,
    plot_group_comparison,
)

# ── directories + fish lists ──────────────────────────────────────────────
dir_analysis  = cfg.dir_analysis
EXPT_FISH     = cfg.expt_fish
CTRL_FISH     = cfg.ctrl_fish
EXPT_TAG      = cfg.EXPT_TAG
CTRL_TAG      = cfg.CTRL_TAG
CLIP_ABS_DZ   = cfg.CLIP_ABS_DZ   # clip extreme ΔZ before voxelization

COMPARISON_FIG_DIR = (
    Path(dir_analysis) / "comparisons" / cfg.COMPARISON_TAG / "figures"
)
COMPARISON_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── USER SETTINGS (match notebook defaults) ───────────────────────────────
DS     = (5, 5, 2)       # coarse-grid downsampling (X, Y, Z)
N_JOBS = 25              # parallel workers for voxelization

VMIN_DZ = -3.0
VMAX_DZ =  3.0
VMIN_DP = -0.6
VMAX_DP =  0.6

# Template brain shape (X, Y, Z) in full-resolution template voxels.
BRAIN_SHAPE = (288, 568, 40)   # confirmed from template_mean_brain.nii.gz

RECOMPUTE_EPOCHS = False   # True → recompute even if epoch arrays exist

# ── HCRT reference traces — lifted from hcrt_regressor.ipynb ─────────────
# Source: raw mean HCRT fluorescence from hcrt_all.csv per fish
# Pipeline: F_tonic  = rolling 20th-percentile (600 vols, centered)
#            F_phasic = (F - F_tonic) / F_tonic
#            Both z-scored on baseline (vols 0:CSN_ONSET_VOL)

import pandas as pd

HCRT_BASE = "/resnick/home/ychiu/yun/lightsheet/hcrt-trpv1_hcrt-h2b-g8m_120min"
HCRT_FISH_PATHS = [
    f"{HCRT_BASE}/260426_hcrt-trpv1_hcrt-h2b-g8m_csn_10uM_fish1/hcrt_all.csv",
    f"{HCRT_BASE}/260426_hcrt-trpv1_hcrt-h2b-g8m_csn_10uM_fish2/hcrt_all.csv",
    f"{HCRT_BASE}/260426_hcrt-trpv1_hcrt-h2b-g8m_csn_10uM_fish3/hcrt_all.csv",
]
HCRT_FISH_LABELS   = ["fish1", "fish2", "fish3"]
HCRT_TONIC_WINDOW  = 600    # vols (~10 min), matches voluseg convention
HCRT_TONIC_PCTILE  = 0.20
CSN_ONSET_VOL      = 2700
TOTAL_VOLS_HCRT    = 7200

hcrt_df        = None
hcrt_window_dz = None
hcrt_window_dp = None

if all(Path(p).exists() for p in HCRT_FISH_PATHS):
    fish_tonic  = {}
    fish_phasic = {}

    for label, path in zip(HCRT_FISH_LABELS, HCRT_FISH_PATHS):
        raw = pd.read_csv(path)["Mean"].values   # raw HCRT mean fluorescence

        # F_tonic: rolling 20th-percentile, centered — matches notebook exactly
        f_tonic = (pd.Series(raw)
                     .rolling(HCRT_TONIC_WINDOW, center=True, min_periods=1)
                     .quantile(HCRT_TONIC_PCTILE)
                     .values)

        # F_phasic: (F - F_tonic) / F_tonic
        f_phasic = (raw - f_tonic) / np.maximum(f_tonic, 1e-6)

        # z-score each on baseline (vols 0:CSN_ONSET_VOL)
        for sig, store in [(f_tonic, fish_tonic), (f_phasic, fish_phasic)]:
            mu    = sig[:CSN_ONSET_VOL].mean()
            sigma = sig[:CSN_ONSET_VOL].std(ddof=1)
            store[label] = (sig - mu) / max(sigma, 1e-6)

        print(f"  {label}: tonic bl z={fish_tonic[label][:CSN_ONSET_VOL].mean():.4f}  "
              f"phasic bl z={fish_phasic[label][:CSN_ONSET_VOL].mean():.4f}")

    t_min      = np.arange(TOTAL_VOLS_HCRT) / 60.0
    tonic_mat  = np.stack([fish_tonic[l]  for l in HCRT_FISH_LABELS])  # (3, T)
    phasic_mat = np.stack([fish_phasic[l] for l in HCRT_FISH_LABELS])  # (3, T)
    n_hcrt     = tonic_mat.shape[0]

    hcrt_df = pd.DataFrame({
        "time_min":        t_min,
        "hcrt_tonic_z":   tonic_mat.mean(axis=0),
        "hcrt_tonic_sem":  tonic_mat.std(axis=0, ddof=1) / np.sqrt(n_hcrt),
        "hcrt_phasic_z":  phasic_mat.mean(axis=0),
        "hcrt_phasic_sem": phasic_mat.std(axis=0, ddof=1) / np.sqrt(n_hcrt),
    })
    for label in HCRT_FISH_LABELS:
        hcrt_df[f"tonic_{label}"]  = fish_tonic[label]
        hcrt_df[f"phasic_{label}"] = fish_phasic[label]

    # per-window mean z during window (already baseline-normalised via z-score)
    hcrt_window_dz = [float(hcrt_df["hcrt_tonic_z"].values[s:e].mean())
                      for _, s, e in TEMPORAL_WINDOWS]
    hcrt_window_dp = [float(hcrt_df["hcrt_phasic_z"].values[s:e].mean())
                      for _, s, e in TEMPORAL_WINDOWS]

    print(f"HCRT tonic  ΔZ per window: {[f'{v:.2f}' for v in hcrt_window_dz]}")
    print(f"HCRT phasic d' per window: {[f'{v:.2f}' for v in hcrt_window_dp]}")
else:
    print("⚠️  One or more hcrt_all.csv not found — HCRT row will be skipped")
    print(f"    Expected base: {HCRT_BASE}")

HCRT_WINDOW_VOLS = [(s, e) for _, s, e in TEMPORAL_WINDOWS]


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
        vals = np.clip(arr[:, w], -CLIP_ABS_DZ, CLIP_ABS_DZ)
        vol  = _voxelize_cell_values(coords, vals, BRAIN_SHAPE, DS)
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


# ── run for both tonic ΔZ and phasic d' ──────────────────────────────────
group_maps = {}

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

def _diff_maps(metric):
    """Compute per-window EXPT − CTRL difference volumes."""
    diff = []
    for w in range(N_WINDOWS):
        e = group_maps[metric][EXPT_TAG][w]
        c = group_maps[metric][CTRL_TAG][w]
        diff.append((e - c).astype(np.float32) if e is not None and c is not None else None)
    return diff

# ── tonic ΔZ: CTRL | EXPT | EXPT−CTRL ───────────────────────────────────
plot_group_comparison(
    ctrl_maps         = group_maps["dz"][CTRL_TAG],
    expt_maps         = group_maps["dz"][EXPT_TAG],
    diff_maps         = _diff_maps("dz"),
    window_labels     = WINDOW_LABELS,
    brain_shape       = BRAIN_SHAPE,
    DS                = DS,
    ctrl_tag          = f"{CTRL_TAG}\n(N={n_ctrl})",
    expt_tag          = f"{EXPT_TAG}\n(N={n_expt})",
    diff_tag          = "EXPT − CTRL",
    metric_label      = "Tonic ΔZ",
    title             = f"Tonic ΔZ | DS={DS} | 15min window 10min step | clip={CLIP_ABS_DZ}",
    vmin              = VMIN_DZ,
    vmax              = VMAX_DZ,
    hcrt_df           = hcrt_df,
    hcrt_trace_col    = "hcrt_tonic_z",
    hcrt_sem_col      = "hcrt_tonic_sem",
    hcrt_fish_cols    = [f"tonic_{l}" for l in HCRT_FISH_LABELS],
    hcrt_trace_label  = "Hcrt F_tonic",
    hcrt_window_vols  = HCRT_WINDOW_VOLS,
    fig_dir           = COMPARISON_FIG_DIR,
    filename          = "temporal_dz_CTRL_vs_EXPT.png",
)

# ── phasic d': CTRL | EXPT | EXPT−CTRL ───────────────────────────────────
plot_group_comparison(
    ctrl_maps         = group_maps["dprime"][CTRL_TAG],
    expt_maps         = group_maps["dprime"][EXPT_TAG],
    diff_maps         = _diff_maps("dprime"),
    window_labels     = WINDOW_LABELS,
    brain_shape       = BRAIN_SHAPE,
    DS                = DS,
    ctrl_tag          = f"{CTRL_TAG}\n(N={n_ctrl})",
    expt_tag          = f"{EXPT_TAG}\n(N={n_expt})",
    diff_tag          = "EXPT − CTRL",
    metric_label      = "Phasic d′",
    title             = f"Phasic d′ | DS={DS} | 15min window 10min step",
    vmin              = VMIN_DP,
    vmax              = VMAX_DP,
    hcrt_df           = hcrt_df,
    hcrt_trace_col    = "hcrt_phasic_z",
    hcrt_sem_col      = "hcrt_phasic_sem",
    hcrt_fish_cols    = [f"phasic_{l}" for l in HCRT_FISH_LABELS],
    hcrt_trace_label  = "Hcrt F_phasic",
    hcrt_window_vols  = HCRT_WINDOW_VOLS,
    fig_dir           = COMPARISON_FIG_DIR,
    filename          = "temporal_dprime_CTRL_vs_EXPT.png",
)

print(f"\nAll figures saved → {COMPARISON_FIG_DIR}")
