"""
run_decompose.py
================
SLURM entry point: compute F_tonic and F_phasic for all fish.

Reads raw voluseg HDF5 from dir_voluseg, writes .npy arrays to dir_analysis.
Skips any fish where outputs already exist (set overwrite=True to force).

Pipeline per fish:
    Step 0: estimate camera background F_dark from raw volume corner patches
    Step 1: load voluseg traces, subtract F_dark → F_corrected
    Step 2: compute F_tonic (sliding percentile of F_corrected)
    Step 3: compute F_phasic (F_corrected - F_tonic) / F_tonic

Usage
-----
On HPC (sbatch):
    sbatch submit_decompose.sh

Interactively (login node or salloc session):
    python run_decompose.py

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_decompose.py
"""

import argparse
import gc
import importlib
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make repo root importable regardless of working directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Parse --config argument and load config dynamically
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compute F_tonic and F_phasic for all fish in a config."
)
parser.add_argument(
    "--config", required=True,
    help="Config module name under chemogenetic/config/, e.g. config_hcrt_trpv1_csn_120min"
)
args = parser.parse_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

all_fish            = cfg.all_fish
dir_analysis        = cfg.dir_analysis
dir_voluseg         = cfg.dir_voluseg
f_tonic_percentile  = cfg.f_tonic_percentile
f_tonic_window_size = cfg.f_tonic_window_size
sampling_rate_hz    = cfg.sampling_rate_hz
baseline_end        = cfg.baseline_end
drug_start          = cfg.drug_start
drug_end            = cfg.drug_end
from utils.data_io import fish_dir, read_data
from utils.preprocess import (
    compute_f_tonic,
    compute_f_phasic,
    estimate_background,
    subtract_background,
)

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE_TONIC  = False
OVERWRITE_PHASIC = False

DENOM_MODE   = "legacy"    # "legacy" (matches notebook) or "fixed_floor"
EPS_FLOOR    = 1e-6

CHUNK_CELLS  = 20000
N_JOBS       = 28
DTYPE_OUT    = np.float32

# Background estimation settings
N_BG_VOLUMES = 100    # number of evenly-spaced volumes to sample
BG_PATCH_SIZE = 10    # corner patch size in pixels (10x10)


# ============================================================
# DECOMPOSITION QC
# ============================================================
BL_START_VOL = 1200   # vol 1200 = 20 min — skip habituation transient

def _plot_decompose_qc(expt_ID, fish_out, tonic_path, phasic_path):
    """
    Save a decomposition QC figure for one fish.

    Layout (2 rows × 2 cols):
        Row 0: population mean ± std  (F_tonic left | F_phasic right)
        Row 1: top-5 tonic-ΔZ cells, overlapping z-scored traces
                (F_tonic left | F_phasic right, same 5 cells)

    Cell selection: subsample 20K cells, compute ΔZ = (μ_drug - μ_bl) / σ_bl,
    pick top 5 by |ΔZ|. Z-score uses baseline vols BL_START_VOL:baseline_end.
    Loads from disk — never re-runs decomposition.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not tonic_path.exists() or not phasic_path.exists():
        print(f"  ⚠️  QC skipped — f_tonic or f_phasic not found")
        return

    Ft = np.load(str(tonic_path),  mmap_mode="r")   # (n_cells, T)
    Fp = np.load(str(phasic_path), mmap_mode="r")   # (n_cells, T)
    n_cells, T = Ft.shape
    t = np.arange(T) / (60 * sampling_rate_hz)       # minutes

    bl_s = BL_START_VOL
    bl_e = int(baseline_end)
    dr_s = int(drug_start)
    dr_e = int(drug_end)

    # ── find top-5 tonic-ΔZ cells via subsample ───────────────────────
    rng  = np.random.default_rng(42)
    sub  = rng.choice(n_cells, size=min(20_000, n_cells), replace=False)

    Ft_sub = Ft[sub].astype(np.float32)
    mu_bl  = Ft_sub[:, bl_s:bl_e].mean(axis=1)
    sg_bl  = Ft_sub[:, bl_s:bl_e].std(axis=1).clip(1e-6)
    mu_dr  = Ft_sub[:, dr_s:dr_e].mean(axis=1)
    dz_sub = (mu_dr - mu_bl) / sg_bl

    top5_in_sub = np.argsort(np.abs(dz_sub))[-5:][::-1]
    top5_idx    = sub[top5_in_sub]          # actual cell indices
    top5_dz     = dz_sub[top5_in_sub]

    # ── z-score top-5 traces ──────────────────────────────────────────
    def zscore_traces(arr):
        """arr: (5, T); z-score each row using baseline window."""
        mu = arr[:, bl_s:bl_e].mean(axis=1, keepdims=True)
        sg = arr[:, bl_s:bl_e].std(axis=1, keepdims=True).clip(1e-6)
        return (arr - mu) / sg

    Ft_top5 = zscore_traces(Ft[top5_idx].astype(np.float32))  # (5, T)
    Fp_top5 = zscore_traces(Fp[top5_idx].astype(np.float32))  # (5, T)

    # ── population mean ± std (subsample 5K) ─────────────────────────
    sub5k  = rng.choice(n_cells, size=min(5_000, n_cells), replace=False)
    Ft_s   = Ft[sub5k].astype(np.float32)
    Fp_s   = Fp[sub5k].astype(np.float32)
    Ft_mean, Ft_std = Ft_s.mean(axis=0), Ft_s.std(axis=0)
    Fp_mean, Fp_std = Fp_s.mean(axis=0), Fp_s.std(axis=0)

    # ── epoch lines ───────────────────────────────────────────────────
    ep = {
        "drug start": (dr_s / (60 * sampling_rate_hz), "tomato"),
        "drug end":   (dr_e / (60 * sampling_rate_hz), "steelblue"),
    }

    def _add_epochs(ax):
        for lbl, (xv, col) in ep.items():
            ax.axvline(xv, color=col, lw=1.2, ls="--", alpha=0.8, label=lbl)

    # ── figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 8),
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    fig.suptitle(f"Decomposition QC: {expt_ID}", fontsize=12)
    cmap = plt.cm.tab10

    # Row 0: population mean ± std
    ax = axes[0, 0]
    ax.fill_between(t, Ft_mean - Ft_std, Ft_mean + Ft_std,
                    alpha=0.25, color="orange")
    ax.plot(t, Ft_mean, color="orange", lw=1.2)
    _add_epochs(ax)
    ax.set_title("F_tonic — population mean ± std", fontsize=10)
    ax.set_ylabel("F_tonic (counts)")
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.fill_between(t, Fp_mean - Fp_std, Fp_mean + Fp_std,
                    alpha=0.2, color="steelblue")
    ax.plot(t, Fp_mean, color="steelblue", lw=1.2)
    _add_epochs(ax)
    ax.set_title("F_phasic — population mean ± std", fontsize=10)
    ax.set_ylabel("F_phasic (ΔF/F)")
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=7)

    # Row 1: top-5 overlapping z-scored traces
    ax = axes[1, 0]
    for k in range(5):
        ci, dz = top5_idx[k], top5_dz[k]
        ax.plot(t, Ft_top5[k], color=cmap(k), lw=0.9, alpha=0.9,
                label=f"cell {ci}  ΔZ={dz:+.1f}")
    _add_epochs(ax)
    ax.set_title("F_tonic — top-5 |ΔZ| cells (z-scored)", fontsize=10)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Z-score")
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="k", lw=0.5, ls=":")

    ax = axes[1, 1]
    for k in range(5):
        ci, dz = top5_idx[k], top5_dz[k]
        ax.plot(t, Fp_top5[k], color=cmap(k), lw=0.9, alpha=0.9,
                label=f"cell {ci}  ΔZ={dz:+.1f}")
    _add_epochs(ax)
    ax.set_title("F_phasic — same top-5 cells (z-scored)", fontsize=10)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Z-score")
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="k", lw=0.5, ls=":")

    plt.tight_layout()

    qc_path = fish_out / "QC_figures" / "decompose_QC.png"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(qc_path), dpi=150, bbox_inches="tight")
    print(f"  ✅ Decompose QC saved → {qc_path}")
    plt.close(fig)
    del Ft, Fp, Ft_sub, Ft_s, Fp_s, Ft_top5, Fp_top5


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print(f"Running decompose for {len(all_fish)} fish")
    print(f"  dir_voluseg  : {dir_voluseg}")
    print(f"  dir_analysis : {dir_analysis}")
    print(f"  window       : {f_tonic_window_size}s | percentile: {f_tonic_percentile}")
    print(f"  denom_mode   : {DENOM_MODE}")
    print(f"  n_jobs       : {N_JOBS}")
    print()

    for fish in all_fish:
        proj_ID, expt_ID = fish
        print(f"── {expt_ID} ──────────────────────────────")

        fish_out = fish_dir(dir_analysis, fish)
        fish_out.mkdir(parents=True, exist_ok=True)

        tonic_path   = fish_out / "f_tonic.npy"
        phasic_path  = fish_out / "f_phasic.npy"
        bg_path      = fish_out / "f_dark_scalar.npy"

        try:
            # ----------------------------------------------------------
            # Step 0: estimate camera background F_dark
            # ----------------------------------------------------------
            if bg_path.exists() and not OVERWRITE_TONIC:
                f_dark = float(np.load(str(bg_path)))
                print(f"  ⏩ F_dark exists: {f_dark:.2f} counts (loaded from disk)")
            else:
                print(f"  Estimating background from {N_BG_VOLUMES} volumes...")
                f_dark, patch_medians = estimate_background(
                    fish=fish,
                    dir_voluseg=dir_voluseg,
                    n_volumes=N_BG_VOLUMES,
                    patch_size=BG_PATCH_SIZE,
                )
                np.save(str(bg_path), np.array(f_dark, dtype=np.float32))
                print(f"  ✅ F_dark = {f_dark:.2f} counts")
                print(f"     top-left median:    {patch_medians['top_left']:.2f}")
                print(f"     bottom-left median: {patch_medians['bottom_left']:.2f}")

            # ----------------------------------------------------------
            # Step 1 + 2: F_tonic
            # ----------------------------------------------------------
            if not OVERWRITE_TONIC and tonic_path.exists():
                print(f"  ⏩ F_tonic exists, skipping.")
            else:
                data_array, _, _, _, _ = read_data(fish, dir_voluseg)
                X = np.asarray(data_array, dtype=np.float32)

                # subtract camera background before decomposition
                X = subtract_background(X, f_dark, clip_min=0.0)

                Ft = compute_f_tonic(
                    X,
                    sampling_rate_hz=sampling_rate_hz,
                    window_seconds=f_tonic_window_size,
                    f_tonic_percentile=f_tonic_percentile,
                    chunk_cells=CHUNK_CELLS,
                    n_jobs=N_JOBS,
                    dtype_out=DTYPE_OUT,
                    show_pbar=True,
                    desc=f"{expt_ID} F_tonic",
                )

                np.save(str(tonic_path), Ft)
                print(f"  ✅ Saved F_tonic  → {tonic_path}")

                del X, Ft
                gc.collect()

            # ----------------------------------------------------------
            # Step 3: F_phasic  (requires F_tonic to exist)
            # ----------------------------------------------------------
            if not OVERWRITE_PHASIC and phasic_path.exists():
                print(f"  ⏩ F_phasic exists, skipping.")
            else:
                if not tonic_path.exists():
                    raise FileNotFoundError(
                        f"F_tonic not found for {expt_ID}. "
                        f"Run with OVERWRITE_TONIC=True first.\n{tonic_path}"
                    )

                data_array, _, _, _, _ = read_data(fish, dir_voluseg)
                X  = np.asarray(data_array, dtype=np.float32)

                # subtract camera background before decomposition
                X  = subtract_background(X, f_dark, clip_min=0.0)
                Ft = np.load(str(tonic_path), mmap_mode="r")

                Fp = compute_f_phasic(
                    X, Ft,
                    denom_mode=DENOM_MODE,
                    eps_floor=EPS_FLOOR,
                    chunk_cells=CHUNK_CELLS,
                    n_jobs=N_JOBS,
                    dtype_out=DTYPE_OUT,
                    show_pbar=True,
                    desc=f"{expt_ID} F_phasic",
                )

                np.save(str(phasic_path), Fp)
                print(f"  ✅ Saved F_phasic → {phasic_path}")

                del X, Ft, Fp
                gc.collect()

            # ----------------------------------------------------------
            # Step 4: QC figure (always — loads from disk, never recomputes)
            # ----------------------------------------------------------
            _plot_decompose_qc(expt_ID, fish_out, tonic_path, phasic_path)

        except Exception as e:
            print(f"  ❌ {expt_ID} failed: {e}")

        finally:
            gc.collect()

    print("\nDecompose run complete.")


if __name__ == "__main__":
    main()
