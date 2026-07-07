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

        except Exception as e:
            print(f"  ❌ {expt_ID} failed: {e}")

        finally:
            gc.collect()

    print("\nDecompose run complete.")


if __name__ == "__main__":
    main()
