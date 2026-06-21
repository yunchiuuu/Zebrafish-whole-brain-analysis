"""
run_decompose.py
================
SLURM entry point: compute F_tonic and F_phasic for all fish.

Reads raw voluseg HDF5 from dir_voluseg, writes .npy arrays to dir_analysis.
Skips any fish where outputs already exist (set overwrite=True to force).

Usage
-----
On HPC (sbatch):
    sbatch submit_decompose.sh

    where submit_decompose.sh contains something like:
        #!/bin/bash
        #SBATCH --job-name=decompose
        #SBATCH --output=logs/decompose-%j.out
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=30
        #SBATCH --mem=128G
        #SBATCH --time=06:00:00
        #SBATCH --partition=expansion
        source ~/.bashrc
        conda activate proberlab
        python ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_decompose.py

Interactively (login node or salloc session):
    python run_decompose.py

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_decompose.py
"""

import gc
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make repo root importable regardless of working directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]   # .../Zebrafish-whole-brain-analysis
sys.path.insert(0, str(REPO_ROOT))

from chemogenetic.config.hcrt_trpv1_csn_120min import (
    all_fish,
    dir_analysis,
    dir_voluseg,
    f_tonic_percentile,
    f_tonic_window_size,
    sampling_rate_hz,
)
from utils.data_io import fish_dir, read_data
from utils.preprocess import compute_f_tonic, compute_f_phasic

# ============================================================
# SETTINGS  (edit here or override via env vars if needed)
# ============================================================
OVERWRITE_TONIC  = False   # set True to recompute even if file exists
OVERWRITE_PHASIC = False   # set True to recompute even if file exists

DENOM_MODE   = "legacy"    # "legacy" (matches notebook) or "fixed_floor"
EPS_FLOOR    = 1e-6        # only used when denom_mode="fixed_floor"

CHUNK_CELLS  = 20000
N_JOBS       = 28          # tune to match --cpus-per-task in your sbatch script
DTYPE_OUT    = np.float32


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

        tonic_path  = fish_out / "data_array_f_tonic.npy"
        phasic_path = fish_out / "data_array_f_phasic.npy"

        try:
            # ----------------------------------------------------------
            # Step 1: F_tonic
            # ----------------------------------------------------------
            if not OVERWRITE_TONIC and tonic_path.exists():
                print(f"  ⏩ F_tonic exists, skipping.")
            else:
                data_array, _, _, _, _ = read_data(fish, dir_voluseg)
                X = np.asarray(data_array, dtype=np.float32)

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
            # Step 2: F_phasic  (requires F_tonic to exist)
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
