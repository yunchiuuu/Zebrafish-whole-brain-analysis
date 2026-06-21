"""
run_dprime.py
=============
SLURM entry point: compute phasic d′ per cell for all fish.

Runs all fish in parallel. Each fish reads data_array_f_phasic.npy and
writes phasic_dprime_cells_{mode}.npy and phasic_deltaMean_cells_{mode}.npy.

Usage
-----
    sbatch submit_dprime.sh
    # or interactively:
    python run_dprime.py

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_dprime.py
"""

import sys
from pathlib import Path

from joblib import Parallel, delayed
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chemogenetic.config.hcrt_trpv1_csn_120min import (
    all_fish,
    baseline_end,
    baseline_start,
    dir_analysis,
    drug_end,
    drug_start,
    sampling_rate_hz,
)
from chemogenetic.dprime import dprime_one_fish

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE      = False
AMPLITUDE_MODE = "raw"       # "raw" (signed, recommended), "abs", or "rms"
OFFSET_SEC     = 15 * 60     # skip first 15 min of each epoch window
VAR_FLOOR      = 1e-4        # variance floor for denominator stability
EPS_VAR        = 1e-8
CLIP_ABS       = None        # set e.g. 1.5 to clip extreme d′ values

# Drug window extension: your notebook added 20 min lag after drug_end
# to capture delayed responses. Adjust here if needed.
DRUG_END_EXTENDED = drug_end + 20 * 60

N_JOBS = 28


# ============================================================
# WORKER
# ============================================================
def _run_one(fish):
    try:
        return dprime_one_fish(
            fish=fish,
            dir_analysis=dir_analysis,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            drug_start=drug_start,
            drug_end=DRUG_END_EXTENDED,
            sampling_rate_hz=sampling_rate_hz,
            offset_sec=OFFSET_SEC,
            amplitude_mode=AMPLITUDE_MODE,
            var_floor=VAR_FLOOR,
            eps_var=EPS_VAR,
            clip_abs=CLIP_ABS,
            overwrite=OVERWRITE,
        )
    except Exception as e:
        return {"fish": fish, "status": f"ERROR: {e}"}


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Running d′ for {len(all_fish)} fish (n_jobs={N_JOBS})")
    print(f"  dir_analysis   : {dir_analysis}")
    print(f"  amplitude_mode : {AMPLITUDE_MODE}")
    print(f"  offset_sec     : {OFFSET_SEC // 60} min")
    print(f"  baseline       : frames [{baseline_start}, {baseline_end})")
    print(f"  drug (extended): frames [{drug_start}, {DRUG_END_EXTENDED})")
    print()

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_run_one)(fish) for fish in tqdm(all_fish, desc="d′ (fish)")
    )

    print("\n── Summary ──────────────────────────────")
    for r in results:
        status = r["status"]
        if status == "ok":
            print(
                f"  {r['fish'][1]:50s}  "
                f"mean={r['mean']:+.3f}  median={r['median']:+.3f}  "
                f"n={r['n_cells']}"
            )
        else:
            print(f"  {r['fish'][1]:50s}  {status}")


if __name__ == "__main__":
    main()
