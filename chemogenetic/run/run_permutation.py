"""
run_permutation.py
==================
SLURM entry point: run permutation test + BH-FDR for all fish.

Two sequential steps per fish:
    1. run_permutation_one_fish  — raw p-values for tonic + phasic
    2. run_bh_one_fish           — BH-FDR correction, save responder indices

Permutation is the slow step (n_cells × n_resamples). Each fish is run
serially to avoid CPU oversubscription — all cores are used *within* each
fish via n_jobs in parallel_permutation_all_cells.

Usage
-----
    sbatch submit_permutation.sh
    # or interactively:
    python run_permutation.py

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_permutation.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chemogenetic.config.hcrt_trpv1_csn_120min import (
    all_fish,
    baseline_end,
    baseline_start,
    BH_Q,
    dir_analysis,
    drug_end,
    drug_start,
    n_resample_permutation,
)
from chemogenetic.permutation import run_bh_one_fish, run_permutation_one_fish

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE_PERM = False    # set True to re-run permutation test
OVERWRITE_BH   = False    # set True to re-run BH correction only

N_JOBS_CELLS   = 28       # cores used within each fish (tune to --cpus-per-task)
SAVE_ADJ_P     = True     # also save full-length adjusted p-value arrays
BH_SUFFIX      = "perm_BH"


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Running permutation test + BH-FDR for {len(all_fish)} fish")
    print(f"  dir_analysis  : {dir_analysis}")
    print(f"  baseline      : frames [{baseline_start}, {baseline_end})")
    print(f"  drug          : frames [{drug_start}, {drug_end})")
    print(f"  n_resamples   : {n_resample_permutation}")
    print(f"  BH q          : {BH_Q}")
    print(f"  n_jobs_cells  : {N_JOBS_CELLS}")
    print()

    for fish in all_fish:
        proj_ID, expt_ID = fish
        print(f"── {expt_ID} ──────────────────────────────")

        try:
            # Step 1: permutation p-values
            run_permutation_one_fish(
                fish=fish,
                dir_analysis=dir_analysis,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                drug_start=drug_start,
                drug_end=drug_end,
                n_resamples=n_resample_permutation,
                n_jobs=N_JOBS_CELLS,
                overwrite=OVERWRITE_PERM,
            )

            # Step 2: BH-FDR
            run_bh_one_fish(
                fish=fish,
                dir_analysis=dir_analysis,
                q=BH_Q,
                save_adj_p=SAVE_ADJ_P,
                suffix=BH_SUFFIX,
                overwrite=OVERWRITE_BH,
            )

        except Exception as e:
            print(f"  ❌ {expt_ID} failed: {e}")

    print("\nPermutation run complete.")


if __name__ == "__main__":
    main()
