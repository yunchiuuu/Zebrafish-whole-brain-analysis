"""
run_spearman.py
===============
SLURM entry point: compute lagged Spearman tonic correlations for all fish.

Reads F_tonic from dir_analysis, writes rho arrays back to the same fish folder.
Runs all fish in parallel via joblib.

Usage
-----
    sbatch submit_spearman.sh
    # or interactively:
    python run_spearman.py

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_spearman.py
"""

import sys
from pathlib import Path

from joblib import Parallel, delayed
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chemogenetic.config.hcrt_trpv1_csn_120min import (
    all_fish,
    dir_analysis,
    drug_end,
    drug_start,
    drug_uM_per_fish,
    Q_ml_min,
    sampling_rate_hz,
    V_ml,
    wash_end,
)
from chemogenetic.spearman import compute_spearman_one_fish

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE       = False

LAG_MAX_SEC     = 20 * 60    # 0..20 min lag scan
LAG_STEP_SEC    = 5  * 60    # 5 min steps
BASELINE_PRE_SEC = 15 * 60   # baseline extension before drug_start

CHUNK_CELLS     = 2000
N_JOBS_FISH     = 25         # parallel fish (tune to node CPU count)


# ============================================================
# WORKER
# ============================================================
def _run_one(fish):
    proj_ID, expt_ID = fish
    fish_drug_uM = drug_uM_per_fish.get(fish, 10.0)

    try:
        compute_spearman_one_fish(
            fish=fish,
            dir_analysis=dir_analysis,
            sampling_rate=sampling_rate_hz,
            drug_start_frame=drug_start,
            drug_end_frame=drug_end,
            wash_end_frame=wash_end,
            drug_uM=fish_drug_uM,
            V_ml=V_ml,
            Q_ml_min=Q_ml_min,
            lag_max_sec=LAG_MAX_SEC,
            lag_step_sec=LAG_STEP_SEC,
            baseline_pre_sec=BASELINE_PRE_SEC,
            chunk_cells=CHUNK_CELLS,
            overwrite=OVERWRITE,
        )
        return fish, "ok"
    except Exception as e:
        return fish, f"ERROR: {e}"


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Running Spearman for {len(all_fish)} fish (n_jobs={N_JOBS_FISH})")
    print(f"  dir_analysis  : {dir_analysis}")
    print(f"  lag scan      : 0..{LAG_MAX_SEC//60} min, step {LAG_STEP_SEC//60} min")
    print(f"  baseline_pre  : {BASELINE_PRE_SEC//60} min")
    print()

    results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
        delayed(_run_one)(fish)
        for fish in tqdm(all_fish, desc="Spearman (fish)")
    )

    print("\n── Summary ──────────────────────────────")
    for fish, status in results:
        print(f"  {fish[1]:50s}  {status}")


if __name__ == "__main__":
    main()
