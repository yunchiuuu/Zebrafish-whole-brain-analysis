"""
run_responders_effectsize.py
============================
SLURM entry point: compute fixed-window ΔZ and plateau ΔZ for all fish.

Two stages, both togglable:
    A. Fixed-window ΔZ  — fast, runs all fish in parallel
    B. Plateau ΔZ       — slightly heavier, also parallel

Both read responder indices from glm.py outputs and write per-fish
ΔZ arrays back to dir_analysis.

Usage
-----
    sbatch submit_responders_effectsize.sh
    # or interactively:
    python run_responders_effectsize.py

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_responders_effectsize.py
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
    CLIP_ABS_DZ,
    dir_analysis,
    drug_end,
    drug_start,
    L_MIN,
    NULL_TAG,
    RESPONDER_NULL_THRESH,
    sampling_rate_hz,
)
from chemogenetic.responders_effectsize import (
    fixed_dz_one_fish,
    frames_to_min_pair,
    plateau_dz_one_fish,
)

# ============================================================
# STAGE TOGGLES
# ============================================================
RUN_FIXED_DZ   = True
RUN_PLATEAU_DZ = True

OVERWRITE_FIXED   = False
OVERWRITE_PLATEAU = False

# ============================================================
# SETTINGS
# ============================================================
# Convert config frame indices to minute pairs (used by both methods)
BASELINE_MIN = frames_to_min_pair(baseline_start, baseline_end, sampling_rate_hz)
DRUG_MIN     = frames_to_min_pair(drug_start,     drug_end,     sampling_rate_hz)

# Fixed-window cache root (proj-level, matching notebook convention)
# One folder per proj_ID, not per fish — fish results live in subfolders inside
from chemogenetic.config.hcrt_trpv1_csn_120min import EXPT_PROJ, CTRL_PROJ
CACHE_ROOT_EXPT = Path(dir_analysis) / EXPT_PROJ / "results_dz_vectors"
CACHE_ROOT_CTRL = Path(dir_analysis) / CTRL_PROJ / "results_dz_vectors"

N_JOBS = 28


# ============================================================
# HELPERS
# ============================================================
def _cache_root_for(fish):
    """Each proj_ID gets its own results_dz_vectors folder."""
    proj_ID, _ = fish
    return Path(dir_analysis) / proj_ID / "results_dz_vectors"


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Responders effect size | {len(all_fish)} fish")
    print(f"  dir_analysis    : {dir_analysis}")
    print(f"  null_tag        : {NULL_TAG}")
    print(f"  null_percentile : {RESPONDER_NULL_THRESH}")
    print(f"  baseline        : {BASELINE_MIN[0]:g}–{BASELINE_MIN[1]:g} min")
    print(f"  drug            : {DRUG_MIN[0]:g}–{DRUG_MIN[1]:g} min")
    print(f"  clip_abs        : {CLIP_ABS_DZ}")
    print(f"  L_min (plateau) : {L_MIN} min")
    print()

    # ── A: Fixed-window ΔZ ───────────────────────────────────
    if RUN_FIXED_DZ:
        print("── Stage A: Fixed-window ΔZ ────────────────────────────")

        def _fixed(fish):
            try:
                return fixed_dz_one_fish(
                    fish=fish,
                    dir_analysis=dir_analysis,
                    sampling_rate_hz=sampling_rate_hz,
                    baseline_min_pair=BASELINE_MIN,
                    drug_min_pair=DRUG_MIN,
                    null_tag=NULL_TAG,
                    null_percentile=RESPONDER_NULL_THRESH,
                    clip_abs=CLIP_ABS_DZ,
                    cache_root=_cache_root_for(fish),
                    overwrite=OVERWRITE_FIXED,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}", "n_pos": 0, "n_neg": 0}

        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_fixed)(fish) for fish in tqdm(all_fish, desc="Fixed ΔZ")
        )

        for r in results:
            print(
                f"  {r['fish'][1]:50s}  {r['status']:10s}  "
                f"pos n={r.get('n_pos', '?'):5}  neg n={r.get('n_neg', '?'):5}"
            )

    # ── B: Plateau ΔZ ────────────────────────────────────────
    if RUN_PLATEAU_DZ:
        print("\n── Stage B: Plateau ΔZ ─────────────────────────────────")

        def _plateau(fish):
            try:
                return plateau_dz_one_fish(
                    fish=fish,
                    dir_analysis=dir_analysis,
                    sampling_rate_hz=sampling_rate_hz,
                    baseline_min_pair=BASELINE_MIN,
                    drug_epoch_min_pair=DRUG_MIN,
                    null_tag=NULL_TAG,
                    null_percentile=RESPONDER_NULL_THRESH,
                    L_min=L_MIN,
                    save_per_cell=True,
                    overwrite=OVERWRITE_PLATEAU,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}",
                        "fish_pos": float("nan"), "fish_neg": float("nan")}

        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_plateau)(fish) for fish in tqdm(all_fish, desc="Plateau ΔZ")
        )

        print("\n── Plateau ΔZ summary ───────────────────────────────────")
        for r in results:
            if r["status"] == "ok":
                print(
                    f"  {r['fish'][1]:50s}  "
                    f"pos={r['fish_pos']:+.3f}  neg={r['fish_neg']:+.3f}"
                )
            else:
                print(f"  {r['fish'][1]:50s}  {r['status']}")

    print("\n✅ Responders effect size complete.")


if __name__ == "__main__":
    main()
