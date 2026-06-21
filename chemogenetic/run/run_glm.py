"""
run_glm.py
==========
SLURM entry point: run the full GLM pipeline for all fish.

Stages run in order:
    A. CV            — per-fish hyperparameter selection (parallel across fish)
    B. Choose global — aggregate CV results into K_global, drift_global, lam_global
    C. Refit         — refit all cells with global params (parallel across fish)
    D. Ablation      — drift-only ΔR² (parallel across fish)
    E. IAAFT null    — surrogate null distribution (serial per fish, heavy compute)
    F. Responders    — threshold + save pos/neg idx (parallel across fish)

Each stage can be toggled independently via the RUN_* flags, so you can
re-run, e.g., just the null (E) without re-running CV or refit.

Usage
-----
    sbatch submit_glm.sh
    # or interactively:
    python run_glm.py

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_glm.py
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
    INCLUDED_BASELINE,
    input_tag,
    K_global,
    drift_global,
    lam_global,
    lag_global,
    NULL_TAG,
    param_folder_name,
    Q_ml_min,
    RESPONDER_NULL_THRESH,
    sampling_rate_hz,
    V_ml,
)
from chemogenetic.glm import (
    choose_global_params,
    cv_one_fish,
    refit_one_fish,
    ablation_one_fish,
    iaaft_null_one_fish,
    save_responder_idx,
)

# ============================================================
# STAGE TOGGLES  — set False to skip a stage
# ============================================================
RUN_CV         = True
RUN_REFIT      = True
RUN_ABLATION   = True
RUN_NULL       = True
RUN_RESPONDERS = True

# ============================================================
# SETTINGS
# ============================================================
OVERWRITE_CV         = False
OVERWRITE_REFIT      = False
OVERWRITE_ABLATION   = False
OVERWRITE_NULL       = False
OVERWRITE_RESPONDERS = True    # always overwrite so idx reflect current null

# CV grid (set narrower for quick reruns)
K_LIST        = (60, 120, 300, 600, 900, 1200)
DRIFT_ORDERS  = (1, 2, 3)
LAM_LIST      = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)
BLOCK_LEN_SEC = 10 * 60
N_CELLS_CV    = 2000
CHUNK_CELLS   = 2000

# IAAFT null settings
N_SURROGATES   = 200
N_ITER_IAAFT   = 50
N_CELLS_NULL   = 20000

# Responder sign windows (in minutes within the fit window)
BASELINE_WIN_MIN = (0.0,  15.0)
DRUG_WIN_MIN     = (30.0, 45.0)

# Parallelism
N_JOBS_FISH = 10   # parallel fish for CV / refit / ablation / responders
              # NOTE: IAAFT null runs serially (each fish uses all cores internally)

FIT_BASELINE_SEC = INCLUDED_BASELINE * 60


# ============================================================
# HELPERS
# ============================================================
def _drug_uM(fish):
    return drug_uM_per_fish.get(fish, 10.0)


def _common_kwargs(fish):
    return dict(
        fish=fish,
        dir_analysis=dir_analysis,
        sampling_rate_hz=sampling_rate_hz,
        drug_start_frame_full=drug_start,
        drug_end_frame_full=drug_end,
        drug_uM=_drug_uM(fish),
        V_ml=V_ml,
        Q_ml_min=Q_ml_min,
        input_tag=input_tag,
        lag_global=lag_global,
        fit_baseline_sec=FIT_BASELINE_SEC,
        chunk_cells_fit=CHUNK_CELLS,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"GLM pipeline | {len(all_fish)} fish | dir_analysis={dir_analysis}")
    print(f"  input_tag={input_tag}  K={K_global}  drift={drift_global}  "
          f"lam={lam_global}  lag={lag_global}  null={NULL_TAG}\n")

    # ── A: CV ────────────────────────────────────────────────
    if RUN_CV:
        print("── Stage A: CV ─────────────────────────────────────")

        def _cv(fish):
            try:
                return cv_one_fish(
                    **_common_kwargs(fish),
                    K_list=K_LIST, drift_orders=DRIFT_ORDERS, lam_list=LAM_LIST,
                    block_len_sec=BLOCK_LEN_SEC, n_cells_cv=N_CELLS_CV,
                    overwrite=OVERWRITE_CV, show_progress=False, verbose_cv=False,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_cv)(fish) for fish in tqdm(all_fish, desc="CV")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── B: Choose globals ─────────────────────────────────────
    # Always recompute from saved JSON — fast, no toggle needed.
    print("\n── Stage B: Choose global params ───────────────────────")
    gp = choose_global_params(all_fish, dir_analysis)
    K_g, drift_g, lam_g = gp["K_global"], gp["drift_global"], gp["lam_global"]
    print(f"  K_global={K_g}  drift_global={drift_g}  lam_global={lam_g:.3g}")
    print(f"  (config values: K={K_global} drift={drift_global} lam={lam_global})")
    print("  NOTE: using config values (manually set). Set RUN_CV=True to use CV globals.\n")

    # Use config globals (manual), not auto-computed, to match param_folder_name in config.
    # If you want auto globals, replace K_global/drift_global/lam_global below with K_g/drift_g/lam_g.

    # ── C: Refit ──────────────────────────────────────────────
    if RUN_REFIT:
        print("── Stage C: Global refit ───────────────────────────────")

        def _refit(fish):
            try:
                return refit_one_fish(
                    **_common_kwargs(fish),
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    overwrite=OVERWRITE_REFIT, show_progress=False,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_refit)(fish) for fish in tqdm(all_fish, desc="Refit")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── D: Ablation ───────────────────────────────────────────
    if RUN_ABLATION:
        print("\n── Stage D: Drift-only ablation ────────────────────────")

        def _ablation(fish):
            try:
                return ablation_one_fish(
                    **_common_kwargs(fish),
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    overwrite=OVERWRITE_ABLATION, show_progress=False,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_ablation)(fish) for fish in tqdm(all_fish, desc="Ablation")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    # ── E: IAAFT null ─────────────────────────────────────────
    # Serial: each fish is compute-heavy (n_surrogates × ridge fits).
    if RUN_NULL:
        print("\n── Stage E: IAAFT null ─────────────────────────────────")
        for fish in all_fish:
            try:
                result = iaaft_null_one_fish(
                    **_common_kwargs(fish),
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    n_surrogates=N_SURROGATES, n_iter_iaaft=N_ITER_IAAFT,
                    n_cells_null=N_CELLS_NULL,
                    null_percentile=RESPONDER_NULL_THRESH,
                    overwrite=OVERWRITE_NULL, show_progress=True,
                )
                print(f"  {fish[1]:50s}  {result['status']}")
            except Exception as e:
                print(f"  {fish[1]:50s}  ERROR: {e}")

    # ── F: Responder indices ──────────────────────────────────
    if RUN_RESPONDERS:
        print("\n── Stage F: Save responder indices ─────────────────────")

        def _responders(fish):
            try:
                return save_responder_idx(
                    fish=fish,
                    dir_analysis=dir_analysis,
                    sampling_rate_hz=sampling_rate_hz,
                    K_global=K_global, drift_global=drift_global, lam_global=lam_global,
                    input_tag=input_tag, lag_global=lag_global,
                    null_tag=NULL_TAG,
                    null_percentile=RESPONDER_NULL_THRESH,
                    baseline_win_min=BASELINE_WIN_MIN,
                    drug_win_min=DRUG_WIN_MIN,
                    overwrite=OVERWRITE_RESPONDERS,
                )
            except Exception as e:
                return {"fish": fish, "status": f"ERROR: {e}"}

        results = Parallel(n_jobs=N_JOBS_FISH, backend="loky")(
            delayed(_responders)(fish) for fish in tqdm(all_fish, desc="Responders")
        )
        for r in results:
            print(f"  {r['fish'][1]:50s}  {r['status']}")

    print("\n✅ GLM pipeline complete.")


if __name__ == "__main__":
    main()
