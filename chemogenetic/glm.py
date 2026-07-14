"""
glm.py
======
Kernel ridge GLM for tonic whole-brain analysis.

Pipeline stages (all in this file):
    A. cv_one_fish          — time-block cross-validation to select per-fish hyperparams
    B. choose_global_params — aggregate per-fish CV results into global K, drift, lam
    C. refit_one_fish       — refit all cells with global hyperparams, save kernels + R²
    D. ablation_one_fish    — drift-only ablation → ΔR² = R²_full - R²_drift
    E. iaaft_null_one_fish  — IAAFT surrogate null distribution for ΔR²
    F. save_responder_idx   — threshold ΔR² vs null, separate pos/neg by kernel sign

Model:
    F_tonic(t) = Σ_k h[k] · u(t - k) + B(t) · β + ε
    where u(t) is C(t) or dC(t), h is the kernel, B is polynomial drift.

Outputs per fish live under:
    dir_analysis / proj_ID / expt_ID /
        C_capsaicin.npy, dC_capsaicin.npy
        kernel_ridge_best_params.json
        kernel_ridge_cv_table.json
        glm / <param_tag> /
            kernel_h_hat.npy              (n_cells, K)
            kernel_beta_hat.npy           (n_cells, nb)
            kernel_fit_r2.npy             (n_cells,)
            kernel_fit_r2_drift.npy       (n_cells,)
            kernel_delta_r2_fit.npy       (n_cells,)   ΔR²
            kernel_delta_r2_null__iaaft.npy
            kernel_delta_r2_null_thresh_p{N}__iaaft.npy
            X_mu.npy, X_sd.npy
            RUN_META.json
            RUN_META_ABLATION.json
            RUN_META_NULL__iaaft.json
        tonic_pos_glm_iaaft_nullp{N}_idxs.npy
        tonic_neg_glm_iaaft_nullp{N}_idxs.npy

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/analysis/glm.py
"""

import gc
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from utils.data_io import fish_dir
from chemogenetic.drug_regressor import build_drug_regressor, delta_signal


# ============================================================
# PATH HELPERS
# ============================================================

def make_param_tag(K, drift, lam, input_tag="C", lag=0):
    """Canonical subfolder name for a GLM run. Must match across all pipeline stages."""
    return f"in{input_tag}_K{int(K)}_drift{int(drift)}_lam{float(lam):.3g}_lag{int(lag)}"


def get_run_dir(dir_analysis, fish, K, drift, lam, input_tag="C", lag=0):
    """
    Return (base_dir, run_dir) for a fish's GLM outputs.

    base_dir : dir_analysis / proj_ID / expt_ID
    run_dir  : base_dir / glm / <param_tag>
    """
    base_dir = fish_dir(dir_analysis, fish)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / "glm" / make_param_tag(K, drift, lam, input_tag, lag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, run_dir


# ============================================================
# DESIGN MATRIX
# ============================================================

def shift_with_zeros(u, shift_frames):
    """
    Causal lag: delay u by shift_frames (zero-pad at start).
    Negative shift_frames advances u (zero-pad at end).
    """
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    T = u.shape[0]
    s = int(shift_frames)
    if s == 0:
        return u.copy()
    out = np.zeros(T, dtype=np.float32)
    if s > 0:
        if s < T:
            out[s:] = u[:-s]
    else:
        s = -s
        if s < T:
            out[:-s] = u[s:]
    return out


def build_U_from_input(u, K):
    """
    Build convolution design matrix U from input signal u.

    U[t, k] = u[t - k]  for t - k >= 0, else 0.

    Parameters
    ----------
    u : np.ndarray, shape (T,)
    K : int
        Kernel length.

    Returns
    -------
    U : np.ndarray, shape (T, K), dtype float32
    """
    u = np.asarray(u, dtype=np.float32)
    T = u.shape[0]
    U = np.zeros((T, K), dtype=np.float32)
    for k in range(K):
        U[k:, k] = u[:T - k]
    return U


def build_poly_drift_basis(T, order):
    """
    Polynomial drift basis, time normalised to [-1, 1]. Includes intercept.

    Returns
    -------
    B : np.ndarray, shape (T, order + 1), dtype float32
    """
    t = np.linspace(-1, 1, T, dtype=np.float32)
    cols = [np.ones(T, dtype=np.float32)]
    for p in range(1, order + 1):
        cols.append(t ** p)
    return np.stack(cols, axis=1).astype(np.float32)


def get_fit_window_indices(T_full, sampling_rate_hz, drug_start_frame_full, fit_baseline_sec):
    """
    Fit window = [drug_start - fit_baseline_sec, end_of_recording).

    Returns
    -------
    fit_start : int
    fit_end   : int
    drug_start_fit : int
        Position of drug_start within the fit window (= drug_start_frame_full - fit_start).
    """
    fit_start = int(round(drug_start_frame_full - fit_baseline_sec * sampling_rate_hz))
    fit_start = int(np.clip(fit_start, 0, T_full))
    fit_end   = int(T_full)
    if fit_end <= fit_start:
        raise ValueError(f"Invalid fit window: [{fit_start}, {fit_end})")
    drug_start_fit = int(drug_start_frame_full - fit_start)
    return fit_start, fit_end, drug_start_fit


# ============================================================
# RIDGE REGRESSION
# ============================================================

def standardize_X(X_train, X_test=None, eps=1e-8):
    """
    Z-score X_train column-wise; apply same transform to X_test if provided.

    Returns (X_train_z, (mu, sd), X_test_z_or_None)
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0,  keepdims=True) + eps
    Xtr_z = (X_train - mu) / sd
    if X_test is None:
        return Xtr_z, (mu, sd), None
    return Xtr_z, (mu, sd), (X_test - mu) / sd


def ridge_fit_predict_chunked(X_train, Y_train, X_test=None, lam=1.0,
                               chunk_cells=2000, show_progress=False, desc="ridge"):
    """
    Chunked ridge regression: solves (XᵀX + λI) W = XᵀY per chunk of cells.

    Parameters
    ----------
    X_train : (Ttr, P)
    Y_train : (n_cells, Ttr)
    X_test  : (Tte, P) or None
    lam     : float, ridge penalty
    chunk_cells : int

    Returns
    -------
    W_hat     : (n_cells, P)
    Yhat_test : (n_cells, Tte) or None
    """
    Ttr, P    = X_train.shape
    n_cells   = Y_train.shape[0]
    A         = X_train.T @ X_train + lam * np.eye(P, dtype=np.float32)
    W_hat     = np.zeros((n_cells, P), dtype=np.float32)
    Yhat_test = np.zeros((n_cells, X_test.shape[0]), dtype=np.float32) \
        if X_test is not None else None

    it = tqdm(range(0, n_cells, chunk_cells), desc=desc, leave=False) \
        if show_progress else range(0, n_cells, chunk_cells)

    for s in it:
        e   = min(s + chunk_cells, n_cells)
        Yc  = Y_train[s:e].T.astype(np.float32)     # (Ttr, chunk)
        Wc  = np.linalg.solve(A, X_train.T @ Yc).astype(np.float32)
        W_hat[s:e] = Wc.T
        if X_test is not None:
            Yhat_test[s:e] = (X_test @ Wc).T

    return W_hat, Yhat_test


def r2_score_per_cell(Y, Yhat, eps_var=1e-8):
    """
    In-sample R² per cell.

    Parameters
    ----------
    Y, Yhat : (n_cells, T)

    Returns
    -------
    r2 : (n_cells,), NaN for degenerate traces (SST < eps_var)
    """
    Y    = np.asarray(Y,    dtype=np.float32)
    Yhat = np.asarray(Yhat, dtype=np.float32)
    ybar = Y.mean(axis=1, keepdims=True)
    sst  = ((Y - ybar) ** 2).sum(axis=1)
    sse  = ((Y - Yhat) ** 2).sum(axis=1)
    r2   = 1.0 - sse / (sst + 1e-12)
    r2[sst < eps_var] = np.nan
    return r2


def _r2_from_W(Xz, Y, W_hat, chunk_cells=2000):
    """Compute R² for a pre-fit W_hat without re-solving the system."""
    n_cells = Y.shape[0]
    r2 = np.zeros(n_cells, dtype=np.float32)
    for s in range(0, n_cells, chunk_cells):
        e = min(s + chunk_cells, n_cells)
        Yhat = (Xz @ W_hat[s:e].T).T
        r2[s:e] = r2_score_per_cell(Y[s:e], Yhat).astype(np.float32)
    return r2


def build_drug_concentration(base_dir, T, sampling_rate_hz,
                              drug_start_frame_full, drug_end_frame_full,
                              drug_uM, V_ml, Q_ml_min):
    """
    Build C(t) and dC(t) regressors and save them to base_dir for inspection.

    Always recomputes from parameters — never reads cached files.
    This avoids the footgun where a stale C_capsaicin.npy (e.g. from a
    different drug_uM run) would be silently loaded and corrupt the regressor.

    Parameters
    ----------
    base_dir : Path
        Fish output directory. C_capsaicin.npy and dC_capsaicin.npy are
        saved here as outputs for reproducibility.
    T : int
        Total number of timepoints.
    sampling_rate_hz, drug_start_frame_full, drug_end_frame_full,
    drug_uM, V_ml, Q_ml_min : see build_drug_regressor.

    Returns
    -------
    C_full  : np.ndarray, shape (T,), float32
    dC_full : np.ndarray, shape (T,), float32
    """
    C_full = build_drug_regressor(
        T=T, sampling_rate_hz=sampling_rate_hz,
        drug_start_frame=int(round(drug_start_frame_full)),
        drug_end_frame=int(round(drug_end_frame_full)),
        drug_uM=drug_uM, V_ml=V_ml, Q_ml_min=Q_ml_min,
    ).astype(np.float32).reshape(-1)
    dC_full = delta_signal(C_full).astype(np.float32).reshape(-1)
    np.save(str(base_dir / "C_capsaicin.npy"),  C_full)
    np.save(str(base_dir / "dC_capsaicin.npy"), dC_full)
    return C_full, dC_full


def _select_input(C_full, dC_full, input_tag, fit_start, fit_end, lag):
    """Slice regressor to fit window and apply causal lag."""
    if input_tag in ("C", "HCRT"):   # HCRT uses C_full set to u_ext by caller
        u = C_full[fit_start:fit_end].astype(np.float32)
    elif input_tag == "dC":
        u = dC_full[fit_start:fit_end].astype(np.float32)
    else:
        raise ValueError(f"input_tag must be 'C', 'dC', or 'HCRT', got {input_tag!r}")
    return shift_with_zeros(u, lag)


# ============================================================
# STAGE A: PER-FISH CV
# ============================================================

def make_time_blocks(T, block_len_sec, sampling_rate_hz):
    """Partition T frames into non-overlapping time blocks for CV."""
    block_len = int(round(block_len_sec * sampling_rate_hz))
    if block_len < 10:
        raise ValueError(f"Block too small: block_len={block_len}")
    return [(b0, min(T, b0 + block_len))
            for b0 in range(0, T, block_len)
            if min(T, b0 + block_len) - b0 >= 10]


def cv_select_hyperparams(Y, u, K_list, drift_orders, lam_list,
                           sampling_rate_hz=1.0, block_len_sec=600,
                           n_cells_cv=2000, seed=0, chunk_cells=500,
                           show_progress=True, verbose=False, desc="CV"):
    """
    Time-block cross-validation to select (K, drift_order, lam).

    Score = mean across folds of median per-cell held-out R².

    Parameters
    ----------
    Y : (n_cells, Tfit)
    u : (Tfit,) — already sliced and lagged input signal
    K_list, drift_orders, lam_list : iterables of hyperparameter values

    Returns
    -------
    best : dict with keys K, drift_order, lam, score
    results : list of dicts (full CV table)
    idx_cv : indices of cells used for CV, or None
    """
    rng = np.random.default_rng(seed)
    n_cells, Tfit = Y.shape

    if n_cells_cv is not None and n_cells > n_cells_cv:
        idx_cv = rng.choice(n_cells, size=n_cells_cv, replace=False)
        Ycv = Y[idx_cv]
    else:
        idx_cv = None
        Ycv = Y

    blocks = make_time_blocks(Tfit, block_len_sec, sampling_rate_hz)
    if len(blocks) < 3:
        raise ValueError(f"Need ≥3 time blocks for CV, got {len(blocks)}")

    best    = {"score": -np.inf, "K": None, "drift_order": None, "lam": None}
    results = []

    total = len(K_list) * len(drift_orders) * len(lam_list)
    pbar  = tqdm(total=total, desc=desc, unit="combo", leave=False) if show_progress else None

    for K in K_list:
        U = build_U_from_input(u, K)
        for drift_order in drift_orders:
            B      = build_poly_drift_basis(Tfit, drift_order)
            X_full = np.concatenate([U, B], axis=1).astype(np.float32)

            for lam in lam_list:
                fold_scores = []
                for (b0, b1) in blocks:
                    test_mask  = np.zeros(Tfit, dtype=bool)
                    test_mask[b0:b1] = True
                    Xtr_z, _, Xte_z = standardize_X(X_full[~test_mask], X_full[test_mask])
                    _, Yhat = ridge_fit_predict_chunked(
                        Xtr_z, Ycv[:, ~test_mask], Xte_z, lam=lam, chunk_cells=chunk_cells,
                    )
                    fold_scores.append(float(np.nanmedian(r2_score_per_cell(Ycv[:, test_mask], Yhat))))

                score = float(np.nanmean(fold_scores))
                results.append({"K": int(K), "drift_order": int(drift_order),
                                 "lam": float(lam), "cv_r2_median": score})
                if verbose:
                    print(f"  K={K:4d} drift={drift_order} lam={lam:.2e} → {score:.6f}")
                if score > best["score"]:
                    best.update({"score": score, "K": int(K),
                                 "drift_order": int(drift_order), "lam": float(lam)})
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    return best, results, idx_cv


def cv_one_fish(fish, dir_analysis, sampling_rate_hz,
                drug_start_frame_full, drug_end_frame_full,
                drug_uM=10.0, V_ml=15.0, Q_ml_min=4.5,
                K_list=(60, 120, 300, 600, 900, 1200),
                drift_orders=(1, 2, 3),
                lam_list=(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3),
                block_len_sec=600, n_cells_cv=2000, seed=0,
                chunk_cells_cv=500, fit_baseline_sec=15 * 60,
                overwrite=False, show_progress=True, verbose_cv=False):
    """
    Run CV for one fish and save best params + CV table.

    Reads  : dir_analysis / proj_ID / expt_ID / f_tonic.npy
    Writes : dir_analysis / proj_ID / expt_ID /
                C_capsaicin.npy, dC_capsaicin.npy
                kernel_ridge_best_params.json
                kernel_ridge_cv_table.json
                kernel_ridge_meta.txt
    """
    proj_ID, expt_ID = fish
    base_dir = fish_dir(dir_analysis, fish)
    base_dir.mkdir(parents=True, exist_ok=True)

    params_path  = base_dir / "kernel_ridge_best_params.json"
    cvtable_path = base_dir / "kernel_ridge_cv_table.json"

    if not overwrite and params_path.exists() and cvtable_path.exists():
        print(f"⏩ {expt_ID}: CV results exist, skipping.")
        return {"fish": fish, "status": "skipped_cv"}

    Ft_path = base_dir / "f_tonic.npy"
    if not Ft_path.exists():
        raise FileNotFoundError(f"Missing F_tonic: {Ft_path}")

    print(f"▶️  {expt_ID}: CV start")
    Ft = np.load(str(Ft_path), mmap_mode="r")
    n_cells, Tfull = Ft.shape

    C_full, dC_full = build_drug_concentration(
        base_dir, Tfull, sampling_rate_hz,
        drug_start_frame_full, drug_end_frame_full,
        drug_uM, V_ml, Q_ml_min,
    )

    fit_start, fit_end, drug_start_fit = get_fit_window_indices(
        Tfull, sampling_rate_hz, drug_start_frame_full, fit_baseline_sec,
    )
    Y_fit  = np.asarray(Ft[:, fit_start:fit_end], dtype=np.float32)
    dC_fit = dC_full[fit_start:fit_end].astype(np.float32)
    Tfit   = Y_fit.shape[1]

    best, cv_table, _ = cv_select_hyperparams(
        Y=Y_fit, u=dC_fit,
        K_list=K_list, drift_orders=drift_orders, lam_list=lam_list,
        sampling_rate_hz=sampling_rate_hz,
        block_len_sec=block_len_sec, n_cells_cv=n_cells_cv,
        seed=seed, chunk_cells=chunk_cells_cv,
        show_progress=show_progress, verbose=verbose_cv,
        desc=f"CV {expt_ID}",
    )

    with open(str(params_path),  "w") as f: json.dump(best, f, indent=2)
    with open(str(cvtable_path), "w") as f: json.dump(cv_table, f, indent=2)

    meta_path = base_dir / "kernel_ridge_meta.txt"
    with open(str(meta_path), "w") as f:
        f.write(f"expt_ID={expt_ID}\nsampling_rate_hz={sampling_rate_hz}\n"
                f"Tfull={Tfull}\nfit_start={fit_start}\nfit_end={fit_end}\n"
                f"Tfit={Tfit}\ndrug_start_full={drug_start_frame_full}\n"
                f"drug_end_full={drug_end_frame_full}\ndrug_start_fit={drug_start_fit}\n"
                f"model=F=U_dC*h + B*beta + eps\n"
                f"K_best={best['K']}\ndrift_order_best={best['drift_order']}\n"
                f"lambda_best={best['lam']}\nblock_len_sec={block_len_sec}\n"
                f"n_cells_cv={n_cells_cv}\nbest_cv_score={best['score']}\n")

    print(f"✅ {expt_ID}: CV done | K={best['K']} drift={best['drift_order']} "
          f"lam={best['lam']:.2e} score={best['score']:.6f}")

    del Ft, Y_fit, C_full, dC_full, dC_fit
    gc.collect()
    return {"fish": fish, "status": "ok_cv", "best": best}


# ============================================================
# STAGE B: CHOOSE GLOBAL PARAMS
# ============================================================

def choose_global_params(fish_list, dir_analysis):
    """
    Aggregate per-fish CV results into global hyperparameters.

    Rules: K_global = max, drift_global = max, lam_global = median.

    Parameters
    ----------
    fish_list : list of (proj_ID, expt_ID) tuples

    Returns
    -------
    dict with K_global, drift_global, lam_global (and per-fish lists)
    """
    K_list, drift_list, lam_list, missing = [], [], [], []

    for fish in fish_list:
        p = fish_dir(dir_analysis, fish) / "kernel_ridge_best_params.json"
        if not p.exists():
            missing.append(fish[1])
            continue
        with open(str(p)) as f:
            b = json.load(f)
        K_list.append(int(b["K"]))
        drift_list.append(int(b["drift_order"]))
        lam_list.append(float(b["lam"]))

    if missing:
        raise FileNotFoundError(
            f"Missing kernel_ridge_best_params.json for {len(missing)} fish: {missing[:5]}"
        )

    return {
        "K_global":     int(np.max(K_list)),
        "drift_global": int(np.max(drift_list)),
        "lam_global":   float(np.median(lam_list)),
        "K_bests":      K_list,
        "drift_bests":  drift_list,
        "lam_bests":    lam_list,
    }


# ============================================================
# STAGE C: GLOBAL REFIT
# ============================================================

def refit_one_fish(fish, dir_analysis, sampling_rate_hz,
                   drug_start_frame_full, drug_end_frame_full,
                   K_global, drift_global, lam_global,
                   drug_uM=10.0, V_ml=15.0, Q_ml_min=4.5,
                   input_tag="C", lag_global=0,
                   fit_baseline_sec=15 * 60,
                   chunk_cells_fit=2000,
                   u_ext=None,
                   overwrite=False, show_progress=True):
    """
    Refit all cells using global hyperparameters. Save kernels, drift weights, R², meta.

    Writes to: base_dir / glm / <param_tag> /
        kernel_h_hat.npy, kernel_beta_hat.npy, kernel_fit_r2.npy
        X_mu.npy, X_sd.npy
        kernel_ridge_global_params.json, RUN_META.json
    """
    proj_ID, expt_ID = fish
    base_dir, run_dir = get_run_dir(
        dir_analysis, fish, K_global, drift_global, lam_global, input_tag, lag_global,
    )

    h_path    = run_dir / "kernel_h_hat.npy"
    beta_path = run_dir / "kernel_beta_hat.npy"
    r2_path   = run_dir / "kernel_fit_r2.npy"
    x_mu_path = run_dir / "X_mu.npy"
    x_sd_path = run_dir / "X_sd.npy"
    meta_path = run_dir / "RUN_META.json"

    if not overwrite and all(p.exists() for p in
                              [h_path, beta_path, r2_path, x_mu_path, x_sd_path, meta_path]):
        print(f"⏩ {expt_ID}: refit exists, skipping.")
        return {"fish": fish, "status": "skipped_refit", "run_dir": str(run_dir)}

    print(f"▶️  {expt_ID}: global refit ({run_dir.name})")
    Ft_path = base_dir / "f_tonic.npy"
    if not Ft_path.exists():
        raise FileNotFoundError(f"Missing F_tonic: {Ft_path}")

    Ft = np.load(str(Ft_path), mmap_mode="r")
    n_cells, Tfull = Ft.shape

    if u_ext is not None:
        C_full  = np.asarray(u_ext, dtype=np.float32).reshape(-1)[:Tfull]
        dC_full = None
    else:
        C_full, dC_full = build_drug_concentration(
            base_dir, Tfull, sampling_rate_hz,
            drug_start_frame_full, drug_end_frame_full,
            drug_uM, V_ml, Q_ml_min,
        )

    fit_start, fit_end, drug_start_fit = get_fit_window_indices(
        Tfull, sampling_rate_hz, drug_start_frame_full, fit_baseline_sec,
    )
    Y_fit = np.asarray(Ft[:, fit_start:fit_end], dtype=np.float32)
    Tfit  = Y_fit.shape[1]

    u_fit = _select_input(C_full, dC_full, input_tag, fit_start, fit_end, lag_global)
    U     = build_U_from_input(u_fit, int(K_global)).astype(np.float32)
    B     = build_poly_drift_basis(Tfit, int(drift_global)).astype(np.float32)
    X     = np.concatenate([U, B], axis=1).astype(np.float32)
    Xz, (mu, sd), _ = standardize_X(X)

    np.save(str(x_mu_path), np.asarray(mu, dtype=np.float32))
    np.save(str(x_sd_path), np.asarray(sd, dtype=np.float32))

    W_hat, _ = ridge_fit_predict_chunked(
        Xz, Y_fit, lam=lam_global, chunk_cells=chunk_cells_fit, show_progress=False,
    )

    K = int(K_global)
    h_hat    = W_hat[:, :K].astype(np.float32)
    beta_hat = W_hat[:, K:].astype(np.float32)
    fit_r2   = _r2_from_W(Xz, Y_fit, W_hat, chunk_cells=chunk_cells_fit)

    np.save(str(h_path),    h_hat)
    np.save(str(beta_path), beta_hat)
    np.save(str(r2_path),   fit_r2)

    run_meta = {
        "proj_ID": proj_ID, "expt_ID": expt_ID,
        "input_tag": input_tag, "lag_global_frames": int(lag_global),
        "param_tag": run_dir.name, "out_dir": str(run_dir),
        "K_global": K, "drift_global": int(drift_global), "lam_global": float(lam_global),
        "Tfull": int(Tfull), "Tfit": int(Tfit),
        "fit_start": int(fit_start), "fit_end": int(fit_end),
        "drug_start_frame_full": float(drug_start_frame_full),
        "drug_end_frame_full":   float(drug_end_frame_full),
        "sampling_rate_hz": float(sampling_rate_hz),
        "chunk_cells_fit": int(chunk_cells_fit),
    }
    with open(str(meta_path), "w") as f:
        json.dump(run_meta, f, indent=2)
    with open(str(run_dir / "kernel_ridge_global_params.json"), "w") as f:
        json.dump({"input_tag": input_tag, "lag_global_frames": int(lag_global),
                   "K_global": K, "drift_global": int(drift_global),
                   "lam_global": float(lam_global)}, f, indent=2)

    print(f"✅ {expt_ID}: refit done → {run_dir}")
    del Ft, Y_fit, U, B, X, Xz, W_hat, h_hat, beta_hat, fit_r2, C_full, dC_full, u_fit
    gc.collect()
    return {"fish": fish, "status": "ok_refit", "run_dir": str(run_dir)}


# ============================================================
# STAGE D: DRIFT-ONLY ABLATION
# ============================================================

def ablation_one_fish(fish, dir_analysis, sampling_rate_hz,
                      drug_start_frame_full, drug_end_frame_full,
                      K_global, drift_global, lam_global,
                      drug_uM=10.0, V_ml=15.0, Q_ml_min=4.5,
                      input_tag="C", lag_global=0,
                      fit_baseline_sec=15 * 60,
                      chunk_cells_fit=2000,
                      u_ext=None,
                      overwrite=False, show_progress=True):
    """
    Compute ΔR² = R²_full - R²_drift_only for one fish.

    Reuses kernel_fit_r2.npy from refit if available; recomputes otherwise.

    Writes to: run_dir /
        kernel_fit_r2_drift.npy
        kernel_delta_r2_fit.npy
        RUN_META_ABLATION.json
    """
    proj_ID, expt_ID = fish
    base_dir, run_dir = get_run_dir(
        dir_analysis, fish, K_global, drift_global, lam_global, input_tag, lag_global,
    )

    r2_drift_path = run_dir / "kernel_fit_r2_drift.npy"
    dR2_path      = run_dir / "kernel_delta_r2_fit.npy"
    r2_full_path  = run_dir / "kernel_fit_r2.npy"

    if not overwrite and r2_drift_path.exists() and dR2_path.exists():
        print(f"⏩ {expt_ID}: ablation exists, skipping.")
        return {"fish": fish, "status": "skipped_ablation"}

    print(f"▶️  {expt_ID}: drift-only ablation ({run_dir.name})")
    Ft_path = base_dir / "f_tonic.npy"
    if not Ft_path.exists():
        raise FileNotFoundError(f"Missing F_tonic: {Ft_path}")

    Ft = np.load(str(Ft_path), mmap_mode="r")
    n_cells, Tfull = Ft.shape

    if u_ext is not None:
        C_full  = np.asarray(u_ext, dtype=np.float32).reshape(-1)[:Tfull]
        dC_full = None
    else:
        C_full, dC_full = build_drug_concentration(
            base_dir, Tfull, sampling_rate_hz,
            drug_start_frame_full, drug_end_frame_full,
            drug_uM, V_ml, Q_ml_min,
        )

    fit_start, fit_end, _ = get_fit_window_indices(
        Tfull, sampling_rate_hz, drug_start_frame_full, fit_baseline_sec,
    )
    Y_fit = np.asarray(Ft[:, fit_start:fit_end], dtype=np.float32)
    Tfit  = Y_fit.shape[1]
    u_fit = _select_input(C_full, dC_full, input_tag, fit_start, fit_end, lag_global)

    # drift-only
    B     = build_poly_drift_basis(Tfit, int(drift_global)).astype(np.float32)
    Xd_z, _, _ = standardize_X(B)
    W_drift, _ = ridge_fit_predict_chunked(
        Xd_z, Y_fit, lam=lam_global, chunk_cells=chunk_cells_fit, show_progress=False,
    )
    r2_drift = _r2_from_W(Xd_z, Y_fit, W_drift, chunk_cells=chunk_cells_fit)
    np.save(str(r2_drift_path), r2_drift.astype(np.float32))

    # full model R² — reuse from refit if available
    if r2_full_path.exists() and not overwrite:
        r2_full = np.load(str(r2_full_path)).astype(np.float32).reshape(-1)
        if r2_full.shape[0] != n_cells:
            raise ValueError(f"{expt_ID}: kernel_fit_r2.npy length mismatch")
    else:
        U     = build_U_from_input(u_fit, int(K_global)).astype(np.float32)
        X     = np.concatenate([U, B], axis=1).astype(np.float32)
        Xf_z, _, _ = standardize_X(X)
        W_full, _ = ridge_fit_predict_chunked(
            Xf_z, Y_fit, lam=lam_global, chunk_cells=chunk_cells_fit, show_progress=False,
        )
        r2_full = _r2_from_W(Xf_z, Y_fit, W_full, chunk_cells=chunk_cells_fit)
        np.save(str(r2_full_path), r2_full.astype(np.float32))
        del U, X, Xf_z, W_full
        gc.collect()

    dR2 = (r2_full - r2_drift).astype(np.float32)
    np.save(str(dR2_path), dR2)

    with open(str(run_dir / "RUN_META_ABLATION.json"), "w") as f:
        json.dump({"proj_ID": proj_ID, "expt_ID": expt_ID,
                   "input_tag": input_tag, "lag_global_frames": int(lag_global),
                   "K_global": int(K_global), "drift_global": int(drift_global),
                   "lam_global": float(lam_global), "Tfit": int(Tfit),
                   "fit_start": int(fit_start), "fit_end": int(fit_end),
                   "sampling_rate_hz": float(sampling_rate_hz)}, f, indent=2)

    print(f"✅ {expt_ID}: ablation done | ΔR² p99={float(np.nanpercentile(dR2, 99)):.4f}")
    del Ft, Y_fit, C_full, dC_full, u_fit, B, Xd_z, W_drift, r2_drift, r2_full, dR2
    gc.collect()
    return {"fish": fish, "status": "ok_ablation"}


# ============================================================
# STAGE E: IAAFT NULL
# ============================================================

def iaaft_surrogate(x, n_iter=50, seed=0):
    """
    IAAFT surrogate of x.

    Preserves: exact amplitude distribution + approximately power spectrum.
    """
    rng  = np.random.default_rng(seed)
    x    = np.asarray(x, dtype=np.float32).reshape(-1)
    n    = x.size
    if n < 2:
        return x.copy()
    x_sorted   = np.sort(x)
    target_mag = np.abs(np.fft.rfft(x))
    y = rng.permutation(x).astype(np.float32)
    for _ in range(int(n_iter)):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(target_mag * np.exp(1j * np.angle(Y)), n=n).astype(np.float32)
        y_new = np.empty_like(y)
        y_new[np.argsort(y)] = x_sorted
        y = y_new
    return y.astype(np.float32)


def _null_output_paths(run_dir, null_tag, null_percentile):
    suffix = f"__{null_tag}"
    return (
        run_dir / f"kernel_delta_r2_null{suffix}.npy",
        run_dir / f"kernel_delta_r2_null_idx{suffix}.npy",
        run_dir / f"kernel_delta_r2_null_thresh_p{int(null_percentile)}{suffix}.npy",
        run_dir / f"RUN_META_NULL{suffix}.json",
    )


def iaaft_null_one_fish(fish, dir_analysis, sampling_rate_hz,
                         drug_start_frame_full, drug_end_frame_full,
                         K_global, drift_global, lam_global,
                         drug_uM=10.0, V_ml=15.0, Q_ml_min=4.5,
                         input_tag="C", lag_global=0,
                         fit_baseline_sec=15 * 60,
                         n_surrogates=200, n_iter_iaaft=50,
                         n_cells_null=20000, null_percentile=99,
                         seed=0, chunk_cells_fit=2000,
                         u_ext=None,
                         overwrite=False, show_progress=True):
    """
    Compute IAAFT null ΔR² distribution for one fish.

    Requires kernel_delta_r2_fit.npy to already exist (run ablation first).

    Writes to: run_dir /
        kernel_delta_r2_null__iaaft.npy       (n_surrogates, n_cells_null)
        kernel_delta_r2_null_idx__iaaft.npy   (n_cells_null,)
        kernel_delta_r2_null_thresh_p{N}__iaaft.npy  (scalar)
        RUN_META_NULL__iaaft.json
    """
    NULL_TAG = "iaaft"
    proj_ID, expt_ID = fish
    base_dir, run_dir = get_run_dir(
        dir_analysis, fish, K_global, drift_global, lam_global, input_tag, lag_global,
    )

    dR2_real_path = run_dir / "kernel_delta_r2_fit.npy"
    if not dR2_real_path.exists():
        raise FileNotFoundError(f"Run ablation first — missing: {dR2_real_path}")

    null_path, idx_path, thresh_path, meta_path = _null_output_paths(
        run_dir, NULL_TAG, null_percentile,
    )

    if not overwrite and all(p.exists() for p in [null_path, idx_path, thresh_path, meta_path]):
        print(f"⏩ {expt_ID}: IAAFT null exists, skipping.")
        return {"fish": fish, "status": f"skipped_null_{NULL_TAG}"}

    Ft_path = base_dir / "f_tonic.npy"
    if not Ft_path.exists():
        raise FileNotFoundError(f"Missing F_tonic: {Ft_path}")

    Ft = np.load(str(Ft_path), mmap_mode="r")
    n_cells, Tfull = Ft.shape

    if u_ext is not None:
        C_full  = np.asarray(u_ext, dtype=np.float32).reshape(-1)[:Tfull]
        dC_full = None
    else:
        C_full, dC_full = build_drug_concentration(
            base_dir, Tfull, sampling_rate_hz,
            drug_start_frame_full, drug_end_frame_full,
            drug_uM, V_ml, Q_ml_min,
        )

    fit_start, fit_end, _ = get_fit_window_indices(
        Tfull, sampling_rate_hz, drug_start_frame_full, fit_baseline_sec,
    )
    Tfit  = fit_end - fit_start
    u_fit = _select_input(C_full, dC_full, input_tag, fit_start, fit_end, lag_global)
    rng = np.random.default_rng(seed)
    n_null = int(min(n_cells_null, n_cells))
    idx_null = np.sort(rng.choice(n_cells, size=n_null, replace=False))
    np.save(str(idx_path), idx_null.astype(np.int64))

    Y_null = np.asarray(Ft[idx_null, fit_start:fit_end], dtype=np.float32)

    # drift baseline
    B     = build_poly_drift_basis(Tfit, int(drift_global)).astype(np.float32)
    Xd_z, _, _ = standardize_X(B)
    W_drift, _ = ridge_fit_predict_chunked(
        Xd_z, Y_null, lam=lam_global, chunk_cells=chunk_cells_fit, show_progress=False,
    )
    r2_drift = _r2_from_W(Xd_z, Y_null, W_drift, chunk_cells=chunk_cells_fit)

    # surrogate loop
    dR2_null = np.zeros((int(n_surrogates), n_null), dtype=np.float32)
    it = tqdm(range(int(n_surrogates)), desc=f"IAAFT null {expt_ID}", unit="surr", leave=False) \
        if show_progress else range(int(n_surrogates))

    for j in it:
        u_surr = iaaft_surrogate(u_fit, n_iter=int(n_iter_iaaft), seed=int(seed + 1000 * j))
        U      = build_U_from_input(u_surr, int(K_global)).astype(np.float32)
        X      = np.concatenate([U, B], axis=1).astype(np.float32)
        Xz, _, _ = standardize_X(X)
        W_full, _ = ridge_fit_predict_chunked(
            Xz, Y_null, lam=lam_global, chunk_cells=chunk_cells_fit, show_progress=False,
        )
        dR2_null[j] = (_r2_from_W(Xz, Y_null, W_full, chunk_cells=chunk_cells_fit) - r2_drift
                       ).astype(np.float32)
        del u_surr, U, X, Xz, W_full
        gc.collect()

    thr = float(np.nanpercentile(dR2_null.reshape(-1), int(null_percentile)))
    np.save(str(null_path),   dR2_null.astype(np.float32))
    np.save(str(thresh_path), np.array(thr, dtype=np.float32))

    with open(str(meta_path), "w") as f:
        json.dump({"proj_ID": proj_ID, "expt_ID": expt_ID,
                   "null_tag": NULL_TAG, "input_tag": input_tag,
                   "lag_global_frames": int(lag_global),
                   "K_global": int(K_global), "drift_global": int(drift_global),
                   "lam_global": float(lam_global),
                   "n_surrogates": int(n_surrogates), "n_iter_iaaft": int(n_iter_iaaft),
                   "n_cells_null_used": int(n_null), "null_percentile": int(null_percentile),
                   "seed": int(seed), "sampling_rate_hz": float(sampling_rate_hz),
                   "Tfit": int(Tfit), "fit_start": int(fit_start), "fit_end": int(fit_end),
                   "drug_uM": float(drug_uM), "V_ml": float(V_ml), "Q_ml_min": float(Q_ml_min)},
                  f, indent=2)

    print(f"✅ {expt_ID}: IAAFT null done | thr(p{null_percentile})={thr:.6f}")
    del Ft, Y_null, C_full, dC_full, u_fit, B, Xd_z, W_drift, r2_drift, dR2_null
    gc.collect()
    return {"fish": fish, "status": "ok_iaaft_null", "thresh": thr}


# ============================================================
# STAGE F: SAVE RESPONDER INDICES
# ============================================================

def _window_frames(sr_hz, t0_min, t1_min, Tfit):
    s = int(np.clip(int(round(t0_min * 60.0 * sr_hz)), 0, Tfit))
    e = int(np.clip(int(round(t1_min * 60.0 * sr_hz)), 0, Tfit))
    if e <= s:
        raise ValueError(f"Window [{t0_min},{t1_min}) min is empty after clipping")
    return s, e


def save_responder_idx(fish, dir_analysis, sampling_rate_hz,
                        K_global, drift_global, lam_global,
                        input_tag="C", lag_global=0,
                        null_tag="iaaft", null_percentile=95,
                        baseline_win_min=(0.0, 15.0),
                        drug_win_min=(30.0, 45.0),
                        yhat_sign_eps=0.0, chunk_cells=20000,
                        u_ext=None,
                        overwrite=True):
    """
    Threshold ΔR² vs null and split responders into pos/neg by kernel sign.

    Sign is determined by comparing yhat(drug window) vs yhat(baseline window)
    using only the drug-locked component (U · h), so it reflects the kernel
    direction rather than the drift.

    Writes to: base_dir /
        tonic_pos_glm_{null_tag}_nullp{N}_idxs.npy
        tonic_neg_glm_{null_tag}_nullp{N}_idxs.npy
    """
    proj_ID, expt_ID = fish
    base_dir, run_dir = get_run_dir(
        dir_analysis, fish, K_global, drift_global, lam_global, input_tag, lag_global,
    )

    ptag     = int(null_percentile)
    pos_path = base_dir / f"tonic_pos_glm_{null_tag}_nullp{ptag}_idxs.npy"
    neg_path = base_dir / f"tonic_neg_glm_{null_tag}_nullp{ptag}_idxs.npy"

    if not overwrite and pos_path.exists() and neg_path.exists():
        print(f"⏩ {expt_ID}: responder idx exist, skipping.")
        return {"fish": fish, "status": "skipped_responders"}

    # load ΔR² and null threshold
    dR2_path    = run_dir / "kernel_delta_r2_fit.npy"
    _, _, thresh_path, _ = _null_output_paths(run_dir, null_tag, null_percentile)

    if not dR2_path.exists():
        raise FileNotFoundError(f"Missing ΔR²: {dR2_path}")
    if not thresh_path.exists():
        raise FileNotFoundError(f"Missing null threshold: {thresh_path}")

    dR2_real = np.load(str(dR2_path)).astype(np.float32).ravel()
    thr      = float(np.load(str(thresh_path)))

    is_resp  = np.isfinite(dR2_real) & (dR2_real > thr)
    resp_idx = np.where(is_resp)[0]
    n_resp   = resp_idx.size
    n_cells  = dR2_real.size

    if n_resp == 0:
        np.save(str(pos_path), np.array([], dtype=np.int64))
        np.save(str(neg_path), np.array([], dtype=np.int64))
        print(f"⚠️ {expt_ID}: 0 responders at {null_tag} p{ptag}")
        return {"fish": fish, "status": "ok_responders", "n_pos": 0, "n_neg": 0}

    # load kernel and RUN_META to reconstruct u_fit
    h_hat     = np.load(str(run_dir / "kernel_h_hat.npy")).astype(np.float32)[:n_cells]
    meta_path = run_dir / "RUN_META.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing RUN_META.json: {meta_path}")
    with open(str(meta_path)) as f:
        meta = json.load(f)

    fit_start  = int(meta["fit_start"])
    fit_end    = int(meta["fit_end"])
    Tfit       = int(meta["Tfit"])
    lag_frames = int(meta.get("lag_global_frames", 0))
    it         = meta.get("input_tag", "C")

    C_path  = base_dir / "C_capsaicin.npy"
    dC_path = base_dir / "dC_capsaicin.npy"

    if u_ext is not None:
        C_full = np.asarray(u_ext, dtype=np.float32).reshape(-1)
        u_fit  = C_full[fit_start:fit_end]
    elif it in ("C", "HCRT"):
        C_full = np.load(str(C_path)).astype(np.float32).ravel()
        u_fit  = C_full[fit_start:fit_end]
    else:
        u_fit  = np.load(str(dC_path)).astype(np.float32).ravel()[fit_start:fit_end]
    u_fit = shift_with_zeros(u_fit, lag_frames)

    K = int(K_global)
    U = build_U_from_input(u_fit, K).astype(np.float32)

    b0, b1 = _window_frames(sampling_rate_hz, *baseline_win_min, Tfit)
    d0, d1 = _window_frames(sampling_rate_hz, *drug_win_min,     Tfit)

    sign_vals = np.zeros(n_resp, dtype=np.float32)
    for s in range(0, n_resp, chunk_cells):
        e     = min(s + chunk_cells, n_resp)
        H     = h_hat[resp_idx[s:e]].T      # (K, chunk)
        yhat  = U @ H                        # (Tfit, chunk)
        sign_vals[s:e] = (np.mean(yhat[d0:d1], axis=0) -
                          np.mean(yhat[b0:b1], axis=0)).astype(np.float32)

    pos_idx = resp_idx[sign_vals >  yhat_sign_eps]
    neg_idx = resp_idx[sign_vals < -yhat_sign_eps]

    np.save(str(pos_path), pos_idx.astype(np.int64))
    np.save(str(neg_path), neg_idx.astype(np.int64))

    print(f"✅ {expt_ID}: responders saved | null={null_tag} p{ptag} thr={thr:.4g} | "
          f"total={n_resp}/{n_cells} pos={pos_idx.size} neg={neg_idx.size}")
    return {"fish": fish, "status": "ok_responders",
            "n_resp": n_resp, "n_pos": pos_idx.size, "n_neg": neg_idx.size}
