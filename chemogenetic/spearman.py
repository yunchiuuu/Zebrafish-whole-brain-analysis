"""
spearman.py
===========
Tonic Spearman correlation between F_tonic and the lagged drug regressor C(t).

Pipeline (STEP 2):
    1. Build C(t) via drug_regressor.build_drug_regressor
    2. Scan lags 0..lag_max_sec in lag_step_sec increments
    3. For each lag, compute rank-correlation of F_tonic(t) vs C(t - lag)
       over the window [drug_start - 15min, wash_end)
    4. Save best-lag, second-best-lag, mean-across-lags, and full (n_cells, n_lags) matrix

Outputs saved per fish under dir_analysis / proj_ID / expt_ID/:
    C_capsaicin.npy              — drug regressor C(t), shape (T,)
    C_capsaicin_meta.txt         — metadata about the regressor and correlation window
    rho_tonic_lagmax.npy         — signed rho at best lag,  shape (n_cells,)
    rho_tonic_lagsecond.npy      — signed rho at 2nd-best lag
    rho_tonic_lagmean.npy        — mean rho across all lags
    rho_tonic_lag_argmax_sec.npy — best lag in seconds per cell
    rho_tonic_lag_arg2nd_sec.npy — 2nd-best lag in seconds per cell
    rho_tonic_all_lags.npy       — full matrix, shape (n_cells, n_lags)
    rho_tonic_lag_grid_sec.npy   — lag values, shape (n_lags,)

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/tonic/spearman.py
"""

import gc
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tqdm.auto import tqdm

from utils.data_io import fish_dir
from chemogenetic.drug_regressor import build_drug_regressor, delta_signal


# ============================================================
# LAGGED SPEARMAN CORRELATION
# ============================================================

def lagged_spearman_tonic(
    Ft,
    C,
    sampling_rate,
    lag_max_sec,
    lag_step_sec,
    corr_start_frame,
    corr_end_frame,
    chunk_cells=2000,
    desc="lagged spearman",
    return_all_lags=True,
):
    """
    Compute lagged Spearman correlation between F_tonic and C(t - lag).

    For each lag in [0, lag_max_sec] (stepped by lag_step_sec), computes
    the rank correlation of each cell's F_tonic trace with the lagged
    regressor over the window [corr_start_frame, corr_end_frame).

    Best and second-best lags are selected by |rho| (magnitude), but the
    *signed* rho is returned so directionality is preserved.

    Parameters
    ----------
    Ft : np.ndarray or memmap, shape (n_cells, T)
        F_tonic traces.
    C : np.ndarray, shape (T,)
        Drug concentration regressor (from build_drug_regressor).
    sampling_rate : float
        Sampling rate in Hz.
    lag_max_sec : float
        Maximum lag to scan in seconds.
    lag_step_sec : float
        Step between lags in seconds.
    corr_start_frame : int
        Start frame of the correlation window (inclusive).
    corr_end_frame : int
        End frame of the correlation window (exclusive).
    chunk_cells : int
        Cells processed per chunk (memory vs. speed tradeoff).
    desc : str
        Progress bar label.
    return_all_lags : bool
        If True, return the full (n_cells, n_lags) rho matrix.

    Returns
    -------
    rho_max : np.ndarray (n_cells,)
        Signed rho at the best lag (largest |rho|).
    rho_second : np.ndarray (n_cells,)
        Signed rho at the second-best lag.
    rho_mean : np.ndarray (n_cells,)
        Mean signed rho across all lags.
    lag_argmax_sec : np.ndarray (n_cells,)
        Best lag in seconds per cell.
    lag_arg2nd_sec : np.ndarray (n_cells,)
        Second-best lag in seconds per cell.
    rho_all : np.ndarray (n_cells, n_lags) or None
        Full rho matrix. None if return_all_lags=False.
    lag_sec_grid : np.ndarray (n_lags,)
        Lag values in seconds corresponding to columns of rho_all.
    """
    n_cells, T = Ft.shape
    if C.shape[0] != T:
        raise ValueError(f"C length {C.shape[0]} != T {T}")

    corr_start_frame = int(np.clip(int(round(corr_start_frame)), 0, T))
    corr_end_frame   = int(np.clip(int(round(corr_end_frame)),   0, T))
    if corr_end_frame <= corr_start_frame:
        raise ValueError(
            f"Invalid correlation window: start={corr_start_frame}, end={corr_end_frame}"
        )

    step_frames = max(1, int(round(lag_step_sec * sampling_rate)))
    lag_frames  = np.arange(
        0,
        int(round(lag_max_sec * sampling_rate)) + 1,
        step_frames,
        dtype=int,
    )
    lag_sec_grid = (lag_frames / sampling_rate).astype(np.float32)
    n_lags = len(lag_frames)

    rho_max    = np.full(n_cells, np.nan, dtype=np.float32)
    rho_second = np.full(n_cells, np.nan, dtype=np.float32)
    rho_mean   = np.full(n_cells, np.nan, dtype=np.float32)
    lag_argmax_sec = np.full(n_cells, np.nan, dtype=np.float32)
    lag_arg2nd_sec = np.full(n_cells, np.nan, dtype=np.float32)
    rho_all = np.full((n_cells, n_lags), np.nan, dtype=np.float32) \
        if return_all_lags else None

    pbar = tqdm(total=n_cells, desc=desc, unit="cell")

    for s in range(0, n_cells, chunk_cells):
        e = min(s + chunk_cells, n_cells)
        X_chunk = np.asarray(Ft[s:e])          # (n_chunk, T)
        rho_lags = np.full((X_chunk.shape[0], n_lags), np.nan, dtype=np.float32)

        for j, lag in enumerate(lag_frames):
            t0 = max(corr_start_frame, lag)
            t1 = min(corr_end_frame, T)
            L  = t1 - t0
            if L < 3:
                continue

            Cseg = C[t0 - lag : t1 - lag]          # (L,)
            Xseg = X_chunk[:, t0:t1]               # (n_chunk, L)

            # rank-zscore the regressor
            ry = rankdata(Cseg, method="average").astype(np.float32)
            ry = (ry - ry.mean()) / (ry.std(ddof=0) + 1e-12)

            # rank-zscore each cell segment
            rX = np.empty_like(Xseg, dtype=np.float32)
            for i in range(Xseg.shape[0]):
                r = rankdata(Xseg[i], method="average").astype(np.float32)
                r = (r - r.mean()) / (r.std(ddof=0) + 1e-12)
                rX[i] = r

            rho_lags[:, j] = np.nanmean(rX * ry[None, :], axis=1)

        if return_all_lags:
            rho_all[s:e, :] = rho_lags

        # reduce across lags
        abs_rho = np.abs(rho_lags)
        max_abs = np.nanmax(abs_rho, axis=1)
        all_nan = ~np.isfinite(max_abs)

        best_j  = np.nanargmax(abs_rho, axis=1).astype(int)
        rho_best = rho_lags[np.arange(rho_lags.shape[0]), best_j].astype(np.float32)
        best_lag_sec = lag_sec_grid[best_j].astype(np.float32)

        abs_rho2 = abs_rho.copy()
        abs_rho2[np.arange(abs_rho2.shape[0]), best_j] = -np.inf
        second_j = np.nanargmax(abs_rho2, axis=1).astype(int)
        rho_second_chunk = rho_lags[np.arange(rho_lags.shape[0]), second_j].astype(np.float32)
        second_lag_sec = lag_sec_grid[second_j].astype(np.float32)

        rho_mean_chunk = np.nanmean(rho_lags, axis=1).astype(np.float32)

        # blank out all-NaN cells
        rho_best[all_nan]        = np.nan
        rho_second_chunk[all_nan] = np.nan
        rho_mean_chunk[all_nan]  = np.nan
        best_lag_sec[all_nan]    = np.nan
        second_lag_sec[all_nan]  = np.nan

        rho_max[s:e]        = rho_best
        rho_second[s:e]     = rho_second_chunk
        rho_mean[s:e]       = rho_mean_chunk
        lag_argmax_sec[s:e] = best_lag_sec
        lag_arg2nd_sec[s:e] = second_lag_sec

        pbar.update(e - s)

    pbar.close()
    return rho_max, rho_second, rho_mean, lag_argmax_sec, lag_arg2nd_sec, rho_all, lag_sec_grid


# ============================================================
# PER-FISH RUNNER  (called by run/run_spearman.py)
# ============================================================

def compute_spearman_one_fish(
    fish,
    dir_analysis,
    sampling_rate,
    drug_start_frame,
    drug_end_frame,
    wash_end_frame,
    drug_uM=10.0,
    V_ml=15.0,
    Q_ml_min=4.5,
    lag_max_sec=20 * 60,
    lag_step_sec=5 * 60,
    baseline_pre_sec=15 * 60,
    chunk_cells=2000,
    overwrite=False,
):
    """
    Compute and save tonic Spearman outputs for one fish.

    Reads  : dir_analysis / proj_ID / expt_ID / data_array_f_tonic.npy
    Writes : dir_analysis / proj_ID / expt_ID /
                C_capsaicin.npy
                C_capsaicin_meta.txt
                rho_tonic_lagmax.npy
                rho_tonic_lagsecond.npy
                rho_tonic_lagmean.npy
                rho_tonic_lag_argmax_sec.npy
                rho_tonic_lag_arg2nd_sec.npy
                rho_tonic_all_lags.npy
                rho_tonic_lag_grid_sec.npy

    Parameters
    ----------
    fish : tuple of (str, str)
        (proj_ID, expt_ID).
    dir_analysis : str
        Base analysis output path (config.dir_analysis).
    sampling_rate : float
        Sampling rate in Hz (config.sampling_rate_hz).
    drug_start_frame, drug_end_frame, wash_end_frame : int
        Frame indices from config (drug_start, drug_end, wash_end).
    drug_uM, V_ml, Q_ml_min : float
        CSTR parameters from config.
    lag_max_sec, lag_step_sec : float
        Lag scan range in seconds.
    baseline_pre_sec : float
        Baseline extension before drug start included in correlation window.
    chunk_cells : int
        Cells per chunk for Spearman computation.
    overwrite : bool
        If False and all outputs exist, skip computation.
    """
    proj_ID, expt_ID = fish
    out_dir = fish_dir(dir_analysis, fish)
    out_dir.mkdir(parents=True, exist_ok=True)

    # output paths
    paths = {
        "rho_max":   out_dir / "rho_tonic_lagmax.npy",
        "rho_2nd":   out_dir / "rho_tonic_lagsecond.npy",
        "rho_mean":  out_dir / "rho_tonic_lagmean.npy",
        "lag_max":   out_dir / "rho_tonic_lag_argmax_sec.npy",
        "lag_2nd":   out_dir / "rho_tonic_lag_arg2nd_sec.npy",
        "rho_all":   out_dir / "rho_tonic_all_lags.npy",
        "lag_grid":  out_dir / "rho_tonic_lag_grid_sec.npy",
        "C":         out_dir / "C_capsaicin.npy",
        "meta":      out_dir / "C_capsaicin_meta.txt",
    }

    if not overwrite and all(p.exists() for p in paths.values()):
        print(f"⏩ {expt_ID}: Spearman outputs exist, skipping.")
        return

    Ft_path = out_dir / "data_array_f_tonic.npy"
    if not Ft_path.exists():
        raise FileNotFoundError(f"Missing F_tonic for {expt_ID}: {Ft_path}")

    Ft = np.load(str(Ft_path), mmap_mode="r")
    n_cells, T = Ft.shape

    # build regressor
    C = build_drug_regressor(
        T=T,
        sampling_rate_hz=sampling_rate,
        drug_start_frame=int(round(drug_start_frame)),
        drug_end_frame=int(round(drug_end_frame)),
        drug_uM=drug_uM,
        V_ml=V_ml,
        Q_ml_min=Q_ml_min,
    )
    np.save(str(paths["C"]), C.astype(np.float32))

    # correlation window: [drug_start - baseline_pre, wash_end)
    pre_frames       = int(round(baseline_pre_sec * sampling_rate))
    corr_start_frame = int(np.clip(int(round(drug_start_frame - pre_frames)), 0, T))
    corr_end_frame   = int(np.clip(int(round(wash_end_frame)), 0, T))
    if corr_end_frame <= corr_start_frame:
        raise ValueError(
            f"{expt_ID}: invalid correlation window "
            f"[{corr_start_frame}, {corr_end_frame}) for T={T}"
        )

    # write meta
    with open(str(paths["meta"]), "w") as f:
        f.write(f"sampling_rate_hz={sampling_rate}\n")
        f.write(f"drug_start={drug_start_frame}\n")
        f.write(f"drug_end={drug_end_frame}\n")
        f.write(f"wash_end={wash_end_frame}\n")
        f.write(f"baseline_pre_sec={baseline_pre_sec}\n")
        f.write(f"corr_start_frame={corr_start_frame}\n")
        f.write(f"corr_end_frame={corr_end_frame}\n")
        f.write(f"lag_max_sec={lag_max_sec}\n")
        f.write(f"lag_step_sec={lag_step_sec}\n")
        f.write(f"drug_uM={drug_uM}\n")
        f.write(f"V_ml={V_ml}\n")
        f.write(f"Q_ml_min={Q_ml_min}\n")
        f.write("model=dC/dt=(Q/V)*(Cin-C)\n")
        f.write("NOTE=Spearman computed in [drug_start-baseline_pre, wash_end)\n")

    # run Spearman
    rho_max, rho_second, rho_mean, lag_argmax, lag_arg2nd, rho_all, lag_sec_grid = \
        lagged_spearman_tonic(
            Ft=Ft,
            C=C,
            sampling_rate=sampling_rate,
            lag_max_sec=lag_max_sec,
            lag_step_sec=lag_step_sec,
            corr_start_frame=corr_start_frame,
            corr_end_frame=corr_end_frame,
            chunk_cells=chunk_cells,
            desc=f"{expt_ID} tonic~C(t) lagged",
            return_all_lags=True,
        )

    np.save(str(paths["rho_max"]),  rho_max.astype(np.float32))
    np.save(str(paths["rho_2nd"]),  rho_second.astype(np.float32))
    np.save(str(paths["rho_mean"]), rho_mean.astype(np.float32))
    np.save(str(paths["lag_max"]),  lag_argmax.astype(np.float32))
    np.save(str(paths["lag_2nd"]),  lag_arg2nd.astype(np.float32))
    np.save(str(paths["rho_all"]),  rho_all.astype(np.float32))
    np.save(str(paths["lag_grid"]), lag_sec_grid.astype(np.float32))

    print(f"✅ {expt_ID}: Spearman done | corr window frames [{corr_start_frame}, {corr_end_frame})")

    del Ft, C, rho_max, rho_second, rho_mean, lag_argmax, lag_arg2nd, rho_all
    gc.collect()
