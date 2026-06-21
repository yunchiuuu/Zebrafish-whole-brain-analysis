"""
preprocess.py
=============
Generic signal preprocessing utilities for whole-brain fluorescence data.

Functions here are pure math — no file I/O, no config dependencies, no
fish-specific logic. They operate on numpy arrays and can be called from
any run script or notebook regardless of experiment type.

More preprocessing functions (e.g. dF/F normalization, baseline correction)
can be added here as the pipeline grows.

Location:
    ~/Zebrafish-whole-brain-analysis/utils/preprocess.py
"""

import gc

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import percentile_filter
from tqdm.auto import tqdm


# ============================================================
# F_TONIC: sliding percentile baseline
# ============================================================

def compute_f_tonic(
    activity_matrix,
    sampling_rate_hz,
    window_seconds,
    f_tonic_percentile=20,
    chunk_cells=20000,
    n_jobs=20,
    dtype_out=np.float32,
    show_pbar=True,
    desc="F_tonic",
):
    """
    Compute F_tonic = sliding percentile baseline along the time axis, per cell.

    Uses scipy.ndimage.percentile_filter with reflect padding, applied in
    parallel cell chunks via joblib threads. Window length is forced to odd
    for symmetric centering behavior.

    Parameters
    ----------
    activity_matrix : np.ndarray, shape (n_cells, T)
        Raw fluorescence traces.
    sampling_rate_hz : float
        Sampling rate in Hz (e.g. 1.0 for 1 volume/sec).
    window_seconds : float
        Duration of the sliding window in seconds (e.g. 600 for 10 min).
    f_tonic_percentile : int
        Percentile for the baseline filter. Default 20 (20th percentile).
    chunk_cells : int
        Cells per parallel chunk. Default 20000.
    n_jobs : int
        Number of parallel threads. Default 20.
    dtype_out : np.dtype
        Output dtype. float32 recommended for memory efficiency.
    show_pbar : bool
        Show tqdm progress bars for dispatch and write phases.
    desc : str
        Label for the progress bar.

    Returns
    -------
    F_tonic : np.ndarray, shape (n_cells, T), dtype dtype_out
    """
    X = np.asarray(activity_matrix, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"activity_matrix must be 2D (n_cells, T). Got shape {X.shape}")
    n_cells, T = X.shape

    window_samples = int(round(window_seconds * sampling_rate_hz))
    if window_samples < 1:
        raise ValueError(f"window_seconds too small → window_samples={window_samples}")
    if window_samples % 2 == 0:
        window_samples += 1  # enforce odd for symmetric centering

    chunks = [(s, min(s + chunk_cells, n_cells)) for s in range(0, n_cells, chunk_cells)]

    def _work(s, e):
        Ft = percentile_filter(
            X[s:e],
            percentile=f_tonic_percentile,
            size=(1, window_samples),
            mode="reflect",
        ).astype(dtype_out, copy=False)
        return s, e, Ft

    iterator = tqdm(chunks, total=len(chunks), desc=f"{desc} (dispatch)", unit="chunk") \
        if show_pbar else chunks

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_work)(s, e) for (s, e) in iterator
    )

    Ft_all = np.empty((n_cells, T), dtype=dtype_out)
    write_iter = tqdm(results, total=len(results), desc=f"{desc} (write)", unit="chunk") \
        if show_pbar else results
    for s, e, Ft in write_iter:
        Ft_all[s:e] = Ft

    print(
        f"[F_tonic] done → shape={Ft_all.shape} | "
        f"window_samples={window_samples}, percentile={f_tonic_percentile}, "
        f"chunk_cells={chunk_cells}, n_jobs={n_jobs}"
    )
    return Ft_all


# ============================================================
# F_PHASIC: (F - F_tonic) / F_tonic
# ============================================================

def compute_f_phasic(
    activity_matrix,
    f_tonic,
    denom_mode="legacy",
    eps_floor=1e-6,
    chunk_cells=20000,
    n_jobs=20,
    dtype_out=np.float32,
    show_pbar=True,
    desc="F_phasic",
):
    """
    Compute F_phasic = (F - F_tonic) / denom, per cell per timepoint.

    Parameters
    ----------
    activity_matrix : np.ndarray, shape (n_cells, T)
        Raw fluorescence traces (same array used to compute F_tonic).
    f_tonic : np.ndarray, shape (n_cells, T)
        Tonic baseline, as returned by compute_f_tonic().
    denom_mode : str
        "legacy"      — original formula: denom = F_tonic; NaN where F_tonic == 0.
                        Matches the notebook exactly. Can blow up if F_tonic is tiny.
        "fixed_floor" — denom = max(F_tonic, eps_floor). More stable, no NaNs
                        from near-zero tonic values.
    eps_floor : float
        Floor value used when denom_mode="fixed_floor". Default 1e-6.
    chunk_cells, n_jobs, dtype_out, show_pbar, desc
        Same as compute_f_tonic.

    Returns
    -------
    F_phasic : np.ndarray, shape (n_cells, T), dtype dtype_out

    Notes
    -----
    Use denom_mode="legacy" to reproduce existing results exactly.
    Use denom_mode="fixed_floor" for new experiments or when F_tonic
    contains near-zero values that cause numerical blowup.
    """
    X  = np.asarray(activity_matrix, dtype=np.float32)
    Ft = np.asarray(f_tonic,         dtype=np.float32)

    if X.shape != Ft.shape:
        raise ValueError(
            f"activity_matrix shape {X.shape} must match f_tonic shape {Ft.shape}"
        )

    n_cells, T = X.shape
    chunks = [(s, min(s + chunk_cells, n_cells)) for s in range(0, n_cells, chunk_cells)]

    def _work(s, e):
        Xc  = X[s:e].astype(dtype_out, copy=False)
        Ftc = Ft[s:e].astype(dtype_out, copy=False)

        if denom_mode == "legacy":
            with np.errstate(divide="ignore", invalid="ignore"):
                Fp = np.where(
                    Ftc != 0,
                    (Xc - Ftc) / Ftc,
                    np.nan,
                ).astype(dtype_out, copy=False)

        elif denom_mode == "fixed_floor":
            den = np.maximum(Ftc, dtype_out(eps_floor))
            with np.errstate(divide="ignore", invalid="ignore"):
                Fp = ((Xc - Ftc) / den).astype(dtype_out, copy=False)

        else:
            raise ValueError(
                f"denom_mode must be 'legacy' or 'fixed_floor', got {denom_mode!r}"
            )

        return s, e, Fp

    iterator = tqdm(chunks, total=len(chunks), desc=f"{desc} (dispatch)", unit="chunk") \
        if show_pbar else chunks

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_work)(s, e) for (s, e) in iterator
    )

    Fp_all = np.empty((n_cells, T), dtype=dtype_out)
    write_iter = tqdm(results, total=len(results), desc=f"{desc} (write)", unit="chunk") \
        if show_pbar else results
    for s, e, Fp in write_iter:
        Fp_all[s:e] = Fp

    print(
        f"[F_phasic] done → shape={Fp_all.shape} | "
        f"denom_mode={denom_mode!r}, chunk_cells={chunk_cells}, n_jobs={n_jobs}"
    )
    return Fp_all
