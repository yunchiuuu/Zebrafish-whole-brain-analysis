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
import os
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import percentile_filter
from tqdm.auto import tqdm
import h5py
from pathlib import Path


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




def estimate_background(
    fish,
    dir_voluseg,
    n_volumes=100,
    patch_size=10,
    seed=0,
):
    """
    Estimate camera background (dark offset) for one fish from raw volume HDF5s.

    Samples N evenly-spaced volumes from {dir_voluseg}/{proj_ID}/{expt_ID}/input/,
    extracts 10x10 pixel patches from the top-left and bottom-left corners across
    all z-planes, and returns the median as a single scalar F_dark.

    Corner choice rationale:
        - Left side corners are away from the eye (right side) which can
          contaminate higher z-planes with fluorescence bleed
        - Both corners are in the dark region outside the brain/agar

    Parameters
    ----------
    fish : tuple of (str, str)
        (proj_ID, expt_ID)
    dir_voluseg : str or Path
        Base voluseg directory (config.dir_voluseg).
    n_volumes : int
        Number of evenly-spaced volumes to sample (default 100).
    patch_size : int
        Side length of corner patch in pixels (default 10 → 10x10 patch).
    seed : int
        Not used (deterministic even-spacing), kept for API consistency.

    Returns
    -------
    f_dark : float
        Median pixel value across all sampled patches — the camera background scalar.
    patch_medians : dict
        Per-corner median values for diagnostic use:
        {'top_left': float, 'bottom_left': float}

    Notes
    -----
    Volume files are expected at:
        {dir_voluseg}/{proj_ID}/{expt_ID}/input/volume*.h5
    Each file contains dataset 'default' with shape (n_planes, dim1, dim2), dtype uint16.
    """
    proj_ID, expt_ID = fish
    input_dir = Path(dir_voluseg) / proj_ID / expt_ID / "input"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # find and sort all volume HDF5 files
    vol_files = sorted(input_dir.glob("volume*.h5"))
    if len(vol_files) == 0:
        raise FileNotFoundError(f"No volume*.h5 files found in {input_dir}")

    # sample N evenly-spaced volumes
    indices = np.linspace(0, len(vol_files) - 1, min(n_volumes, len(vol_files)),
                          dtype=int)
    sampled_files = [vol_files[i] for i in indices]

    # read one file to get shape
    with h5py.File(str(sampled_files[0]), "r") as f:
        vol0 = f["default"][:]
    n_planes, dim1, dim2 = vol0.shape

    # corner patch indices — top-left and bottom-left only
    # rows: first and last `patch_size` rows; cols: first `patch_size` cols
    patches_def = {
        "top_left":    (slice(0, patch_size),          slice(0, patch_size)),
        "bottom_left": (slice(dim1 - patch_size, dim1), slice(0, patch_size)),
    }

    all_values = {k: [] for k in patches_def}

    for fpath in sampled_files:
        with h5py.File(str(fpath), "r") as f:
            vol = f["default"][:]    # shape (n_planes, dim1, dim2)

        for corner, (row_sl, col_sl) in patches_def.items():
            # extract patch across all z-planes: shape (n_planes, patch_size, patch_size)
            patch = vol[:, row_sl, col_sl].astype(np.float32)
            all_values[corner].append(patch.ravel())

    # compute per-corner medians for diagnostics
    patch_medians = {
        corner: float(np.median(np.concatenate(vals)))
        for corner, vals in all_values.items()
    }

    # pool all corners → single scalar
    all_pooled = np.concatenate([
        np.concatenate(vals) for vals in all_values.values()
    ])
    f_dark = float(np.median(all_pooled))

    return f_dark, patch_medians


def subtract_background(f_raw, f_dark, clip_min=0.0):
    """
    Subtract camera background offset from raw fluorescence traces.

    Applied before F_tonic/F_phasic decomposition so that both
    downstream computations operate on true fluorescence (offset-removed).

    Parameters
    ----------
    f_raw : np.ndarray, shape (n_cells, T)
        Raw fluorescence traces from voluseg.
    f_dark : float
        Camera background scalar from estimate_background().
    clip_min : float
        Minimum value after subtraction (default 0.0 — clips negative values
        that can arise from read noise). Set to None to disable clipping.

    Returns
    -------
    f_corrected : np.ndarray, shape (n_cells, T), dtype float32
        Background-subtracted fluorescence traces.

    Notes
    -----
    This implements:
        F_corrected = F_raw - F_dark

    After this correction:
        F_tonic  = percentile_10(F_corrected, sliding window)
        F_phasic = (F_corrected - F_tonic) / F_tonic
    Both denominators are now in true-fluorescence units.
    """
    f_corrected = np.asarray(f_raw, dtype=np.float32) - float(f_dark)
    if clip_min is not None:
        np.clip(f_corrected, clip_min, None, out=f_corrected)
    return f_corrected