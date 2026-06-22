"""
responders_effectsize.py
========================
Effect size quantification for tonic GLM responders.

Two methods for measuring how much F_tonic changes in drug vs baseline,
computed only within the responder subsets (pos/neg) identified by glm.py:

    1. Fixed-window ΔZ  (STEP 4 in notebook, cell 58)
       ΔZ_i = (mean(F_tonic[drug]) - mean(F_tonic[baseline])) / std(F_tonic[baseline])
       — straightforward mean shift normalized to baseline variability.
       — saved as per-cell vectors, fish-level mean is the comparison metric.

    2. Plateau ΔZ  (STEP 4 plateau, cell 60)
       Scans a sliding window of length L_min within the drug epoch and finds
       the peak (pos) or trough (neg) sustained mean z-score per cell.
       — captures the *peak sustained response* rather than the overall mean.
       — more robust to cells that respond transiently or with a lag.

Outputs per fish under dir_analysis / proj_ID / expt_ID/:

    Fixed-window ΔZ (saved under results_dz_vectors/):
        tonic_dz_glm_{null_tag}_nullp{p}_b{b0}-{b1}_d{d0}-{d1}_clip{c}_pos.npy
        tonic_dz_glm_{null_tag}_nullp{p}_b{b0}-{b1}_d{d0}-{d1}_clip{c}_neg.npy

    Plateau ΔZ (saved directly in fish dir):
        tonic_pos_plateauDz_{null_tag}_L{L}min_nullp{p}.npy
        tonic_neg_plateauDz_{null_tag}_L{L}min_nullp{p}.npy
        tonic_pos_plateauWin_{null_tag}_L{L}min_nullp{p}.npy  (start/end frames)
        tonic_neg_plateauWin_{null_tag}_L{L}min_nullp{p}.npy

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/responders_effectsize.py
"""

import gc
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from utils.data_io import fish_dir
from utils.stats import tqdm_joblib


EPS_SD = 1e-6


# ============================================================
# SHARED HELPERS
# ============================================================

def frames_to_min_pair(start_frame, end_frame, sampling_rate_hz):
    """Convert frame indices to (start_min, end_min) tuple."""
    return (
        float(start_frame) / float(sampling_rate_hz) / 60.0,
        float(end_frame)   / float(sampling_rate_hz) / 60.0,
    )


def minutes_to_frames(min_pair, sr_hz, Tfull):
    """Convert (start_min, end_min) to frame indices, clipped to [0, Tfull]."""
    s = int(np.clip(int(round(min_pair[0] * 60.0 * sr_hz)), 0, Tfull))
    e = int(np.clip(int(round(min_pair[1] * 60.0 * sr_hz)), 0, Tfull))
    if e <= s:
        raise ValueError(
            f"Window {min_pair} min is empty after clipping: [{s},{e}) / Tfull={Tfull}"
        )
    return s, e


def _load_responder_idx(fish_dir_path, null_tag, null_percentile):
    """
    Load pos/neg responder index files for a fish.

    Tries tagged filename first (tonic_pos_glm_{null_tag}_nullp{p}_idxs.npy),
    then legacy fallback.
    """
    ptag = int(null_percentile)

    candidates = [
        (
            fish_dir_path / f"tonic_pos_glm_{null_tag}_nullp{ptag}_idxs.npy",
            fish_dir_path / f"tonic_neg_glm_{null_tag}_nullp{ptag}_idxs.npy",
        ),
        (
            fish_dir_path / f"tonic_pos_glm_nullp{ptag}_idxs.npy",
            fish_dir_path / f"tonic_neg_glm_nullp{ptag}_idxs.npy",
        ),
    ]

    for pos_path, neg_path in candidates:
        if pos_path.exists() and neg_path.exists():
            return (
                np.load(str(pos_path)).astype(np.int64),
                np.load(str(neg_path)).astype(np.int64),
                pos_path,
                neg_path,
            )

    tried = [str(p) for pair in candidates for p in pair]
    raise FileNotFoundError(
        f"Missing responder idx files for null_tag={null_tag!r} p{ptag}.\nTried:\n  "
        + "\n  ".join(tried)
    )


# ============================================================
# METHOD 1: FIXED-WINDOW ΔZ
# ============================================================

def _dz_output_paths(cache_dir, null_tag, null_percentile,
                     baseline_min_pair, drug_min_pair, clip_abs):
    """Build canonical output filenames for fixed-window ΔZ vectors."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    b0, b1 = baseline_min_pair
    d0, d1 = drug_min_pair
    clip = "none" if clip_abs is None else f"{float(clip_abs):g}"
    stem = (
        f"tonic_dz_glm_{null_tag}_nullp{int(null_percentile)}"
        f"_b{b0:g}-{b1:g}_d{d0:g}-{d1:g}_clip{clip}"
    )
    return cache_dir / f"{stem}_pos.npy", cache_dir / f"{stem}_neg.npy"


def _compute_dz_for_idxs(Ft_full, idx, b0, b1, d0, d1, clip_abs):
    """Compute ΔZ = (mean_drug - mean_base) / (std_base + eps) for a set of cell indices."""
    if idx.size == 0:
        return np.zeros(0, dtype=np.float32)

    Ft_sub   = np.asarray(Ft_full[idx, :], dtype=np.float32)
    mu_base  = np.mean(Ft_sub[:, b0:b1], axis=1)
    sd_base  = np.std( Ft_sub[:, b0:b1], axis=1, ddof=0)
    mu_drug  = np.mean(Ft_sub[:, d0:d1], axis=1)

    dz = (mu_drug - mu_base) / (sd_base + EPS_SD)

    if clip_abs is not None:
        dz = np.clip(dz, -float(clip_abs), float(clip_abs))

    return dz[np.isfinite(dz)].astype(np.float32)


def fixed_dz_one_fish(
    fish,
    dir_analysis,
    sampling_rate_hz,
    baseline_min_pair,
    drug_min_pair,
    null_tag,
    null_percentile,
    clip_abs=50.0,
    cache_root=None,
    overwrite=False,
):
    """
    Compute and save fixed-window ΔZ vectors for one fish's pos/neg responders.

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    sampling_rate_hz : float
    baseline_min_pair : tuple (start_min, end_min)
    drug_min_pair : tuple (start_min, end_min)
    null_tag : str  — "iaaft" or "shift"
    null_percentile : int  — e.g. 95
    clip_abs : float or None  — clip ΔZ to [-clip_abs, +clip_abs]
    cache_root : str or None  — defaults to dir_analysis/proj_ID/results_dz_vectors
    overwrite : bool

    Returns
    -------
    dict with fish, status, n_pos, n_neg, pos_path, neg_path
    """
    proj_ID, expt_ID = fish
    base_dir  = fish_dir(dir_analysis, fish)

    if cache_root is None:
        # store under the proj_ID level, not per-fish, matching notebook convention
        cache_root = Path(dir_analysis) / proj_ID / "results_dz_vectors"

    fish_cache = Path(cache_root) / expt_ID
    pos_out, neg_out = _dz_output_paths(
        fish_cache, null_tag, null_percentile,
        baseline_min_pair, drug_min_pair, clip_abs,
    )

    if not overwrite and pos_out.exists() and neg_out.exists():
        pos = np.load(str(pos_out), mmap_mode="r")
        neg = np.load(str(neg_out), mmap_mode="r")
        return {"fish": fish, "status": "loaded",
                "n_pos": pos.size, "n_neg": neg.size,
                "pos_path": pos_out, "neg_path": neg_out}

    Ft_path = base_dir / "data_array_f_tonic.npy"
    if not Ft_path.exists():
        return {"fish": fish, "status": "missing_Ft", "n_pos": 0, "n_neg": 0}

    Ft = np.load(str(Ft_path), mmap_mode="r")
    n_cells, Tfull = Ft.shape

    b0, b1 = minutes_to_frames(baseline_min_pair, sampling_rate_hz, Tfull)
    d0, d1 = minutes_to_frames(drug_min_pair,     sampling_rate_hz, Tfull)

    try:
        pos_idx, neg_idx, _, _ = _load_responder_idx(base_dir, null_tag, null_percentile)
    except FileNotFoundError:
        return {"fish": fish, "status": "missing_idx", "n_pos": 0, "n_neg": 0}

    pos_idx = pos_idx[(pos_idx >= 0) & (pos_idx < n_cells)]
    neg_idx = neg_idx[(neg_idx >= 0) & (neg_idx < n_cells)]

    dz_pos = _compute_dz_for_idxs(Ft, pos_idx, b0, b1, d0, d1, clip_abs)
    dz_neg = _compute_dz_for_idxs(Ft, neg_idx, b0, b1, d0, d1, clip_abs)

    np.save(str(pos_out), dz_pos)
    np.save(str(neg_out), dz_neg)

    return {"fish": fish, "status": "computed",
            "n_pos": dz_pos.size, "n_neg": dz_neg.size,
            "pos_path": pos_out, "neg_path": neg_out}


# ============================================================
# METHOD 2: PLATEAU ΔZ
# ============================================================

def _running_mean_matrix(Z, W):
    """
    Fast running mean over axis=1 using cumsum.

    Parameters
    ----------
    Z : np.ndarray, shape (n_cells, T)
    W : int, window length

    Returns
    -------
    np.ndarray, shape (n_cells, T - W + 1)
    """
    cs = np.cumsum(Z, axis=1, dtype=np.float64)
    cs = np.concatenate([np.zeros((Z.shape[0], 1), dtype=np.float64), cs], axis=1)
    return ((cs[:, W:] - cs[:, :-W]) / float(W)).astype(np.float32)


def plateau_delta_z_cells(
    Ft_full,
    idx_cells,
    sr_hz,
    L_min,
    baseline_min_pair,
    drug_epoch_min_pair,
    sign="pos",
    chunk_cells=4000,
):
    """
    Compute plateau ΔZ per cell by scanning a sustained window within the drug epoch.

    For each cell:
      1. z-score F_tonic relative to baseline mean/std
      2. compute running mean z-score with window length L_min within drug epoch
      3. plateau ΔZ = max running mean (pos) or min running mean (neg)

    Also returns the start/end frames of the plateau window per cell.

    Parameters
    ----------
    Ft_full : np.ndarray or memmap, shape (n_cells, T)
    idx_cells : np.ndarray of int
        Indices of the responder subset (pos or neg).
    sr_hz : float
    L_min : float
        Plateau window duration in minutes.
    baseline_min_pair : tuple (start_min, end_min)
    drug_epoch_min_pair : tuple (start_min, end_min)
    sign : str
        "pos" — find max plateau (positive responders).
        "neg" — find min plateau (negative responders).
    chunk_cells : int

    Returns
    -------
    dz_out : np.ndarray, shape (n_resp,), float32
    ws_out : np.ndarray, shape (n_resp,), int64  — plateau start frame
    we_out : np.ndarray, shape (n_resp,), int64  — plateau end frame
    """
    idx_cells = np.asarray(idx_cells, dtype=np.int64)
    if idx_cells.size == 0:
        return np.array([], np.float32), np.array([], np.int64), np.array([], np.int64)

    n_cells, Tfull = Ft_full.shape
    idx_cells = idx_cells[(idx_cells >= 0) & (idx_cells < n_cells)]
    if idx_cells.size == 0:
        return np.array([], np.float32), np.array([], np.int64), np.array([], np.int64)

    b0, b1 = minutes_to_frames(baseline_min_pair,    sr_hz, Tfull)
    d0, d1 = minutes_to_frames(drug_epoch_min_pair,  sr_hz, Tfull)

    W     = int(round(L_min * 60.0 * sr_hz))
    Tdrug = d1 - d0
    if W < 2:
        raise ValueError(f"L_min={L_min} too small → W={W} frames")
    if W > Tdrug:
        raise ValueError(
            f"L_min={L_min} min (W={W}) exceeds drug epoch ({Tdrug} frames). Reduce L_min."
        )

    n_resp = idx_cells.size
    dz_out = np.zeros(n_resp, dtype=np.float32)
    ws_out = np.zeros(n_resp, dtype=np.int64)
    we_out = np.zeros(n_resp, dtype=np.int64)

    for s in range(0, n_resp, chunk_cells):
        e   = min(s + chunk_cells, n_resp)
        idx = idx_cells[s:e]

        Ft_sub   = np.asarray(Ft_full[idx, :], dtype=np.float32)
        base_seg = Ft_sub[:, b0:b1]
        mu_b     = np.mean(base_seg, axis=1)
        sd_b     = np.std( base_seg, axis=1, ddof=0) + EPS_SD

        Zdrug = (Ft_sub[:, d0:d1] - mu_b[:, None]) / sd_b[:, None]
        R     = _running_mean_matrix(Zdrug, W)

        if sign == "pos":
            k  = np.argmax(R, axis=1)
            dz = R[np.arange(R.shape[0]), k]
        elif sign == "neg":
            k  = np.argmin(R, axis=1)
            dz = R[np.arange(R.shape[0]), k]
        else:
            raise ValueError(f"sign must be 'pos' or 'neg', got {sign!r}")

        dz_out[s:e] = dz.astype(np.float32)
        ws_out[s:e] = (d0 + k).astype(np.int64)
        we_out[s:e] = (d0 + k + W).astype(np.int64)

    return dz_out, ws_out, we_out


def plateau_dz_one_fish(
    fish,
    dir_analysis,
    sampling_rate_hz,
    baseline_min_pair,
    drug_epoch_min_pair,
    null_tag,
    null_percentile,
    L_min=20.0,
    chunk_cells=4000,
    save_per_cell=True,
    overwrite=False,
):
    """
    Compute and save plateau ΔZ per cell for one fish's pos/neg responders.

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    sampling_rate_hz : float
    baseline_min_pair : tuple (start_min, end_min)
    drug_epoch_min_pair : tuple (start_min, end_min)
    null_tag : str
    null_percentile : int
    L_min : float
        Plateau window duration in minutes (config.L_MIN).
    save_per_cell : bool
        Save per-cell ΔZ and window arrays to disk.
    overwrite : bool

    Returns
    -------
    dict with fish, fish_pos, fish_neg, status, log
    """
    proj_ID, expt_ID = fish
    base_dir = fish_dir(dir_analysis, fish)
    ptag     = int(null_percentile)
    Ltag     = f"{int(round(L_min))}min"
    nt       = f"_{null_tag}"

    pos_dz_path  = base_dir / f"tonic_pos_plateauDz{nt}_L{Ltag}_nullp{ptag}.npy"
    neg_dz_path  = base_dir / f"tonic_neg_plateauDz{nt}_L{Ltag}_nullp{ptag}.npy"

    if not overwrite and pos_dz_path.exists() and neg_dz_path.exists():
        dz_pos = np.load(str(pos_dz_path))
        dz_neg = np.load(str(neg_dz_path))
        return {
            "fish": fish, "status": "loaded",
            "fish_pos": float(np.nanmean(dz_pos)) if dz_pos.size else np.nan,
            "fish_neg": float(np.nanmean(dz_neg)) if dz_neg.size else np.nan,
        }

    Ft_path = base_dir / "data_array_f_tonic.npy"
    if not Ft_path.exists():
        return {"fish": fish, "status": "missing_Ft", "fish_pos": np.nan, "fish_neg": np.nan}

    Ft = np.load(str(Ft_path), mmap_mode="r")

    try:
        pos_idx, neg_idx, _, _ = _load_responder_idx(base_dir, null_tag, null_percentile)
    except FileNotFoundError as e:
        return {"fish": fish, "status": f"missing_idx: {e}", "fish_pos": np.nan, "fish_neg": np.nan}

    dz_pos, ws_pos, we_pos = plateau_delta_z_cells(
        Ft, pos_idx, sampling_rate_hz, L_min,
        baseline_min_pair, drug_epoch_min_pair, sign="pos", chunk_cells=chunk_cells,
    )
    dz_neg, ws_neg, we_neg = plateau_delta_z_cells(
        Ft, neg_idx, sampling_rate_hz, L_min,
        baseline_min_pair, drug_epoch_min_pair, sign="neg", chunk_cells=chunk_cells,
    )

    fish_pos = float(np.nanmean(dz_pos)) if dz_pos.size else np.nan
    fish_neg = float(np.nanmean(dz_neg)) if dz_neg.size else np.nan

    log = (
        f"{expt_ID} | plateau Δz L={L_min:g}min | "
        f"base={baseline_min_pair[0]:g}-{baseline_min_pair[1]:g}min | "
        f"drug={drug_epoch_min_pair[0]:g}-{drug_epoch_min_pair[1]:g}min | "
        f"pos mean={fish_pos:.3f} (n={dz_pos.size}) | "
        f"neg mean={fish_neg:.3f} (n={dz_neg.size})"
    )

    if save_per_cell:
        np.save(str(pos_dz_path), dz_pos.astype(np.float32))
        np.save(str(neg_dz_path), dz_neg.astype(np.float32))
        np.save(
            str(base_dir / f"tonic_pos_plateauWin{nt}_L{Ltag}_nullp{ptag}.npy"),
            np.stack([ws_pos, we_pos], axis=1).astype(np.int64),
        )
        np.save(
            str(base_dir / f"tonic_neg_plateauWin{nt}_L{Ltag}_nullp{ptag}.npy"),
            np.stack([ws_neg, we_neg], axis=1).astype(np.int64),
        )

    del Ft
    gc.collect()

    return {
        "fish": fish, "status": "ok",
        "fish_pos": fish_pos, "fish_neg": fish_neg,
        "log": log,
    }
