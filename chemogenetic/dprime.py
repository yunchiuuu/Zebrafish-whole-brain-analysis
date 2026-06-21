"""
dprime.py
=========
Phasic d′ (sensitivity index) per cell for the whole-brain analysis pipeline.

d′ measures how much the phasic fluorescence amplitude changes between
baseline and drug epochs, normalized by the pooled within-epoch variability:

    d′_i = (μ_drug - μ_baseline) / sqrt((σ²_baseline + σ²_drug) / 2)

Three amplitude modes are supported:
    "raw" — signed F_phasic (recommended; default)
    "abs" — |F_phasic| (unsigned amplitude)
    "rms" — F_phasic² (energy proxy)

Outputs saved per fish under dir_analysis / proj_ID / expt_ID/:
    phasic_dprime_cells_{mode}.npy      (n_cells,)
    phasic_deltaMean_cells_{mode}.npy   (n_cells,)

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/dprime.py
"""

import gc
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from utils.data_io import fish_dir


# ============================================================
# CORE: D′ PER CELL
# ============================================================

def dprime_per_cell(
    Fphasic,
    idx_base,
    idx_drug,
    amplitude_mode="raw",
    var_floor=1e-4,
    eps_var=1e-8,
    clip_abs=None,
):
    """
    Compute d′ and Δmean per cell between baseline and drug epochs.

    Parameters
    ----------
    Fphasic : np.ndarray or memmap, shape (n_cells, T)
        F_phasic traces.
    idx_base : slice or array-like
        Time indices for the baseline epoch.
    idx_drug : slice or array-like
        Time indices for the drug epoch.
    amplitude_mode : str
        "raw" — use F_phasic directly (signed; default).
        "abs" — use |F_phasic| (unsigned amplitude).
        "rms" — use F_phasic² (energy proxy; delta_mean in squared units).
    var_floor : float
        Minimum variance floor to stabilize the denominator. Prevents
        blowup in quiet windows where both epochs have near-zero variance.
    eps_var : float
        Additional epsilon added inside sqrt for numerical safety.
    clip_abs : float or None
        If set, clip d′ to [-clip_abs, +clip_abs] per cell.

    Returns
    -------
    dprime : np.ndarray, shape (n_cells,), float32
        Sensitivity index per cell. NaN for cells with degenerate windows.
    delta_mean : np.ndarray, shape (n_cells,), float32
        Raw mean difference (μ_drug - μ_baseline) in the amplitude domain.
    """
    # amplitude transform
    A = np.asarray(Fphasic, dtype=np.float32)
    if amplitude_mode == "raw":
        pass
    elif amplitude_mode == "abs":
        A = np.abs(A)
    elif amplitude_mode == "rms":
        A = A ** 2
    else:
        raise ValueError(f"amplitude_mode must be 'raw', 'abs', or 'rms', got {amplitude_mode!r}")

    base = A[:, idx_base]
    drug = A[:, idx_drug]

    if base.shape[1] < 2 or drug.shape[1] < 2:
        nan = np.full(A.shape[0], np.nan, dtype=np.float32)
        return nan, nan.copy()

    mu_b   = np.nanmean(base, axis=1)
    mu_d   = np.nanmean(drug, axis=1)
    var_b  = np.maximum(np.nanvar(base, axis=1, ddof=1), var_floor)
    var_d  = np.maximum(np.nanvar(drug, axis=1, ddof=1), var_floor)
    denom  = np.sqrt((var_b + var_d) / 2.0 + eps_var).astype(np.float32)

    dprime     = ((mu_d - mu_b) / denom).astype(np.float32)
    delta_mean = (mu_d - mu_b).astype(np.float32)

    dprime[~np.isfinite(dprime)] = np.nan
    dprime[denom <= 0]           = np.nan

    if clip_abs is not None:
        dprime = np.clip(dprime, -float(clip_abs), float(clip_abs))

    return dprime, delta_mean


def summarize_fish_dprime(dprime_cells):
    """
    Compute summary statistics for one fish's d′ distribution.

    Returns
    -------
    dict with keys: mean, median, p5, p25, p75, p95, p99, n_cells
    """
    x = dprime_cells[np.isfinite(dprime_cells)]
    if x.size == 0:
        return {k: np.nan for k in ("mean", "median", "p5", "p25", "p75", "p95", "p99")} | \
               {"n_cells": 0}
    p5, p25, p75, p95, p99 = np.nanpercentile(x, [5, 25, 75, 95, 99])
    return {
        "mean":    float(np.nanmean(x)),
        "median":  float(np.nanmedian(x)),
        "p5":      float(p5),
        "p25":     float(p25),
        "p75":     float(p75),
        "p95":     float(p95),
        "p99":     float(p99),
        "n_cells": int(x.size),
    }


# ============================================================
# PER-FISH RUNNER
# ============================================================

def _get_window(start_full, end_full, sr_hz, offset_sec):
    """Apply a frame offset to a window (e.g. skip early baseline frames)."""
    off = int(round(offset_sec * sr_hz))
    s   = int(start_full + off)
    e   = int(end_full)
    if e <= s:
        raise ValueError(f"Window is empty after offset: start={s}, end={e}")
    return s, e


def dprime_one_fish(
    fish,
    dir_analysis,
    baseline_start,
    baseline_end,
    drug_start,
    drug_end,
    sampling_rate_hz=1.0,
    offset_sec=15 * 60,
    amplitude_mode="raw",
    var_floor=1e-4,
    eps_var=1e-8,
    clip_abs=None,
    overwrite=False,
):
    """
    Compute and save d′ per cell for one fish.

    The offset_sec parameter skips the first `offset_sec` of each window
    (e.g. 15 min) to avoid edge artefacts from F_phasic decomposition.

    Reads  : dir_analysis / proj_ID / expt_ID / data_array_f_phasic.npy
    Writes : dir_analysis / proj_ID / expt_ID /
                phasic_dprime_cells_{amplitude_mode}.npy
                phasic_deltaMean_cells_{amplitude_mode}.npy

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    baseline_start, baseline_end : int
        Frame indices from config (before offset is applied).
    drug_start, drug_end : int
        Frame indices from config (before offset is applied).
    sampling_rate_hz : float
    offset_sec : float
        Seconds to skip at the start of each window. Default 15 min.
    amplitude_mode : str
        "raw", "abs", or "rms".
    overwrite : bool

    Returns
    -------
    dict with fish, status, and summary stats.
    """
    proj_ID, expt_ID = fish
    out_dir = fish_dir(dir_analysis, fish)

    dprime_path = out_dir / f"phasic_dprime_cells_{amplitude_mode}.npy"
    dmean_path  = out_dir / f"phasic_deltaMean_cells_{amplitude_mode}.npy"

    if not overwrite and dprime_path.exists() and dmean_path.exists():
        print(f"⏩ {expt_ID}: d′ exists, skipping.")
        return {"fish": fish, "status": "skipped"}

    Fp_path = out_dir / "data_array_f_phasic.npy"
    if not Fp_path.exists():
        raise FileNotFoundError(f"Missing F_phasic for {expt_ID}: {Fp_path}")

    Fp = np.load(str(Fp_path), mmap_mode="r")
    n_cells, Tfull = Fp.shape

    b_s, b_e = _get_window(baseline_start, baseline_end, sampling_rate_hz, offset_sec)
    d_s, d_e = _get_window(drug_start,     drug_end,     sampling_rate_hz, offset_sec)

    if b_e > Tfull or d_e > Tfull:
        raise ValueError(
            f"{expt_ID}: window exceeds Tfull={Tfull} "
            f"(baseline [{b_s},{b_e}), drug [{d_s},{d_e}))"
        )

    dprime_cells, delta_mean_cells = dprime_per_cell(
        Fp,
        idx_base=slice(b_s, b_e),
        idx_drug=slice(d_s, d_e),
        amplitude_mode=amplitude_mode,
        var_floor=var_floor,
        eps_var=eps_var,
        clip_abs=clip_abs,
    )

    np.save(str(dprime_path), dprime_cells)
    np.save(str(dmean_path),  delta_mean_cells)

    summ = summarize_fish_dprime(dprime_cells)

    print(
        f"✅ {expt_ID} | mean d′={summ['mean']:.3f} median={summ['median']:.3f} "
        f"p95={summ['p95']:.3f} n={summ['n_cells']}"
    )

    del Fp, dprime_cells, delta_mean_cells
    gc.collect()

    return {"fish": fish, "status": "ok", **summ}
