"""
temporal_windows.py
===================
Compute per-cell metrics for each temporal window.

Tonic ΔZ:
    ΔZ_i(w) = (μ_drug_window_w - μ_baseline) / σ_baseline

Phasic d':
    d'_i(w) = (μ_phasic_window_w - μ_phasic_baseline) / σ_phasic_baseline

Baseline: vols 1200–2699  (min 20–45; last 25 min of baseline epoch,
          avoids the ~20 min habituation transient after fish mounting)

Temporal windows (15-min / 900-vol wide, 10-min / 600-vol step):
    W0   0–15 min  vols 2700–3600   drug
    W1  10–25 min  vols 3300–4200   drug
    W2  20–35 min  vols 3900–4800   drug
    W3  30–45 min  vols 4500–5400   drug → washout transition
    W4  40–55 min  vols 5100–6000   washout
    W5  50–65 min  vols 5700–6600   washout
    W6  60–75 min  vols 6300–7200   washout  (last vol = end of recording)

I/O conventions (must match run_decompose.py):
    Input  reads from:  fish_dir(dir_analysis, fish) / "f_tonic.npy"
                        fish_dir(dir_analysis, fish) / "f_phasic.npy"
    Output writes to:   fish_dir(dir_analysis, fish) / "epoch_dz.npy"
                        fish_dir(dir_analysis, fish) / "epoch_dprime.npy"

Location:
    ~/zwba/chemogenetic/temporal_windows.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ── recording constants (1 vol / sec) ─────────────────────────────────────
BASELINE_START: int = 1200   # inclusive  (20 min)
BASELINE_END:   int = 2700   # exclusive  (45 min = csn_onset)

# (label, start_vol_inclusive, end_vol_exclusive)
TEMPORAL_WINDOWS: list[tuple[str, int, int]] = [
    ("0–15 min",  2700, 3600),
    ("10–25 min", 3300, 4200),
    ("20–35 min", 3900, 4800),
    ("30–45 min", 4500, 5400),
    ("40–55 min", 5100, 6000),
    ("50–65 min", 5700, 6600),
    ("60–75 min", 6300, 7200),
]
N_WINDOWS:     int       = len(TEMPORAL_WINDOWS)
WINDOW_LABELS: list[str] = [w[0] for w in TEMPORAL_WINDOWS]


# ── tonic ΔZ ──────────────────────────────────────────────────────────────

def compute_epoch_dz(
    f_tonic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-cell tonic ΔZ for each temporal window.

    Parameters
    ----------
    f_tonic : np.ndarray, shape (n_cells, n_vols)
        Tonic fluorescence traces (output of run_decompose → f_tonic.npy).

    Returns
    -------
    dz : np.ndarray, shape (n_cells, N_WINDOWS), dtype float32
    mu_baseline : np.ndarray, shape (n_cells,)
        Per-cell baseline mean (for QC).
    sigma_baseline : np.ndarray, shape (n_cells,)
        Per-cell baseline SD (for QC).
    """
    baseline = f_tonic[:, BASELINE_START:BASELINE_END]
    mu_bl    = baseline.mean(axis=1)
    sigma_bl = baseline.std(axis=1)
    # Guard against silent / flat cells
    sigma_bl_safe = np.where(sigma_bl < 1e-6, 1e-6, sigma_bl)

    dz = np.empty((f_tonic.shape[0], N_WINDOWS), dtype=np.float32)
    for w, (_, start, end) in enumerate(TEMPORAL_WINDOWS):
        mu_drug  = f_tonic[:, start:end].mean(axis=1)
        dz[:, w] = (mu_drug - mu_bl) / sigma_bl_safe

    return dz, mu_bl.astype(np.float32), sigma_bl.astype(np.float32)


# ── phasic d' per window ───────────────────────────────────────────────────

def compute_epoch_dprime(f_phasic: np.ndarray) -> np.ndarray:
    """
    Compute per-cell phasic d' for each temporal window.

    d'_i(w) = (μ_phasic_w - μ_phasic_baseline) / σ_phasic_baseline

    Parameters
    ----------
    f_phasic : np.ndarray, shape (n_cells, n_vols)
        Phasic fluorescence traces (output of run_decompose → f_phasic.npy).

    Returns
    -------
    dprime : np.ndarray, shape (n_cells, N_WINDOWS), dtype float32
    """
    baseline_p  = f_phasic[:, BASELINE_START:BASELINE_END]
    mu_bl_p     = baseline_p.mean(axis=1)
    sigma_bl_p  = baseline_p.std(axis=1)
    sigma_safe  = np.where(sigma_bl_p < 1e-6, 1e-6, sigma_bl_p)

    dprime = np.empty((f_phasic.shape[0], N_WINDOWS), dtype=np.float32)
    for w, (_, start, end) in enumerate(TEMPORAL_WINDOWS):
        mu_w        = f_phasic[:, start:end].mean(axis=1)
        dprime[:, w] = (mu_w - mu_bl_p) / sigma_safe

    return dprime.astype(np.float32)


# ── I/O helpers ────────────────────────────────────────────────────────────
# All paths follow run_decompose.py conventions:
#   fish_dir(dir_analysis, fish)  →  the per-fish output directory
#   filenames are lowercase (f_tonic.npy, not F_tonic.npy)

def load_ftonic(fish_analysis_dir: Path) -> np.ndarray:
    """Load f_tonic.npy from the fish analysis directory."""
    p = Path(fish_analysis_dir) / "f_tonic.npy"
    if not p.exists():
        raise FileNotFoundError(f"f_tonic.npy not found: {p}\n"
                                "Run run_decompose.py first.")
    return np.load(str(p))


def load_fphasic(fish_analysis_dir: Path) -> np.ndarray:
    """Load f_phasic.npy from the fish analysis directory."""
    p = Path(fish_analysis_dir) / "f_phasic.npy"
    if not p.exists():
        raise FileNotFoundError(f"f_phasic.npy not found: {p}\n"
                                "Run run_decompose.py first.")
    return np.load(str(p))


def save_epoch_dz(dz: np.ndarray, fish_analysis_dir: Path) -> Path:
    """Save dz (n_cells × N_WINDOWS) → epoch_dz.npy"""
    out = Path(fish_analysis_dir) / "epoch_dz.npy"
    np.save(str(out), dz)
    print(f"  Saved epoch_dz:     {out}  shape={dz.shape}")
    return out


def save_epoch_dprime(dprime: np.ndarray, fish_analysis_dir: Path) -> Path:
    """Save dprime (n_cells × N_WINDOWS) → epoch_dprime.npy"""
    out = Path(fish_analysis_dir) / "epoch_dprime.npy"
    np.save(str(out), dprime)
    print(f"  Saved epoch_dprime: {out}  shape={dprime.shape}")
    return out


def load_epoch_dz(fish_analysis_dir: Path) -> np.ndarray:
    """Load epoch_dz.npy, shape (n_cells, N_WINDOWS)."""
    p = Path(fish_analysis_dir) / "epoch_dz.npy"
    if not p.exists():
        raise FileNotFoundError(f"epoch_dz.npy not found: {p}\n"
                                "Run run_temporal_intensity_map.py first.")
    return np.load(str(p))


def load_epoch_dprime(fish_analysis_dir: Path) -> np.ndarray:
    """Load epoch_dprime.npy, shape (n_cells, N_WINDOWS)."""
    p = Path(fish_analysis_dir) / "epoch_dprime.npy"
    if not p.exists():
        raise FileNotFoundError(f"epoch_dprime.npy not found: {p}\n"
                                "Run run_temporal_intensity_map.py first.")
    return np.load(str(p))
