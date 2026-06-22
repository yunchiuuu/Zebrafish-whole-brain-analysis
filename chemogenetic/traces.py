"""
traces.py
=========
Per-fish trace visualizations:
    - plot_tonic_phasic_zscore : mean z-scored F_tonic + F_phasic per fish
    - plot_glm_qc              : per-fish GLM QC overlay (data vs fit, drug term vs drift)

All functions write PDF/PNG to dir_analysis/proj_ID/expt_ID/figures/ and
optionally call plt.show() for interactive use.

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/visualization/traces.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.data_io import fish_dir


# ============================================================
# TONIC / PHASIC Z-SCORE TRACE
# ============================================================

def plot_tonic_phasic_zscore(
    fish,
    dir_analysis,
    drug_start,
    drug_end,
    drug_label="Drug",
    ylim=(-3, 5),
    figsize=(10, 4),
    save=True,
    show=True,
):
    """
    Plot mean z-scored F_tonic and F_phasic traces for one fish.

    Reads  : dir_analysis / proj_ID / expt_ID /
                data_array_f_tonic.npy
                data_array_f_phasic.npy
    Writes : dir_analysis / proj_ID / expt_ID / figures /
                F_tonic_phasic_zscore_trace.pdf

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    drug_start, drug_end : int
        Frame indices for the drug epoch (used for the highlighted window).
    drug_label : str
        Label for the drug epoch span.
    ylim : tuple
    figsize : tuple
    save : bool
    show : bool
    """
    proj_ID, expt_ID = fish
    base_dir = fish_dir(dir_analysis, fish)

    tonic_path  = base_dir / "data_array_f_tonic.npy"
    phasic_path = base_dir / "data_array_f_phasic.npy"

    if not tonic_path.exists() or not phasic_path.exists():
        print(f"⚠️  {expt_ID}: missing F_tonic or F_phasic, skipping.")
        return

    tonic  = np.load(str(tonic_path),  mmap_mode="r")
    phasic = np.load(str(phasic_path), mmap_mode="r")
    n_cells = tonic.shape[0]

    # mean across cells
    mu_t = np.nanmean(tonic,  axis=0)
    mu_p = np.nanmean(phasic, axis=0)

    # z-score each mean trace
    def _zscore(x):
        m, s = np.nanmean(x), np.nanstd(x)
        return (x - m) / (s + 1e-8)

    z_t = _zscore(mu_t)
    z_p = _zscore(mu_p)
    t   = np.arange(z_t.size)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, z_p, color="salmon",     lw=0.4, label="Phasic z-score", alpha=1.0)
    ax.plot(t, z_t, color="dodgerblue", lw=2.0, label="Tonic z-score",  alpha=0.9)

    ax.axvspan(drug_start, drug_end, color="lemonchiffon", alpha=0.5, label=drug_label)
    ax.axhline(0, color="gray", lw=0.8, ls="--")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(f"{expt_ID}\nF_Tonic vs F_Phasic mean trace (n={n_cells} cells)")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("F (z-score)")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", lw=0.5, alpha=0.7)
    plt.tight_layout()

    if save:
        fig_dir = base_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / "F_tonic_phasic_zscore_trace.pdf"
        plt.savefig(str(out), bbox_inches="tight")
        print(f"  Saved: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# GLM QC OVERLAY
# ============================================================

def _zscore_1d(y, eps=1e-8):
    y = np.asarray(y, dtype=np.float32)
    m, s = np.nanmean(y), np.nanstd(y)
    if not np.isfinite(s) or s < eps:
        return y * 0.0
    return (y - m) / (s + eps)


def _build_U(u, K):
    """Convolution design matrix. Duplicate of glm.py helper — kept local."""
    u = np.asarray(u, dtype=np.float32).ravel()
    T = u.size
    U = np.zeros((T, K), dtype=np.float32)
    for k in range(K):
        U[k:, k] = u[:T - k]
    return U


def _build_drift(T, order):
    t    = np.linspace(-1, 1, T, dtype=np.float32)
    cols = [np.ones(T, dtype=np.float32)]
    for p in range(1, int(order) + 1):
        cols.append(t ** p)
    return np.stack(cols, axis=1)


def plot_glm_qc(
    fish,
    dir_analysis,
    param_folder_name,
    n_show=10,
    random_seed=0,
    plot_mode="zscore",
    figsize=(14, 3),
    save=True,
    show=True,
):
    """
    For a random subset of responder cells, overlay:
        - F_tonic (data)
        - full GLM fit (U·h + B·β)
        - drug-locked component only (U·h)
        - drift component only (B·β)

    Reads from: dir_analysis / proj_ID / expt_ID / glm / <param_folder_name> /
        RUN_META.json, kernel_h_hat.npy, kernel_beta_hat.npy,
        X_mu.npy, X_sd.npy, kernel_delta_r2_fit.npy
    and: dir_analysis / proj_ID / expt_ID /
        data_array_f_tonic.npy, C_capsaicin.npy, dC_capsaicin.npy

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    param_folder_name : str  — config.param_folder_name
    n_show : int  — number of random responder cells to plot
    random_seed : int
    plot_mode : str  — "zscore" or "raw"
    save : bool
    show : bool
    """
    proj_ID, expt_ID = fish
    base_dir = fish_dir(dir_analysis, fish)
    run_dir  = base_dir / "glm" / param_folder_name

    meta_path = run_dir / "RUN_META.json"
    if not meta_path.exists():
        print(f"⚠️  {expt_ID}: missing RUN_META.json, skipping GLM QC.")
        return

    with open(str(meta_path)) as f:
        meta = json.load(f)

    fit_start  = int(meta["fit_start"])
    fit_end    = int(meta["fit_end"])
    Tfit       = int(meta["Tfit"])
    K          = int(meta["K_global"])
    drift_ord  = int(meta["drift_global"])
    lam        = float(meta["lam_global"])
    lag_frames = int(meta.get("lag_global_frames", 0))
    it         = meta.get("input_tag", "C")

    # load regressor
    C_path  = base_dir / "C_capsaicin.npy"
    dC_path = base_dir / "dC_capsaicin.npy"
    if not C_path.exists():
        print(f"⚠️  {expt_ID}: missing C_capsaicin.npy, skipping.")
        return

    C_full = np.load(str(C_path)).astype(np.float32).ravel()
    u_raw  = C_full[fit_start:fit_end] if it == "C" \
             else np.load(str(dC_path)).astype(np.float32).ravel()[fit_start:fit_end]

    # apply lag
    if lag_frames > 0:
        u = np.zeros_like(u_raw)
        u[lag_frames:] = u_raw[:-lag_frames]
    else:
        u = u_raw.copy()

    U = _build_U(u, K)
    B = _build_drift(Tfit, drift_ord)
    X = np.concatenate([U, B], axis=1).astype(np.float32)

    # standardize with saved mu/sd
    mu_path = run_dir / "X_mu.npy"
    sd_path = run_dir / "X_sd.npy"
    if mu_path.exists() and sd_path.exists():
        mu_X = np.load(str(mu_path))
        sd_X = np.load(str(sd_path))
        Xz = (X - mu_X) / (sd_X + 1e-8)
    else:
        Xz = X

    # load weights and ΔR²
    h_path   = run_dir / "kernel_h_hat.npy"
    beta_path = run_dir / "kernel_beta_hat.npy"
    dR2_path  = run_dir / "kernel_delta_r2_fit.npy"

    if not all(p.exists() for p in [h_path, beta_path, dR2_path]):
        print(f"⚠️  {expt_ID}: missing GLM weight files, skipping.")
        return

    h_hat    = np.load(str(h_path)).astype(np.float32)
    beta_hat = np.load(str(beta_path)).astype(np.float32)
    dR2      = np.load(str(dR2_path)).astype(np.float32).ravel()

    # load F_tonic
    Ft_path = base_dir / "data_array_f_tonic.npy"
    Ft      = np.load(str(Ft_path), mmap_mode="r")
    n_cells = Ft.shape[0]

    # pick random responder cells
    resp_idx = np.where(np.isfinite(dR2) & (dR2 > 0))[0]
    rng      = np.random.default_rng(random_seed)
    sample   = rng.choice(resp_idx, size=min(n_show, resp_idx.size), replace=False)

    Uh = Xz[:, :K]   @ h_hat.T     # drug term: (Tfit, n_cells)
    Bb = Xz[:, K:]   @ beta_hat.T  # drift term: (Tfit, n_cells)

    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(fit_start, fit_end)

    for ci in sample:
        y_raw  = np.asarray(Ft[ci, fit_start:fit_end], dtype=np.float32)
        y_fit  = Uh[:, ci] + Bb[:, ci]
        y_drug = Uh[:, ci]
        y_drft = Bb[:, ci]

        if plot_mode == "zscore":
            y_raw  = _zscore_1d(y_raw)
            y_fit  = _zscore_1d(y_fit)
            y_drug = _zscore_1d(y_drug)
            y_drft = _zscore_1d(y_drft)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, y_raw,  color="black",    lw=1.0, alpha=0.7, label="F_tonic (data)")
        ax.plot(t, y_fit,  color="red",      lw=1.5, alpha=0.9, label="Full fit")
        ax.plot(t, y_drug, color="steelblue",lw=1.2, alpha=0.8, label="Drug term (U·h)")
        ax.plot(t, y_drft, color="gray",     lw=1.0, alpha=0.6, ls="--", label="Drift (B·β)")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"{expt_ID} | cell {ci} | ΔR²={dR2[ci]:.4f}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("F" + (" z-score" if plot_mode == "zscore" else ""))
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        if save:
            out = fig_dir / f"glm_qc_cell{ci}_{param_folder_name}.pdf"
            plt.savefig(str(out), bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
