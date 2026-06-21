"""
permutation.py
==============
Permutation test + Benjamini-Hochberg FDR correction for tonic and phasic
drug responses (STEP X in the analysis pipeline).

For each cell, tests whether the mean fluorescence in the drug epoch is
significantly different from the baseline epoch using a label-permutation
test (scipy.stats.permutation_test). Cells with any NaN values in either
window are skipped and assigned NaN p-values.

BH-FDR correction is then applied across all cells, preserving the full
cell index so outputs can be directly indexed back into the original arrays.

Outputs saved per fish under dir_analysis / proj_ID / expt_ID/:
    perm_p_values_and_diff_f_tonic.npy   — shape (n_cells, 2): [p-value, mean_diff]
    perm_p_values_and_diff_f_phasic.npy  — same for F_phasic
    perm_BH_tonic_pos_idxs_q{q}.npy     — BH-significant increase indices
    perm_BH_tonic_neg_idxs_q{q}.npy     — BH-significant decrease indices
    perm_BH_phasic_pos_idxs_q{q}.npy
    perm_BH_phasic_neg_idxs_q{q}.npy
    perm_BH_tonic_p_adj_q{q}.npy        — full-length BH-adjusted p-values
    perm_BH_phasic_p_adj_q{q}.npy

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/tonic/permutation.py
"""

from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from utils.data_io import fish_dir


# ============================================================
# CORE STAT FUNCTIONS
# ============================================================

def permutation_p_value(pre_data, drug_data, n_resamples=500):
    """
    Two-sample permutation test on means for a single cell.

    Skips cells with any NaN in either window.

    Parameters
    ----------
    pre_data : np.ndarray, shape (T_baseline,)
    drug_data : np.ndarray, shape (T_drug,)
    n_resamples : int

    Returns
    -------
    list of [p_value, mean_diff]
        p_value is NaN if any NaN values are present.
        mean_diff = mean(drug) - mean(pre).
    """
    if np.isnan(pre_data).any() or np.isnan(drug_data).any():
        return [np.nan, np.nan]

    mean_diff_fn = lambda x, y, axis: np.mean(y, axis=axis) - np.mean(x, axis=axis)
    result = permutation_test(
        (pre_data, drug_data),
        mean_diff_fn,
        n_resamples=n_resamples,
        vectorized=True,
    )
    mean_diff = float(np.mean(drug_data) - np.mean(pre_data))
    return [float(result.pvalue), mean_diff]


def parallel_permutation_all_cells(
    data_matrix,
    baseline_start,
    baseline_end,
    drug_start,
    drug_end,
    n_resamples=500,
    n_jobs=-1,
):
    """
    Run permutation test on all cells in parallel.

    Parameters
    ----------
    data_matrix : np.ndarray, shape (n_cells, T)
        F_tonic or F_phasic traces.
    baseline_start, baseline_end : int
        Baseline window frame indices [start, end).
    drug_start, drug_end : int
        Drug window frame indices [start, end).
    n_resamples : int
        Number of permutations per cell.
    n_jobs : int
        Parallel workers (-1 = all cores).

    Returns
    -------
    np.ndarray, shape (n_cells, 2)
        Column 0: p-values (NaN for cells with NaN data).
        Column 1: mean difference drug - baseline (NaN for skipped cells).
        Full-length output — indices align to original cell axis.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(permutation_p_value)(
            data_row[baseline_start:baseline_end],
            data_row[drug_start:drug_end],
            n_resamples=n_resamples,
        )
        for data_row in tqdm(data_matrix, desc="Permutation test (cells)", unit="cell")
    )
    return np.array(results, dtype=np.float64)  # (n_cells, 2)


def apply_bh_fdr(pval_diff, q=0.05):
    """
    Apply Benjamini-Hochberg FDR correction to raw permutation p-values.

    NaN p-values are excluded from BH correction but preserved in outputs.
    All outputs are aligned to the original full cell axis.

    Parameters
    ----------
    pval_diff : np.ndarray, shape (n_cells, 2)
        Column 0: raw p-values. Column 1: mean differences.
    q : float
        Target FDR level (e.g. 0.05).

    Returns
    -------
    inc_idx : np.ndarray of int
        Indices of BH-significant cells with mean_diff > 0 (increase).
    dec_idx : np.ndarray of int
        Indices of BH-significant cells with mean_diff < 0 (decrease).
    p_adj_full : np.ndarray, shape (n_cells,)
        BH-adjusted p-values. NaN where input p-value was NaN.
    reject_full : np.ndarray of bool, shape (n_cells,)
        True where BH rejects the null hypothesis.
    """
    pvals = pval_diff[:, 0].astype(float)
    diffs = pval_diff[:, 1].astype(float)

    ok = np.isfinite(pvals)
    p_adj_full  = np.full_like(pvals, np.nan, dtype=float)
    reject_full = np.zeros_like(pvals, dtype=bool)

    if ok.sum() == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            p_adj_full,
            reject_full,
        )

    reject, p_adj, _, _ = multipletests(pvals[ok], alpha=q, method="fdr_bh")
    p_adj_full[ok]  = p_adj
    reject_full[ok] = reject

    inc_idx = np.where(reject_full & (diffs > 0))[0]
    dec_idx = np.where(reject_full & (diffs < 0))[0]

    return inc_idx, dec_idx, p_adj_full, reject_full


# ============================================================
# PER-FISH RUNNERS  (called by run/run_permutation.py)
# ============================================================

def run_permutation_one_fish(
    fish,
    dir_analysis,
    baseline_start,
    baseline_end,
    drug_start,
    drug_end,
    n_resamples=500,
    n_jobs=28,
    overwrite=False,
):
    """
    Run permutation test on F_tonic and F_phasic for one fish and save results.

    Reads  : dir_analysis / proj_ID / expt_ID / data_array_f_tonic.npy
             dir_analysis / proj_ID / expt_ID / data_array_f_phasic.npy
    Writes : dir_analysis / proj_ID / expt_ID /
                perm_p_values_and_diff_f_tonic.npy
                perm_p_values_and_diff_f_phasic.npy

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    baseline_start, baseline_end, drug_start, drug_end : int
        Frame indices from config.
    n_resamples : int
        Permutations per cell (config.n_resample_permutation).
    n_jobs : int
        Parallel workers.
    overwrite : bool
    """
    proj_ID, expt_ID = fish
    out_dir = fish_dir(dir_analysis, fish)

    tonic_perm_path  = out_dir / "perm_p_values_and_diff_f_tonic.npy"
    phasic_perm_path = out_dir / "perm_p_values_and_diff_f_phasic.npy"

    if not overwrite and tonic_perm_path.exists() and phasic_perm_path.exists():
        print(f"⏩ {expt_ID}: permutation results exist, skipping.")
        return

    tonic_path  = out_dir / "data_array_f_tonic.npy"
    phasic_path = out_dir / "data_array_f_phasic.npy"

    if not tonic_path.exists() or not phasic_path.exists():
        raise FileNotFoundError(
            f"Missing F_tonic or F_phasic for {expt_ID}. Run decompose first."
        )

    print(f"▶️  {expt_ID}: running permutation test...")

    for trace_path, out_path, label in [
        (tonic_path,  tonic_perm_path,  "tonic"),
        (phasic_path, phasic_perm_path, "phasic"),
    ]:
        data = np.load(str(trace_path))
        results = parallel_permutation_all_cells(
            data,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            drug_start=drug_start,
            drug_end=drug_end,
            n_resamples=n_resamples,
            n_jobs=n_jobs,
        )
        np.save(str(out_path), results)
        print(f"  ✅ Saved {label} permutation results → {out_path}")

    print(f"✅ {expt_ID}: permutation done.")


def run_bh_one_fish(
    fish,
    dir_analysis,
    q=0.05,
    save_adj_p=True,
    suffix="perm_BH",
    overwrite=False,
):
    """
    Apply BH-FDR correction to saved permutation results for one fish.

    Reads  : dir_analysis / proj_ID / expt_ID /
                perm_p_values_and_diff_f_tonic.npy
                perm_p_values_and_diff_f_phasic.npy
    Writes : dir_analysis / proj_ID / expt_ID /
                {suffix}_tonic_pos_idxs_q{q}.npy
                {suffix}_tonic_neg_idxs_q{q}.npy
                {suffix}_phasic_pos_idxs_q{q}.npy
                {suffix}_phasic_neg_idxs_q{q}.npy
                {suffix}_tonic_p_adj_q{q}.npy     (if save_adj_p)
                {suffix}_phasic_p_adj_q{q}.npy    (if save_adj_p)

    Parameters
    ----------
    fish : tuple of (str, str)
    dir_analysis : str
    q : float
        BH FDR level (config.BH_Q).
    save_adj_p : bool
        Also save full-length adjusted p-value arrays.
    suffix : str
        Filename prefix for BH outputs.
    overwrite : bool
    """
    proj_ID, expt_ID = fish
    out_dir = fish_dir(dir_analysis, fish)

    for trace_label in ("tonic", "phasic"):
        perm_path = out_dir / f"perm_p_values_and_diff_f_{trace_label}.npy"
        if not perm_path.exists():
            raise FileNotFoundError(
                f"Missing permutation results for {expt_ID} ({trace_label}). "
                f"Run run_permutation_one_fish first."
            )

        pos_path = out_dir / f"{suffix}_{trace_label}_pos_idxs_q{q}.npy"
        neg_path = out_dir / f"{suffix}_{trace_label}_neg_idxs_q{q}.npy"

        if not overwrite and pos_path.exists() and neg_path.exists():
            print(f"⏩ {expt_ID} ({trace_label}): BH results exist, skipping.")
            continue

        pval_diff = np.load(str(perm_path))
        inc_idx, dec_idx, p_adj_full, reject_full = apply_bh_fdr(pval_diff, q=q)

        np.save(str(pos_path), inc_idx)
        np.save(str(neg_path), dec_idx)

        if save_adj_p:
            np.save(str(out_dir / f"{suffix}_{trace_label}_p_adj_q{q}.npy"), p_adj_full)
            np.save(str(out_dir / f"{suffix}_{trace_label}_reject_q{q}.npy"), reject_full)

        print(
            f"  ✅ {expt_ID} ({trace_label}) BH(q={q}): "
            f"↑={len(inc_idx)}, ↓={len(dec_idx)}"
        )
