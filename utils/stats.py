"""
stats.py
========
Shared statistical helper functions used across the analysis pipeline.

All functions are pure (no side effects, no file I/O) and have no
dependency on config or any other pipeline module.

Location:
    ~/Zebrafish-whole-brain-analysis/shared/stats.py
"""

import contextlib

import joblib
import numpy as np
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm


# ============================================================
# MANN-WHITNEY U TEST
# ============================================================

def mw_test_and_stars(x, y):
    """
    Two-sided Mann-Whitney U test with star annotation.

    Filters non-finite values before testing. Returns (nan, 'n/a') if
    either group has fewer than 2 finite values.

    Parameters
    ----------
    x, y : array-like
        Per-fish values for two groups.

    Returns
    -------
    p : float
        Two-sided p-value (or np.nan if test could not be run).
    stars : str
        Significance annotation: '***', '**', '*', 'ns', or 'n/a'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size < 2 or y.size < 2:
        return np.nan, "n/a"

    _, p = mannwhitneyu(x, y, alternative="two-sided")
    return float(p), star_from_p(p)


# ============================================================
# SIGNIFICANCE ANNOTATIONS
# ============================================================

def star_from_p(p):
    """
    Convert a p-value to a star annotation string.

    Parameters
    ----------
    p : float

    Returns
    -------
    str : '****', '***', '**', '*', 'ns', or 'n/a' (if non-finite).
    """
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def add_sig_bar(ax, x1, x2, y, text, bar_h=None, text_pad=None):
    """
    Draw a significance bracket on a matplotlib axis.

    The bracket runs horizontally between x1 and x2 at height y, with
    short vertical ticks at each end and text centered above.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x1, x2 : float
        X positions of the two groups being compared.
    y : float
        Bottom of the bracket in data coordinates.
    text : str
        Label to display above the bracket (e.g. '** (p=0.003)').
    bar_h : float or None
        Height of the bracket ticks in data coordinates.
        Defaults to 2% of the current y-axis range.
    text_pad : float or None
        Vertical offset of text above the bracket top.
        Defaults to 1% of the current y-axis range.
    """
    ylim = ax.get_ylim()
    y_range = abs(ylim[1] - ylim[0])

    if bar_h is None:
        bar_h = 0.02 * y_range if y_range > 0 else 0.1
    if text_pad is None:
        text_pad = 0.01 * y_range if y_range > 0 else 0.05

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + bar_h, y + bar_h, y],
        lw=1.5, color="black"
    )
    ax.text(
        (x1 + x2) / 2,
        y + bar_h + text_pad,
        text,
        ha="center", va="bottom", fontsize=10
    )


# ============================================================
# JOBLIB + TQDM INTEGRATION
# ============================================================

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager that patches joblib to report progress into a tqdm bar.

    Usage
    -----
    with tqdm_joblib(tqdm(total=n_fish, desc="Processing fish")):
        results = Parallel(n_jobs=8)(
            delayed(process_one_fish)(fish) for fish in all_fish
        )

    Parameters
    ----------
    tqdm_object : tqdm instance
        A tqdm progress bar (e.g. from tqdm.auto import tqdm).
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()
