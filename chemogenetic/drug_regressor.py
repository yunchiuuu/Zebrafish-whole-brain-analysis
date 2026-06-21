"""
drug_regressor.py
=================
Drug concentration regressors for the tonic analysis pipeline.

Implements the CSTR (continuous stirred-tank reactor) model used to estimate
chamber drug concentration over time, and the first-difference operator used
to produce the dC/dt regressor variant.

Both C(t) and dC(t) are used by:
    - analysis/spearman.py : as the signal correlated against F_tonic
    - analysis/glm.py      : as the input u(t) convolved with the kernel h,
                             via glm.build_drug_concentration which wraps
                             build_drug_regressor and saves outputs

Location:
    ~/Zebrafish-whole-brain-analysis/chemogenetic/analysis/drug_regressor.py
"""

import numpy as np


def build_drug_regressor(
    T,
    sampling_rate_hz,
    drug_start_frame,
    drug_end_frame,
    drug_uM=10.0,
    V_ml=15.0,
    Q_ml_min=4.5,
):
    """
    Build capsaicin concentration regressor C(t) via CSTR model.

    Model: dC/dt = (Q/V) * (Cin(t) - C(t))

    Cin(t) = drug_uM during [drug_start_frame, drug_end_frame), 0 otherwise.
    C(0)   = 0 (chamber starts drug-free).

    Parameters
    ----------
    T : int
        Total number of timepoints.
    sampling_rate_hz : float
        Sampling rate in Hz (e.g. 1.0 for 1 volume/sec).
    drug_start_frame : int
        Frame at which drug perfusion begins (Cin switches to drug_uM).
    drug_end_frame : int
        Frame at which drug perfusion ends (Cin switches back to 0).
    drug_uM : float
        Drug concentration in the input line (µM). Default 10.0.
    V_ml : float
        Chamber volume in mL. Default 15.0.
    Q_ml_min : float
        Perfusion flow rate in mL/min. Default 4.5.

    Returns
    -------
    C : np.ndarray, shape (T,), dtype float32
        Estimated chamber drug concentration at each timepoint.

    Notes
    -----
    For fish run at 5 µM instead of 10 µM, pass drug_uM=5.0 explicitly.
    Use drug_uM_per_fish from config to look up the correct value per fish.
    """
    drug_start_frame = int(np.clip(drug_start_frame, 0, T))
    drug_end_frame   = int(np.clip(drug_end_frame,   0, T))

    dt_min = (1.0 / sampling_rate_hz) / 60.0
    k = Q_ml_min / V_ml          # washout rate constant (per minute)

    Cin = np.zeros(T, dtype=np.float32)
    Cin[drug_start_frame:drug_end_frame] = np.float32(drug_uM)

    C = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        C[t] = C[t - 1] + k * (Cin[t - 1] - C[t - 1]) * dt_min

    return C


def delta_signal(x):
    """
    First difference: dC(t) = C(t) - C(t-1), with dC(0) = 0.

    Used to produce the dC/dt regressor variant when input_tag = "dC"
    in the GLM. Represents the rate of change of drug concentration.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input signal (typically C(t) from build_drug_regressor).

    Returns
    -------
    np.ndarray, shape (T,), dtype float32
    """
    x = np.asarray(x, dtype=np.float32)
    dx = np.zeros_like(x)
    dx[1:] = x[1:] - x[:-1]
    return dx
