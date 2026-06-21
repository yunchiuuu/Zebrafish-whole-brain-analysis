"""
coordinates.py
==============
Cell coordinate geometry: filtering, structuring, voxel ↔ physical transforms,
mean coordinate computation, and padding-aware index shifts.

All _old variants have been dropped; only the current versions are kept.

Location:
    ~/Zebrafish-whole-brain-analysis/registration/coordinates.py
"""

import itertools

import ants
import numpy as np
import pandas as pd


# ============================================================
# COORDINATE EXTRACTION
# ============================================================

def get_coordinates_of_cells(cell_x, cell_y, cell_z):
    """
    Filter out negative coordinate values (voluseg uses -1 as a sentinel
    for pixels that don't belong to a cell).

    Parameters
    ----------
    cell_x, cell_y, cell_z : array-like of array-like
        Per-cell arrays of pixel coordinates.

    Returns
    -------
    cell_x_positive, cell_y_positive, cell_z_positive : list of list
        Filtered coordinates (non-negative only).
    """
    cell_x_positive = [[v for v in row if v >= 0] for row in cell_x]
    cell_y_positive = [[v for v in row if v >= 0] for row in cell_y]
    cell_z_positive = [[v for v in row if v >= 0] for row in cell_z]
    return cell_x_positive, cell_y_positive, cell_z_positive


def structure_coordinates(cell_x, cell_y, cell_z):
    """
    Flatten per-cell pixel coordinates into a single DataFrame and
    compute per-cell pixel counts.

    Parameters
    ----------
    cell_x, cell_y, cell_z : array-like of array-like
        Raw per-cell coordinate arrays.

    Returns
    -------
    points_df : pd.DataFrame
        Columns ['x', 'y', 'z'] with one row per pixel.
    pixel_list : list of int
        Number of pixels per cell (for reconstructing cell identity).
    """
    cell_x_pos, cell_y_pos, cell_z_pos = get_coordinates_of_cells(cell_x, cell_y, cell_z)

    data = {
        "x": np.array(list(itertools.chain.from_iterable(cell_x_pos))),
        "y": np.array(list(itertools.chain.from_iterable(cell_y_pos))),
        "z": np.array(list(itertools.chain.from_iterable(cell_z_pos))),
    }
    points_df = pd.DataFrame(data)
    pixel_list = [len(row) for row in cell_x_pos]

    return points_df, pixel_list


# ============================================================
# VOXEL ↔ PHYSICAL TRANSFORMS
# ============================================================

def voxel_to_physical(row, image):
    """
    Convert voxel indices (i, j, k) to physical (x, y, z) coordinates.

    Parameters
    ----------
    row : pd.Series or dict-like
        Must have keys 'i', 'j', 'k'.
    image : ants.ANTsImage
        The image whose spacing/origin defines the coordinate system.

    Returns
    -------
    pd.Series with index ['x', 'y', 'z']
    """
    i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
    x, y, z = ants.transform_index_to_physical_point(image, (i, j, k))
    return pd.Series([x, y, z], index=["x", "y", "z"])


def physical_to_voxel(row, image):
    """
    Convert physical (x, y, z) coordinates to voxel indices (i, j, k).

    Parameters
    ----------
    row : pd.Series or dict-like
        Must have keys 'x', 'y', 'z'.
    image : ants.ANTsImage
        The image whose spacing/origin defines the coordinate system.

    Returns
    -------
    pd.Series with index ['i', 'j', 'k']
    """
    x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
    i, j, k = ants.transform_physical_point_to_index(image, (x, y, z))
    return pd.Series([i, j, k], index=["i", "j", "k"])


# ============================================================
# PADDING INDEX ADJUSTMENT
# ============================================================

def get_new_index_after_padding(coord, padding_tuples):
    """
    Shift coordinates to account for padding added by reshape_image().

    Parameters
    ----------
    coord : tuple of (int, int, int) or pd.DataFrame
        If tuple: (i, j, k) indices.
        If DataFrame: must have columns 'i', 'j', 'k'.
    padding_tuples : list of (int, int)
        [(pad_i_before, pad_i_after), (pad_j_before, ...), (pad_k_before, ...)].
        As returned by reshape_image().

    Returns
    -------
    tuple or pd.DataFrame
        Shifted coordinates.
    """
    pad_i_left, _ = padding_tuples[0]
    pad_j_left, _ = padding_tuples[1]
    pad_k_left, _ = padding_tuples[2]

    if isinstance(coord, tuple):
        i, j, k = coord
        return (i + pad_i_left, j + pad_j_left, k + pad_k_left)

    if isinstance(coord, pd.DataFrame):
        df = coord.copy()
        df["i"] = df["i"] + pad_i_left
        df["j"] = df["j"] + pad_j_left
        df["k"] = df["k"] + pad_k_left
        return df

    raise ValueError("coord must be a tuple or pd.DataFrame")


# ============================================================
# MEAN COORDINATES
# ============================================================

def _mean_of_non_negative(values):
    """Mean of non-negative values, rounded to int. Returns NaN if none."""
    non_neg = [v for v in values if v >= 0]
    if non_neg:
        return int(np.round(np.mean(non_neg)))
    return np.nan


def calculate_mean(cell_x, cell_y, cell_z):
    """
    Compute per-cell mean coordinates using masked arrays (ignoring negatives).

    Returns
    -------
    cell_x_means, cell_y_means, cell_z_means : np.ma.MaskedArray
    """
    cell_x_means = np.ma.mean(np.ma.masked_less(cell_x, 0), axis=1)
    cell_y_means = np.ma.mean(np.ma.masked_less(cell_y, 0), axis=1)
    cell_z_means = np.ma.mean(np.ma.masked_less(cell_z, 0), axis=1)
    return cell_x_means, cell_y_means, cell_z_means


def calculate_mean_coordinates(cell_x, cell_y, cell_z):
    """
    Compute per-cell mean coordinates as a DataFrame.

    Uses _mean_of_non_negative for integer-rounded means.

    Returns
    -------
    pd.DataFrame with columns ['x', 'y', 'z'], one row per cell.
    """
    mean_x = np.array([_mean_of_non_negative(x) for x in cell_x])
    mean_y = np.array([_mean_of_non_negative(y) for y in cell_y])
    mean_z = np.array([_mean_of_non_negative(z) for z in cell_z])

    return pd.DataFrame({"x": mean_x, "y": mean_y, "z": mean_z})
