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


# ============================================================
# MEDOID COMPUTATION  (matches notebook Step 6 exactly)
# ============================================================

def compute_medoids(cell_x: np.ndarray,
                    cell_y: np.ndarray,
                    cell_z: np.ndarray) -> np.ndarray:
    """
    Compute per-cell true medoid in raw voluseg voxel space.

    The medoid is the pixel within each cell that minimises the SUM OF
    DISTANCES to all other pixels in that cell (via pairwise cdist).
    This is the true L1-medoid, not the closest-to-centroid approximation.

    Parameters
    ----------
    cell_x, cell_y, cell_z : np.ndarray, shape (n_cells, max_pixels_per_cell)
        Per-cell pixel coordinate arrays from voluseg (via read_data).
        Uses -1 as sentinel for empty/padding entries.
        All three coordinate arrays are masked jointly:
            valid = (x >= 0) & (y >= 0) & (z >= 0)

    Returns
    -------
    medoids : np.ndarray, shape (n_cells, 3), dtype int
        Per-cell medoid as [x, y, z] in voluseg voxel convention.
        Rows with no valid pixels are set to [-1, -1, -1].

    Notes
    -----
    For h2b nuclear labelling, cells have ~1–20 pixels each, so the
    O(M²) pairwise distance per cell is fast enough in a Python loop.
    """
    from scipy.spatial.distance import cdist

    N   = cell_x.shape[0]
    medoids = np.full((N, 3), fill_value=-1, dtype=int)

    for i in range(N):
        # joint validity mask across all three coordinate arrays
        valid = (cell_x[i] >= 0) & (cell_y[i] >= 0) & (cell_z[i] >= 0)
        xs = cell_x[i][valid]
        ys = cell_y[i][valid]
        zs = cell_z[i][valid]

        if xs.size == 0:
            continue

        pts = np.stack((xs, ys, zs), axis=1)          # (n_pix, 3)
        D   = cdist(pts, pts, metric='euclidean')      # (n_pix, n_pix)
        j   = np.argmin(D.sum(axis=1))                 # true medoid index
        medoids[i] = pts[j].astype(int)

    print(f"[compute_medoids] done — shape={medoids.shape}, "
          f"valid={np.sum(~np.all(medoids == -1, axis=1)):,}/{N:,}")
    return medoids


def rotate_medoids(medoids_all: np.ndarray,
                   vol_shape: tuple,
                   rotation_k: int = 2) -> np.ndarray:
    """
    Apply 180° rotation in X–Y to align medoid coordinates with the
    ANTs registration convention (rot90 k=2 applied during registration).

    Parameters
    ----------
    medoids_all : np.ndarray, shape (n_cells, 3), dtype int
        Raw medoids from compute_medoids(). Sentinel value = -1.
    vol_shape : tuple
        Shape of the reference volume as (X, Y, Z).
        Use volume_mean_raw.shape from read_data() — typically (280, 544, 40).
    rotation_k : int
        Number of 90° counter-clockwise rotations. Default 2 (180°).

    Returns
    -------
    medoids_rot : np.ndarray, shape (n_cells, 3), dtype int
        Rotated medoids. Same sentinel convention (-1 for invalid cells).
    """
    medoids_rot = medoids_all.copy()
    if rotation_k % 4 == 2:
        valid = ~np.all(medoids_rot == -1, axis=1)
        medoids_rot[valid, 0] = (vol_shape[0] - 1) - medoids_rot[valid, 0]
        medoids_rot[valid, 1] = (vol_shape[1] - 1) - medoids_rot[valid, 1]
        # z (axis=2) is unchanged
    elif rotation_k % 4 != 0:
        print(f"[rotate_medoids] ⚠️  rotation_k={rotation_k} not implemented "
              f"→ returning unrotated copy")
    return medoids_rot


# ============================================================
# MEDOID TRANSFORMATION TO TEMPLATE SPACE
# ============================================================

def transform_medoids_to_template(
    medoids_rot:    np.ndarray,
    transform_list: list,
    template_img,
    mov_ref,
    res_x:          float = 1.52,
    res_y:          float = 1.52,
    res_z:          float = 6.25,
) -> tuple:
    """
    Transform already-rotated medoid coordinates to template voxel space
    via ANTs point transforms.

    Coordinate pipeline (matches notebook Step 6 exactly):
        medoids_rot [x, y, z] in (X, Y, Z) voluseg voxel space
        → ants.transform_index_to_physical_point(mov_ref, (x, y, z))  → physical (µm)
        → ants.apply_transforms_to_points([affine, warp])              → template physical
        → ants.transform_physical_point_to_index(template_img, pt)    → template voxel

    Parameters
    ----------
    medoids_rot : np.ndarray, shape (n_cells, 3), dtype int
        Rotated medoids from rotate_medoids(). Sentinel = -1 (invalid cells).
    transform_list : list of str
        ANTs transforms in POINTS order: [affine.mat, warp.nii.gz].
        Note: this is the REVERSE of the image transform list.
        For images:  [SyN_Warp.nii.gz, SyN_GenericAffine.mat]
        For points:  [SyN_GenericAffine.mat, SyN_Warp.nii.gz]  ← this arg
        No whichtoinvert needed — transforms are stored in forward direction.
    template_img : ants.ANTsImage
        Template brain (fixed image used during registration).
        Used for physical→voxel conversion in template space.
    mov_ref : ants.ANTsImage
        Reference ANTs image for the moving (fish) volume with correct spacing.
        Used for voxel→physical conversion of medoid coordinates.
        Build from volume_mean_raw: ants.from_numpy(volume_mean_raw, spacing=(res_x, res_y, res_z))
    res_x, res_y, res_z : float
        Voxel size (µm) of the original pre-resample fish volume.
        Only used as a fallback if mov_ref is None.

    Returns
    -------
    transformed_all : np.ndarray, shape (n_cells, 3), dtype float32
        Template voxel coordinates [i, j, k] for each cell.
        NaN where input was invalid (-1) or mapped out-of-bounds in template.
    """
    import ants
    import pandas as pd

    n_cells = medoids_rot.shape[0]
    transformed_all = np.full((n_cells, 3), np.nan, dtype=np.float32)

    # valid: not all -1
    valid  = ~np.any(medoids_rot == -1, axis=1)
    ijk    = np.round(medoids_rot).astype(np.int32)
    vol_shape = mov_ref.shape   # (X, Y, Z)

    # also bounds-check against moving volume
    in_mov = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < vol_shape[0]) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < vol_shape[1]) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < vol_shape[2])
    )

    use     = valid & in_mov
    use_idx = np.where(use)[0]
    print(f"[transform_medoids] valid={valid.sum():,}  in_mov={in_mov.sum():,}  "
          f"to_transform={use.sum():,}/{n_cells:,}")

    if use_idx.size == 0:
        print("[transform_medoids] ⚠️  No cells to transform.")
        return transformed_all

    # Step 1: voxel → physical using ANTs index-to-physical (respects spacing + origin)
    phys_pts = np.array([
        ants.transform_index_to_physical_point(mov_ref, (int(ijk[idx, 0]),
                                                          int(ijk[idx, 1]),
                                                          int(ijk[idx, 2])))
        for idx in use_idx
    ], dtype=np.float32)

    # Step 2: apply ANTs point transforms — fish physical → template physical
    # transform_list for POINTS = [affine, warp] (forward application order)
    # No whichtoinvert needed; transforms are stored fish→template forward.
    warped_df = ants.apply_transforms_to_points(
        dim          = 3,
        points       = pd.DataFrame(phys_pts, columns=["x", "y", "z"]),
        transformlist= transform_list,
    )
    warped_pts = warped_df[["x", "y", "z"]].to_numpy().astype(np.float32)

    # Step 3: template physical → template voxel index
    atlas_shape = np.array(template_img.shape)
    atlas_ijk = []
    for pt in warped_pts:
        try:
            idx_pt = ants.transform_physical_point_to_index(
                template_img, (float(pt[0]), float(pt[1]), float(pt[2]))
            )
            atlas_ijk.append(idx_pt)
        except Exception:
            atlas_ijk.append((np.nan, np.nan, np.nan))
    atlas_ijk = np.array(atlas_ijk, dtype=np.float32)

    # bounds check in template
    atlas_ijk_round = np.round(atlas_ijk).astype(np.float32)
    atlas_in = (
        np.isfinite(atlas_ijk_round[:, 0]) &
        (atlas_ijk_round[:, 0] >= 0) & (atlas_ijk_round[:, 0] < atlas_shape[0]) &
        (atlas_ijk_round[:, 1] >= 0) & (atlas_ijk_round[:, 1] < atlas_shape[1]) &
        (atlas_ijk_round[:, 2] >= 0) & (atlas_ijk_round[:, 2] < atlas_shape[2])
    )

    # write back into full-length aligned array
    good_src = use_idx[atlas_in]
    transformed_all[good_src] = atlas_ijk_round[atlas_in]

    print(f"[transform_medoids] ✅ {int(np.isfinite(transformed_all[:, 0]).sum()):,} "
          f"cells successfully mapped to template space")
    return transformed_all