"""
registration.py
===============
ANTs-based brain registration: loading references, padding, multi-stage
registration (rigid → affine → SyN), and visualization.

All hardcoded paths have been replaced with explicit parameters.
Callers pass paths from config or from the calling script.

Location:
    ~/Zebrafish-whole-brain-analysis/registration/registration.py
"""

import math
import os
import shutil

import ants
import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage import io as skio


# ============================================================
# LOADING
# ============================================================

def load_mapzebrain_reference_brain(dir_voluseg):
    """
    Load the MapZebrain GCaMP reference brain as an ANTs image.

    Parameters
    ----------
    dir_voluseg : str
        Base voluseg path (config.dir_voluseg).
        Expects: dir_voluseg/mapzebrain/ants_template/T_AVG_HuCH2BGCaMP2-tg_ch0.tif

    Returns
    -------
    mapZebrain_ants : ants.ANTsImage
        Reference brain with spacing set to (0.9940709, 0.9939616, 1).
    """
    fname = os.path.join(
        dir_voluseg, "mapzebrain", "ants_template", "T_AVG_HuCH2BGCaMP2-tg_ch0.tif"
    )
    raw = skio.imread(fname)
    print(f"Raw mapZebrain tif shape (Z, Y, X): {raw.shape}")

    transposed = np.transpose(raw, axes=(2, 1, 0))
    print(f"Transposed mapZebrain tif shape (X, Y, Z): {transposed.shape}")

    mapZebrain_ants = ants.from_numpy(np.asarray(transposed, dtype="float32"))
    mapZebrain_ants.set_spacing((0.9940709, 0.9939616, 1))

    plt.figure()
    plt.imshow(mapZebrain_ants.numpy()[:, :, 130])
    plt.title("Reference Brain")
    plt.show()

    print(mapZebrain_ants)
    return mapZebrain_ants


def load_registered_brain(registered_fname):
    """
    Load a previously registered brain from a .nii.gz file.

    Parameters
    ----------
    registered_fname : str
        Full path to the registered .nii.gz file.

    Returns
    -------
    ants.ANTsImage
    """
    img = ants.image_read(registered_fname)

    plt.figure()
    plt.imshow(img.numpy()[:, :, 130])
    plt.title("Registered Brain (ANTsPy)")
    plt.show()

    print(f"Loaded registered ANTs image: {img}")
    return img


def load_volume_mean(fish, dir_voluseg, res_x, res_y, res_z, rotation_k):
    """
    Load volume_mean from voluseg HDF5, transpose, rotate, and convert to ANTs.

    Parameters
    ----------
    fish : tuple of (str, str)
        (proj_ID, expt_ID).
    dir_voluseg : str
        Base voluseg path (config.dir_voluseg).
    res_x, res_y, res_z : float
        Voxel spacing in microns.
    rotation_k : int
        Number of 90-degree rotations for np.rot90 (k=2 → 180°).

    Returns
    -------
    volume_mean_ants : ants.ANTsImage
        Ready for registration.
    """
    proj_ID, expt_ID = fish
    output_dir = os.path.join(dir_voluseg, proj_ID, expt_ID, "output")

    with h5py.File(os.path.join(output_dir, "volume0.hdf5"), "r") as f:
        volume_mean_raw = f["volume_mean"][()]
    print(f"Raw HDF5 shape (Z, X, Y): {volume_mean_raw.shape}")

    volume_mean = np.transpose(volume_mean_raw, (1, 2, 0))
    print(f"Transposed shape (X, Y, Z): {volume_mean.shape}")

    volume_mean_rotated = np.rot90(volume_mean, k=rotation_k, axes=(0, 1))
    print(f"Rotated shape (X, Y, Z): {volume_mean_rotated.shape}")

    volume_mean_ants = ants.from_numpy(np.asarray(volume_mean_rotated, dtype="float32"))
    volume_mean_ants.set_spacing((res_x, res_y, res_z))

    plt.figure()
    plt.imshow(volume_mean_ants.numpy()[:, :, 20])
    plt.title("volume mean ants example slice")
    plt.show()

    print(volume_mean_ants)
    return volume_mean_ants


def load_mean_expt_brain(proj_ID, dir_registration):
    """
    Load the mean experiment brain for a project.

    Parameters
    ----------
    proj_ID : str
        Project folder name.
    dir_registration : str
        Registration output base path (config.dir_registration).

    Returns
    -------
    ants.ANTsImage
    """
    mean_path = os.path.join(dir_registration, proj_ID, "mean_expt_brain.nii.gz")
    img = ants.image_read(mean_path)

    plt.figure()
    plt.imshow(img.numpy()[:, :, 20])
    plt.title("mean expt brain ants example slice")
    plt.show()

    print(img)
    return img


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def normalize_image_intensity(img_np):
    """Robust min-max normalization using 1st/99th percentiles."""
    img_min = np.percentile(img_np, 1)
    img_max = np.percentile(img_np, 99)
    return np.clip((img_np - img_min) / (img_max - img_min), 0, 1)


def get_padding_width(mapZebrain_ants, volume_mean_ants):
    """
    Compute padding needed to match volume_mean shape to reference brain shape.

    Returns
    -------
    pad_width : list of (int, int)
        Padding tuples for each axis [(y_lo, y_hi), (x_lo, x_hi), (z_lo, z_hi)].
    """
    y1, x1, z1 = mapZebrain_ants.shape
    y2, x2, z2 = volume_mean_ants.shape

    print(f"Reference shape: {mapZebrain_ants.shape}")
    print(f"Volume shape:    {volume_mean_ants.shape}")

    pad_width = [
        (math.floor((y1 - y2) / 2), math.floor((y1 - y2) / 2)),
        (math.floor((x1 - x2) / 2), math.floor((x1 - x2) / 2)),
        (math.floor((z1 - z2) / 2), math.floor((z1 - z2) / 2)),
    ]
    print(f"Padding: {pad_width}")

    plt.figure()
    plt.imshow(volume_mean_ants.numpy()[:, :, 15], cmap="gray")
    plt.title("Padding Width")
    plt.colorbar()
    plt.show()

    return pad_width


def reshape_image(mapZebrain_ants, volume_mean_ants):
    """
    Pad moving image so its physical extent matches the reference brain.

    Origin is shifted to account for padding added before the data.
    Z padding is added only at the start (convention).

    Returns
    -------
    padded_image : ants.ANTsImage
    padding_tuples : list of (int, int)
        Padding applied per axis, for use with get_new_index_after_padding().
    """
    physical_sizes_ref = [
        dim * sp for dim, sp in zip(mapZebrain_ants.shape, mapZebrain_ants.spacing)
    ]
    physical_sizes_volume = [
        dim * sp for dim, sp in zip(volume_mean_ants.shape, volume_mean_ants.spacing)
    ]

    padding_physical = [ref - vol for ref, vol in zip(physical_sizes_ref, physical_sizes_volume)]

    padding_voxels = [
        int(np.round(pad / sp)) if pad > 0 else 0
        for pad, sp in zip(padding_physical, volume_mean_ants.spacing)
    ]
    pad_x, pad_y, pad_z = padding_voxels

    pad_x_start = pad_x // 2
    pad_x_end = pad_x - pad_x_start
    pad_y_start = pad_y // 2
    pad_y_end = pad_y - pad_y_start
    pad_z_start = pad_z
    pad_z_end = 0
    padding_tuples = [
        (pad_x_start, pad_x_end),
        (pad_y_start, pad_y_end),
        (pad_z_start, pad_z_end),
    ]

    padded_array = np.pad(
        volume_mean_ants.numpy(),
        padding_tuples,
        mode="constant",
        constant_values=0,
    )

    # Shift origin to account for pre-data padding
    old_origin = np.array(volume_mean_ants.origin)
    spacing = np.array(volume_mean_ants.spacing)
    padding_start_voxels = np.array([pad_x_start, pad_y_start, pad_z_start])
    new_origin = old_origin - padding_start_voxels * spacing

    padded_image = ants.from_numpy(
        padded_array,
        origin=tuple(new_origin),
        spacing=volume_mean_ants.spacing,
        direction=volume_mean_ants.direction,
    )

    return padded_image, padding_tuples


# ============================================================
# REGISTRATION STAGES
# ============================================================

def register_to_template(moving, fixed, target_spacing, save_dir):
    """
    Full SyN registration of moving to fixed, with outputs saved to save_dir.

    Returns
    -------
    ants.ANTsImage
        The warped moving image.
    """
    if target_spacing is None:
        target_spacing = moving.spacing

    moving_resampled = ants.resample_image(moving, target_spacing, use_voxels=False, interp_type=1)
    fixed_resampled = ants.resample_image(fixed, target_spacing, use_voxels=False, interp_type=1)

    moving_norm = ants.iMath_normalize(moving_resampled)
    fixed_norm = ants.iMath_normalize(fixed_resampled)

    registration = ants.registration(
        fixed=fixed_norm, moving=moving_norm,
        type_of_transform="SyN", verbose=True,
    )

    ants.image_write(
        registration["warpedmovout"],
        os.path.join(save_dir, "expt_to_temp_registered.nii.gz"),
    )
    shutil.copy(
        registration["fwdtransforms"][0],
        os.path.join(save_dir, "expt_to_temp_warp.nii.gz"),
    )
    shutil.copy(
        registration["fwdtransforms"][1],
        os.path.join(save_dir, "expt_to_temp_affine.mat"),
    )

    return registration["warpedmovout"]


def manual_scaling(scale_factors, dir_ants_output, file_name):
    """Apply a manual scaling transform and save to dir_ants_output/file_name."""
    scale_transform = ants.new_ants_transform(
        dimension=3,
        transform_type="AffineTransform",
    )
    params = np.array([
        scale_factors[0], 0, 0,
        0, scale_factors[1], 0,
        0, 0, scale_factors[2],
        0, 0, 0,
    ])
    scale_transform.set_parameters(params)

    save_path = os.path.join(dir_ants_output, file_name)
    ants.write_transform(scale_transform, save_path)
    print(f"Manual scaling transform saved: {save_path}")

    return save_path


def rigid_registration(moving_norm, fixed_norm, dir_ants_output, file_name):
    """
    Rigid registration with cross-correlation metric.

    Returns
    -------
    str
        Path to the saved rigid transform matrix.
    """
    result = ants.registration(
        fixed=fixed_norm,
        moving=moving_norm,
        type_of_transform="Rigid",
        reg_template="Rigid",
        reg={
            "iterations": [500, 500, 500, 200],
            "smoothing_sigmas": [2, 1, 0.5, 0],
            "shrink_factors": [8, 4, 2, 1],
            "metric": "CC",
        },
        sampling_strategy="Regular",
        sampling_percentage=1,
        interpolation="WelchWindowedSinc",
        use_histogram_matching=False,
        verbose=True,
    )

    rigid_matrix = result["fwdtransforms"][0]
    save_path = os.path.join(dir_ants_output, file_name)
    shutil.copy(rigid_matrix, save_path)
    print(f"Rigid transform matrix saved: {save_path}")

    return save_path


def affine_registration(moving_norm, fixed_norm, dir_ants_output, file_name):
    """
    TRSAA affine registration with mutual information metric.

    Returns
    -------
    str
        Path to the saved affine transform matrix.
    """
    result = ants.registration(
        fixed=fixed_norm,
        moving=moving_norm,
        type_of_transform="TRSAA",
        aff_metric="MI",
        verbose=True,
    )

    affine_matrix = result["fwdtransforms"][0]
    save_path = os.path.join(dir_ants_output, file_name)
    shutil.copy(affine_matrix, save_path)
    print(f"Affine transform matrix saved: {save_path}")

    return save_path


def syn_registration(
    moving_norm, fixed_norm, dir_ants_output,
    file_name_warp, file_name_affine, file_name_syn_registration,
    init_tx, fixed_mask, moving_mask,
):
    """
    SyNRA registration (affine MI + SyN CC) with masks and initial transform.

    Returns
    -------
    tuple of (str, str)
        Paths to saved warp field and affine transform.
    """
    result = ants.registration(
        moving=moving_norm,
        fixed=fixed_norm,
        type_of_transform="SyNRA",
        aff_metric="MI",
        syn_metric="CC",
        shrink_factors=(8, 4, 2, 1),
        smoothing_sigmas=(3, 2, 1, 0),
        reg_iterations=(120, 80, 40, 0),
        interpolator="linear",
        mask=fixed_mask,
        moving_mask=moving_mask,
        initial_transform=init_tx,
        verbose=True,
    )

    syn_warp = result["fwdtransforms"][0]
    syn_affine = result["fwdtransforms"][1]

    warp_path = os.path.join(dir_ants_output, file_name_warp)
    affine_path = os.path.join(dir_ants_output, file_name_affine)

    shutil.copy(syn_warp, warp_path)
    shutil.copy(syn_affine, affine_path)
    ants.image_write(
        result["warpedmovout"],
        os.path.join(dir_ants_output, file_name_syn_registration),
        ri=False,
    )

    print(f"SyN warp field saved: {warp_path}")
    print(f"SyN affine transform saved: {affine_path}")

    return warp_path, affine_path


def elastic_registration(moving_norm, fixed_norm, dir_ants_output):
    """
    Elastic registration (ANTs default parameters).

    Returns
    -------
    dict
        Full ANTs registration output dict.
    """
    result = ants.registration(
        fixed=fixed_norm,
        moving=moving_norm,
        type_of_transform="Elastic",
    )

    shutil.copy(
        result["fwdtransforms"][0],
        os.path.join(dir_ants_output, "elastic_warp_field.nii.gz"),
    )
    shutil.copy(
        result["fwdtransforms"][1],
        os.path.join(dir_ants_output, "elastic_affine_transform.mat"),
    )
    ants.image_write(
        result["warpedmovout"],
        os.path.join(dir_ants_output, "elastic_registered.nii.gz"),
        ri=False,
    )

    return result


def robust_affine_registration(
    original_moving_norm, fixed_norm, initial_alignment_type,
    dir_ants_output, output_warped_image_filename, output_affine_filename,
):
    """
    Multi-resolution affine registration with Mattes mutual information.

    Returns
    -------
    str
        Path to the saved affine transform matrix.
    """
    result = ants.registration(
        fixed=fixed_norm,
        moving=original_moving_norm,
        type_of_transform="Affine",
        initial_transform=initial_alignment_type,
        metric="Mattes",
        sampling_strategy="Regular",
        sampling_percentage=0.25,
        reg_iterations=(1000, 500, 250, 100),
        shrink_factors=(8, 4, 2, 1),
        smoothing_sigmas=(3, 2, 1, 0),
        use_histogram_matching=False,
        interpolation="Linear",
        verbose=True,
    )

    temp_affine = result["fwdtransforms"][0]
    save_path = os.path.join(dir_ants_output, output_affine_filename)
    shutil.copy(temp_affine, save_path)
    print(f"Robust Affine transform saved to: {save_path}")

    return save_path


# ============================================================
# VISUALIZATION
# ============================================================

def create_image(z, mapZebrain_ants, dir_ants_output, input_file, method):
    """
    Overlay a registered image on the reference brain at slice z and save.

    Parameters
    ----------
    z : int
        Slice index to visualize.
    mapZebrain_ants : ants.ANTsImage
        Reference brain.
    dir_ants_output : str
        Directory containing input_file and where the figure is saved.
    input_file : str
        Filename of the registered .nii.gz to overlay.
    method : str
        Label for the registration method (used in title and filename).
    """
    file_path = os.path.join(dir_ants_output, input_file)
    if not os.path.exists(file_path):
        print(f"Path does not exist: {file_path}")
        return

    img = nib.load(file_path)
    registered_data = img.get_fdata()
    mapzebrain_data = mapZebrain_ants.numpy()

    print(f"mapzebrain shape: {mapzebrain_data.shape}")
    print(f"registered shape: {registered_data.shape}")

    plt.figure()
    plt.imshow(mapzebrain_data[:, :, z], cmap="gray", alpha=1)
    plt.imshow(registered_data[:, :, z], cmap="viridis", alpha=0.8)
    plt.tight_layout()
    plt.title(f"Transformed Image (z = {z})")
    plt.savefig(os.path.join(dir_ants_output, f"transformed_image_{z}_{method}.pdf"))
    plt.show()
