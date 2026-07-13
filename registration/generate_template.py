"""
generate_template.py
====================
Registers three 300 µm Zstack fish to a fixed reference (fish2) and
averages all three in template space to produce a mean template brain.

Pipeline per moving fish (mirrors run_registration_syn.py):
    reorient LPS → resample isotropic → normalize → mask →
    affine_initializer → syn_registration → apply_transforms full-res →
    QC overlay

Usage:
    python registration/generate_template.py

Outputs (all written to TEMPLATE_DIR):
    template_mean_brain.nii.gz              — final averaged template
    fish2_fixed.nii.gz                      — fixed reference
    fish{1,3}/fish{1,3}_syn_registered.nii.gz      — warped (downsampled)
    fish{1,3}/fish{1,3}_registered_final.nii.gz    — warped (full-res)
    fish{1,3}/fish{1,3}_syn_warp.nii.gz            — SyN warp field
    fish{1,3}/fish{1,3}_syn_affine.mat             — SyN affine
    fish{1,3}/qc_registration_overlay.pdf          — QC overlay

Location:
    ~/Zebrafish-whole-brain-analysis/registration/generate_template.py
"""

import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ITK threading
try:
    n_threads = str(len(os.sched_getaffinity(0)) - 2)
except AttributeError:
    n_threads = str(multiprocessing.cpu_count() - 2)

HPC_USERNAME = "yun"
os.environ["TMPDIR"] = f"/resnick/scratch/{HPC_USERNAME}/tmp_ants"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = n_threads
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import ants
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import syn_registration

print(f"✅ TMPDIR={os.environ['TMPDIR']} | ITK threads={n_threads}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path("/resnick/groups/Proberlab/yun/lightsheet/hcrt-trpv1_huc-h2b-g8m_csn_120min_audio")

FISH = {
    "fish1": BASE / "260707_hcrt-trpv1_huc-h2b-g8m_csn_10uM_audio_fish1" / "output_Zstack" / "volume0.hdf5",
    "fish2": BASE / "260707_hcrt-trpv1_huc-h2b-g8m_csn_10uM_audio_fish2" / "output_Zstack" / "volume0.hdf5",
    "fish3": BASE / "260708_hcrt-trpv1_huc-h2b-g8m_csn_10uM_audio_fish1" / "output_Zstack" / "volume0.hdf5",
}
FIXED_KEY = "fish2"

TEMPLATE_DIR = BASE / "zstack_template"

# Voxel spacing for 300 µm Zstacks
RES_X, RES_Y, RES_Z = 1.52, 1.52, 7.5
ROTATION_K = 2

# Isotropic target spacing for registration (match your config_registration)
TARGET_SPACING = (3.0, 3.0, 3.0)


# ---------------------------------------------------------------------------
# Load Zstack volume0.hdf5 → ANTsImage
# ---------------------------------------------------------------------------
def load_zstack_volume(h5_path):
    """
    Load volume_mean from Zstack volume0.hdf5.
    Axis convention follows registration.load_volume_mean:
        raw (Z, X, Y) → transpose(1,2,0) → (X, Y, Z) → rot90(k=2)
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print(f"  HDF5 keys: {keys}")
        if "volume_mean" in keys:
            vol = f["volume_mean"][()]
        else:
            print(f"  WARNING: 'volume_mean' not found, using first key '{keys[0]}'")
            vol = f[keys[0]][()]

    print(f"  Raw HDF5 shape (Z, X, Y): {vol.shape}")

    vol = np.transpose(vol, (1, 2, 0))
    print(f"  Transposed shape (X, Y, Z): {vol.shape}")

    vol = np.rot90(vol, k=ROTATION_K, axes=(0, 1))
    print(f"  Rotated shape (X, Y, Z): {vol.shape}")

    img = ants.from_numpy(np.asarray(vol, dtype="float32"))
    img.set_spacing((RES_X, RES_Y, RES_Z))
    return img


# ---------------------------------------------------------------------------
# QC overlay (from run_registration_syn.py)
# ---------------------------------------------------------------------------
def _norm_slice(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def save_qc_overlay(fixed_atlas, registered_path, out_dir, label, n_planes=10):
    if not os.path.exists(registered_path):
        print(f"  ⚠️ Registered image not found for QC: {registered_path}")
        return

    registered = ants.image_read(registered_path, pixeltype="float")
    atlas_np = fixed_atlas.numpy()
    reg_np = registered.numpy()

    if (not np.allclose(registered.shape, fixed_atlas.shape) or
        not np.allclose(registered.spacing, fixed_atlas.spacing) or
        not np.allclose(registered.origin, fixed_atlas.origin)):
        print("  ⚠️ Grid mismatch — resampling to atlas grid for QC.")
        registered = ants.resample_image_to_target(registered, fixed_atlas, interp_type="linear")
        reg_np = registered.numpy()

    n_z = atlas_np.shape[2]
    z_slices = np.linspace(0, n_z - 1, min(n_planes, n_z), dtype=int)
    ncols = 5
    nrows = int(np.ceil(len(z_slices) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 8), squeeze=False)
    axes = axes.ravel()

    for i, z in enumerate(z_slices):
        ax = axes[i]
        ax.imshow(_norm_slice(atlas_np[:, :, z].T), cmap="gray", origin="lower", aspect="auto")
        ax.imshow(_norm_slice(reg_np[:, :, z].T), cmap="viridis", origin="lower", aspect="auto", alpha=0.7)
        ax.set_title(f"Z={z}", fontsize=9)
        ax.invert_yaxis()
        ax.axis("off")

    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    fig.suptitle(f"Template Registration QC: {label}\nFixed (gray) | Warped (green overlay)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = os.path.join(out_dir, "qc_registration_overlay.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  ✅ QC overlay saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Register one moving fish → fixed (mirrors run_registration_syn.py)
# ---------------------------------------------------------------------------
def register_to_fixed(moving_raw, fixed_raw, fish_label, save_dir):
    """
    Full pipeline for one moving fish:
        LPS → isotropic resample → normalize → mask →
        affine_initializer → syn_registration →
        apply_transforms at full resolution → QC
    """
    save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Reorient to LPS
    fixed_lps = ants.reorient_image2(fixed_raw, orientation="LPS")
    mov_lps = ants.reorient_image2(moving_raw, orientation="LPS")
    print(f"  [{fish_label}] fixed LPS spacing={fixed_lps.spacing} origin={fixed_lps.origin}")
    print(f"  [{fish_label}] mov   LPS spacing={mov_lps.spacing} origin={mov_lps.origin}")

    # Resample to isotropic target spacing
    print(f"  [{fish_label}] Resampling to {TARGET_SPACING} µm ...")
    fixed_ds = ants.resample_image(fixed_lps, TARGET_SPACING, use_voxels=False, interp_type=1)
    moving_ds = ants.resample_image(mov_lps, TARGET_SPACING, use_voxels=False, interp_type=1)

    # Normalize
    fixed_norm = ants.iMath_normalize(fixed_ds)
    moving_norm = ants.iMath_normalize(moving_ds)

    # Brain masks
    fixed_mask = ants.iMath(ants.get_mask(fixed_norm, cleanup=True), "MD", 1)
    moving_mask = ants.iMath(ants.get_mask(moving_norm, cleanup=True), "MD", 1)

    # Affine initializer
    print(f"  [{fish_label}] Affine initializer ...")
    init_tx = ants.affine_initializer(
        fixed_norm, moving_norm, search_factor=20, radian_fraction=0.1
    )

    # SyN registration
    print(f"  [{fish_label}] SyN registration ...")
    warp_path, affine_path = syn_registration(
        moving_norm=moving_norm,
        fixed_norm=fixed_norm,
        dir_ants_output=save_dir,
        file_name_warp=f"{fish_label}_syn_warp.nii.gz",
        file_name_affine=f"{fish_label}_syn_affine.mat",
        file_name_syn_registration=f"{fish_label}_syn_registered.nii.gz",
        init_tx=init_tx,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
    )

    # Apply composite transform at full resolution on fixed grid
    print(f"  [{fish_label}] Applying transforms at full resolution ...")
    final_registered = ants.apply_transforms(
        fixed=fixed_raw,
        moving=mov_lps,
        transformlist=[warp_path, affine_path],
        interpolator="welchWindowedSinc",
        verbose=True,
    )
    final_path = os.path.join(save_dir, f"{fish_label}_registered_final.nii.gz")
    ants.image_write(final_registered, final_path)
    print(f"  ✅ Full-res registered image: {final_path}")

    # QC overlay
    print(f"  [{fish_label}] Saving QC overlay ...")
    save_qc_overlay(
        fixed_atlas=fixed_raw,
        registered_path=final_path,
        out_dir=save_dir,
        label=fish_label,
        n_planes=10,
    )

    return final_registered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    # Load all Zstack volumes
    print("=" * 60)
    print("Loading Zstack volumes")
    print("=" * 60)
    imgs = {}
    for name, path in FISH.items():
        print(f"\n{name}: {path}")
        imgs[name] = load_zstack_volume(path)
        print(f"  ANTs image: {imgs[name]}")

    fixed_raw = imgs[FIXED_KEY]
    moving_keys = [k for k in FISH if k != FIXED_KEY]

    # Save fixed reference
    fixed_path = TEMPLATE_DIR / "fish2_fixed.nii.gz"
    ants.image_write(fixed_raw, str(fixed_path))
    print(f"\n✅ Fixed reference saved: {fixed_path}")

    # Register each moving fish → fixed
    warped = {FIXED_KEY: fixed_raw}
    for mk in moving_keys:
        print(f"\n{'=' * 60}")
        print(f"  Registering {mk} → {FIXED_KEY}")
        print(f"{'=' * 60}")
        fish_dir = TEMPLATE_DIR / mk
        warped[mk] = register_to_fixed(
            moving_raw=imgs[mk],
            fixed_raw=fixed_raw,
            fish_label=mk,
            save_dir=fish_dir,
        )

    # Average all three in template space
    print(f"\n{'=' * 60}")
    print("  Averaging → mean template brain")
    print(f"{'=' * 60}")
    mean_vol = sum(w.numpy() for w in warped.values()) / len(warped)
    mean_img = ants.from_numpy(
        mean_vol,
        origin=fixed_raw.origin,
        spacing=fixed_raw.spacing,
        direction=fixed_raw.direction,
    )

    out_path = TEMPLATE_DIR / "template_mean_brain.nii.gz"
    ants.image_write(mean_img, str(out_path))
    print(f"\n✅ Template written to: {out_path}")
    print(f"   All outputs in:     {TEMPLATE_DIR}")


if __name__ == "__main__":
    main()
