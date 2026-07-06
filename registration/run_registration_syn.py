"""
run_registration_syn.py
=======================
SyN registration of one fish to the shared mean brain, with QC overlay
saved immediately after registration.

Takes fish identity from the experiment config + CLI argument.
The mean brain path comes from config_registration.py.

Usage (interactive):
    python registration/run_registration_syn.py \
        --config registration/config_registration.py \
        --expt_ID 251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4

Usage (sbatch one fish):
    sbatch --cpus-per-task=16 --mem=64G --wrap="\
        python registration/run_registration_syn.py \
        --config registration/config_registration.py \
        --expt_ID 251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4"

Outputs per fish under {dir_registration}/{proj_ID}/{expt_ID}/:
    expt_to_mean_warp.nii.gz
    expt_to_mean_affine.mat
    expt_to_mean_registered.nii.gz
    expt_to_mean_registered_final.nii.gz
    qc_registration_overlay.pdf

Location:
    ~/Zebrafish-whole-brain-analysis/registration/run_registration_syn.py
"""

import argparse
import importlib.util
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — safe for sbatch; PDFs saved to disk
import matplotlib.pyplot as plt

# Set ITK threads before importing ants
try:
    n_threads = str(len(os.sched_getaffinity(0)) - 2)
except AttributeError:
    n_threads = str(multiprocessing.cpu_count() - 2)

HPC_USERNAME = "yun"
os.environ["TMPDIR"] = f"/resnick/scratch/{HPC_USERNAME}/tmp_ants"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = n_threads
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import ants

sys.path.insert(0, str(Path(__file__).resolve().parent))    # for registration.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for utils/

from registration import (
    load_volume_mean,
    syn_registration,
)
from utils.data_io import create_folder

print(f"✅ TMPDIR={os.environ['TMPDIR']} | ITK threads={n_threads}")

# ============================================================
# PARSE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="SyN registration of one fish to the shared mean brain."
)
parser.add_argument(
    "--config", required=True,
    help="Path to experiment config .py (e.g. chemogenetic/config/hcrt_trpv1_csn_120min.py)"
)
parser.add_argument(
    "--expt_ID", required=True,
    help="Experiment ID of the fish to register."
)
args = parser.parse_args()

# ============================================================
# LOAD EXPERIMENT CONFIG DYNAMICALLY
# ============================================================

config_path = Path(args.config).resolve()
spec = importlib.util.spec_from_file_location("expt_config", config_path)
cfg  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

# Resolve fish tuple from expt_ID
expt_ID  = args.expt_ID
all_fish = cfg.all_fish
fish     = next((f for f in all_fish if f[1] == expt_ID), None)
if fish is None:
    raise ValueError(
        f"{expt_ID} not found in all_fish from {config_path.name}.\n"
        f"Available: {[f[1] for f in all_fish]}"
    )
proj_ID = fish[0]

# Imaging params from experiment config
res_x      = cfg.res_x
res_y      = cfg.res_y
res_z      = cfg.res_z
n_slices   = cfg.n_slices
rotation_k = cfg.rotation_k

# ============================================================
# LOAD REGISTRATION CONFIG
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
from config_registration import (
    dir_voluseg,
    dir_registration,
    MEAN_BRAIN_PATH,
    target_spacing,
)

# ============================================================
# HELPERS
# ============================================================

def _norm_slice(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def save_qc_overlay(fixed_atlas, registered_path, out_dir, expt_ID, n_planes=10):
    """Save QC overlay PDF: mean brain (gray) + registered fish (viridis)."""
    if not os.path.exists(registered_path):
        print(f"  ⚠️ Registered image not found for QC: {registered_path}")
        return

    registered = ants.image_read(registered_path, pixeltype="float")
    atlas_np   = fixed_atlas.numpy()
    reg_np     = registered.numpy()

    # resample to atlas grid if grids don't match
    if (not np.allclose(registered.shape,   fixed_atlas.shape) or
        not np.allclose(registered.spacing, fixed_atlas.spacing) or
        not np.allclose(registered.origin,  fixed_atlas.origin)):
        print("  ⚠️ Grid mismatch — resampling to atlas grid for QC visualization.")
        registered = ants.resample_image_to_target(registered, fixed_atlas, interp_type="linear")
        reg_np = registered.numpy()

    n_z      = atlas_np.shape[2]
    z_slices = np.linspace(0, n_z - 1, min(n_planes, n_z), dtype=int)
    ncols    = 5
    nrows    = int(np.ceil(len(z_slices) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 8),
                             squeeze=False)
    axes = axes.ravel()

    for i, z in enumerate(z_slices):
        ax = axes[i]
        ax.imshow(_norm_slice(atlas_np[:, :, z].T), cmap="gray",
                  origin="lower", aspect="auto")
        ax.imshow(_norm_slice(reg_np[:, :, z].T),   cmap="viridis",
                  origin="lower", aspect="auto", alpha=0.7)
        ax.set_title(f"Z={z}", fontsize=9)
        ax.invert_yaxis()
        ax.axis("off")

    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Registration QC: {expt_ID}\nMean brain (gray) | Registered fish (green overlay)",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = os.path.join(out_dir, "qc_registration_overlay.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  ✅ QC overlay saved: {pdf_path}")

# ============================================================
# LOAD MEAN BRAIN
# ============================================================

if not os.path.exists(MEAN_BRAIN_PATH):
    raise FileNotFoundError(
        f"Mean brain not found: {MEAN_BRAIN_PATH}\n"
        "Run run_registration_mean_brain.py first."
    )

print(f"\nLoading mean brain: {MEAN_BRAIN_PATH}")
fixed_atlas = ants.image_read(MEAN_BRAIN_PATH)
print(f"Mean brain shape={fixed_atlas.shape}, spacing={fixed_atlas.spacing}")

# ============================================================
# SyN REGISTRATION
# ============================================================

print(f"\n{'='*60}")
print(f"  Registering: {expt_ID}")
print(f"  proj_ID:     {proj_ID}")
print(f"{'='*60}")

out_dir = os.path.join(dir_registration, proj_ID, expt_ID)
create_folder(proj_ID, expt_ID, dir_registration)

# Load moving image
print("Loading volume mean (moving)...")
mov = load_volume_mean(
    (proj_ID, expt_ID), dir_voluseg, res_x, res_y, res_z, rotation_k
)

# Reorient both to LPS
fixed_lps = ants.reorient_image2(fixed_atlas, orientation="LPS")
mov_lps   = ants.reorient_image2(mov,         orientation="LPS")
print(f"  [atlas] spacing={fixed_lps.spacing} origin={fixed_lps.origin}")
print(f"  [mov  ] spacing={mov_lps.spacing} origin={mov_lps.origin}")

# Downsample to isotropic target spacing
print(f"Resampling to {target_spacing} µm...")
fixed_ds  = ants.resample_image(fixed_lps, target_spacing, use_voxels=False, interp_type=1)
moving_ds = ants.resample_image(mov_lps,   target_spacing, use_voxels=False, interp_type=1)

# Intensity normalization
fixed_norm  = ants.iMath_normalize(fixed_ds)
moving_norm = ants.iMath_normalize(moving_ds)

# Brain masks
fixed_mask  = ants.iMath(ants.get_mask(fixed_norm,  cleanup=True), "MD", 1)
moving_mask = ants.iMath(ants.get_mask(moving_norm, cleanup=True), "MD", 1)

# Affine initializer
init_tx = ants.affine_initializer(
    fixed_norm, moving_norm, search_factor=20, radian_fraction=0.1
)

# SyN registration
print("Running SyN...")
warp_path, affine_path = syn_registration(
    moving_norm,
    fixed_norm,
    out_dir,
    file_name_warp="expt_to_mean_warp.nii.gz",
    file_name_affine="expt_to_mean_affine.mat",
    file_name_syn_registration="expt_to_mean_registered.nii.gz",
    init_tx=init_tx,
    fixed_mask=fixed_mask,
    moving_mask=moving_mask,
)

# Apply composite transform at full resolution on atlas grid
print("Applying composite transform at full resolution...")
final_registered = ants.apply_transforms(
    fixed=fixed_atlas,
    moving=mov_lps,
    transformlist=[warp_path, affine_path],
    interpolator="welchWindowedSinc",
    verbose=True,
)
final_path = os.path.join(out_dir, "expt_to_mean_registered_final.nii.gz")
ants.image_write(final_registered, final_path)
print(f"✅ Registered image saved: {final_path}")

# ============================================================
# QC OVERLAY — saved immediately after registration
# ============================================================

print("Saving QC overlay...")
save_qc_overlay(
    fixed_atlas=fixed_atlas,
    registered_path=final_path,
    out_dir=out_dir,
    expt_ID=expt_ID,
    n_planes=10,
)

print(f"\n=== Registration complete: {expt_ID} ===")
