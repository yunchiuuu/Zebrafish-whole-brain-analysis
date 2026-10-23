"""
run_registration_syn.py
=======================
SyN registration of one fish to the shared mean brain, with QC overlay
saved immediately after registration.

--config is a MODULE NAME under registration/config/ (no folder, no .py), the
same convention as the chemogenetic scripts. Fish list, imaging specs, template
path and resampling all come from that one config.

Usage (interactive):
    python registration/run_registration_syn.py \
        --config config_registration \
        --expt_ID 251008_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish4

Usage (sbatch all fish in the config; one job per fish):
    bash registration/submit_registration_syn.sh --config config_registration

Outputs per fish under {dir_registration}/{proj_ID}/{expt_ID}/:
    expt_to_mean_warp.nii.gz
    expt_to_mean_affine.mat
    expt_to_mean_registered.nii.gz
    expt_to_mean_registered_final.nii.gz
    qc_registration_overlay.pdf
    registration_meta.json   (template path + sha256, parameters, git commit)

Location:
    ~/Zebrafish-whole-brain-analysis/registration/run_registration_syn.py
"""

import argparse
import datetime
import hashlib
import importlib
import json
import multiprocessing
import os
import subprocess
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import (
    SYN_PARAMS,
    load_volume_mean,
    syn_registration,
)

print(f"✅ TMPDIR={os.environ['TMPDIR']} | ITK threads={n_threads}")

# ============================================================
# PARSE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="SyN registration of one fish to the shared mean brain."
)
parser.add_argument(
    "--config", required=True,
    help="Config MODULE NAME under registration/config/, no folder and no .py "
         "(e.g. config_registration)"
)
parser.add_argument(
    "--expt_ID", required=True,
    help="Experiment ID of the fish to register."
)
args = parser.parse_args()

# ============================================================
# LOAD CONFIG (module name, same convention as chemogenetic scripts)
# ============================================================

if args.config.endswith(".py") or "/" in args.config:
    raise ValueError(
        f"--config must be a module name under registration/config/ "
        f"(e.g. config_registration), not a path: {args.config!r}"
    )

sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
cfg = importlib.import_module(args.config)

# Resolve fish tuple from expt_ID
expt_ID  = args.expt_ID
all_fish = cfg.all_fish
fish     = next((f for f in all_fish if f[1] == expt_ID), None)
if fish is None:
    raise ValueError(
        f"{expt_ID} not found in all_fish from {args.config}.\n"
        f"Available: {[f[1] for f in all_fish]}"
    )
proj_ID = fish[0]

# Everything below comes from the one config
res_x            = cfg.res_x
res_y            = cfg.res_y
res_z            = cfg.res_z
n_slices         = cfg.n_slices
rotation_k       = cfg.rotation_k
dir_voluseg      = cfg.dir_voluseg
dir_registration = cfg.dir_registration
MEAN_BRAIN_PATH  = cfg.MEAN_BRAIN_PATH
target_spacing   = cfg.target_spacing
TEMPLATE_VERSION = cfg.TEMPLATE_VERSION

# Affine initializer settings (recorded in registration_meta.json)
AFFINE_INIT_PARAMS = dict(search_factor=20, radian_fraction=0.1)

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
print(f"\nLoading template brain: {MEAN_BRAIN_PATH}")
fixed_atlas = ants.image_read(MEAN_BRAIN_PATH)
print(f"Template shape={fixed_atlas.shape}, spacing={fixed_atlas.spacing}")

# ============================================================
# SyN REGISTRATION
# ============================================================

print(f"\n{'='*60}")
print(f"  Registering: {expt_ID}")
print(f"  proj_ID:     {proj_ID}")
print(f"{'='*60}")

out_dir = os.path.join(dir_registration, proj_ID, expt_ID)
os.makedirs(out_dir, exist_ok=True)

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
    fixed_norm, moving_norm, **AFFINE_INIT_PARAMS
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

# ============================================================
# PROVENANCE SIDECAR — what exactly this registration used
# ============================================================

def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(*cmd):
    """Run a git command in the repo root; return stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *cmd], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _jsonable(d):
    """Tuples -> lists so the dict serializes cleanly."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


template_stat = os.stat(MEAN_BRAIN_PATH)
git_status = _git("status", "--porcelain")
meta = {
    "expt_ID": expt_ID,
    "proj_ID": proj_ID,
    "registered_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "config_module": args.config,
    "template": {
        "version": TEMPLATE_VERSION,
        "path": str(MEAN_BRAIN_PATH),
        "sha256": _sha256(MEAN_BRAIN_PATH),
        "size_bytes": template_stat.st_size,
        "mtime_utc": datetime.datetime.fromtimestamp(
            template_stat.st_mtime, datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        "shape": list(fixed_atlas.shape),
        "spacing": list(fixed_atlas.spacing),
    },
    "imaging": {
        "res_x": res_x, "res_y": res_y, "res_z": res_z,
        "n_slices": n_slices, "rotation_k": rotation_k,
    },
    "target_spacing": list(target_spacing),
    "affine_init": AFFINE_INIT_PARAMS,
    "syn": _jsonable(SYN_PARAMS),
    "apply_transforms_interpolator": "welchWindowedSinc",
    "git_commit": _git("rev-parse", "HEAD"),
    "git_dirty": bool(git_status) if git_status is not None else None,
    "ants_version": getattr(ants, "__version__", None),
}
meta_path = os.path.join(out_dir, "registration_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"✅ Registration metadata saved: {meta_path}")

print(f"\n=== Registration complete: {expt_ID} ===")
