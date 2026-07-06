"""
run_registration_mean_brain_batch.py
=====================================
Batch (non-interactive) version of run_registration_mean_brain.py Cell 4.

Run this AFTER completing the interactive QC in run_registration_mean_brain.py
and deciding on TEMPLATE_IDX. This script registers all fish to the chosen
template and saves the mean brain NIfTI.

Usage:
    sbatch registration/submit_mean_brain.sh

    Or directly:
    sbatch --job-name=mean_brain \
           --cpus-per-task=16 \
           --mem=128G \
           --output=logs/registration/mean_brain.log \
           --wrap="python registration/run_registration_mean_brain_batch.py"

Location:
    ~/Zebrafish-whole-brain-analysis/registration/run_registration_mean_brain_batch.py
"""

import multiprocessing
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np

# ITK threads before importing ants
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
    load_volume_mean,
    register_to_template,
    normalize_image_intensity,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "config"))
from config_registration import (
    dir_voluseg,
    dir_registration,
    MEAN_BRAIN_PATH,
    res_x, res_y, res_z,
    rotation_k,
    all_fish_for_mean_brain,
    TEMPLATE_IDX,
)

print(f"✅ TMPDIR={os.environ['TMPDIR']} | ITK threads={n_threads}")
print(f"Mean brain will be saved to: {MEAN_BRAIN_PATH}")
print(f"Total fish: {len(all_fish_for_mean_brain)}")

# ── Settings ──────────────────────────────────────────────────────────────────
# TEMPLATE_IDX and fish_for_mean_brain come from config_registration.py.
# To change the template, update TEMPLATE_IDX there and resubmit.

# All fish passed QC — edit this list in config if you discarded any during QC.
fish_for_mean_brain = list(all_fish_for_mean_brain)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_vol(fish):
    return load_volume_mean(fish, dir_voluseg, res_x, res_y, res_z, rotation_k)

# ── Load template ─────────────────────────────────────────────────────────────

template_idx  = min(TEMPLATE_IDX, len(fish_for_mean_brain) - 1)
template_fish = fish_for_mean_brain[template_idx]
print(f"\nLoading template: {template_fish[1]}")
template_img = load_vol(template_fish)
print(f"Template shape: {template_img.shape}, spacing: {template_img.spacing}")
plt.close("all")   # discard any plots from load_volume_mean

# ── Register all fish to template ─────────────────────────────────────────────

print(f"\n=== Registering {len(fish_for_mean_brain)} fish to template ===")
registered_imgs = []

for i, fish in enumerate(fish_for_mean_brain):
    proj_ID, expt_ID = fish
    print(f"\n[{i+1}/{len(fish_for_mean_brain)}] {expt_ID}")

    save_dir = os.path.join(dir_registration, proj_ID, expt_ID)
    os.makedirs(save_dir, exist_ok=True)

    registered_path = os.path.join(save_dir, "expt_to_temp_registered.nii.gz")

    if os.path.exists(registered_path):
        print(f"  ⏭️  Already registered — loading from disk.")
        warped = ants.image_read(registered_path)
        registered_imgs.append(warped)
        continue

    moving_img = load_vol(fish)
    plt.close("all")

    warped = register_to_template(
        moving_img, template_img, target_spacing=None, save_dir=save_dir
    )
    registered_imgs.append(warped)
    print(f"  ✅ Done")

# ── Masked average → mean brain ───────────────────────────────────────────────

print("\n=== Creating masked mean brain ===")
registered_np = [
    normalize_image_intensity(img.numpy())
    for img in registered_imgs
]
masks    = [(img > 0).astype(np.float32) for img in registered_np]
sum_img  = np.sum(registered_np, axis=0)
sum_mask = np.sum(masks, axis=0)
mean_np  = np.divide(sum_img, sum_mask,
                     out=np.zeros_like(sum_img), where=sum_mask > 0)
print(f"Mean brain shape: {mean_np.shape}")

mean_ants = ants.from_numpy(mean_np)
mean_ants.set_spacing(registered_imgs[0].spacing)

os.makedirs(os.path.dirname(MEAN_BRAIN_PATH), exist_ok=True)
ants.image_write(mean_ants, MEAN_BRAIN_PATH)
print(f"\n✅ Mean brain saved: {MEAN_BRAIN_PATH}")
print(f"   Built from {len(fish_for_mean_brain)} fish.")
print(f"   Template: {template_fish[1]} (index {template_idx})")
