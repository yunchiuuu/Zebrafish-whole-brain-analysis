"""
run_registration_mean_brain.py
==============================
Build the shared mean brain from all fish across all experiment groups.

Run interactively via VS Code Remote-SSH using the interactive window (# %% cells).
Do NOT sbatch — the QC step requires visual inspection.

Pipeline:
    Cell 1: imports + env setup
    Cell 2: config
    Cell 3: (optional) interactive QC — plot volume means, keep/discard fish
    Cell 4: generate + save mean brain

Output:
    {dir_registration} / {MEAN_BRAIN_FNAME}
    e.g. .../registration/mean_brain_HCRT_TRPV1_ALL.nii.gz

Location:
    ~/Zebrafish-whole-brain-analysis/registration/run_registration_mean_brain.py
"""

# %% ============================================================
# CELL 1: Imports + env setup
# ============================================================

import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registration.registration import (
    load_volume_mean,
    register_to_template,
    normalize_image_intensity,
)

print(f"✅ TMPDIR={os.environ['TMPDIR']} | ITK threads={n_threads}")

# %% ============================================================
# CELL 2: Config
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registration.config_registration import (
    dir_registration,
    MEAN_BRAIN_PATH,
    n_slices,
    res_x, res_y, res_z,
    rotation_k,
    all_fish_for_mean_brain,
)

# Index of fish to use as initial template (change if that fish has bad quality)
TEMPLATE_IDX = 1
RUN_QC       = True    # set False to skip interactive QC and use all_fish_for_mean_brain

print(f"Mean brain will be saved to: {MEAN_BRAIN_PATH}")
print(f"Total fish available: {len(all_fish_for_mean_brain)}")
print(f"Template fish: {all_fish_for_mean_brain[TEMPLATE_IDX][1]}")

# %% ============================================================
# CELL 3: Interactive QC — inspect volume means, keep/discard fish
# (set RUN_QC = False in Cell 2 to skip)
# ============================================================

def plot_volume_mean_qc(img, expt_ID, n_planes=10):
    arr = img.numpy()
    z_indices = np.linspace(0, arr.shape[2] - 1, n_planes, dtype=int)
    fig, axes = plt.subplots(1, n_planes, figsize=(30, 2))
    fig.suptitle(expt_ID, fontsize=11)
    for ax, z in zip(axes, z_indices):
        ax.imshow(arr[:, :, z], cmap="gray")
        ax.set_title(f"Z={z}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show(block=True)


def load_vol(fish):
    proj_ID, expt_ID = fish
    return load_volume_mean(
        proj_ID, expt_ID, res_x, res_y, res_z, n_slices, binning=1, rot=rotation_k
    )


fish_for_mean_brain = list(all_fish_for_mean_brain)

if RUN_QC:
    print("=== Interactive QC: inspect volume means ===")
    print("Close each plot window, then type y to keep or n to discard.\n")
    fish_for_mean_brain = []

    for fish in all_fish_for_mean_brain:
        proj_ID, expt_ID = fish
        vol_img = load_vol(fish)
        plot_volume_mean_qc(vol_img, expt_ID)
        answer = input(f"Keep {expt_ID}? (y/n): ").strip().lower()
        if answer == "y":
            fish_for_mean_brain.append(fish)
            print(f"  ✅ Kept")
        else:
            print(f"  ❌ Discarded")

    print(f"\nFish kept ({len(fish_for_mean_brain)} / {len(all_fish_for_mean_brain)}):")
    for proj_ID, expt_ID in fish_for_mean_brain:
        print(f"  {expt_ID}")

    if len(fish_for_mean_brain) == 0:
        raise RuntimeError("No fish kept — aborting.")

# %% ============================================================
# CELL 4: Generate + save mean brain
# ============================================================

# Clip template index to kept fish list
template_idx  = min(TEMPLATE_IDX, len(fish_for_mean_brain) - 1)
template_fish = fish_for_mean_brain[template_idx]
print(f"\nLoading initial template: {template_fish[1]}")
template_img = load_vol(template_fish)
print(f"Template shape: {template_img.shape}, spacing: {template_img.spacing}")

# Register all fish to template
print(f"\n=== Registering {len(fish_for_mean_brain)} fish to template ===")
registered_imgs = []

for fish in fish_for_mean_brain:
    proj_ID, expt_ID = fish
    print(f"\n  {expt_ID}")
    save_dir = os.path.join(dir_registration, proj_ID, expt_ID)
    os.makedirs(save_dir, exist_ok=True)

    moving_img = load_vol(fish)
    warped = register_to_template(
        moving_img, template_img, target_spacing=None, save_dir=save_dir
    )
    registered_imgs.append(warped)
    print(f"  ✅ Done")

# Masked average → mean brain
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
