"""
run_medoids.py
==============
Compute per-cell medoid coordinates and transform them to template space.
Matches notebook Step 6 logic exactly.

Per-fish output (written to {dir_analysis}/{proj_ID}/{expt_ID}/):
    medoids_all.npy               (n_cells, 3) int    — raw medoids, sentinel=-1
    medoids_rot_all.npy           (n_cells, 3) int    — after 180° XY rotation
    medoids_template_vox.npy     (n_cells, 3) float32 — template voxel [i,j,k], NaN=invalid

Step sequence per fish:
    1. read_data()         → cell_x, cell_y, cell_z (N×M arrays) + volume_mean_raw
    2. compute_medoids()   → medoids_all.npy  (true medoid via pairwise cdist)
    3. rotate_medoids()    → medoids_rot_all.npy  (180° XY flip to match ANTs convention)
    4. transform_medoids_to_template()  → medoids_template_vox.npy

Transform pipeline (Step 4):
    medoids_rot [x, y, z] in fish ANTs voxel space
    → ants.transform_index_to_physical_point(mov_ref, ...)  → fish physical (µm)
    → ants.apply_transforms_to_points([affine, warp])        → template physical (µm)
    → ants.transform_physical_point_to_index(template, ...) → template voxel [i,j,k]

    ⚠️  transform_list for POINTS is [affine.mat, warp.nii.gz]
        (REVERSED from image list — no whichtoinvert needed)

Usage
-----
    python run_medoids.py --config config_hcrt_trpv1_csn_120min

Location:
    ~/zwba/chemogenetic/run/run_medoids.py
"""

import argparse
import gc
import importlib
import sys
from pathlib import Path

import ants
import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compute and transform cell medoids for all fish in a config."
)
parser.add_argument(
    "--config", required=True,
    help="Config module under chemogenetic/config/, "
         "e.g. config_hcrt_trpv1_csn_120min"
)
args = parser.parse_args()

cfg = importlib.import_module(f"chemogenetic.config.{args.config}")

all_fish     = cfg.all_fish
dir_analysis = cfg.dir_analysis
dir_voluseg  = cfg.dir_voluseg

from utils.data_io import fish_dir, read_data
from utils.preprocess import (
    compute_medoids,
    rotate_medoids,
    transform_medoids_to_template,
)

# ---------------------------------------------------------------------------
# PATHS — imported from configs, no hardcoding
# ---------------------------------------------------------------------------
# Per-fish ANTs transforms expected at:
#   {DIR_REGISTRATION}/{proj_ID}/{expt_ID}/expt_to_mean_affine.mat
#   {DIR_REGISTRATION}/{proj_ID}/{expt_ID}/expt_to_mean_warp.nii.gz
DIR_REGISTRATION = Path(cfg.dir_registration)

# Template brain — from registration config
sys.path.insert(0, str(REPO_ROOT / "registration" / "config"))
from config_registration import MEAN_BRAIN_PATH
TEMPLATE_PATH = Path(MEAN_BRAIN_PATH)

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
OVERWRITE  = False  # True → recompute even if output files already exist
TEST_MODE  = False   # True → run only the first fish for QC; set False for full batch

# Voluseg raw volume geometry
# volume_mean_raw.shape from read_data is (X, Y, Z) = (280, 544, 40)
# These spacings are for the ORIGINAL (pre-resample) volume
RES_X = 1.52   # µm/pixel (lateral)
RES_Y = 1.52   # µm/pixel (lateral)
RES_Z = 6.25   # µm/plane  (250 µm depth / 40 planes)

ROTATION_K = 2  # 180° flip of X–Y axes

# read_data return order (confirmed from notebook):
#   data_array, volume_mean_raw, cell_x, cell_y, cell_z = read_data(...)
# run_decompose only uses data_array and discards the rest with _.


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QC HELPER — runs for every fish regardless of TEST_MODE
# ---------------------------------------------------------------------------
def _plot_medoid_qc(
    expt_ID, fish_out, vol_mean_raw, medoids_rot, vox_path, reg_final_path
):
    """
    Save a 4-panel QC figure for one fish:
        [Before XY | Before YZ | After XY | After YZ]
    Background: grayscale brain (volume_mean_raw before, registered_final after).
    Medoids: yellow (before), blue (after), alpha=0.4.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for SLURM nodes
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import ants as _ants

    if not vox_path.exists():
        print(f"  ⚠️  QC skipped — medoids_template_vox.npy not found")
        return

    transformed = np.load(str(vox_path))                         # (n_cells,3) float32
    finite_mask = np.isfinite(transformed[:, 0])
    valid_mov   = ~np.all(medoids_rot == -1, axis=1)

    # load registered brain background
    reg_brain = None
    if reg_final_path.exists():
        reg_brain = _ants.image_read(str(reg_final_path)).numpy().astype(np.float32)
    else:
        print(f"  ⚠️  registered_final not found, after-panel will show dots only")

    vol_mean_raw = np.asarray(vol_mean_raw, dtype=np.float32)    # (X, Y, Z)

    # projections
    mov_xy  = np.nanmean(vol_mean_raw, axis=2)                   # (X, Y)
    mov_yz  = np.nanmean(vol_mean_raw, axis=0)                   # (Y, Z)
    tmpl_xy = np.nanmean(reg_brain, axis=2) if reg_brain is not None else None
    tmpl_yz = np.nanmean(reg_brain, axis=0) if reg_brain is not None else None

    # subsample for scatter speed
    MAX_PTS = 80_000
    rng = np.random.default_rng(0)

    def _sub(mask):
        idx = np.where(mask)[0]
        return rng.choice(idx, MAX_PTS, replace=False) if idx.size > MAX_PTS else idx

    mv = medoids_rot[_sub(valid_mov)].astype(float)              # (n, 3): x, y, z
    tr = transformed[_sub(finite_mask)]                           # (n, 3): i, j, k

    n_mov  = int(valid_mov.sum())
    n_tmpl = int(finite_mask.sum())

    DOT_B = dict(s=0.5, color="yellow",  alpha=0.4, linewidths=0)
    DOT_A = dict(s=0.5, color="#4a9ee8", alpha=0.4, linewidths=0)

    X_mov, Y_mov, Z_dim = vol_mean_raw.shape
    X_tmpl = reg_brain.shape[0] if reg_brain is not None else X_mov
    Y_tmpl = reg_brain.shape[1] if reg_brain is not None else Y_mov

    # flip before-scatter y to match origin="lower"
    mv_y_flip = (Y_mov - 1) - mv[:, 1]

    width_ratios = [X_mov / Y_mov, Z_dim / Y_mov,
                    X_tmpl / Y_tmpl, Z_dim / Y_tmpl]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"QC: {expt_ID} — medoid transform", fontsize=13)
    gs = GridSpec(1, 4, width_ratios=width_ratios, wspace=0.03)
    ax_bxy = fig.add_subplot(gs[0, 0])
    ax_byz = fig.add_subplot(gs[0, 1], sharey=ax_bxy)
    ax_axy = fig.add_subplot(gs[0, 2])
    ax_ayz = fig.add_subplot(gs[0, 3], sharey=ax_axy)

    ax_bxy.imshow(mov_xy.T, cmap="gray", origin="lower",
                  aspect="equal", interpolation="nearest")
    ax_bxy.scatter(mv[:, 0], mv_y_flip, **DOT_B)
    ax_bxy.set_title(f"Before — XY top-down\nN={n_mov:,}", fontsize=10)
    ax_bxy.axis("off")

    ax_byz.imshow(np.fliplr(mov_yz), cmap="gray", origin="lower",
                  aspect="auto", interpolation="nearest")
    ax_byz.scatter(mv[:, 2], mv_y_flip, **DOT_B)
    ax_byz.set_title(f"Before — YZ side\nN={n_mov:,}", fontsize=10)
    ax_byz.axis("off")

    if tmpl_xy is not None:
        ax_axy.imshow(tmpl_xy.T, cmap="gray", origin="upper",
                      aspect="equal", interpolation="nearest")
    ax_axy.scatter(tr[:, 0], tr[:, 1], **DOT_A)
    ax_axy.set_title(f"After — XY top-down\nN={n_tmpl:,}", fontsize=10)
    ax_axy.axis("off")

    if tmpl_yz is not None:
        ax_ayz.imshow(tmpl_yz, cmap="gray", origin="upper",
                      aspect="auto", interpolation="nearest")
    ax_ayz.scatter(tr[:, 2], tr[:, 1], **DOT_A)
    ax_ayz.set_title(f"After — YZ side\nN={n_tmpl:,}", fontsize=10)
    ax_ayz.axis("off")

    plt.tight_layout()

    qc_path = fish_out / "QC_figures" / "medoids_transform_QC.png"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(qc_path), dpi=200, bbox_inches="tight")
    print(f"  ✅ QC figure saved → {qc_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Template not found: {TEMPLATE_PATH}\n"
            "Update TEMPLATE_PATH at top of this script."
        )
    if not DIR_REGISTRATION.exists():
        raise FileNotFoundError(
            f"Registration dir not found: {DIR_REGISTRATION}\n"
            "Update DIR_REGISTRATION at top of this script."
        )

    template_img = ants.image_read(str(TEMPLATE_PATH))
    print(f"Template: {TEMPLATE_PATH.name}  shape={template_img.shape}")

    fish_to_run = all_fish[:1] if TEST_MODE else all_fish
    print(f"Running medoids for {len(fish_to_run)} fish"
          f"  {'(TEST MODE — first fish only)' if TEST_MODE else ''}\n")

    for fish in fish_to_run:
        proj_ID, expt_ID = fish
        print(f"── {expt_ID}")

        fish_out  = fish_dir(dir_analysis, fish)
        fish_out.mkdir(parents=True, exist_ok=True)

        raw_path  = fish_out / "medoids_all.npy"
        rot_path  = fish_out / "medoids_rot_all.npy"
        vox_path  = fish_out / "medoids_template_vox.npy"

        try:
            # Step 1: load cell coordinates
            _, volume_mean_raw, cell_x, cell_y, cell_z = read_data(fish, dir_voluseg)
            vol_shape = volume_mean_raw.shape
            print(f"  volume_mean_raw.shape: {vol_shape}")

            # Step 2: compute medoids
            if raw_path.exists() and not OVERWRITE:
                print(f"  ⏩ medoids_all.npy exists, loading")
                medoids_all = np.load(str(raw_path))
            else:
                medoids_all = compute_medoids(cell_x, cell_y, cell_z)
                np.save(str(raw_path), medoids_all)
                print(f"  ✅ Saved medoids_all.npy  → {raw_path}")

            # Step 3: rotate
            if rot_path.exists() and not OVERWRITE:
                print(f"  ⏩ medoids_rot_all.npy exists, loading")
                medoids_rot = np.load(str(rot_path))
            else:
                medoids_rot = rotate_medoids(medoids_all, vol_shape, ROTATION_K)
                np.save(str(rot_path), medoids_rot)
                print(f"  ✅ Saved medoids_rot_all.npy → {rot_path}")

            # Step 4: transform to template space
            if vox_path.exists() and not OVERWRITE:
                print(f"  ⏩ medoids_template_vox.npy exists, skipping")
            else:
                reg_dir = DIR_REGISTRATION / proj_ID / expt_ID
                affine  = reg_dir / "expt_to_mean_affine.mat"
                warp    = reg_dir / "expt_to_mean_warp.nii.gz"

                if not affine.exists() or not warp.exists():
                    raise FileNotFoundError(
                        f"ANTs transforms not found in {reg_dir}\n"
                        "Run run_registration_syn.py for this fish first."
                    )

                # POINTS list = [affine, warp] (reversed from image list, no whichtoinvert)
                transform_list_points = [str(affine), str(warp)]

                mov_ref = ants.from_numpy(
                    volume_mean_raw.astype(np.float32),
                    spacing=(RES_X, RES_Y, RES_Z),
                )

                medoids_vox = transform_medoids_to_template(
                    medoids_rot    = medoids_rot,
                    transform_list = transform_list_points,
                    template_img   = template_img,
                    mov_ref        = mov_ref,
                    res_x          = RES_X,
                    res_y          = RES_Y,
                    res_z          = RES_Z,
                )

                np.save(str(vox_path), medoids_vox)
                print(f"  ✅ Saved medoids_template_vox.npy → {vox_path}")

            # Step 5: QC figure — always, for every fish
            reg_final = DIR_REGISTRATION / proj_ID / expt_ID / "expt_to_mean_registered_final.nii.gz"
            _plot_medoid_qc(
                expt_ID        = expt_ID,
                fish_out       = fish_out,
                vol_mean_raw   = volume_mean_raw,
                medoids_rot    = medoids_rot,
                vox_path       = vox_path,
                reg_final_path = reg_final,
            )

        except Exception as e:
            print(f"  ❌ {expt_ID} failed: {e}")
            raise

        finally:
            gc.collect()

    print("\nMedoid run complete.")


if __name__ == "__main__":
    main()
