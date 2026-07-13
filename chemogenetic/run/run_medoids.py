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
TEST_MODE  = True   # True → run only the first fish for QC; set False for full batch

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
            # ----------------------------------------------------------
            # Step 1: load cell coordinates from voluseg
            # ----------------------------------------------------------
            # read_data returns: (data_array, volume_mean_raw, cell_x, cell_y, cell_z)
            # volume_mean_raw.shape = (X, Y, Z) = (280, 544, 40) — already reoriented
            _, volume_mean_raw, cell_x, cell_y, cell_z = read_data(fish, dir_voluseg)
            vol_shape = volume_mean_raw.shape    # (280, 544, 40)
            print(f"  volume_mean_raw.shape: {vol_shape}")

            # ----------------------------------------------------------
            # Step 2: compute true medoids (pairwise cdist)
            # ----------------------------------------------------------
            if raw_path.exists() and not OVERWRITE:
                print(f"  ⏩ medoids_all.npy exists, loading")
                medoids_all = np.load(str(raw_path))
            else:
                medoids_all = compute_medoids(cell_x, cell_y, cell_z)
                np.save(str(raw_path), medoids_all)
                print(f"  ✅ Saved medoids_all.npy  → {raw_path}")

            # ----------------------------------------------------------
            # Step 3: rotate medoids 180° in X–Y to match ANTs convention
            # ----------------------------------------------------------
            if rot_path.exists() and not OVERWRITE:
                print(f"  ⏩ medoids_rot_all.npy exists, loading")
                medoids_rot = np.load(str(rot_path))
            else:
                medoids_rot = rotate_medoids(medoids_all, vol_shape, ROTATION_K)
                np.save(str(rot_path), medoids_rot)
                print(f"  ✅ Saved medoids_rot_all.npy → {rot_path}")

            # ----------------------------------------------------------
            # Step 4: transform rotated medoids to template space
            # ----------------------------------------------------------
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

    # ⚠️  POINTS list = [affine, warp]  (reversed from image list)
                # No whichtoinvert — transforms stored fish→template forward direction
                transform_list_points = [str(affine), str(warp)]

                # Build ANTs moving reference image with correct spacing
                # (used for transform_index_to_physical_point)
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

        except Exception as e:
            print(f"  ❌ {expt_ID} failed: {e}")
            raise   # re-raise so the error is visible; remove for batch runs

        finally:
            gc.collect()

    print("\nMedoid run complete.")

    # ------------------------------------------------------------------
    # QC plots (TEST_MODE only) — 2x2 layout:
    #   Col 0: Before transform (moving space, volume_mean_raw background)
    #   Col 1: After transform  (template space, registered_final background)
    #   Row 0: XY top-down view
    #   Row 1: YZ side view
    # Medoids: yellow scatter, alpha=0.7, overlaid on grayscale brain
    # ------------------------------------------------------------------
    if TEST_MODE:
        import matplotlib.pyplot as plt
        import ants as _ants

        qc_fish       = fish_to_run[0]
        proj_ID_qc, expt_ID_qc = qc_fish
        qc_f_dir      = fish_dir(dir_analysis, qc_fish)
        vox_path      = qc_f_dir / "medoids_template_vox.npy"
        rot_path      = qc_f_dir / "medoids_rot_all.npy"
        reg_final     = DIR_REGISTRATION / proj_ID_qc / expt_ID_qc / "expt_to_mean_registered_final.nii.gz"

        if not (vox_path.exists() and rot_path.exists()):
            print("QC skipped — output files not found.")
        else:
            transformed = np.load(str(vox_path))   # (n_cells, 3) float32, NaN=invalid
            medoids_rot = np.load(str(rot_path))    # (n_cells, 3) int,     -1=invalid

            finite_mask = np.isfinite(transformed[:, 0])
            valid_mov   = ~np.all(medoids_rot == -1, axis=1)

            # load brain backgrounds
            _, vol_mean_raw, _, _, _ = read_data(qc_fish, dir_voluseg)
            vol_mean_raw = np.asarray(vol_mean_raw, dtype=np.float32)  # (X, Y, Z)

            if reg_final.exists():
                reg_brain = _ants.image_read(str(reg_final)).numpy().astype(np.float32)
            else:
                print(f"  ⚠️  registered_final not found: {reg_final}")
                reg_brain = None

            # projections — mean over the orthogonal axis
            mov_xy  = np.nanmean(vol_mean_raw, axis=2)            # (X, Y)
            mov_yz  = np.nanmean(vol_mean_raw, axis=0)            # (Y, Z)
            tmpl_xy = np.nanmean(reg_brain, axis=2) if reg_brain is not None else None
            tmpl_yz = np.nanmean(reg_brain, axis=0) if reg_brain is not None else None

            # subsample medoids so scatter isn't too slow
            MAX_PTS = 80_000
            rng = np.random.default_rng(0)

            def _sub(mask):
                idx = np.where(mask)[0]
                return rng.choice(idx, MAX_PTS, replace=False) if idx.size > MAX_PTS else idx

            mv = medoids_rot[_sub(valid_mov)].astype(float)   # (n, 3): x, y, z
            tr = transformed[_sub(finite_mask)]                # (n, 3): i, j, k

            # figure
            n_mov  = int(valid_mov.sum())
            n_tmpl = int(finite_mask.sum())

            DOT  = dict(s=0.5, color="yellow", alpha=0.4, linewidths=0)
            GRAY = dict(origin="upper", cmap="gray", interpolation="nearest")

            # Layout: 1 row × 4 cols
            #   [Before XY | Before YZ | After XY | After YZ]
            # Y axis is vertical in all panels → sharey pairs XY and YZ
            # width_ratios proportional to X and Z dimensions so brains aren't distorted
            X_mov, Y_mov, Z_dim = vol_mean_raw.shape   # e.g. (296, 532, 40)
            X_tmpl = reg_brain.shape[0] if reg_brain is not None else X_mov

            fig = plt.figure(figsize=(18, 8))
            fig.suptitle(f"QC: {expt_ID_qc} — medoid transform", fontsize=13)
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(
                1, 4,
                width_ratios=[X_mov, Z_dim, X_tmpl, Z_dim],
                wspace=0.03,
            )
            ax_bxy = fig.add_subplot(gs[0, 0])
            ax_byz = fig.add_subplot(gs[0, 1], sharey=ax_bxy)
            ax_axy = fig.add_subplot(gs[0, 2])
            ax_ayz = fig.add_subplot(gs[0, 3], sharey=ax_axy)

            # Before XY — Y vertical (rows), X horizontal (cols)
            ax_bxy.imshow(mov_xy.T, **GRAY, aspect="auto")
            ax_bxy.scatter(mv[:, 0], mv[:, 1], **DOT)
            ax_bxy.set_title(f"Before — XY top-down\nN={n_mov:,}", fontsize=10)
            ax_bxy.axis("off")

            # Before YZ — Y vertical (rows), Z horizontal (cols); no transpose
            ax_byz.imshow(mov_yz, **GRAY, aspect="auto")
            ax_byz.scatter(mv[:, 2], mv[:, 1], **DOT)
            ax_byz.set_title(f"Before — YZ side\nN={n_mov:,}", fontsize=10)
            ax_byz.axis("off")

            # After XY
            if tmpl_xy is not None:
                ax_axy.imshow(tmpl_xy.T, **GRAY, aspect="auto")
            ax_axy.scatter(tr[:, 0], tr[:, 1], **DOT)
            ax_axy.set_title(f"After — XY top-down\nN={n_tmpl:,}", fontsize=10)
            ax_axy.axis("off")

            # After YZ — Y vertical, Z horizontal; no transpose
            if tmpl_yz is not None:
                ax_ayz.imshow(tmpl_yz, **GRAY, aspect="auto")
            ax_ayz.scatter(tr[:, 2], tr[:, 1], **DOT)
            ax_ayz.set_title(f"After — YZ side\nN={n_tmpl:,}", fontsize=10)
            ax_ayz.axis("off")

            plt.tight_layout()

            qc_path = qc_f_dir / "figures" / "medoids_transform_QC.png"
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(qc_path), dpi=200, bbox_inches="tight")
            print(f"\nQC figure saved → {qc_path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
