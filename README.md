# Zebrafish whole-brain analysis (zwba)

Analysis pipeline for whole-brain light-sheet GCaMP imaging in larval zebrafish

Raw TIFF -> HDF5 (manual steps; not included)
-> voluseg cell segmentation (https://github.com/mikarubi/voluseg)
-> ANTs registration to a shared template (included)
-> tonic/phasic decomposition (included)
-> GLM / d' responder statistics (included)
-> brain maps (included)

## Layout

```
run_voluseg.py            voluseg steps 0-5, one fish per job
registration/             template generation, SyN registration, MapZebrain -> template
  config/config_registration.py   fish lists for registration (single source of truth for step 3)
chemogenetic/
  config/config_<exp>.py          per-experiment config (paths, fish tuples, epochs, GLM params)
  run/                            run scripts (sbatch targets)
  submit/                         sbatch wrappers, one per run script
  *.py                            modules (operate on one fish passed as argument)
utils/                    data_io (fish_dir, read_data), preprocess, stats
```

Conventions:
- A fish is a `(proj_ID, expt_ID)` tuple; paths resolve through `utils.data_io.fish_dir()`.
- Three path trees (see config): `dir_voluseg` (voluseg in/out), `dir_registration`
  (ANTs products), `dir_analysis` (everything the analysis writes).
- `--config` means a **module name** under `chemogenetic/config/` for chemogenetic scripts
  (e.g. `config_hcrt_trpv1_csn_120min`), but a **file path** for registration scripts
  (`registration/config/config_registration.py`).
- All submit scripts run from the repo root.

## Running a new dataset

### 0. Register the dataset in two configs
- Copy `chemogenetic/config/config_template.py` to `config_<exp>.py`; set fish tuples,
  epoch windows (frames), group tags.
- Add the fish to `registration/config/config_registration.py`. `submit_registration_syn.sh`
  submits exactly the fish that are **uncommented** there.

### 1. Raw TIFF -> HDF5 (analysis PC, not HPC)
- `run_preprocess_batch.py` (VS Code `# %%` cells) + `tif_to_h5.py`  *(TODO: commit to repo)*.
- Per fish: set `start_plane`, QC z-order, batch-convert to
  `{proj}/{expt}/input/volume1000.h5, volume1001.h5, ...` (dataset key `'default'`).
- Run **serially** (parallel conversion truncated H5 files). Then sync to the HPC.

### 2. Voluseg (HPC, one sbatch per fish)
Apply the voluseg `step2.py` ANTs-glob patch first (`apply_voluseg_patch.py`, `--dry-run` /
`--revert`; *TODO: commit to repo*), clear `__pycache__`, restart any Spark session.

```bash
sbatch --job-name=voluseg_<fish> --cpus-per-task=24 --mem=200G --time=8:00:00 \
  --output=logs/voluseg/<expt_ID>.log \
  --wrap="/resnick/home/ychiu/miniconda3/envs/voluseg/bin/python run_voluseg.py \
          --proj_ID <proj_ID> --expt_ID <expt_ID>"
```
Writes `output/cells0_clean.hdf5` and `output/volume0.hdf5`. Never run several fish in one process.

### 3. Registration (HPC)
One-time, per template:
```bash
python registration/generate_template.py          # 3 audio Z-stack fish -> template_mean_brain.nii.gz
bash registration/submit_mapzebrain_registration.sh   # MapZebrain ref + region masks -> template space
```
`generate_template.py`: registers `260707_fish1` and `260708_fish1` to `260707_fish2`
(single pass, SyN), then averages the two warped volumes with fish2 unwarped.
Not iterative; intensities not normalized before averaging.

Per fish:
```bash
bash registration/submit_registration_syn.sh     # 1 job/fish, 32 CPUs, 256G
```
Writes `expt_to_mean_warp.nii.gz`, `expt_to_mean_affine.mat` under
`{dir_registration}/{proj}/{expt}/`. **If the template changes, re-register ALL fish and
re-run medoids with `OVERWRITE=True`.**

### 4. Chemogenetic preprocessing (HPC)
```bash
bash chemogenetic/submit/submit_decompose.sh --config config_<exp>
    # f_dark_scalar.npy, f_tonic.npy, f_phasic.npy
bash chemogenetic/submit/submit_medoids.sh --config config_<exp>
    # needs step 3 transforms -> medoids_all.npy, medoids_rot_all.npy, medoids_template_vox.npy
```

### 5. Responder statistics
```bash
bash chemogenetic/submit/submit_glm.sh --config config_<exp>
    # CV, global params, refit, ablation, IAAFT null, responders
    # -> glm/inHCRT_K600_drift1_lam0.5_lag0/, tonic_pos/neg_glm_iaaft_nullp99_idxs.npy
bash chemogenetic/submit/submit_dprime.sh --config config_<exp>
    # -> phasic_dprime_cells_*.npy, phasic IAAFT responder idxs, summary figures
bash chemogenetic/submit/submit_figures_glm.sh --config config_<exp>   # after the GLM
```
`run_glm.py` uses the empirical HCRT regressor from `hcrt_all.csv` (paths hardcoded in the
script). If those files are missing it **silently falls back to the CSTR capsaicin model** -
check the log for which one ran.

### 6. Maps (all require `medoids_template_vox.npy`)
```bash
bash chemogenetic/submit/submit_temporal_intensity_map.sh --config config_<exp>
    # caches epoch_dz.npy / epoch_dprime.npy, per-window maps
bash chemogenetic/submit/submit_glm_dprime_map.sh --config config_<exp>   # N-fish presence maps
bash chemogenetic/submit/submit_kernel_map.sh --config config_<exp>       # kernel maps
bash chemogenetic/submit/submit_survey_by_regions.sh ...                   # needs epoch_*.npy + MapZebrain masks
bash chemogenetic/submit/submit_combined_fish_map.sh --config config_hcrt_trpv1_pooled_csn_120min
```
`run_glm_brainmap.py` has no submit script.

## Key conventions
- dF: `F_corrected = F_raw - F_dark`; `F_tonic` = rolling 20th percentile (600 vols);
  `F_phasic = (F_corrected - F_tonic) / F_tonic`. `F_dark` from 10x10 corner patches.
- DeltaZ uses the last 25 min of baseline and of the drug phase (`OFFSET_MIN = 20`, applied
  from config values, never hardcoded). If the fit window changes, set `OVERWRITE_REFIT`,
  `OVERWRITE_ABLATION`, `OVERWRITE_NULL` all True.
- Responder thresholds: ctrl-pooled IAAFT p99 for group fraction comparisons; per-fish IAAFT p99
  for per-fish clustering; phasic uses |d'| >= 0.5 with IAAFT p99 as validation.

## Known issues / TODO
- `run_voluseg.py` line 40 (`import log_step from data_io`) is a syntax error; should be
  `from utils.data_io import log_step`. Docstring location also disagrees with file location.
- Preprocessing scripts (`tif_to_h5.py`, `run_preprocess_batch.py`, `apply_voluseg_patch.py`)
  are not yet in git.
- `submit_individual_fish_map.sh` calls `run_individual_fish_map.py`, which is not in the repo.
- Hardcoded paths: `HCRT_BASE` in `run_glm.py`, `MASKS_DIR` in `run_survey_by_regions.py`,
  `HPC_USERNAME` in registration scripts.
- Registered fish may predate the current Z-stack template; verify warp vs template dates.
- Legacy Spearman / permutation / effect-size pipeline removed (superseded by GLM + IAAFT);
  recover from git tag `pre-spearman-removal`.
