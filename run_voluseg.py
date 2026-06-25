"""
run_voluseg.py
==============
Run voluseg segmentation (steps 0–5) for one fish.

Designed to be submitted as a separate sbatch job per fish.
Voluseg uses Spark internally for parallelism — do NOT run multiple
fish in the same process.

Usage (interactive):
python run/run_voluseg.py --proj_ID hcrt-trpv1_huc-h2b-g8m_csn_120min_May26 \
                          --expt_ID 260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1

Usage (SLURM):
sbatch \
  --cpus-per-task=24 \
  --mem=200G \
  --time=8:00:00 \   
  --wrap="python run/run_voluseg.py \
    --proj_ID hcrt-trpv1_huc-h2b-g8m_csn_120min_May26 \
    --expt_ID 260514_hcrt-trpv1_huc-h2b-g8m_csn_10uM_fish1"

Check SLURM job ID:
squeue -j 12345678
squeue -u $USER

After job finishes:
sacct -j 12345678 --format=JobID,JobName,State,Elapsed,MaxRSS,AllocCPUS,ReqMem
seff 12345678

Location:
~/Zebrafish-whole-brain-analysis/chemogenetic/run/run_voluseg.py
"""

import argparse
import gc
import os
import time
from pathlib import Path
import log_step from data_io

# ============================================================
# PARSE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="Run voluseg segmentation for one fish.")
parser.add_argument("--proj_ID", required=True, help="Project folder name under dir_voluseg")
parser.add_argument("--expt_ID", required=True, help="Experiment folder name")
args = parser.parse_args()

proj_ID = args.proj_ID
expt_ID = args.expt_ID

# ============================================================
# CONFIG
# ============================================================

# Base path on HPC
BASE = "/resnick/groups/Proberlab/yun/lightsheet/"

# Imaging parameters — update per experiment batch
FRAMES_PER_VOLUME = 40
SECONDS_PER_VOLUME = 1
Z_RANGE_UM = 250
BINNING = 1
PIXEL_SIZE_UM = 1.52

# Derived
RES_X = PIXEL_SIZE_UM * BINNING
RES_Y = PIXEL_SIZE_UM * BINNING
RES_Z = Z_RANGE_UM / FRAMES_PER_VOLUME

DIR_INPUT  = os.path.join(BASE, proj_ID, expt_ID, "input")
DIR_OUTPUT = os.path.join(BASE, proj_ID, expt_ID, "output")
DIR_ANTS   = os.path.join(BASE, "voluseg_setup", "install", "bin")

# Spark config (local[24] + parallelism=9 avoids OOM on high-core machines)
SPARK_CORES        = 24
SPARK_PARALLELISM  = 9
SPARK_EXECUTOR_MEM = "80g"
SPARK_DRIVER_MEM   = "80g"
SPARK_OFFHEAP_SIZE = "40g"

# ============================================================
# SPARK SESSION
# ============================================================

import findspark
findspark.init()

from pyspark.sql import SparkSession

try:
    spark.stop()
except NameError:
    pass

spark = (
    SparkSession.builder
    .master(f"local[{SPARK_CORES}]")
    .config("spark.executor.instances",       "8")
    .config("spark.executor.cores",           "5")
    .config("spark.driver.maxResultSize",     "0")
    .config("spark.executor.memory",          SPARK_EXECUTOR_MEM)
    .config("spark.driver.memory",            SPARK_DRIVER_MEM)
    .config("spark.memory.offHeap.enabled",   True)
    .config("spark.memory.offHeap.size",      SPARK_OFFHEAP_SIZE)
    .config("spark.default.parallelism",      str(SPARK_PARALLELISM))
    .config("spark.sql.shuffle.partitions",   "96")
    .appName(f"voluseg_{expt_ID}")
    .getOrCreate()
)

# ============================================================
# VOLUSEG
# ============================================================

import voluseg

print(f"\n{'='*60}")
print(f"  voluseg: {proj_ID} / {expt_ID}")
print(f"{'='*60}\n")

# --- Step 0: set and save parameters ---
print("Step 0: process parameters.")
parameters0 = voluseg.parameter_dictionary()

parameters0["dir_ants"]   = DIR_ANTS
parameters0["dir_input"]  = DIR_INPUT
parameters0["dir_output"] = DIR_OUTPUT

parameters0["registration"]   = "high"
parameters0["diam_cell"]      = 5.0
parameters0["f_volume"]       = 1 / FRAMES_PER_VOLUME
parameters0["t_section"]      = SECONDS_PER_VOLUME / FRAMES_PER_VOLUME
parameters0["ds"]             = 1          # no downsampling
parameters0["res_x"]          = RES_X
parameters0["res_y"]          = RES_Y
parameters0["res_z"]          = RES_Z
parameters0["parallel_clean"] = False      # critical: avoids accumulator bug in step 5

voluseg.step0_process_parameters(parameters0)

# --- Load saved parameters ---
import json
filename_parameters = os.path.join(DIR_OUTPUT, "parameters.json")
parameters = voluseg.load_parameters(filename_parameters)

# --- Step 1–5 ---
log_step("Step 1: process volumes.")
voluseg.step1_process_volumes(parameters)

log_step("\nStep 2: align volumes.")
voluseg.step2_align_volumes(parameters)

log_step("\nStep 3: mask volumes.")
voluseg.step3_mask_volumes(parameters)

log_step("\nStep 4: detect cells.")
voluseg.step4_detect_cells(parameters)

log_step("\nStep 5: clean cells.")
voluseg.step5_clean_cells(parameters)

# --- Teardown ---
try:
    spark.stop()
except Exception:
    pass

gc.collect()
time.sleep(2)

print(f"\n✅ voluseg complete: {expt_ID}")
