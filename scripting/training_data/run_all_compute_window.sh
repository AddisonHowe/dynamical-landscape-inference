#!/bin/bash
#=============================================================================
#
# FILE: run_all_compute_window.sh
#
# USAGE: run_all_compute_window.sh
#
# DESCRIPTION: Run the compute_window.py script on all facs datasets in 
# data/facs/ directory.
#
# EXAMPLE: sh run_all_compute_window.sh
#=============================================================================

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

datdirs=(
    "facs_dec1_v1"
    "facs_dec1_v2"
    "facs_dec1_v3"
    "facs_dec1_v4"
    "facs_dec2_v1"
    "facs_dec2_v2"
    "facs_dec2_v3"
)

for d in "${datdirs[@]}"; do
    echo $d
    python scripting/training_data/compute_window.py -d data/facs/$d
done
