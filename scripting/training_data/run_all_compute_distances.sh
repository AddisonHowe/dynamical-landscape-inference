#!/bin/bash
#=============================================================================
#
# FILE: run_all_compute_distances.sh
#
# USAGE: run_all_compute_distances.sh
#
# DESCRIPTION: Run the compute_distances.py script on all facs datasets in 
# data/facs/ directory.
#
# EXAMPLE: sh run_all_compute_distances.sh
#=============================================================================

eval "$(conda shell.bash hook)"
conda activate env

# datdirs=(
#     "facs_dec1_v1"
#     "facs_dec1_v2"
#     "facs_dec1_v3"
#     "facs_dec1_v4"
#     "facs_dec2_v1"
#     "facs_dec2_v2"
#     "facs_dec2_v3"
# )

# for d in "${datdirs[@]}"; do
#     echo $d
#     python scripting/training_data/compute_distances.py -d data/facs/$d -n 800
# done

datdirs=(
    "data_phi1_4a_distortion_v1_0"
    # "data_phi1_4a_distortion_v1_1"
    # "data_phi1_4a_distortion_v1_2"
    # "data_phi1_4a_distortion_v1_3"
    # "data_phi1_4a_distortion_v1_4"
    "data_phi1_4a_distortion_v1_5"
    "data_phi1_4a_distortion_v1_6"
    "data_phi1_4a_distortion_v2_0"
    # "data_phi1_4a_distortion_v2_1"
    # "data_phi1_4a_distortion_v2_2"
    # "data_phi1_4a_distortion_v2_3"
    # "data_phi1_4a_distortion_v2_4"
    "data_phi1_4a_distortion_v2_5"
    "data_phi1_4a_distortion_v2_6"
)

for d in "${datdirs[@]}"; do
    echo $d
    python scripting/training_data/compute_distances.py \
        -d data/training_data/distortions/paraboloids/$d -n 0
done
