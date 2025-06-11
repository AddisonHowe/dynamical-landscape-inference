#!/usr/bin/env bash
#=============================================================================
#
# FILE: run_all_compute_distances.sh
#
# USAGE: run_all_compute_distances.sh
#
# DESCRIPTION: Run the compute_distances.py script on all facs datasets in 
# data/training_data/facs/ directory.
#
# EXAMPLE: sh run_all_compute_distances.sh
#=============================================================================

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
#     python scripting/training_data/compute_distances.py -d data/training_data/facs/$d -n 800
# done

# datdirs=(
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_0"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_1"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_2"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_3"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_4"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_5"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_6"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_7"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_8"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_9"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_10"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1_11"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_0"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_1"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_2"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_3"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_4"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_5"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_6"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_7"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_8"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_9"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_10"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2_11"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1r_1"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1r_2"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1r_3"
    # "distortions/paraboloids/data_phi1_4a_distortion_v1r_4"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2r_1"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2r_2"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2r_3"
    # "distortions/paraboloids/data_phi1_4a_distortion_v2r_4"
    # "distortions/paraboloids/data_phi1_1a_distortion_v1r_1"
    # "distortions/paraboloids/data_phi1_1a_distortion_v1r_2"
    # "distortions/paraboloids/data_phi1_1a_distortion_v1r_3"
    # "distortions/paraboloids/data_phi1_1a_distortion_v1r_4"
    # "distortions/paraboloids/data_phi1_1a_distortion_v2r_1"
    # "distortions/paraboloids/data_phi1_1a_distortion_v2r_2"
    # "distortions/paraboloids/data_phi1_1a_distortion_v2r_3"
    # "distortions/paraboloids/data_phi1_1a_distortion_v2r_4"
# )

datdirs=(
    # saddle_v1/data_phi1_1a_saddle_v1a_0
    # saddle_v1/data_phi1_1a_saddle_v1a_1
    # saddle_v1/data_phi1_1a_saddle_v1a_3
    # saddle_v1/data_phi1_1a_saddle_v1b_0
    # saddle_v1/data_phi1_1a_saddle_v1b_1
    # saddle_v1/data_phi1_1a_saddle_v1b_3
    # saddle_v1/data_phi1_1a_saddle_v1c_0
    # saddle_v1/data_phi1_1a_saddle_v1c_1
    # saddle_v1/data_phi1_1a_saddle_v1c_3
    # saddle_v1/data_phi1_1a_saddle_v1d_0
    # saddle_v1/data_phi1_1a_saddle_v1d_1
    saddle_v1/data_phi1_1a_saddle_v1d_3
)
for d in "${datdirs[@]}"; do
    echo $d
    python scripting/training_data/compute_distances.py \
        -d data/training_data/$d -n 0
done
