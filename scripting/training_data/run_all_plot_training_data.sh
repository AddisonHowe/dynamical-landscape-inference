#!/usr/bin/env bash
#=============================================================================
#
# FILE: run_all_plot_training_data.sh
#
# USAGE: run_all_plot_training_data.sh
#
# DESCRIPTION: Run the plot_training_data.py script on all datasets.
#
# EXAMPLE: sh scripting/run_all_plot_training_data.sh
#=============================================================================

datdirs=(
    saddle_v1/data_phi1_1a_saddle_v1a_0
    saddle_v1/data_phi1_1a_saddle_v1a_1
    saddle_v1/data_phi1_1a_saddle_v1a_3
    saddle_v1/data_phi1_1a_saddle_v1b_0
    saddle_v1/data_phi1_1a_saddle_v1b_1
    saddle_v1/data_phi1_1a_saddle_v1b_3
    saddle_v1/data_phi1_1a_saddle_v1c_0
    saddle_v1/data_phi1_1a_saddle_v1c_1
    saddle_v1/data_phi1_1a_saddle_v1c_3
    saddle_v1/data_phi1_1a_saddle_v1d_0
    saddle_v1/data_phi1_1a_saddle_v1d_1
    saddle_v1/data_phi1_1a_saddle_v1d_3
)

for d in "${datdirs[@]}"; do
    echo "----------------------------------------------------------"
    echo "-----" $d
    echo "----------------------------------------------------------"
    datdir="data/training_data/$d"
    python scripting/training_data/plot_training_data.py \
        -d ${datdir} -o ${datdir}/images
done
