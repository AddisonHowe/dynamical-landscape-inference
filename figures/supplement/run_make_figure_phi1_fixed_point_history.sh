#!/bin/bash

source figures/.config

eval "$(conda shell.bash hook)"
conda activate env

modeldir=$SF_PHI1_1A_MODEL

python figures/supplement/make_figure_phi_fixed_point_history.py \
    -i $modeldir \
    -o model_phi1 \
    -p phi1 \
    --truesigma 0.1 \
    -s 0 0.1
