#!/bin/bash

source figures/.config

eval "$(conda shell.bash hook)"
conda activate env

modeldir=$SF_PHI2_1A_MODEL

python figures/supplement/make_figure_phi_training.py \
    -i $modeldir \
    -o model_phi2 \
    --truesigma 0.3 #--no-logloss --startidx 50
