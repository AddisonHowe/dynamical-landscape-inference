#!/bin/bash

source figures/.config

eval "$(conda shell.bash hook)"
conda activate env

modeldir=$SF_PHI1_1A_MODEL

python figures/supplement/make_figure_phi1_results.py \
    -i $modeldir \
    -o model_phi1 \
    --truesigma 0.1
