#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

###############################################################################
##  Binary choice dynamics on a saddle with parameterization version 1

k1=1
k2=2
model="model_phi1_4a_distortion_v1_9_v_mmd4_20250519_185958"

python figures/supplement/nonlinear_manifolds/make_figure_latent_geometry.py \
    -k1 $k1 -k2 $k2 -m $model \
    -ul -2 2 \
    -vl -2 2 \
    -xl -3 3 \
    -yl -3 3
