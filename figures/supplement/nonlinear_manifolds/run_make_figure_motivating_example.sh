#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

###############################################################################
##  Overview using saddle manifold set 1

k1=1
k2=4
model="model_phi1_4a_distortion_v1_1_v_mmd1_20250430_112623"

python figures/supplement/nonlinear_manifolds/make_figure_motivating_example.py \
    -k1 $k1 -k2 $k2 -m $model \
    -ul -2 2 \
    -vl -2 2 \
    -xl -3 3 \
    -yl -3 3
