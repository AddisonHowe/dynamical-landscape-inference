#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

###############################################################################
##  Overview using saddle manifold set 1

o1="res1"; k11=1;  k21=4;  model1="model_phi1_4a_distortion_v1_1_v_mmd1_20250430_112623";

python figures/supplement/nonlinear_manifolds/make_figure_nonlinear_manifolds_overview.py \
    -o $o1 -k1 $k11 -k2 $k21 -m $model1 \
    -ul -2 2 -vl -2 2 -xl -3 3 -yl -3 3
