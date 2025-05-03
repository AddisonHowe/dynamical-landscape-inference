#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

o1="res1"; k11=1;  k21=4;  model1="model_phi1_4a_distortion1_v_mmd1_20250430_112623";
o2="res2"; k12=1;  k22=16; model2="model_phi1_4a_distortion2_v_mmd1_20250430_112623";
o3="res3"; k13=4;  k23=1;  model3="model_phi1_4a_distortion3_v_mmd1_20250430_112623";
o4="res4"; k14=16; k24=1;  model4="model_phi1_4a_distortion4_v_mmd1_20250430_112623";


python figures/manuscript/make_fig8_saddle_landscapes.py \
    -o $o1 -k1 $k11 -k2 $k21 -m $model1

# python figures/manuscript/make_fig8_saddle_landscapes.py \
#     -o $o2 -k1 $k12 -k2 $k22 -m $model2

# python figures/manuscript/make_fig8_saddle_landscapes.py \
#     -o $o3 -k1 $k13 -k2 $k23 -m $model3

# python figures/manuscript/make_fig8_saddle_landscapes.py \
#     -o $o4 -k1 $k14 -k2 $k24 -m $model4
