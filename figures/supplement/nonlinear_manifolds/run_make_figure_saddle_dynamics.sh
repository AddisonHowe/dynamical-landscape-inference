#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

###############################################################################
##  Binary choice dynamics on a saddle with parameterization version 1

o1="res1"; k11=1;  k21=1; model1="model_phi1_4a_distortion_v1_0_v_mmd4_20250518_223001";
o2="res2"; k12=1;  k22=4; model2="model_phi1_4a_distortion_v1_1_v_mmd4_20250516_142619";
o3="res3"; k13=4;  k23=1; model3="model_phi1_4a_distortion_v1_3_v_mmd4_20250516_142619";
o4="res4"; k14=16; k24=1; model4="model_phi1_4a_distortion_v1_4_v_mmd4_20250516_143210";
o5="res5"; k15=1; k25=1.1; model5="model_phi1_4a_distortion_v1_5_v_mmd4_20250519_143807";
o6="res6"; k16=1.1; k26=1; model6="model_phi1_4a_distortion_v1_6_v_mmd4_20250518_223520";
o7="res7"; k17=1.1; k27=1.1; model7="model_phi1_4a_distortion_v1_7_v_mmd4_20250519_185958";
o8="res8"; k18=4.0; k28=4.0; model8="model_phi1_4a_distortion_v1_8_v_mmd4_20250519_185958";
o9="res9"; k19=1; k29=2; model9="model_phi1_4a_distortion_v1_9_v_mmd4_20250519_185958";
o10="res10"; k110=2; k210=1; model10="model_phi1_4a_distortion_v1_10_v_mmd4_20250519_185958";
o11="res11"; k111=2; k211=2; model11="model_phi1_4a_distortion_v1_11_v_mmd4_20250519_185958";

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o1 -k1 $k11 -k2 $k21 -m $model1 \
    -ul -2 2 -vl -2 2 -xl -3 3 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o2 -k1 $k12 -k2 $k22 -m $model2 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o3 -k1 $k13 -k2 $k23 -m $model3 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o4 -k1 $k14 -k2 $k24 -m $model4 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o5 -k1 $k15 -k2 $k25 -m $model5 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o6 -k1 $k16 -k2 $k26 -m $model6 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o7 -k1 $k17 -k2 $k27 -m $model7 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o8 -k1 $k18 -k2 $k28 -m $model8 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o9 -k1 $k19 -k2 $k29 -m $model9 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o10 -k1 $k110 -k2 $k210 -m $model10 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization1.py \
    -o $o11 -k1 $k111 -k2 $k211 -m $model11 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -3 3

###############################################################################
##  Binary choice dyamics on a saddle with parameterization version 2

o1="res1"; k11=1;  k21=1; model1="model_phi1_4a_distortion_v2_0_v_mmd4_20250518_223520";
o2="res2"; k12=1;  k22=4; model2="model_phi1_4a_distortion_v2_1_v_mmd4_20250516_150626";
o3="res3"; k13=4;  k23=1; model3="model_phi1_4a_distortion_v2_3_v_mmd4_20250516_155132";
o4="res4"; k14=16; k24=1; model4="model_phi1_4a_distortion_v2_4_v_mmd4_20250516_213103";
o5="res5"; k15=1; k25=1.1; model5="model_phi1_4a_distortion_v2_5_v_mmd4_20250519_143807";
o6="res6"; k16=1.1; k26=1; model6="model_phi1_4a_distortion_v2_6_v_mmd4_20250518_223520";
o7="res7"; k17=1.1; k27=1.1; model7="model_phi1_4a_distortion_v2_7_v_mmd4_20250519_185624";
o8="res8"; k18=4.0; k28=4.0; model8="model_phi1_4a_distortion_v2_8_v_mmd4_20250519_185624";
o9="res9"; k19=1; k29=2; model9="model_phi1_4a_distortion_v2_9_v_mmd4_20250519_185624";
o10="res10"; k110=2; k210=1; model10="model_phi1_4a_distortion_v2_10_v_mmd4_20250519_185624";
o11="res11"; k111=2; k211=2; model11="model_phi1_4a_distortion_v2_11_v_mmd4_20250519_185624";

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o1 -k1 $k11 -k2 $k21 -m $model1 \
    -ul -2 2 -vl -2 2 -xl -3 3 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o2 -k1 $k12 -k2 $k22 -m $model2 \
    -ul -2 2 -vl -2 2 -xl -3 3 -yl -3 3

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o3 -k1 $k13 -k2 $k23 -m $model3 \
    -ul -2 2 -vl -2 2 -xl -2.5 2.5 -yl -2.5 2.5

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o4 -k1 $k14 -k2 $k24 -m $model4 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o5 -k1 $k15 -k2 $k25 -m $model5 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o6 -k1 $k16 -k2 $k26 -m $model6 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o7 -k1 $k17 -k2 $k27 -m $model7 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o8 -k1 $k18 -k2 $k28 -m $model8 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o9 -k1 $k19 -k2 $k29 -m $model9 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o10 -k1 $k110 -k2 $k210 -m $model10 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o11 -k1 $k111 -k2 $k211 -m $model11 \
    -ul -2 2 -vl -2 2 -xl -2 2 -yl -2 2
    