#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env


###############################################################################
##  Overview using saddle manifold set 1

o1="res1"; k11=1;  k21=4;  model1="model_phi1_4a_distortion_v1_1_v_mmd1_20250430_112623";
# o2="res2"; k12=1;  k22=16; model2="model_phi1_4a_distortion_v1_2_v_mmd1_20250430_112623";
# o3="res3"; k13=4;  k23=1;  model3="model_phi1_4a_distortion_v1_3_v_mmd1_20250430_112623";
# o4="res4"; k14=16; k24=1;  model4="model_phi1_4a_distortion_v1_4_v_mmd1_20250430_112623";

python figures/supplement/nonlinear_manifolds/make_figure_nonlinear_manifolds_overview.py \
    -o $o1 -k1 $k11 -k2 $k21 -m $model1 \
    -ul -2 2 -vl -2 2 -xl -3 3 -yl -3 3

# python figures/supplement/nonlinear_manifolds/make_figure_nonlinear_manifolds_overview.py \
#     -o $o2 -k1 $k12 -k2 $k22 -m $model2

# python figures/supplement/nonlinear_manifolds/make_figure_nonlinear_manifolds_overview.py \
#     -o $o3 -k1 $k13 -k2 $k23 -m $model3

# python figures/supplement/nonlinear_manifolds/make_figure_nonlinear_manifolds_overview.py \
#     -o $o4 -k1 $k14 -k2 $k24 -m $model4


###############################################################################
##  Binary choice dynamics on a saddle with parameterization version 1

o1="res1"; k11=1;  k21=1; model1="model_phi1_4a_distortion_v1_0_v_mmd4_20250518_223001";
o2="res2"; k12=1;  k22=4; model2="model_phi1_4a_distortion_v1_1_v_mmd4_20250516_142619";
o3="res3"; k13=4;  k23=1; model3="model_phi1_4a_distortion_v1_3_v_mmd4_20250516_142619";
o4="res4"; k14=16; k24=1; model4="model_phi1_4a_distortion_v1_4_v_mmd4_20250516_143210";
o5="res5"; k15=1; k25=1.1; model5="model_phi1_4a_distortion_v1_5_v_mmd4_20250518_223520";
o6="res6"; k16=1.1; k26=1; model6="model_phi1_4a_distortion_v1_6_v_mmd4_20250518_223520";

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

###############################################################################
##  Binary choice dyamics on a saddle with parameterization version 2

o1="res1"; k11=1;  k21=1; model1="model_phi1_4a_distortion_v2_0_v_mmd4_20250518_223520";
o2="res2"; k12=1;  k22=4; model2="model_phi1_4a_distortion_v2_1_v_mmd4_20250516_150626";
o3="res3"; k13=4;  k23=1; model3="model_phi1_4a_distortion_v2_3_v_mmd4_20250516_155132";
o4="res4"; k14=16; k24=1; model4="model_phi1_4a_distortion_v2_4_v_mmd4_20250516_213103";
o5="res5"; k15=1; k25=1.1; model5="model_phi1_4a_distortion_v2_5_v_mmd4_20250518_223520";
o6="res6"; k16=1.1; k26=1; model6="model_phi1_4a_distortion_v2_6_v_mmd4_20250518_223520";

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o1 -k1 $k11 -k2 $k21 -m $model1 \
    -ul -2 2 -vl -2 2 -xl -5 5 -yl -5 5

python figures/supplement/nonlinear_manifolds/make_figure_saddle_bcd_parameterization2.py \
    -o $o2 -k1 $k12 -k2 $k22 -m $model2 \
    -ul -2 2 -vl -2 2 -xl -5 5 -yl -5 5

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
