#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

sh figures/supplement/nonlinear_manifolds/run_make_figure_overview.sh
sh figures/supplement/nonlinear_manifolds/run_make_figure_saddle_dynamics.sh
sh figures/supplement/nonlinear_manifolds/run_make_figure_transformations_analysis.sh
