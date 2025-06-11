#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

###############################################################################
##  Transformation plots

argsets=(
    "-e embedding1 -k1 1   -k2 1"
    "-e embedding1 -k1 1   -k2 1.1"
    "-e embedding1 -k1 1.1 -k2 1"
    "-e embedding1 -k1 1.1 -k2 1.1"
    "-e embedding1 -k1 1   -k2 2"
    "-e embedding1 -k1 2   -k2 1"
    "-e embedding1 -k1 2   -k2 2"
    "-e embedding2 -k1 1   -k2 1"
    "-e embedding2 -k1 1   -k2 1.1"
    "-e embedding2 -k1 1.1 -k2 1"
    "-e embedding2 -k1 1.1 -k2 1.1"
    "-e embedding2 -k1 1   -k2 2"
    "-e embedding2 -k1 2   -k2 1"
    "-e embedding2 -k1 2   -k2 2"
    "-e embedding3 -k1 1   -k2 1"
    "-e embedding3 -k1 1   -k2 1.1"
    "-e embedding3 -k1 1.1 -k2 1"
    "-e embedding3 -k1 1.1 -k2 1.1"
    "-e embedding3 -k1 1   -k2 2"
    "-e embedding3 -k1 2   -k2 1"
    "-e embedding3 -k1 2   -k2 2"
)

script=figures/supplement/nonlinear_manifolds/make_figure_transformations_analysis.py
for argstr in "${argsets[@]}"; do
    echo Args: "$argstr"
    python $script $argstr
    python $script $argstr -p phi1
done
