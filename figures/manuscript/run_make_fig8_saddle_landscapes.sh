#!/bin/bash

eval "$($CONDA_EXE shell.bash hook)"
conda activate env

python figures/manuscript/make_fig8_saddle_landscapes.py
