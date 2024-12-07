#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

python figures/supplement/make_figure_binary_choice.py
