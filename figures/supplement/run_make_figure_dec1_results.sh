#!/bin/bash

source figures/.config

eval "$(conda shell.bash hook)"
conda activate env

modeldir=$DEC1_MODEL

python figures/supplement/make_figure_facs_results.py \
    -i $modeldir \
    --signal 0.0 1.0 \
    -o model_dec1
