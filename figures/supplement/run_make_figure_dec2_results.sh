#!/bin/bash

source figures/.config

eval "$(conda shell.bash hook)"
conda activate env

modeldir=$DEC2_MODEL

python figures/supplement/make_figure_facs_results.py \
    -i $modeldir \
    --signal 1.0 0.9 \
    -o model_dec2

python figures/supplement/make_plot_facs_tilted.py \
    -i $modeldir \
    --signal 0.0 0.9 \
    -o model_dec2

python figures/supplement/make_plot_facs_tilted.py \
    -i $modeldir \
    --signal 0.0 0.0 \
    -o model_dec2

python figures/supplement/make_plot_facs_tilted.py \
    -i $modeldir \
    --signal 1.0 0.0 \
    -o model_dec2
