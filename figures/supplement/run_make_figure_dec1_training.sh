#!/bin/bash

source figures/.config

eval "$(conda shell.bash hook)"
conda activate env

modeldir=$DEC1_MODEL

python figures/supplement/make_figure_facs_training.py \
    -i $modeldir \
    -o model_dec1
