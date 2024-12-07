#!/bin/bash

source figures/.config

eval "$(conda shell.bash hook)"
conda activate env

modeldir=$FIG3_MODEL
echo modeldir: $modeldir
python figures/manuscript/make_fig3_synthetic_training.py -m $modeldir
