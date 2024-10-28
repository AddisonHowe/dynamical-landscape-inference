#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig0_visual_abstract.py
