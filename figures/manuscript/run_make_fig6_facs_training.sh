#!/bin/bash
#=============================================================================
#
# FILE: run_make_fig6_facs_training.sh
#
# USAGE: run_make_fig6_facs_training.sh
#
# DESCRIPTION: Pulls images from the mESC FACS Data project and runs figure-
# generation script `make_fig6_facs_training.py`. Sources the file `.env` that
# should set an environment variable $MESC_PROJ_PATH pointing to the location
# of the mESC FACS Data project.
#
# EXAMPLE: sh run_make_fig6_facs_training.sh
#=============================================================================

source .env

sh figures/manuscript/pull_fig6_facs_training.sh $MESC_PROJ_PATH

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig6_facs_training.py
