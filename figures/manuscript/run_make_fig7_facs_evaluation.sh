#!/bin/bash
#=============================================================================
#
# FILE: run_make_fig7_facs_evaluation.sh
#
# USAGE: run_make_fig7_facs_evaluation.sh
#
# DESCRIPTION: Pulls images from the mESC FACS Data project and runs figure-
# generation scripts `make_fig7_facs_evaluation.py` and 
# `make_fig7_facs_evaluation_2`. Sources the file `.env` that should set an
# environment variable $MESC_PROJ_PATH pointing to the location of the mESC
# FACS Data project.
#
# EXAMPLE: sh run_make_fig7_facs_evaluation.sh
#=============================================================================

source .env

sh figures/manuscript/pull_fig7_facs_evaluation.sh $MESC_PROJ_PATH

eval "$(conda shell.bash hook)"
conda activate env

python figures/manuscript/make_fig7_facs_evaluation.py
python figures/manuscript/make_fig7_facs_evaluation_2.py
