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
source figures/.config

sh figures/manuscript/pull_fig7_facs_evaluation.sh \
    $MESC_PROJ_PATH $FIG67_PCA_SUBDIR $FIG67_SIGS_SUBDIR

eval "$(conda shell.bash hook)"
conda activate env

MODELDIR=$FIG67_MODEL
python figures/manuscript/make_fig7_facs_evaluation.py -m $MODELDIR
python figures/manuscript/make_fig7_facs_evaluation_2.py -m $MODELDIR
