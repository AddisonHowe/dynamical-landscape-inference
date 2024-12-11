#!/bin/bash

source .env
source figures/.config

sh figures/supplement/pull_figure_dec2_dimred.sh \
    $MESC_PROJ_PATH $DEC2_DIMRED_SUBDIR $DEC2_SIGS_SUBDIR

eval "$(conda shell.bash hook)"
conda activate env
