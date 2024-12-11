#!/bin/bash

source .env
source figures/.config

sh figures/supplement/pull_figure_dec1_dimred.sh \
    $MESC_PROJ_PATH $DEC1_DIMRED_SUBDIR $DEC1_SIGS_SUBDIR

eval "$(conda shell.bash hook)"
conda activate env
