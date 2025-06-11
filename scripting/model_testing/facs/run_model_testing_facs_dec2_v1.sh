#!/bin/bash
#=============================================================================
#
# FILE: run_model_testing_facs_dec2_v1.sh
#
# USAGE: run_model_testing_facs_dec2_v1.sh
#
# DESCRIPTION: Run the model testing script on all model_facs_dec2_v1 models 
#  against testing data.
#
# EXAMPLE: sh run_model_testing_facs_dec2_v1.sh
#=============================================================================

datdirbase=data/training_data/basic/facs
datdir=facs_dec2_v1

key_list="test train valid"
dt0_list="0.025 0.0125 -1"

basedir=data/trained_models/facs
modeldirs=$(ls -d $basedir/model_facs_dec2_v1_*)

for modeldir in ${modeldirs}; do
    modelname=$(basename $modeldir)
    echo $modelname
    sh scripting/model_testing/run_model_testing.sh $basedir $modelname \
        $datdirbase $datdir --key_list $key_list --dt0_list $dt0_list \
        --nresamp 100 --nreps 20 --batch_size 20
    printf "Completed.\n\n"
done
