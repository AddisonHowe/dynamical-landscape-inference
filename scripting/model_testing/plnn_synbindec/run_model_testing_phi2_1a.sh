#!/bin/bash
#=============================================================================
#
# FILE: run_model_testing_phi2_1a.sh
#
# USAGE: run_model_testing_phi2_1a.sh
#
# DESCRIPTION: Run the model testing script on all model_phi2_1[abc] models 
#  using data_phi2_1a as testing data.
#
# EXAMPLE: sh run_model_testing_phi2_1a.sh
#=============================================================================

datdirbase=data/training_data
datdir=data_phi2_1a

key_list="test"
dt0_list="0.025 0.0125 -1"

basedir=data/trained_models/plnn_synbindec
modeldirs=$(ls -d $basedir/model_phi2_1*)

for modeldir in ${modeldirs}; do
    modelname=$(basename $modeldir)
    echo $modelname
    sh scripting/model_testing/run_model_testing.sh $basedir $modelname \
        $datdirbase $datdir --key_list $key_list --dt0_list $dt0_list \
        --nresamp 1 --nreps 10 --batch_size 20
    printf "Completed.\n\n"
done