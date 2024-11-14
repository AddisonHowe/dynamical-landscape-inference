#!/bin/bash
#=============================================================================
#
# FILE: run_model_testing.sh
#
# USAGE: run_model_testing.sh <basedir> <modelname> <datdir> \
#        [--key_list key1 [key2 ...]] [--dt0_list dt01 [dt02 ...]]
#
# DESCRIPTION: Run the plnn model evaluation script on a trained model, 
#  located at <basedir>/<modelname>. The model is applied to every datapoint 
#  in the data directory specified by <datdir> and [key_list], where key_list 
#  is a list of subdirectories (e.g. train, valid, test) of the directory 
#  <datdir>. The model is applied to each subdirectory specified, using each 
#  value of dt0_list for the model's internal timestepping parameter `dt0`. If 
#  not specified, the [dt0_list] defaults to -1, which indicates that the model 
#  should use its internal value already specified.
#
# EXAMPLE: sh run_model_testing.sh data/trained_models/plnn_synbindec \
#                                  model_phi1_1a_v_mmd1 data_phi1_1a \
#                                  --key_list test train --dt0_list -1 0.1
#=============================================================================

# Check if there are at least two positional arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <modelname> <datdir> [--key_list key1 [key2 ...]] " \
                                        "[--dt0_list dt01 [dt02 ...]]"
    exit 1
fi

# Assign positional arguments
basedir=$1
modelname=$2
datdir=$3
shift 3

# Initialize arrays for key_list and dt0_list
key_list=()
dt0_list=()

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --key_list)
            shift
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                key_list+=("$1")
                shift
            done
            ;;
        --dt0_list)
            shift
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                dt0_list+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Assign default values for empty key_list or dt0_list
if [ ${#key_list[@]} -eq 0 ]; then
    key_list=("test")
fi
if [ ${#dt0_list[@]} -eq 0 ]; then
    dt0_list=("-1")
fi

# Echo arguments
echo "basedir: $basedir"
echo "modelname: $modelname"
echo "datdir: $datdir"
echo "key_list values: ${key_list[@]}"
echo "dt0_list values: ${dt0_list[@]}"

# Specify the output directory
outdir=${basedir}/${modelname}/testing/eval_$datdir
mkdir -p $outdir

# Main loop
for key in $key_list; do
    for dt0 in $dt0_list; do
        model_eval --dataset $key --dt0 $dt0 \
            --nresamp 1 --nreps 10 --batch_size 20 \
            --modeldir $modelname \
            --basedir $basedir \
            --datdirbase data/training_data \
            --datdir $datdir \
            --outdir $outdir \
            --nosuboutdir \
            --seed 42
    done
done
echo Done!
