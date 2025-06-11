#!/bin/bash
#=============================================================================
#
# FILE: run_model_testing.sh
#
# USAGE: run_model_testing.sh <basedir> <modelname> <datdirbase> <datdir> \
#        [--key_list key1 [key2 ...]] [--dt0_list dt01 [dt02 ...]] \
#        [--nresamp [nresamp]] [--nreps [nreps]] [--batch_size [batch_size]]
#
# DESCRIPTION: Run the plnn model evaluation script on a trained model, 
#  located at <basedir>/<modelname>. The model is applied to every datapoint 
#  in the data directory specified by <datdir> and [key_list], where key_list 
#  is a list of subdirectories (e.g. train, valid, test) of the directory 
#  <datdirbase>/<datdir>. The model is applied to each subdirectory specified, 
#  using each value of dt0_list for the model's internal timestepping parameter 
#  `dt0`. If not specified, the [dt0_list] defaults to -1, which indicates that 
#  the model should use its internal value already specified.
#
# EXAMPLE: sh run_model_testing.sh data/trained_models/plnn_synbindec \
#                                  model_phi1_1a_v_mmd1 \
#                                  data/training_data/basic \
#                                  data_phi1_1a \
#                                  --key_list test train --dt0_list -1 0.1 \
#                                  --nsamp 20 --nreps 10 --batch_size 5
#=============================================================================

# Check if there are at least four positional arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <basedir> <modelname> <datdirbase> <datdir> " \
                    "[--key_list key1 [key2 ...]] [--dt0_list dt01 [dt02 ...]]"
    exit 1
fi

# Assign positional arguments
basedir=$1
modelname=$2
datdirbase=$3
datdir=$4
shift 4

# Initialize arrays for key_list and dt0_list
key_list=()
dt0_list=()

# Default values for nsamps, nreps, and batch_size
nresamp=1
nreps=1
batch_size=1

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
        --nresamp)
            nresamp="$2"
            shift 2
            ;;
        --nreps)
            nreps="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
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
echo "datdirbase: $datdirbase"
echo "datdir: $datdir"
echo "key_list values: ${key_list[@]}"
echo "dt0_list values: ${dt0_list[@]}"
echo "nresamp: $nresamp"
echo "nreps: $nreps"
echo "batch_size: $batch_size"

# Specify the output directory
outdir=${basedir}/${modelname}/testing/eval_$datdir
mkdir -p $outdir

# Main loop
for key in ${key_list[@]}; do
    for dt0 in ${dt0_list[@]}; do
        model_eval --dataset $key --dt0 $dt0 \
            --nresamp $nresamp --nreps $nreps --batch_size $batch_size \
            --modeldir $modelname \
            --datdirbase $datdirbase \
            --basedir $basedir \
            --datdir $datdir \
            --outdir $outdir \
            --nosuboutdir \
            --seed 42
    done
done
echo Done!
