#!/bin/bash
#=============================================================================
#
# FILE: echo_training_cmd_from_runargs.sh
#
# USAGE: echo_training_cmd_from_runargs.sh [argfile.tsv]
#
# DESCRIPTION: Construct and echo a python command to train a model, using the
# parameter values listed in the given tsv file.
#=============================================================================

tsv_file="$1"

boolean_args=(
    "confine"
    "infer_metric"
    "plot"
    "timestamp"
    "phi_layer_normalize"
    "tilt_layer_normalize"
    "metric_layer_normalize"
    "save_all"
    "enforce_gpu"
    "fix_noise"
    "reduce_dt_on_nan"
    "reduce_cf_on_nan"
    "model_do_sample"
)

cmd="python -m plnn.main "
x=$(awk -F "\t+" 'NR>0 {if (substr($1,1,1)!="#") print "--"$1" "$2}' < "$tsv_file")
cmd="$cmd $x"

# replace boolean flag arguments
for boolarg in ${boolean_args[@]}; do
  cmd=$(echo $cmd | sed "s/--$boolarg False//g")
  cmd=$(echo $cmd | sed "s/--$boolarg True/--$boolarg/g")
done

echo $cmd
