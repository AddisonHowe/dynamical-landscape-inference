#!/bin/bash
#=============================================================================
#
# FILE: gen_data_distorted.sh
#
# USAGE: gen_data_distorted.sh
#
# DESCRIPTION: Apply a distortion to a training data file.
#
# EXAMPLE: sh gen_data_distorted.sh
#=============================================================================


datdir=data/training_data
outdir=data/training_data/distortions/paraboloids

function copy_and_distort_data() {
    indata=$1
    name=$2
    k1=$3
    k2=$4
    i=${datdir}/${indata}
    o=${outdir}/${indata}_${name}
    mkdir -p $o
    cp -r ${i}/* $o
    printf "kappa1\t${k1}\nkappa2\t${k2}\n" > ${o}/distortion_parameters.tsv
    python scripting/training_data/distort_data.py -i $o -o $o \
        -k1 "${k1}" -k2 "${k2}"
}

# Generate data: data_phi1_4a_distortion1
indata=data_phi1_4a
name=distortion1
k1=1.0
k2=4.0
copy_and_distort_data $indata $name $k1 $k2

# Generate data: data_phi1_4a_distortion2
indata=data_phi1_4a
name=distortion2
k1=1.0
k2=16.0
copy_and_distort_data $indata $name $k1 $k2

# Generate data: data_phi1_4a_distortion3
indata=data_phi1_4a
name=distortion3
k1=4.0
k2=1.0
copy_and_distort_data $indata $name $k1 $k2

# Generate data: data_phi1_4a_distortion4
indata=data_phi1_4a
name=distortion4
k1=16.0
k2=1.0
copy_and_distort_data $indata $name $k1 $k2
