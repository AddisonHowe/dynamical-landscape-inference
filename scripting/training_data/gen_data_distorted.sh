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
    transform=$5
    theta=$6
    du=$7
    dv=$8
    : "${theta:=0}"
    : "${du:=0}"
    : "${dv:=0}"
    i=${datdir}/${indata}
    o=${outdir}/${indata}_${name}
    mkdir -p $o
    cp -r ${i}/* $o
    printf "kappa1\t${k1}\nkappa2\t${k2}\n" > ${o}/distortion_parameters.tsv
    python scripting/training_data/distort_data.py -i $o -o $o \
        -k1 "${k1}" -k2 "${k2}" -t ${transform} \
        --theta "${theta}" --du "${du}" --dv "${dv}"
}

# Generate data: data_phi1_4a_distortion_v1_0
indata=data_phi1_4a
name=distortion_v1_0
k1=1.0
k2=1.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_1
indata=data_phi1_4a
name=distortion_v1_1
k1=1.0
k2=4.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_2
indata=data_phi1_4a
name=distortion_v1_2
k1=1.0
k2=16.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_3
indata=data_phi1_4a
name=distortion_v1_3
k1=4.0
k2=1.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_4
indata=data_phi1_4a
name=distortion_v1_4
k1=16.0
k2=1.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_5
indata=data_phi1_4a
name=distortion_v1_5
k1=1.0
k2=1.1
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_6
indata=data_phi1_4a
name=distortion_v1_6
k1=1.1
k2=1.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_7
indata=data_phi1_4a
name=distortion_v1_7
k1=1.1
k2=1.1
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_8
indata=data_phi1_4a
name=distortion_v1_8
k1=4.0
k2=4.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_9
indata=data_phi1_4a
name=distortion_v1_9
k1=1.0
k2=2.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_10
indata=data_phi1_4a
name=distortion_v1_10
k1=2.0
k2=1.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v1_11
indata=data_phi1_4a
name=distortion_v1_11
k1=2.0
k2=2.0
transform=v1
copy_and_distort_data $indata $name $k1 $k2 $transform

#############################################################################
#############################################################################

# Generate data: data_phi1_4a_distortion_v2_0
indata=data_phi1_4a
name=distortion_v2_0
k1=1.0
k2=1.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_1
indata=data_phi1_4a
name=distortion_v2_1
k1=1.0
k2=4.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_2
indata=data_phi1_4a
name=distortion_v2_2
k1=1.0
k2=16.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_3
indata=data_phi1_4a
name=distortion_v2_3
k1=4.0
k2=1.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_4
indata=data_phi1_4a
name=distortion_v2_4
k1=16.0
k2=1.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_5
indata=data_phi1_4a
name=distortion_v2_5
k1=1.0
k2=1.1
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_6
indata=data_phi1_4a
name=distortion_v2_6
k1=1.1
k2=1.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_7
indata=data_phi1_4a
name=distortion_v2_7
k1=1.1
k2=1.1
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_8
indata=data_phi1_4a
name=distortion_v2_8
k1=4.0
k2=4.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_9
indata=data_phi1_4a
name=distortion_v2_9
k1=1.0
k2=2.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_10
indata=data_phi1_4a
name=distortion_v2_10
k1=2.0
k2=1.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform

# Generate data: data_phi1_4a_distortion_v2_11
indata=data_phi1_4a
name=distortion_v2_11
k1=2.0
k2=2.0
transform=v2
copy_and_distort_data $indata $name $k1 $k2 $transform


#############################################################################
#############################################################################

# Generate data: data_phi1_4a_distortion_v1r_1
indata=data_phi1_4a
name=distortion_v1r_1
k1=1.0
k2=1.0
transform=v1_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv

# Generate data: data_phi1_4a_distortion_v1r_2
indata=data_phi1_4a
name=distortion_v1r_1
k1=1.0
k2=2.0
transform=v1_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv

# Generate data: data_phi1_4a_distortion_v1r_3
indata=data_phi1_4a
name=distortion_v1r_1
k1=2.0
k2=1.0
transform=v1_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv

# Generate data: data_phi1_4a_distortion_v1r_4
indata=data_phi1_4a
name=distortion_v1r_1
k1=2.0
k2=2.0
transform=v1_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv


# Generate data: data_phi1_4a_distortion_v2r_1
indata=data_phi1_4a
name=distortion_v2r_1
k1=1.0
k2=1.0
transform=v2_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv

# Generate data: data_phi1_4a_distortion_v2r_2
indata=data_phi1_4a
name=distortion_v2r_1
k1=1.0
k2=2.0
transform=v2_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv

# Generate data: data_phi1_4a_distortion_v2r_3
indata=data_phi1_4a
name=distortion_v2r_1
k1=2.0
k2=1.0
transform=v2_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv

# Generate data: data_phi1_4a_distortion_v2r_4
indata=data_phi1_4a
name=distortion_v2r_1
k1=2.0
k2=2.0
transform=v2_rotation
theta="-0.7853981633974483"
du=1
dv=1
copy_and_distort_data $indata $name $k1 $k2 $transform $theta $du $dv
