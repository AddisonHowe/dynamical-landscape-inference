#!/usr/bin/env bash

indir=data_phi1_3a
datdir_orig="data/training_data/basic"
outdirbase="data/training_data/transforms/basic_3d_v1"
pyscript=data/training_data/transforms/transform_data.py
transform=transform1

mkdir -p ${outdirbase}
cp -r ${datdir_orig}/${indir} ${outdirbase}/

for k in testing training validation; do
    subdirs=$(ls ${outdirbase}/${indir}/${k})
    for sd in ${subdirs[@]}; do
        dpath=${outdirbase}/${indir}/${k}/${sd}
        opath=${outdirbase}/${indir}/${k}/${sd}
        if [[ -d $dpath ]]; then
            echo $sd
            python $pyscript -i ${dpath}/xs.npy -o ${opath}/xs.npy -t $transform
        fi
    done
done
