"""Script to apply a parabolic distortion to input data.

Usage:
    python distort_data.py -i <indir> -o <outdir> -k1 <kappa1> -k2 <kappa2>
"""

# python scripting/training_data/distort_data.py \
#     -i ${datdir}/${indata} -o ${outdir}/${indata}_${name} \
#     -k1 "${k1}" -k2 "${k2}"

import argparse
import os
import numpy as np
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', type=str, required=True)
parser.add_argument('-o', '--outdir', type=str, required=True)
parser.add_argument('-k1', '--kappa1', type=float, required=True)
parser.add_argument('-k2', '--kappa2', type=float, required=True)
args = parser.parse_args()

datdir = args.indir
outdir = args.outdir
k1 = args.kappa1
k2 = args.kappa2

print(f"Distorting input data at path '{datdir}'")
print(f"kappa1: {k1:.5g}")
print(f"kappa2: {k2:.5g}")
print(f"Saving data to output directory '{outdir}'")

# Copy input directory
shutil.copytree(datdir, outdir, dirs_exist_ok=True)

SUBDIRS = ["testing", "training", "validation"]

def f(uvs, k1, k2):
    u, v = uvs[...,0], uvs[...,1]
    x = np.sqrt(2 / k1) * np.sinh((u + v) / 2)
    y = np.sqrt(2 / k2) * np.sinh((u - v) / 2)
    xys = np.zeros_like(uvs)
    xys[...,0] = x
    xys[...,1] = y
    return xys


for subdir in SUBDIRS:
    simdir_list = os.listdir(f"{outdir}/{subdir}")
    simdir_list = [d for d in simdir_list if d.startswith("sim")]
    
    for simdir in simdir_list:
        fpath = f"{outdir}/{subdir}/{simdir}/xs.npy"
        xs = np.load(fpath)
        ys = f(xs, k1, k2)
        np.save(fpath, ys)
