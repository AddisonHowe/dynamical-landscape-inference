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

os.makedirs(outdir, exist_ok=True)

SUBDIRS = ["testing", "training", "validation"]

def f(xs, k1, k2):
    x, y = xs[...,0], xs[...,1]
    t1 = np.arcsinh(np.sqrt(k1/2) * x)
    t2 = np.arcsinh(np.sqrt(k2/2) * y)
    u = t1 + t2
    v = t1 - t2
    ys = np.zeros_like(xs)
    ys[...,0] = u
    ys[...,1] = v
    return ys

for subdir in SUBDIRS:
    simdir_list = os.listdir(f"{datdir}/{subdir}")
    simdir_list = [d for d in simdir_list if d.startswith("sim")]
    
    for simdir in simdir_list:
        fpath = f"{datdir}/{subdir}/{simdir}/xs.npy"
        xs = np.load(fpath)
        ys = f(xs, k1, k2)
        np.save(fpath, ys)
