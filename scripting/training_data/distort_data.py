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
parser.add_argument('-t', '--transform', type=str, required=True)
parser.add_argument('--theta', type=float, default=0)
parser.add_argument('--du', type=float, default=0)
parser.add_argument('--dv', type=float, default=0)
args = parser.parse_args()

datdir = args.indir
outdir = args.outdir
k1 = args.kappa1
k2 = args.kappa2
transform = args.transform
theta = args.theta
du = args.du
dv = args.dv

print(f"Distorting input data at path '{datdir}'")
print(f"Using transform: {transform}")
print(f"kappa1: {k1:.5g}")
print(f"kappa2: {k2:.5g}")
print(f"Transformation parameters: theta={theta}, du={du}, dv={dv}")
print(f"Saving data to output directory '{outdir}'")

# Copy input directory
# shutil.copytree(datdir, outdir, dirs_exist_ok=True)

SUBDIRS = ["testing", "training", "validation"]


if transform == "v1":
    def f(uvs, k1, k2):
        u, v = uvs[...,0], uvs[...,1]
        x = np.sqrt(2 / k1) * np.sinh((u + v) / 2)
        y = np.sqrt(2 / k2) * np.sinh((u - v) / 2)
        xys = np.zeros_like(uvs)
        xys[...,0] = x
        xys[...,1] = y
        return xys
elif transform == "v2":
    def f(uvs, k1, k2):
        u, v = uvs[...,0], uvs[...,1]
        a = 1 / np.sqrt(k1)
        b = 1 / np.sqrt(k2)
        x = a / 2 * (u + v)
        y = b / 2 * (u - v)
        xys = np.zeros_like(uvs)
        xys[...,0] = x
        xys[...,1] = y
        return xys
elif transform == "v1_rotation":
    def f(uvs, k1, k2):
        u, v = uvs[...,0], uvs[...,1]
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        u, v = rot @ [u.flatten(), v.flatten()]
        u += du
        v += dv
        # apply transform v1
        x = np.sqrt(2 / k1) * np.sinh((u + v) / 2)
        y = np.sqrt(2 / k2) * np.sinh((u - v) / 2)
        xys = np.zeros_like(uvs)
        xys[...,0] = x.reshape(xys.shape[0:-1])
        xys[...,1] = y.reshape(xys.shape[0:-1])
        return xys
elif transform == "v2_rotation":
    def f(uvs, k1, k2):
        u, v = uvs[...,0], uvs[...,1]
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        u, v = rot @ [u.flatten(), v.flatten()]
        u += du
        v += dv
        # apply transform v2
        a = 1 / np.sqrt(k1)
        b = 1 / np.sqrt(k2)
        x = a / 2 * (u + v)
        y = b / 2 * (u - v)
        xys = np.zeros_like(uvs)
        xys[...,0] = x.reshape(xys.shape[0:-1])
        xys[...,1] = y.reshape(xys.shape[0:-1])
        return xys
else:
    raise RuntimeError(f"Cannot handle transform: {transform}")


for subdir in SUBDIRS:
    simdir_list = os.listdir(f"{outdir}/{subdir}")
    simdir_list = [d for d in simdir_list if d.startswith("sim")]
    
    for simdir in simdir_list:
        fpath = f"{outdir}/{subdir}/{simdir}/xs.npy"
        xs = np.load(fpath)
        ys = f(xs, k1, k2)
        np.save(fpath, ys)
