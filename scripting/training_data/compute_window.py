import argparse

import warnings
import os
import tqdm as tqdm
import numpy as np

from plnn.dataset import get_dataloaders


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datdir', type=str, required=True,
                    help="e.g. data/training_data/facs/facs_dec1_v1")
parser.add_argument('-o', '--outdir', type=str, default="out/data_analysis")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('-v', '--verbosity', type=int, default=1)
args = parser.parse_args()

DATDIR = args.datdir
OUTDIR = args.outdir
NCELLS_SAMPLE = 800
SEED = args.seed
verbosity = args.verbosity

os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(seed=SEED)

datdir_train = f"{DATDIR}/training"
datdir_valid = f"{DATDIR}/validation"
datdir_test = f"{DATDIR}/testing"

nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)

try:
    nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)
except FileNotFoundError as e:
    msg = f"{e} Reverting to validation data instead."
    warnings.warn(msg)
    datdir_test = f"{DATDIR}/validation"
    nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)


_, _, _, train_dset, valid_dset, test_dset = get_dataloaders(
    datdir_train, datdir_valid, nsims_train, nsims_valid,
    shuffle_train=False,
    shuffle_valid=False,
    return_datasets=True,
    include_test_data=True,
    datdir_test=datdir_test, 
    nsims_test=nsims_test, 
    shuffle_test=False,
    batch_size_test=1,
    ncells_sample=NCELLS_SAMPLE,
    length_multiplier=1,
    seed=rng.integers(2**32)
)

def get_window(xs):
    """xs is shape (n,d), return array of shape (d,2)."""
    mins = np.min(xs, axis=0)
    maxs = np.max(xs, axis=0)
    window = np.array([mins, maxs]).T
    return window

# Compute window for each sample across the datasets

dsets = [train_dset, valid_dset, test_dset]
names = ['training', 'validation', 'testing']
for dset, name in zip(dsets, names):
    sample_windows = []
    for i in tqdm.tqdm(range(dset.get_baselength())):
        d = dset.get_unsampled_item(i)
        t0, x0, t1, sp = d[0]
        x1 = d[1]
        if t0 == 0:
            sample_windows.append(get_window(x0))
        sample_windows.append(get_window(x1))
    sample_windows = np.array(sample_windows)
    if verbosity:
        print(f"{name} data...")
        print("sample windows:\n", sample_windows)
    window_mins = np.min(sample_windows[:,:,0], axis=0)
    window_maxs = np.max(sample_windows[:,:,1], axis=0)
    window = np.array([window_mins, window_maxs]).T
    np.save(f"{DATDIR}/{name}/window.npy", window)

    
