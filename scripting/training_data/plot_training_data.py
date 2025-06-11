"""Script to plot training data

Usage:
    python plot_training_data.py -i <indir> -o <outdir>
"""

import os, sys
import warnings
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from plnn.dataset import get_dataloaders


FIGSIZE = None


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datdir", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("--figsize", type=float, nargs=2, default=FIGSIZE)
    return parser.parse_args(args)


def main(args):
    datdir = args.datdir
    outdir = args.outdir
    figsize = args.figsize

    print(f"Loading data in {datdir}")
    print(f"Saving plots to {outdir}")
    os.makedirs(outdir, exist_ok=True)
    time0 = time.time()

    # Load data
    train_dset, valid_dset, test_dset = load_data(datdir)

    #~~~  Plot data  ~~~#
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = plot_data(train_dset, ax=ax)
    ax.set_title("Training data")
    plt.savefig(f"{outdir}/training_data.pdf")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = plot_data(valid_dset, ax=ax)
    ax.set_title("Validation data")
    plt.savefig(f"{outdir}/validation_data.pdf")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = plot_data(test_dset, ax=ax)
    ax.set_title("Testing data")
    plt.savefig(f"{outdir}/testing_data.pdf")
    plt.close()
    #~~~~~~~~~~~~~~~~~~~#

    time1 = time.time()
    print(f"Finished in {time1 - time0:.4g} sec")
    return


def load_data(datdir):
    datdir_train = f"{datdir}/training"
    datdir_valid = f"{datdir}/validation"
    datdir_test = f"{datdir}/testing"
    ncells_sample=0

    nsims_train = np.genfromtxt(f"{datdir_train}/nsims.txt", dtype=int)
    nsims_valid = np.genfromtxt(f"{datdir_valid}/nsims.txt", dtype=int)
    try:
        nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)
    except FileNotFoundError as e:
        msg = f"{e} Reverting to validation data instead."
        warnings.warn(msg)
        datdir_test = f"{datdir}/validation"
        nsims_test = np.genfromtxt(f"{datdir_test}/nsims.txt", dtype=int)

    _, _, _, train_dset, valid_dset, test_dset  = get_dataloaders(
        datdir_train, datdir_valid, nsims_train, nsims_valid,
        shuffle_train=False,
        shuffle_valid=False,
        return_datasets=True,
        include_test_data=True,
        datdir_test=datdir_test, nsims_test=nsims_test, shuffle_test=False,
        batch_size_test=1,
        ncells_sample=ncells_sample,
        seed=42,
    )
    return train_dset, valid_dset, test_dset


def plot_data(
        dataset, 
        ax,        
):
    # Plot datapoints
    for i, d in enumerate(dataset):
        t0 = d[0][0]
        x0 = d[0][1]
        x1 = d[1]
        # Only plot x0 on first iteration
        if t0 == 0:
            ax.plot(
                x0[:,0], x0[:,1], '.k', 
                markersize=1, 
                alpha=0.5, 
                rasterized=True
            )
        ax.plot(
            x1[:,0], x1[:,1], '.k', 
            markersize=1, 
            alpha=0.5, 
            rasterized=True
        )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return ax


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
