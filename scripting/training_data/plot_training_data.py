"""Script to plot training data

Usage:
    python plot_training_data.py -i <indir> -o <outdir>
"""

import os, sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt




def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datdir", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    datdir = args.datdir
    outdir = args.outdir

    print(f"Loading data in {datdir}")
    print(f"Saving plots to {outdir}")
    os.makedirs(outdir, exist_ok=True)
    time0 = time.time()

    # Load data


    # Do plotting
    plot_training_data()
    plot_validation_data()
    plot_testing_data()


    
    time1 = time.time()
    print(f"Finished in {time1 - time0:.4g} sec")
    return


def plot_training_data():
    return


def plot_validation_data():
    return


def plot_testing_data():
    return


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
