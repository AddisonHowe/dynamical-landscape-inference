"""Figure 6 Script (FACS Training Loss)

Generate plots used in Figure 6 of the accompanying manuscript.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figures/manuscript/styles/fig_6.mplstyle')

import jax
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--modeldir', type=str, required=True, 
                    help="Name of trained model directory w/o prefix.")
args = parser.parse_args()


MODELDIR = args.modeldir

DATDIR = f"data/trained_models/facs/{MODELDIR}/testing/eval_facs_dec1_v1"

OUTDIR = "figures/manuscript/out/fig6_facs_training"

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54

filelist = [f.removesuffix('.npy') for f in os.listdir(DATDIR) if f.endswith(".npy")]
filelist = [s.split('_') for s in filelist]
dt0_values = [float(v[3]) for v in filelist]
for f in filelist:
    print(f)

DT0_LIST = np.flip(np.sort(np.unique(dt0_values))) # [0.05, 0.01, 0.005, 0.001]
KEY_LIST = ['train', 'valid']

CONDITION_NAMES = {
    0  : "NO CHIR",
    1  : "CHIR 2-2.5",
    2  : "CHIR 2-3",
    3  : "CHIR 2-3.5",
    4  : "CHIR 2-4",
    5  : "CHIR 2-5",
    6  : "CHIR 2-5 FGF 2-3",
    7  : "CHIR 2-5 FGF 2-3.5",
    8  : "CHIR 2-5 FGF 2-4",
    9  : "CHIR 2-5 FGF 2-4.5",
    10 : "CHIR 2-5 FGF 2-5",
}

KEY_TO_CONDITION_SPLIT1 = {
    'v1'    : {
        'train'   : [0, 2, 5, 6, 10],
        'valid' : [1, 3, 7, 9],
        'test'    : [4, 8],
    },
    'v2' : {
        'train'   : [0, 2, 5, 6, 10],
        'valid' : [1, 3, 7, 9],
        'test'    : [4, 8],
    },
    'v3' : {
        'train'   : [0, 2, 5, 6, 10],
        'valid' : [1, 3, 7, 9],
        'test'    : [4, 8],
    },
}

KEY_TO_CONDITION_SPLIT2 = {
    'v1'    : {
        'train'   : [2, 5, 6, 10],
        'valid' : [1, 3, 7, 9],
        'test'    : [4, 8],
    },
    'v2' : {
        'train'   : [2, 5, 6, 10],
        'valid' : [1, 3, 7, 9],
        'test'    : [4, 8],
    },
    'v3' : {
        'train'   : [2, 5, 6, 10],
        'valid' : [1, 3, 7, 9],
        'test'    : [4, 8],
    },
}

if "dec1" in MODELDIR:
    DECISION_IDX = 1
elif "dec2" in MODELDIR:
    DECISION_IDX = 2
else:
    raise RuntimeError()


if "v1" in MODELDIR:
    train_valid_test_conds = {
        1: KEY_TO_CONDITION_SPLIT1['v1'],
        2: KEY_TO_CONDITION_SPLIT2['v1'],
    }[DECISION_IDX]
elif "v2" in MODELDIR:
    train_valid_test_conds = {
        1: KEY_TO_CONDITION_SPLIT1['v2'],
        2: KEY_TO_CONDITION_SPLIT2['v2'],
    }[DECISION_IDX]
elif "v3" in MODELDIR:
    train_valid_test_conds = {
        1: KEY_TO_CONDITION_SPLIT1['v3'],
        2: KEY_TO_CONDITION_SPLIT2['v3'],
    }[DECISION_IDX]
else:
    raise RuntimeError("Cannot determine train/test/valid condition split.")


DATASETS = {}
for dataset_key in KEY_LIST:
    for dt0 in DT0_LIST:
        conditions = np.load(f"{DATDIR}/conditions_{dataset_key}_dt_{dt0}.npy")
        times = np.load(f"{DATDIR}/times_{dataset_key}_dt_{dt0}.npy")
        losses = np.load(f"{DATDIR}/losses_{dataset_key}_dt_{dt0}.npy")
        DATASETS[dataset_key, dt0] = {
            'conditions': conditions,
            'times': times,
            'losses': losses,
        }



figsize = (7*sf, 4*sf)
for dataset_key in KEY_LIST:

    for dt0 in DT0_LIST:
        dataset = DATASETS[dataset_key, dt0]
        losses = dataset['losses']
        times = dataset['times']
        nconds, ndata_per_cond, nresamps, nreps = losses.shape

        timepoints = np.sort(np.unique(times))
        timepoints += 2 + (timepoints[1] - timepoints[0]) / 2

        for condidx in range(nconds):
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            cond_name = CONDITION_NAMES[train_valid_test_conds[dataset_key][condidx]]
            for sampidx in range(nresamps):
                vals = losses[condidx, :, sampidx, :].mean(1)
                ax.plot(timepoints, vals)
                ax.set_title(f"{cond_name} ({dataset_key})")
                ax.set_xlim(2, 5)
            
            plt.tight_layout()
            figname = f"resampling_comparisons_{dataset_key}_{cond_name}_dt_{dt0}"
            plt.savefig(f"{OUTDIR}/{figname}.pdf")
            plt.close()

print(f"Each line is the mean loss of a resampled initial condition, averaged over {nreps} simulations.")
print("Each line is the mean loss of a condition, " \
      f"averaged over all {nresamps} resamplings.")
print("Error bars show 2 standard deviations.")

figsize = (9*sf, 4*sf)
for dataset_key in KEY_LIST:
    for dt0 in DT0_LIST:
        dataset = DATASETS[dataset_key, dt0]
        losses = dataset['losses']
        times = dataset['times']
        nconds, ndata_per_cond, nresamps, nreps = losses.shape

        timepoints = np.sort(np.unique(times))
        timepoints += 2 + (timepoints[1] - timepoints[0]) / 2

        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
        legend_handles = []
        legend_labels = []
        for condidx in range(nconds):
            cond_name = CONDITION_NAMES[train_valid_test_conds[dataset_key][condidx]]
            avg_losses_over_reps = losses[condidx].mean(-1)
            mean_loss_over_samps = avg_losses_over_reps.mean(-1)
            std_loss_over_samps = avg_losses_over_reps.std(-1)
            
            l, = ax.plot(
                timepoints, mean_loss_over_samps,
                linestyle='-',
                alpha=0.8,
            )
            legend_handles.append(l)
            legend_labels.append(cond_name)
            ax.errorbar(
                timepoints, mean_loss_over_samps, 
                yerr=2*std_loss_over_samps,
                capsize=4, 
                linestyle="None", 
                color=l.get_color(),
                alpha=0.8
            )

        ax.set_xlim(2, 5)
        ax.legend(
            legend_handles, legend_labels, 
            bbox_to_anchor=(1.05, -0.25), loc='lower left',
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Error")
        title = {
            'train': "Training error", 
            'valid': "Validation error"
        }[dataset_key]
        ax.set_title(title)
        
        figname = f"loss_comparison_{dataset_key}_dt_{dt0}"
        plt.savefig(f"{OUTDIR}/{figname}.pdf", transparent=True)
        plt.close()
