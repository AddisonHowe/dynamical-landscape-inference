"""Figure facs-training

Generate plots used in Figures dec1-training and dec2-training of the supplement.
"""

import argparse
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.style.use('figures/supplement/styles/fig_standard.mplstyle')

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.optimizers import get_dt_schedule
import plnn.pl as pl

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, 
                    help="Name of trained model directory w/o prefix.")
parser.add_argument('-o', '--outdir', type=str, default=None, 
                    help="Name of output directory. Defaults to model name")
parser.add_argument('--logloss', default=True, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--startidx', default=0, type=int)
args = parser.parse_args()

modeldir = args.input  # trained model directory
outdir = args.outdir
if outdir is None:
    outdir = modeldir

logloss = args.logloss
startidx = args.startidx
print("logloss", logloss)
print("startidx", startidx)

SAVEPLOTS = True

MODELDIRBASE = "data/trained_models/facs"
OUTDIRBASE = "figures/supplement/out/fig_facs_training"
SEED = 12345
rng = np.random.default_rng(seed=SEED)

OUTDIR = f"{OUTDIRBASE}/{outdir}/"

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches


COLOR_TRAIN = 'r'
COLOR_VALID = 'b'
MARKER_TRAIN = None
MARKER_VALID = None
LINESTYLE_TRAIN = '-'
LINESTYLE_VALID = '-'
LINEWIDTH_TRAIN = 1
LINEWIDTH_VALID = 1


##############################################################################
##############################################################################
##  Load model and training information

MODELDIR = f"{MODELDIRBASE}/{modeldir}"

model, hps, idx, name, fpath = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR)

tilt_weights = model.get_parameters()['tilt.w'][0]
tilt_bias = model.get_parameters()['tilt.b'][0]
noise_parameter = model.get_sigma()

if tilt_bias is None:
    tilt_bias = np.zeros(tilt_weights.shape[0])
else:
    tilt_bias = tilt_bias[0]

def signal_to_tilts(signal):
    return np.dot(tilt_weights, signal) + tilt_bias

def tilts_to_signals(tilts):
    assert tilts.shape[0] == 2
    y = tilts - tilt_bias[:,None]
    return np.linalg.solve(tilt_weights, y)

print("tilt weights:\n", tilt_weights)
print("tilt bias:\n", tilt_bias)
print("inferred noise:\n", noise_parameter)


##############################################################################
##############################################################################
##  Plot the training history

FIGNAME = "training_history"
FIGSIZE = (8*sf, 16*sf)

fig, axes = plt.subplots(4, 1, figsize=FIGSIZE)

dt_hist = training_info['dt_hist']
sigma_hist = training_info['sigma_hist']
try:
    if dt_hist is None or len(dt_hist) < len(sigma_hist):
        print("Calculuating `dt_hist` to match length of `sigma_hist`")
        dt_schedule = get_dt_schedule(
            logged_args.get('dt_schedule', 'constant'), logged_args
        )
        dt_hist = np.array([dt_schedule(i) for i in range(len(sigma_hist))])
except (RuntimeError, TypeError) as e:
    print("Could not calculate `dt_hist` to match length of `sigma_hist`")
    print(e)


ax=axes[0]
pl.plot_learning_rate_history(
    training_info['learning_rate_hist'],
    log=False, 
    color='k',
    ax=ax
)
ax.set_xlabel("")
ax.set_title("Learning rate schedule")

ax=axes[1]
pl.plot_dt_history(
    dt_hist,
    ax=ax
)
ax.set_xlabel("")
ax.set_title("Timestep schedule")

ax=axes[2]
pl.plot_loss_history(
    training_info['loss_hist_train'],
    training_info['loss_hist_valid'],
    startidx=startidx, log=logloss, 
    color_train=COLOR_TRAIN, color_valid=COLOR_VALID,
    marker_train=MARKER_TRAIN, marker_valid=MARKER_VALID,
    linestyle_train=LINESTYLE_TRAIN, linestyle_valid=LINESTYLE_VALID,
    linewidth_train=LINEWIDTH_TRAIN, linewidth_valid=LINEWIDTH_VALID,
    alpha_train=0.7, alpha_valid=0.6,
    ax=ax
)
ax.set_xlabel("")
ax.set_title("Loss history")

ax=axes[3]
pl.plot_sigma_history(
    sigma_hist,
    log=False, 
    color='k',
    linewidth=2,
    marker=None,
    ax=ax
)
ax.set_xlabel("Epochs")
ax.set_ylabel("$\\sigma$")
ax.set_title("Noise parameter over training")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##############################################################################
##############################################################################
##  Plot parameter history

FIGNAME = "parameter_history"
FIGSIZE = (8*sf, 8*sf)

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

ax11 = axes[0][0]
ax12 = axes[0][1]
ax21 = axes[1][0]
ax22 = axes[1][1]

color = 'k'
linewidth = 1


ax11.plot(
    training_info['tilt_weight_hist'][:,0,0,0],
    color=color,
    linewidth=linewidth,
    label="$A_{11}$"
)

ax12.plot(
    training_info['tilt_weight_hist'][:,0,0,1],
    color=color,
    linewidth=linewidth,
    label="$A_{12}$"
)

ax21.plot(
    training_info['tilt_weight_hist'][:,0,1,0],
    color=color,
    linewidth=linewidth,
    label="$A_{21}$"
)

ax22.plot(
    training_info['tilt_weight_hist'][:,0,1,1],
    color=color,
    linewidth=linewidth,
    label="$A_{22}$"
)

for ax in axes.flatten():
    ax.legend()

for ax in [ax21, ax22]:
    ax.set_xlabel("Epoch")

fig.suptitle("Signal transformation over training")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')
