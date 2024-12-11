"""Plot facs-landscape-tilted

Generate a plot of a tilted landscape given a specified signal.
"""

import argparse
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
plt.style.use('figures/supplement/styles/fig_standard.mplstyle')
import matplotlib.patches as patches

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.pl import plot_phi
from plnn.pl import plot_loss_history, plot_learning_rate_history
from plnn.pl import plot_sigma_history
from plnn.pl import plot_validation_loss_history
from plnn.pl import CHIR_COLOR, FGF_COLOR
from plnn.vectorfields import estimate_minima

from cont.plnn_bifurcations import get_plnn_bifurcation_curves 


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, 
                    help="Name of trained model directory w/o prefix.")
parser.add_argument('-o', '--outdir', type=str, default=None, 
                    help="Name of output directory. Defaults to model name")
parser.add_argument('--signal', type=float, nargs=2, default=[0., 1.])
args = parser.parse_args()

modeldir = args.input  # trained model directory
outdir = args.outdir
if outdir is None:
    outdir = modeldir

SIG_TO_PLOT = args.signal

SAVEPLOTS = True

MODELDIRBASE = "data/trained_models/facs"
OUTDIRBASE = "figures/supplement/out/fig_facs_results"
SEED = 141232
rng = np.random.default_rng(seed=SEED)

OUTDIR = f"{OUTDIRBASE}/{outdir}/"

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches

LEGEND_FONTSIZE = 8
INSET_SCALE = "30%"

COLOR_TRAIN = 'r'
COLOR_VALID = 'b'
MARKER_TRAIN = None
MARKER_VALID = None
LINESTYLE_TRAIN = '-'
LINESTYLE_VALID = '-'
LINEWIDTH_TRAIN = 1
LINEWIDTH_VALID = 1

FP_MARKERS = {
    'saddle': 'x',
    'minimum': 'o',
    'maximum': '^',
}


##############################################################################
##############################################################################
##  Load model and training information

MODELDIR = f"{MODELDIRBASE}/{modeldir}"

model, hps, idx, name, fpath = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR, True)

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


# Get the window limits of the corresponding training data
def add_buffer(arr, buffer):
    width = arr[1] - arr[0]
    return arr + buffer * width * np.array([-1.0, 1.0])
training_data_fpath = logged_args['training_data']
window = np.load(f"{training_data_fpath}/window.npy")
XLIMS, YLIMS = window
buffer = 0.05
XLIMS = add_buffer(XLIMS, buffer)
YLIMS = add_buffer(YLIMS, buffer)


##############################################################################
##############################################################################
##  Heatmap of inferred landscape

FIGNAME = "phi_inferred"
FIGSIZE = (10*sf, 6*sf)

res = 50   # resolution
lognormalize = True
clip = None

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

ax = plot_phi(
    model, signal=SIG_TO_PLOT, 
    xrange=XLIMS,
    yrange=YLIMS,
    res=res,
    lognormalize=lognormalize,
    clip=clip,
    include_tilt_inset=True,
    title=f"CHIR: {SIG_TO_PLOT[0]:.1f}, FGF: {SIG_TO_PLOT[1]:.1f}",
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    # cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    cbar_title="",
    equal_axes=True,
    saveas=None,
    figsize=FIGSIZE,
    show=True,
    tight_layout=False,
    ax=ax,
)

# ax.set_xticks([-2, 0, 2])
# ax.set_yticks([-2, 0, 2])

cax = fig.get_axes()[2]
# cax.set_yticks([0, 1, 2, 3])

mins = estimate_minima(
    model, model.tilt_module(np.array(SIG_TO_PLOT)), 
    n=100, 
    tol=1e-3,
    x0_range=window, 
    rng=rng,
)
for m in mins:
    ax.plot(m[0], m[1], marker='.', color='y', markersize=3)

plt.savefig(f"{OUTDIR}/{FIGNAME}_{SIG_TO_PLOT[0]}_{SIG_TO_PLOT[1]}.pdf", transparent=True)
plt.close()


##############################################################################
##############################################################################
