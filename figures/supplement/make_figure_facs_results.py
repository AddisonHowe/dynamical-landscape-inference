"""Figure facs-results

Generate plots used in Figures dec[12]-results of the supplement.
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
parser.add_argument('--logloss', default=True, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--startidx', default=0, type=int)
parser.add_argument('--signal', type=float, nargs=2, default=[0., 1.])
args = parser.parse_args()

modeldir = args.input  # trained model directory
outdir = args.outdir
if outdir is None:
    outdir = modeldir

logloss = args.logloss
startidx = args.startidx
SIG_TO_PLOT = args.signal
print("logloss", logloss)
print("startidx", startidx)

SAVEPLOTS = True

MODELDIRBASE = "data/trained_models/facs"
OUTDIRBASE = "figures/supplement/out/fig_facs_results"
SEED = 12345
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
##  Plot the training history

FIGNAME = "training_history"
FIGSIZE = (8*sf, 12*sf)

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE)

ax=axes[0]
plot_loss_history(
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

ax=axes[1]
plot_sigma_history(
    training_info['sigma_hist'],
    log=False, 
    color='k',
    linewidth=2,
    marker=None,
    ax=ax
)
ax.set_xlabel("")

ax=axes[2]
plot_learning_rate_history(
    training_info['learning_rate_hist'],
    log=False, 
    ax=ax
)

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

plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()

#################################  Bifurcation diagram of inferred landscape
FIGNAME = "phi_bifs_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_inferred, bifcolors_inferred, aux_info = get_plnn_bifurcation_curves(
    model, 
    num_starts=100, 
    # p1lims=[-1, 1],
    # p2lims=[-1, 1],
    maxiter=1000,
    ds=1e-3,
    min_ds=1e-8,
    max_ds=1e-2,
    max_delta_p=1e-1,
    rho=1e-1,
    return_aux_info=True,
    rng=rng,
    verbosity=0,
)

# Filter out singleton bifurcation curves and remove initial estimate point
keepidxs = [i for i in range(len(bifcurves_inferred)) 
            if len(bifcurves_inferred[i]) > 1]
bifcurves_inferred = [bc[1:] for bc in bifcurves_inferred if len(bc) > 1]
bifcolors_inferred = [bifcolors_inferred[i] for i in keepidxs]

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

# ax.set_xlim(-2, 2)
# ax.set_ylim(-1, 3)
# ax.set_xticks([-2, 0, 2])
# ax.set_yticks([0, 2, 4])
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of inferred landscape in signals
FIGNAME = "phi_bifs_signals_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    curve_signal = tilts_to_signals(curve.T).T
    ax.plot(curve_signal[:,0], curve_signal[:,1], '-', color=color)

ax.set_xlabel("CHIR")
ax.set_ylabel("FGF")
ax.set_title("Bifurcations")

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##################################  Noise History
FIGNAME = "noise_history"
FIGSIZE = (5.75*sf, 4.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

logged_args, run_dict = load_model_training_metadata(MODELDIR)
sigma_hist = run_dict['sigma_hist']

plot_sigma_history(
    sigma_hist, log=False,
    color='k', marker=None, linestyle='-',
    title="", 
    figsize=FIGSIZE,
    ax=ax,
)
ylims = ax.get_ylim()
ax.vlines(
    idx, ylims[0], sigma_hist[idx],
    linestyles='--', colors='grey', linewidth=1, zorder=1, 
    label=f"Inferred $\sigma={model.get_sigma():.2g}$"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("$\sigma$")
ax.set_ylim(*ylims)
ax.set_title("Inferred noise")
ax.legend(fontsize=LEGEND_FONTSIZE)
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()


##################################  Loss History
FIGNAME = "loss_history"
FIGSIZE = (5.75*sf, 4.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')
# plot_loss_history(
#     training_info['loss_hist_train'],
#     training_info['loss_hist_valid'],
#     log=True,
#     color_train='k', color_valid='grey',
#     marker_train=None, marker_valid=None,
#     linestyle_train='-', linestyle_valid='-',
#     linewidth_train=1, linewidth_valid=1,
#     alpha_train=0.7, alpha_valid=0.6,
#     ax=ax
# )
plot_validation_loss_history(
    training_info['loss_hist_valid'],
    log=True,
    color='k',
    marker=None,
    linestyle='-',
    linewidth_train=1,
    ax=ax
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Error (validation)")
ax.set_title("")

ax.axvline(
    idx, 0, 1,
    linestyle='--', color='grey', linewidth=1, zorder=1,
)
# ax.legend(fontsize=LEGEND_FONTSIZE)
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()

##################################  Signal mapping
FIGNAME = "signal_mapping"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.set_aspect('equal')

scale = np.max(np.abs(tilt_weights))

ax.arrow(
    0, 0, -tilt_weights[0,0], -tilt_weights[1,0], 
    width=0.01*scale, 
    fc=CHIR_COLOR, ec=CHIR_COLOR,
    label="CHIR"
)

ax.arrow(
    0, 0, -tilt_weights[0,1], -tilt_weights[1,1], 
    width=0.01*scale, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    label="FGF"
)

ax.set_title("Inferred signal effect")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-scale*1.1, scale*1.1])
ax.set_ylim([-scale*1.1, scale*1.1])

ax.legend(fontsize=LEGEND_FONTSIZE)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)


##############################################################################
##############################################################################
