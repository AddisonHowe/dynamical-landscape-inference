"""Figure phi2-results

Generate plots used in Figure phi2-results of the supplement.
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

from cont.binary_flip import get_binary_flip_curves
from cont.plnn_bifurcations import get_plnn_bifurcation_curves 


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, 
                    help="Name of trained model directory w/o prefix.")
parser.add_argument('-o', '--outdir', type=str, default=None, 
                    help="Name of output directory. Defaults to model name")
parser.add_argument('--truesigma', type=float, required=True)
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
truesigma = args.truesigma
print("logloss", logloss)
print("startidx", startidx)
print("truesigma", truesigma)

SAVEPLOTS = True

MODELDIRBASE = "data/trained_models/plnn_synbindec"
OUTDIRBASE = "figures/supplement/out/fig_phi_results"
SIGMA_TRUE = args.truesigma
SEED = 12456156
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
    sigma_true=truesigma,
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
##  Plot NORMALIZED parameter history

FIGNAME = "parameter_history_difference"
FIGSIZE = (8*sf, 8*sf)

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)

ax11 = axes[0][0]
ax12 = axes[0][1]
ax21 = axes[1][0]
ax22 = axes[1][1]

tilt_weights_hist = training_info['tilt_weight_hist'][:,0,:,:]
tilt_weights_true = np.array([[1., 0.],[0., 1.]])

tilt_weights_diffs = tilt_weights_hist - tilt_weights_true

color = 'k'
linewidth = 1


ax11.plot(
    tilt_weights_diffs[:,0,0],
    color=color,
    linewidth=linewidth,
    label="$A_{11} - A_{11}^*$"
)

ax12.plot(
    tilt_weights_diffs[:,0,1],
    color=color,
    linewidth=linewidth,
    label="$A_{12} - A_{12}^*$"
)

ax21.plot(
    tilt_weights_diffs[:,1,0],
    color=color,
    linewidth=linewidth,
    label="$A_{21} - A_{21}^*$"
)

ax22.plot(
    tilt_weights_diffs[:,1,1],
    color=color,
    linewidth=linewidth,
    label="$A_{22} - A_{22}^*$"
)

for ax in axes.flatten():
    ax.legend()
    # ax.legend(loc='lower right')
    ax.axhline(0, 0, 1, color='k', alpha=0.5, linestyle='--', linewidth=1)

for ax in [ax21, ax22]:
    ax.set_xlabel("Epoch")

fig.suptitle("Signal transformation error")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##############################################################################
##############################################################################
#################################  Bifurcation diagram of true landscape
FIGNAME = "phi2_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_true, bifcolors_true = get_binary_flip_curves(
    rng=rng,
    add_flip_curves=True,
)
for curve, color in zip(bifcurves_true, bifcolors_true):
    ax.plot(
        curve[:,0], curve[:,1], 
        ':' if color=='purple' else '-', 
        color=color,
    )

ax.set_xlim(-2, 1)
ax.set_ylim(-2, 2)
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')

##############################################################################
##############################################################################
##  Heatmap of inferred landscape

TILT_TO_PLOT = [0., 0.]

FIGNAME = "phi2_inferred"
FIGSIZE = (5.75*sf, 5.25*sf)

r = 2.5     # box radius
res = 100   # resolution
lognormalize = True
clip = None

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

ax = plot_phi(
    model, tilt=TILT_TO_PLOT, 
    r=r, res=res,
    lognormalize=lognormalize,
    clip=clip,
    include_tilt_inset=True,
    title="Inferred",
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

ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

cax = fig.get_axes()[2]
cax.set_yticks([0, 1, 2, 3])

mins = estimate_minima(
    model, TILT_TO_PLOT, 
    n=50, 
    tol=1e-2,
    x0_range=[[-3, 3],[-3, 3]], 
    rng=rng,
)
for m in mins:
    ax.plot(m[0], m[1], marker='.', color='y', markersize=3)

plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()

#################################  Bifurcation diagram of inferred landscape
FIGNAME = "phi2_bifs_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

bifcurves_inferred, bifcolors_inferred, aux_info = get_plnn_bifurcation_curves(
    model, 
    num_starts=100, 
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
keepidxs = [i for i in range(len(bifcurves_inferred)) if len(bifcurves_inferred[i]) > 1]
bifcurves_inferred = [bc[1:] for bc in bifcurves_inferred if len(bc) > 1]
bifcolors_inferred = [bifcolors_inferred[i] for i in keepidxs]

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    ax.plot(curve[:,0], curve[:,1], '-', color=color)

ax.set_xlim(-2, 1)
ax.set_ylim(-2, 2)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 2, 1])
ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


#################################  Bif diagram of inferred landscape in signals
FIGNAME = "phi2_bifs_signals_inferred"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    curve_signal = tilts_to_signals(curve.T).T
    ax.plot(curve_signal[:,0], curve_signal[:,1], '-', color=color)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight')


##################################  Combined bif diagram
FIGNAME = "phi2_combined_bifs"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    ln, = ax.plot(
        curve[:,0], curve[:,1], 
        ':' if color == 'purple' else '--', 
        color=color,
    )
    if color != 'purple':
        true_line = ln

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    inf_line, = ax.plot(curve[:,0], curve[:,1], '-', color=color, alpha=0.9)

ax.legend(
    [true_line, inf_line], ['Ground truth', 'Inferred'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',
    fontsize=LEGEND_FONTSIZE
)

ax.set_xlabel("$\\tau_1$")
ax.set_ylabel("$\\tau_2$")

ax.set_xlim([-2, 1])
ax.set_ylim([-2, 2])

ax.set_xticks([-2, -1, 0, 1])
ax.set_yticks([-2, -1, 0, 1, 2])

ax.set_title("Bifurcation diagram")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)


##################################  Combined bif diagram in signals
FIGNAME = "phi2_combined_bifs_signals"
FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    ln, = ax.plot(
        curve[:,0], curve[:,1], 
        ':' if color == 'purple' else '--', 
        color=color,
    )
    if color != 'purple':
        true_line = ln

for curve, color in zip(bifcurves_inferred, bifcolors_inferred):
    curve_signal = tilts_to_signals(curve.T).T
    inf_line, = ax.plot(curve_signal[:,0], curve_signal[:,1], '-', 
                        color=color, alpha=0.9)

ax.legend(
    [true_line, inf_line], ['Ground truth', 'Inferred'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',
    fontsize=LEGEND_FONTSIZE
)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

ax.set_xlim([-2, 1])
ax.set_ylim([-2, 2])

ax.set_xticks([-2, -1, 0, 1])
ax.set_yticks([-2, -1, 0, 1, 2])

ax.set_title("Bifurcation diagram")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)


##################################  Noise History
FIGNAME = "noise_history"
FIGSIZE = (5.75*sf, 4.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

logged_args, run_dict = load_model_training_metadata(MODELDIR)
sigma_hist = run_dict['sigma_hist']

plot_sigma_history(
    sigma_hist, log=False, sigma_true=SIGMA_TRUE,
    color='k', marker=None, linestyle='-',
    title="", sigma_true_legend_label=f'Ground truth $\\sigma^*={SIGMA_TRUE:.3g}$',
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
ax.set_ylabel("Noise $\sigma$")
ax.set_ylim(*ylims)
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
    label="$s_1$"
)

ax.arrow(
    0, 0, -tilt_weights[0,1], -tilt_weights[1,1], 
    width=0.01*scale, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    label="$s_2$"
)

ax.set_title("Inferred signal effect")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-scale*1.1, scale*1.1])
ax.set_ylim([-scale*1.1, scale*1.1])

ax.legend(fontsize=LEGEND_FONTSIZE)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)


##################################  Signal prior
FIGNAME = "signal_prior"
FIGSIZE = (5*sf, 5*sf)

ALPHA = 0.5
COL0 = 'orange'
COL1 = 'purple'

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

for curve, color in zip(bifcurves_true, bifcolors_true):
    ax.plot(
        curve[:,0], curve[:,1], 
        ':' if color == 'purple' else '--', 
        color=color,
    )
    

info_dict = {
    's10': [-1.50, -1.00],
    's20': [-0.75,  0.75],
    's11': [-1.50, -1.00],
    's21': [-0.75,  0.75],
}

# Plot the prior for initial signal values
x0 = info_dict['s10'][0]
y0 = info_dict['s20'][0]
w = info_dict['s10'][1] - info_dict['s10'][0]
h = info_dict['s20'][1] - info_dict['s20'][0]
# Create the rectangle patch with transparency (alpha)
rectangle = patches.Rectangle(
    (x0, y0), w, h, 
    alpha=ALPHA, color=COL0, fill=None, hatch=4*'/',
)
p1 = ax.add_patch(rectangle)

# Plot the prior for final signal values
x0 = info_dict['s11'][0]
y0 = info_dict['s21'][0]
w = info_dict['s11'][1] - info_dict['s11'][0]
h = info_dict['s21'][1] - info_dict['s21'][0]
# Create the rectangle patch with transparency (alpha)
rectangle = patches.Rectangle(
    (x0, y0), w, h, 
    alpha=ALPHA, color=COL1, fill=None, hatch=4*'\\',
)
p2 = ax.add_patch(rectangle)

ax.legend(
    [p1, p2], ['Initial', 'Final'], 
    # bbox_to_anchor=(1.05, 1), loc='upper left',
    fontsize=LEGEND_FONTSIZE,
)

ax.set_xlabel("$s_1$")
ax.set_ylabel("$s_2$")

ax.set_xlim([-2, 1])
ax.set_ylim([-2, 2])

ax.set_xticks([-2, -1, 0, 1])
ax.set_yticks([-2, -1, 0, 1, 2])

ax.set_title("Signal prior")

plt.savefig(f"{OUTDIR}/{FIGNAME}", bbox_inches='tight', transparent=True)

##############################################################################
##############################################################################
