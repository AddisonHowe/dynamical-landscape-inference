"""Figure 0 Script (Visual Abstract)

Generate plots used in Figure 0 of the accompanying manuscript.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
plt.style.use('figures/manuscript/styles/fig_1.mplstyle')
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plnn.pl import plot_landscape, CHIR_COLOR, FGF_COLOR
from plnn.helpers import get_phi1_fixed_points, get_phi2_fixed_points
from plnn.data_generation.signals import get_sigmoid_function
from plnn.models.algebraic_pl import AlgebraicPL


OUTDIR = "figures/manuscript/out/fig0_visual_abstract"
SAVEPLOTS = True

SEED = 1234125123

TEX_CONTEXT = 'figures/manuscript/styles/fig_1_tex.mplstyle'

INSET_SCALE = "80%"


os.makedirs(OUTDIR, exist_ok=True)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y + p1*x + p2*y

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches

FP_MARKERS = {
    'saddle': 'x',
    'minimum': 'o',
    'maximum': '^',
}

ANNOTATION_FONTSIZE = 10
PARAM_MARKERSIZE = 4

def get_marker_edge_width(marker):
    return 0.2 if marker == 'o' else 0.5

# See: https://stackoverflow.com/questions/22867620/
#      putting-arrowheads-on-vectors-in-a-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

##############################################################################
##############################################################################
##  Untilted binary choice landscape.

FIGNAME1 = "phi1_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi1_landscape_untilted"
FIGSIZE2 = (6*sf, 5.8*sf)

r = 3       # box radius
res = 50   # resolution
lognormalize = True
clip = None

plot_tilt = [0, 0.5]

fps, fp_types, fp_colors = get_phi1_fixed_points([plot_tilt])

with plt.style.context(TEX_CONTEXT):
    ax = plot_landscape(
        func_phi1_star, r=r, res=res, params=plot_tilt, 
        lognormalize=lognormalize,
        clip=clip,
        title=f"$\\boldsymbol{{\\tau}}=({plot_tilt[0]}, {plot_tilt[1]})$",
        ncontours=10,
        contour_linewidth=0.5,
        contour_linealpha=0.5,
        include_cbar=True,
        cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
        equal_axes=True,
        figsize=FIGSIZE1,
        show=True
    )
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=get_marker_edge_width(marker),
        
    )
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME1}")
plt.close()

ax = plot_landscape(
    func_phi1_star, r=r, res=res, params=plot_tilt, 
    plot3d=True,
    lognormalize=False,
    normalize=True,
    minimum=50,
    clip=100,
    include_cbar=False,
    title=f"",
    cbar_title="$\ln\phi$",
    alpha=0.75,
    xlims=[-3.5, 3.5],
    ylims=[-3.5, 3.5],
    zlims=[0, 150],
    zlabel="$\phi$",
    view_init=[35, -45],  # elevation, aximuthal 
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    equal_axes=True,
    tight_layout=True,
    figsize=FIGSIZE2,
    show=True,
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1], 0,
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w' if marker == 'o' else 'r',
        markeredgewidth=0.2 if marker == 'o' else 0.5,
        zorder=10,
    )

xlims = ax.get_xlim()
ylims = ax.get_ylim()
zlims = ax.get_zlim()

arrow_start = [-2, -2, zlims[0] - 0.5]
arrow1_dir = [-1.5, 0, 0]
arrow2_dir = [0, -1.5, 0]

arrow1 = Arrow3D(
    *[[arrow_start[i], arrow_start[i] + arrow1_dir[i]] for i in range(3)],
    # width=0.01*1, 
    fc=CHIR_COLOR, ec=CHIR_COLOR,
    arrowstyle="simple, head_width=2, head_length=2",
    label="$s_1$",
)
ax.add_artist(arrow1)
arrow2 = Arrow3D(
    *[[arrow_start[i], arrow_start[i] + arrow2_dir[i]] for i in range(3)],
    # [2, 2], [-2, -3], [zlims[0]-0.5, zlims[0]-0.5],
    # width=0.01*1, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    arrowstyle="simple, head_width=2, head_length=2",
    label="$s_2$",
)
ax.add_artist(arrow2)

ax.set_xlim(*xlims)
ax.set_ylim(*ylims)
ax.set_zlim(*zlims)

# ax.tick_params(axis='x', which='both', pad=-5)
# ax.tick_params(axis='y', which='both', pad=-5)
# ax.tick_params(axis='z', which='both', pad=0)
ax.xaxis.labelpad = -15
ax.yaxis.labelpad = -15
ax.zaxis.labelpad = -15
ax.grid(True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# ax.set_xlabel("")
# ax.set_ylabel("")
# ax.set_zlabel("")
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME2}", bbox_inches='tight')
plt.close()

##############################################################################
##############################################################################
##  Untilted binary flip landscape.

FIGNAME1 = "phi2_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi2_landscape_untilted"
FIGSIZE2 = (6*sf, 6*sf)

r = 3       # box radius
res = 50   # resolution
lognormalize = True
clip = None

fps, fp_types, fp_colors = get_phi2_fixed_points([[0, 0]])

with plt.style.context(TEX_CONTEXT):
    ax = plot_landscape(
        func_phi2_star, r=r, res=res, params=[0, 0], 
        lognormalize=lognormalize,
        clip=clip,
        title=f"$\\boldsymbol{{\\tau}}=(0, 0)$",
        ncontours=10,
        contour_linewidth=0.5,
        include_cbar=True,
        cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
        equal_axes=True,
        figsize=FIGSIZE1,
        show=True,
    )
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=get_marker_edge_width(marker),
    )
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME1}")
plt.close()

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    plot3d=True,
    lognormalize=False,
    normalize=False,
    minimum=50,
    clip=100,
    title=f"$\phi_{{bf}}$",
    include_cbar=False,
    cbar_title="$\phi$",
    alpha=0.75,
    equal_axes=True,
    xlims=[-3.5, 3.5],
    ylims=[-3.5, 3.5],
    zlims=[0, 150],
    view_init=[35, -80],  # elevation, aximuthal 
    ncontours=10,
    contour_linewidth=0.5,
    figsize=FIGSIZE2,
    show=True,
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=get_marker_edge_width(marker),
    )
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME2}")
plt.close()

##############################################################################
##############################################################################
##  Tilted binary choice landscape

FIGNAME = "phi1_heatmap_tilted"  # appended with index i
FIGSIZE = (4*sf, 4*sf)

PARAMS_PHI1 = [
    ([ 0,  0.5], 0),
    ([-1.00,  1.00], 1),
    ([ 1.00,  1.00], 2),
    ([-0.50,  0.00], 3),
    ([ 0.50,  0.00], 4),
]
r = 3       # box radius
res = 50   # resolution
lognormalize = True
clip = None
for i, (p, label_number) in enumerate(PARAMS_PHI1):
    with plt.style.context(TEX_CONTEXT):
        title = f"$\\boldsymbol{{\\tau}}=" + f"({p[0]:.2g},{p[1]:.2g})$"
        ax = plot_landscape(
            func_phi1_star, r=r, res=res, params=p, 
            lognormalize=lognormalize,
            clip=clip,
            title=title,
            cbar_title="$\ln\phi$",
            include_cbar=False,
            ncontours=10,
            contour_linewidth=0.5,
            contour_linealpha=0.5,
            xlabel=None,
            ylabel=None,
            xticks=False,
            yticks=False,
            equal_axes=True,
            figsize=FIGSIZE,
            show=True,
        )
    ax.text(
        0.99, 0, str(label_number), 
        color='k',
        fontsize=ANNOTATION_FONTSIZE, 
        ha='right', 
        va='bottom', 
        transform=ax.transAxes
    )
    fps, fp_types, fp_colors = get_phi1_fixed_points([p])
    for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
        marker = FP_MARKERS[fp_type]
        ax.plot(
            fp[0], fp[1],
            color=fp_color, 
            marker=marker,
            markersize=3,
            markeredgecolor='w',
            markeredgewidth=get_marker_edge_width(marker),
        )
    if SAVEPLOTS:
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}", bbox_inches='tight')
    plt.close()


##############################################################################
##############################################################################
##  Tilt time course for binary choice

sigparams = np.array([
    [2.0, 0.0, 0.5, 2.0],
    [6.0, 0.5, 0.0, 4.0],
])

ts = np.linspace(0, 8, 1001, endpoint=True)
sigfunc = get_sigmoid_function(*sigparams.T)
taus_timecourse = sigfunc(ts)

key = jrandom.PRNGKey(seed=SEED)

key, subkey = jrandom.split(key, 2)
model, hyperparams = AlgebraicPL.make_model(
    key=subkey,
    dtype=jnp.float64,
    algebraic_phi_id="phi1",
    tilt_weights=[[1, 0],[0, 1]],
    tilt_bias=[0, 0],
    sigma_init=0.2,
    signal_type="sigmoid",
    nsigparams=4,
    dt0=0.05,
)

ncells = 10
x0_val = (0.0, -0.5)
tfin = 8
burnin = 1
dt_save = 0.5

# Initial condition
key, subkey = jrandom.split(key, 2)
x0 = np.zeros([ncells, 2])
x0[:] = x0_val

# Simulate particles in the landscape
key, subkey = jrandom.split(key, 2)
ts_saved, xs_saved, sigs_saved, ps_saved = model.run_landscape_simulation(
    x0, tfin, dt_save, sigparams, subkey, 
    burnin=burnin
)

ts_saved = ts_saved[0]
xs_saved = xs_saved[0]
sigs_saved = sigs_saved[0]
ps_saved = ps_saved[0]

plot_ts = [0, 2, 4, 6, 8]

FIGNAME = "phi1_tau_timecourse"  # appended with index i
FIGSIZE = (6*sf, 4.1*sf)
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

for i in range(2):
    ax.plot(
        ts, taus_timecourse[:,i], 
        label=["$\\tau_1$", "$\\tau_2$"][i],
        color=[CHIR_COLOR, FGF_COLOR][i],
    )
# ax.legend(bbox_to_anchor=(1.05, 1.25), loc='upper left', frameon=False)
ax.legend(loc='lower center')

for plot_t in plot_ts:
    ax.axvline(plot_t, color='k', alpha=0.5, linestyle='--')

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()


FIGNAME = "phi1_sig_timecourse"  # appended with index i
FIGSIZE = (4.0*sf, 2.75*sf)
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')

for i in range(2):
    ax.plot(
        ts, taus_timecourse[:,i], 
        label=["$s_1$", "$s_2$"][i],
        color=[CHIR_COLOR, FGF_COLOR][i]
    )
# ax.legend(loc='lower center')
# ax.legend(bbox_to_anchor=(1.05, 1.25), loc='upper left', frameon=False)

for plot_t in plot_ts:
    ax.axvline(plot_t, color='k', alpha=0.5, linestyle='--')

ax.set_xticks(plot_ts)
ax.set_yticks([])

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()


##############################################################################
##############################################################################
##  Landscape heatmaps along tau timecourse
FIGNAME = "phi1_timecourse_heatmaps"  # appended with index i
FIGSIZE = (2.75*sf, 2.75*sf)

res = 50
r = 2.5

for i, plot_t in enumerate(plot_ts):

    idx_in_saved_results = np.where(ts_saved == plot_t)[0][0]
    x_state = xs_saved[idx_in_saved_results]

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    p = sigfunc(plot_t).flatten()
    plot_landscape(
        func_phi1_star, r=r, res=res, params=p,
        lognormalize=lognormalize,
        clip=clip,
        title="",
        include_cbar=False,
        ncontours=10,
        contour_linewidth=0.5,
        contour_linealpha=0.5,
        xlabel=None,
        ylabel=None,
        xticks=False,
        yticks=False,
        equal_axes=True,
        figsize=FIGSIZE,
        show=True,
        ax=ax,
    )

    fps, fp_types, fp_colors = get_phi1_fixed_points([np.round(p, 6)])
    for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
        marker = FP_MARKERS[fp_type]
        ax.plot(
            fp[0], fp[1],
            color=fp_color, 
            marker=marker,
            markersize=3,
            markeredgecolor='w',
            markeredgewidth=get_marker_edge_width(marker),
        )

    # Plot signal effect inset
    subax = inset_axes(ax,
        width=INSET_SCALE,
        height=INSET_SCALE,
        # loc=3,
        bbox_to_anchor=(0, 0, 0.5, 0.5),
        bbox_transform=ax.transAxes,
    )
    subax.set_aspect('equal')
    subax.axis('off')
    scale = 1.0
    subax.arrow(
        0, 0, -1, 0, 
        width=0.05*scale, 
        fc=CHIR_COLOR, ec=CHIR_COLOR,
        alpha=0.9,
        label="$s_1$",
    )
    subax.arrow(
        0, 0, 0, -1, 
        width=0.05*scale, 
        fc=FGF_COLOR, ec=FGF_COLOR,
        alpha=0.9,
        label="$s_2$",
    )
    subax.arrow(
        0, 0, -p[0], -p[1], 
        width=0.05*scale, 
        fc='k', ec='k',
        alpha=0.8,
        label="$s_2$",
    )
    subax.set_xlim([-scale*1.5, scale*1.5])
    subax.set_ylim([-scale*1.5, scale*1.5])
    
    ax.plot(
        x_state[:,0], x_state[:,1], '.',
        color='cyan',
        markersize=1,
        alpha=0.8,
    )

    if SAVEPLOTS:
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}", transparent=True)
    plt.close()

##############################################################################
##############################################################################
##  Tilted binary flip landscape

FIGNAME = "phi2_heatmap_tilted"  # appended with index i
FIGSIZE = (4*sf, 4*sf)

PARAMS_PHI2 = [
    [ 1.0,  1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [-1.0, -1.0],
]
r = 3       # box radius
res = 50   # resolution
lognormalize = True
clip = None

for i, p in enumerate(PARAMS_PHI2):
    with plt.style.context(TEX_CONTEXT):
        title = f"$\\boldsymbol{{\\tau}}=" + f"({p[0]:.2g},{p[1]:.2g})$"
        ax = plot_landscape(
            func_phi2_star, r=r, res=res, params=p, 
            lognormalize=lognormalize,
            clip=clip,
            title=title,
            include_cbar=False,
            cbar_title="$\ln\phi$",
            ncontours=10,
            contour_linewidth=0.5,
            xlabel=None,
            ylabel=None,
            xticks=False,
            yticks=False,
            equal_axes=True,
            figsize=FIGSIZE,
            show=True,
        )
    fps, fp_types, fp_colors = get_phi2_fixed_points([p])
    for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
        marker = FP_MARKERS[fp_type]
        ax.plot(
            fp[0], fp[1],
            color=fp_color, 
            marker=marker,
            markersize=3,
            markeredgecolor='w',
            markeredgewidth=get_marker_edge_width(marker),
        )
    if SAVEPLOTS:
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}")
    plt.close()

##############################################################################
##############################################################################
##  Bifurcation Diagram for Binary Choice (phi1)

from cont.binary_choice import plot_binary_choice_bifurcation_diagram

# FIGSIZE = (5*sf, 5*sf)
FIGSIZE = (6*sf, 6*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

plot_binary_choice_bifurcation_diagram(
    ax=ax,
    xlabel="$\\tau_1$",
    ylabel="$\\tau_2$",
)

ax.plot(
    taus_timecourse[:,0], taus_timecourse[:,1],
    color='grey', 
    linewidth=1.5,
    alpha=0.8,
    linestyle='-',
)

for p, label_number in PARAMS_PHI1:
    ax.plot(*p, '.k', alpha=1.0, markersize=PARAM_MARKERSIZE)
    ax.text(*p, label_number, fontsize=ANNOTATION_FONTSIZE)

ax.set_yticks([-1, 0, 1, 2, 3])

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi1_bifdiagram", bbox_inches='tight')

plt.close()

##############################################################################
##############################################################################
##  Bifurcation Diagram for Binary Flip (phi2)

from cont.binary_flip import plot_binary_flip_bifurcation_diagram

FIGSIZE = (4*sf, 4*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

plot_binary_flip_bifurcation_diagram(
    ax=ax,
)

for p in PARAMS_PHI2:
    ax.plot(*p, '.k', alpha=1.0, markersize=PARAM_MARKERSIZE)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi2_bifdiagram")

plt.close()

##############################################################################
##############################################################################
