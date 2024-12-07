"""Figure binary-flip

"""

import os
import matplotlib.pyplot as plt
plt.style.use('figures/supplement/styles/fig_standard.mplstyle')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plnn.pl import plot_landscape
from plnn.helpers import get_phi2_fixed_points
from plnn.pl import CHIR_COLOR, FGF_COLOR


OUTDIR = "figures/supplement/out/fig_binary_flip"
SAVEPLOTS = True

SEED = 9276578238

TEX_CONTEXT = 'figures/supplement/styles/fig_1_tex.mplstyle'

LEGEND_FONTSIZE = 8
INSET_SCALE = "30%"

os.makedirs(OUTDIR, exist_ok=True)

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

TILT_TO_PLOT = [0., 0.]

##############################################################################
##############################################################################
##  Untilted binary flip landscape.

FIGNAME1 = "phi2_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi2_landscape_untilted"
FIGSIZE2 = (6*sf, 6*sf)

r = 2.5       # box radius
res = 50   # resolution
lognormalize = True
clip = None

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE1, layout='constrained')

fps, fp_types, fp_colors = get_phi2_fixed_points([TILT_TO_PLOT])

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=TILT_TO_PLOT, 
    lognormalize=lognormalize,
    clip=clip,
    title=f"",
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    # cbar_title="$\log\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=None,
    figsize=FIGSIZE1,
    tight_layout=False,
    ax=ax,
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
        markeredgewidth=0.2 if marker == 'o' else 0.5,
    )

ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

# Plot signal effect inset
subax = inset_axes(ax,
    width=INSET_SCALE,
    height=INSET_SCALE,
    loc=3,
    # bbox_to_anchor=(0, 0, 1, 1),
    # bbox_transform=ax.transAxes,
)
subax.set_aspect('equal')
subax.axis('off')
subax.set_xlim([-1.25, 1.25])
subax.set_ylim([-1.25, 1.25])
tilt_arrow_width = 0.001
subax.arrow(
    0, 0, -1, 0, 
    width=tilt_arrow_width, 
    head_width=100*tilt_arrow_width, 
    length_includes_head=True, 
    fc=CHIR_COLOR, ec=CHIR_COLOR,
    label="$\\tau_1$"
)
subax.arrow(
    0, 0, 0, -1, 
    width=tilt_arrow_width, 
    head_width=100*tilt_arrow_width, 
    length_includes_head=True, 
    fc=FGF_COLOR, ec=FGF_COLOR,
    label="$\\tau_2$"
)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME1}")
plt.close()

# fig, ax = plt.subplots(1, 1, figsize=FIGSIZE2, layout='constrained')

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=TILT_TO_PLOT, 
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
    view_init=[35, -80],  # elevation, aximuthal 
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    equal_axes=True,
    tight_layout=True,
    figsize=FIGSIZE2,
    show=True,
)
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
##  Tilted binary flip landscape

FIGNAME = "phi2_heatmap_tilted"  # appended with index i
FIGSIZE = (4*sf, 4*sf)

PARAMS_PHI2 = [
    ([ 0,    0], 0),
    ([-0.75, 0.25], 1),
    ([-0.75, -0.25], 2),
    ([-1.0,  1.0], 3),
    ([-1.0, -1.0], 4),
    ([-1.5,  0.5], 5),
    ([-1.5, -0.5], 6),
]
r = 4       # box radius
res = 100   # resolution
lognormalize = True
clip = None

for i, (p, label_number) in enumerate(PARAMS_PHI2):
    with plt.style.context(TEX_CONTEXT):
        title = f"$\\boldsymbol{{\\tau}}=" + f"({p[0]:.2g},{p[1]:.2g})$"
        ax = plot_landscape(
            func_phi2_star, r=r, res=res, params=p, 
            lognormalize=lognormalize,
            clip=clip,
            title=title,
            include_cbar=False,
            cbar_title="$\log\phi$",
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
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}", bbox_inches='tight')
    plt.close()

##############################################################################
##############################################################################
##  Bifurcation Diagram for Binary Flip (phi2)

from cont.binary_flip import plot_binary_flip_bifurcation_diagram

FIGSIZE = (4*sf, 4*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

plot_binary_flip_bifurcation_diagram(
    ax=ax,
    xlabel="$\\tau_1$",
    ylabel="$\\tau_2$",
    add_flip_curves=True,
)

for p, label_number in PARAMS_PHI2:
    ax.plot(*p, '.k', alpha=1.0, markersize=PARAM_MARKERSIZE)
    ax.text(*p, label_number, fontsize=ANNOTATION_FONTSIZE)

ax.set_yticks([-1, 0, 1, 2, 3])

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi2_bifdiagram", bbox_inches='tight')

plt.close()

##############################################################################
##############################################################################
