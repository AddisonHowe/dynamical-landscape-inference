"""General theory of landscape inference on nonlinear manifolds

Generates plots used for SI Figure.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize
plt.style.use('figures/supplement/styles/nonlinear_manifolds.mplstyle')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

from plnn.io import load_model_from_directory
from plnn.models.algebraic_pl import AlgebraicPL
from plnn.pl import plot_phi, plot_f, CHIR_COLOR, FGF_COLOR

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outdir', type=str, required=True)
parser.add_argument('-k1', '--kappa1', type=int, required=True)
parser.add_argument('-k2', '--kappa2', type=int, required=True)
parser.add_argument('-m', '--modeldir', type=str, required=True)
parser.add_argument('-xl', '--xlims', type=float, nargs=2, default=None)
parser.add_argument('-yl', '--ylims', type=float, nargs=2, default=None)
parser.add_argument('-ul', '--ulims', type=float, nargs=2, default=None)
parser.add_argument('-vl', '--vlims', type=float, nargs=2, default=None)
args = parser.parse_args()

K1 = args.kappa1
K2 = args.kappa2

ULIMS_DEFAULT = [-2, 2]
VLIMS_DEFAULT = [-2, 2]
XLIMS_DEFAULT = [-3, 3]
YLIMS_DEFAULT = [-3, 3]

ULIMS = ULIMS_DEFAULT if args.ulims is None else args.ulims
VLIMS = VLIMS_DEFAULT if args.vlims is None else args.vlims
XLIMS = XLIMS_DEFAULT if args.xlims is None else args.xlims
YLIMS = YLIMS_DEFAULT if args.ylims is None else args.ylims

SEED = 42
rng = np.random.default_rng(seed=SEED)
key = jrandom.PRNGKey(rng.integers(2**32))

OUTDIR = f"figures/supplement/out/nonlinear_manifolds/overview/{args.outdir}"
SAVEPLOTS = True
DATDIR = f"data/trained_models/distorted_plnn_synbindec"

modeldir = f"{DATDIR}/{args.modeldir}"

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches


##############################################################################
##############################################################################
##  Configuration and setup

SIG_TO_PLOT = [0, 0]
RES_PLOT = 100
RES_VECT = 20

greys = plt.colormaps.get_cmap('Greys')(np.linspace(0.1, 1.0, 100))
mygreys = LinearSegmentedColormap.from_list("myGreys", greys)
mygreys_r = mygreys.reversed()

CMAP_SURFACE = mygreys_r # plt.colormaps.get_cmap('Greys_r')
CMAP_VFIELD_UVSPACE = mygreys # ListedColormap([1, 1, 1])  # White vectors
CMAP_VFIELD_PROJ2D = mygreys  # plt.colormaps.get_cmap('Greys')
CMAP_VFIELD_SURFACE = mygreys # plt.colormaps.get_cmap('Greys')
CMAP_VFIELD_INFERRED = mygreys # plt.colormaps.get_cmap('Greys')
CMAP_VFIELD_EXPECTED = mygreys # plt.colormaps.get_cmap('Greys')
CMAP_VFIELD_DIFF = plt.colormaps.get_cmap('RdYlBu')

CONTOUR_COLOR = 'purple'

# Options for circle overlay
R_CIRCLE = 1.5
THETAS = np.linspace(0, 2*np.pi, 1000)
circ_us = R_CIRCLE * np.cos(THETAS)
circ_vs = R_CIRCLE * np.sin(THETAS)
circ_col = 'gold'
circ_linestyle = ':'
circ_linewidth = 0.5
circ_idx0 = int(len(THETAS) / 2)
circ_idx1 = int(3 * len(THETAS) / 4)
circ_mrk0 = 'o'
circ_mrk1 = 'o'
circ_mrkcol0 = CHIR_COLOR
circ_mrkcol1 = FGF_COLOR


##############################################################################
##############################################################################
##  Helper functions

def u_func(x, y, k1, k2):
    return np.arcsinh(np.sqrt(k1/2)*x) + np.arcsinh(np.sqrt(k2/2)*y)

def v_func(x, y, k1, k2):
    return np.arcsinh(np.sqrt(k1/2)*x) - np.arcsinh(np.sqrt(k2/2)*y)

def x_func(u, v, k1, k2):
    return np.sqrt(2/k1) * np.sinh((u+v)/2)

def y_func(u, v, k1, k2):
    return np.sqrt(2/k2) * np.sinh((u-v)/2)

def z_func(x, y, k1, k2):
    return k1/2 * x*x - k2/2 * y*y

def J_emb(u, v, k1, k2):
    return np.array([
        [np.cosh(0.5*(u + v)) / np.sqrt(2 * k1), 
         np.cosh(0.5*(u + v)) / np.sqrt(2 * k1),],
        [np.cosh(0.5*(u - v)) / np.sqrt(2 * k2), 
         -np.cosh(0.5*(u - v)) / np.sqrt(2 * k2),],
        [np.cosh(u) * np.sinh(v), 
         np.sinh(u) * np.cosh(v)],
    ])

J_PI = np.array([[1,0,0],[0,1,0],[0,0,0]])


def plot_circle_2d(xs, ys, ax):
    ax.plot(
        xs, ys,
        color=circ_col,
        linestyle=circ_linestyle,
        linewidth=circ_linewidth,
    )
    ax.plot(
        xs[circ_idx0], ys[circ_idx0],
        marker=circ_mrk0,
        color=circ_mrkcol0,
        markersize=1,
    )
    ax.plot(
        xs[circ_idx1], ys[circ_idx1],
        marker=circ_mrk1,
        color=circ_mrkcol1,
        markersize=1,
    )
    # ax.plot(
    #     xs, ys*0,
    #     color=circ_col,
    #     linestyle=circ_linestyle,
    #     linewidth=circ_linewidth,
    #     alpha=0.5,
    # )
    # ax.plot(
    #     0*xs, ys,
    #     color=circ_col,
    #     linestyle=circ_linestyle,
    #     linewidth=circ_linewidth,
    #     alpha=0.5,
    # )
    return


def plot_circle_3d(xs, ys, zs, ax):
    ax.plot(
        xs, ys, zs, 
        color=circ_col,
        linestyle=circ_linestyle,
        linewidth=circ_linewidth,
    )
    ax.plot(
        xs[circ_idx0], ys[circ_idx0], zs[circ_idx0], 
        marker=circ_mrk0,
        color=circ_mrkcol0,
        markersize=1,
    )
    ax.plot(
        xs[circ_idx1], ys[circ_idx1], zs[circ_idx1], 
        marker=circ_mrk1,
        color=circ_mrkcol1,
        markersize=1,
    )


##############################################################################
##############################################################################
##  Load and construct models

key, subkey = jrandom.split(key, 2)
model_star, hyperparams = AlgebraicPL.make_model(
    key=subkey,
    dtype=jnp.float64,
    algebraic_phi_id="phi1",
    tilt_weights=[[1, 0],[0, 1]],
    tilt_bias=[0, 0],
    sigma_init=0.1,
    signal_type="sigmoid",
    nsigparams=4,
    dt0=0.05,
)

# model_star, hyperparams = AlgebraicPL.make_model(
#     key=subkey,
#     dtype=jnp.float64,
#     algebraic_phi_id="quadratic",
#     phi_args={'a':1, 'b': 2},
#     tilt_weights=[[1, 0],[0, 1]],
#     tilt_bias=[0, 0],
#     sigma_init=0.1,
#     signal_type="sigmoid",
#     nsigparams=4,
#     dt0=0.05,
# )

model, _, epoch_loaded, _, _ = load_model_from_directory(
    modeldir
)

##############################################################################
##############################################################################
##  Computations

DEBUG_IDX = 45

k1, k2 = K1, K2

# Coordinates of the embedded circle
circ_xs = x_func(circ_us, circ_vs, k1, k2)
circ_ys = y_func(circ_us, circ_vs, k1, k2)
circ_zs = z_func(circ_xs, circ_ys, k1, k2)

# Get the coordinate grid points (x,y,z).
x = np.linspace(*XLIMS, RES_PLOT)
y = np.linspace(*YLIMS, RES_PLOT)
xs, ys = np.meshgrid(x, y)
xys = np.array([xs.flatten(), ys.flatten()]).T
zs = z_func(xs, ys, k1, k2)

# Convert the (x,y,z) coordinate grid points to (u,v) points.
us = u_func(xs, ys, k1, k2)
vs = v_func(xs, ys, k1, k2)
uvs = np.array([us.flatten(), vs.flatten()]).T

# Get embedded vector positions in (x,y,z) coordinates.
x_vect = np.linspace(*XLIMS, RES_VECT)
y_vect = np.linspace(*YLIMS, RES_VECT)
xs_vect, ys_vect = np.meshgrid(x_vect, y_vect)
xys_vect = np.array([xs_vect.flatten(), ys_vect.flatten()]).T
zs_vect = z_func(xs_vect, ys_vect, k1, k2)
print("(x,y,z): ", *xys_vect[DEBUG_IDX], zs_vect.flatten()[DEBUG_IDX])

# Convert the (x,y,z) vector positions to (u,v) vector positions
us_vect = u_func(xs_vect, ys_vect, k1, k2)
vs_vect = v_func(xs_vect, ys_vect, k1, k2)
uvs_vect = np.array([us_vect.flatten(), vs_vect.flatten()]).T
print("(u,v): ", *uvs_vect[DEBUG_IDX])

# Compute the value of the potential at (u,v) points and shift to minimum 1.
phi = model_star.phi_with_signal(0, uvs, jnp.array(SIG_TO_PLOT))
phi = phi + 1 - phi.min()

# Compute the gradient of the potential and vector field at the vector positions.
gphi_uv = model_star.grad_phi_with_signal(0, uvs_vect, jnp.array(SIG_TO_PLOT))
f_uv = -gphi_uv[:,:,None]
print("Shape of f_uv:", f_uv.shape)
print(f_uv[DEBUG_IDX])

# Compute the gradient of the inferred potential and vector field.
gphi_inf = model.grad_phi_with_signal(0, xys_vect, jnp.array(SIG_TO_PLOT))
f_inf = -gphi_inf
fx_inf, fy_inf = f_inf.T
f_inf_norms = np.sqrt(fx_inf**2 + fy_inf**2)
print("Shape of f_inf:", f_inf.shape)
print(f_inf[DEBUG_IDX])

# Compute the jacobian of the embedding
jac_emb = J_emb(uvs_vect[:,0], uvs_vect[:,1], k1, k2)
jac_emb = jac_emb.T.swapaxes(1,2)
print("Shape of jac_emb:", jac_emb.shape)
print(jac_emb[DEBUG_IDX])

# Compute the jacobian of the complete transformation (embedding + projection)
jac_trans = J_PI @ jac_emb
print("Shape of jac_trans:", jac_trans.shape)
print(jac_trans[DEBUG_IDX])

# Compute the embedded vector field
f_emb = np.matmul(jac_emb, f_uv).squeeze()
fx_emb, fy_emb, fz_emb = f_emb.T
f_emb_norms = np.sqrt(fx_emb**2 + fy_emb**2 + fz_emb**2)
print("Shape of f_emb:", f_emb.shape)
print(f_emb[DEBUG_IDX])
print("norm:", f_emb_norms[DEBUG_IDX])

# Compute the projected vector field
f_proj = np.matmul(jac_trans, f_uv).squeeze()
fx_proj, fy_proj, fz_proj = f_proj.T
f_proj_norms = np.sqrt(fx_proj**2 + fy_proj**2 + fz_proj**2)
print("Shape of f_proj:", f_proj.shape)
print(f_proj[DEBUG_IDX])
print("norm:", f_proj_norms[DEBUG_IDX])
assert np.allclose(fz_proj, 0), "fz_proj should all be 0"

# Compute the ratio between the norms of the projected and embedded vectors.
proj_scaling_factor = f_proj_norms / f_emb_norms
print("Shape of proj_scaling_factor:", proj_scaling_factor.shape)
print(proj_scaling_factor[DEBUG_IDX])

# Compute the gradient of the transformed potential psi(x,y,z)
jac_trans_inv = np.linalg.inv(jac_trans[:,:-1,:])
f_psi = np.matmul(jac_trans_inv.swapaxes(1, 2), f_uv).squeeze()
fx_psi, fy_psi = f_psi.T
f_psi_norms = np.sqrt(fx_psi**2 + fy_psi**2)

# Compute the inverse metric tensor G.
ginv = jac_trans @ jac_trans.swapaxes(1,2)
print("Shape of ginv:", ginv.shape)
print(ginv[DEBUG_IDX])

# Compute the transformed vector field using the metric tensor ginv.
# Note that this matches the projected vector field, as expected.
f_trans = ginv @ np.array([
    [fx_psi.flatten(), fy_psi.flatten(), 0*fy_psi.flatten()]
]).T
f_trans = f_trans.squeeze()
fx_trans, fy_trans, fz_trans = f_trans.T
f_trans_norms = np.sqrt(fx_trans**2 + fy_trans**2 + fz_trans**2)
assert np.allclose(f_trans, f_proj)


##############################################################################
##############################################################################
##  Latent space plot u, v

FIGSIZE = (5*sf, 5*sf)
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

model_star.plot_phi(
    signal=SIG_TO_PLOT,
    xrange=ULIMS,
    yrange=VLIMS,
    res=50,
    ax=ax,
    title="",
    show=True,
    tight_layout=False,
    include_cbar=False,
    include_tilt_inset=True,
)

model_star.plot_f(
    signal=SIG_TO_PLOT,
    xrange=ULIMS,
    yrange=VLIMS,
    res=RES_VECT,
    ax=ax,
    title="",
    show=True,
    tight_layout=False,
    include_cbar=False,
    cmap=CMAP_VFIELD_UVSPACE,
)

# Plot a circle for perspective
plot_circle_2d(circ_us, circ_vs, ax)

ax.set_xlabel("$u$")
ax.set_ylabel("$v$")
#ax.set_xticks([])
#ax.set_yticks([])
ax.set_xlim(*ULIMS)
ax.set_ylim(*VLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/latent_space.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Transformed potential in 2d

FIGSIZE = (3.5*sf, 3.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

# Plot the transformed potential
sc = ax.pcolormesh(
    xs, ys, phi.reshape(xs.shape),
    cmap=CMAP_SURFACE,
    norm=LogNorm(), 
    shading="gouraud",
)

# Plot contours of the potential
ax.contour(
    xs, ys, phi.reshape(xs.shape),
    alpha=0.7,
    colors='k',
    linewidths=0.5,
    linestyles='--', 
)

# Plot the circle
plot_circle_2d(circ_xs, circ_ys, ax)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

ax.set_xlim(*XLIMS)
ax.set_ylim(*YLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/transformed_potential.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Transformed potential in 2d with gradient vector field

FIGSIZE = (3.5*sf, 3.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

# Plot the transformed potential
sc = ax.pcolormesh(
    xs, ys, phi.reshape(xs.shape),
    cmap=CMAP_SURFACE,
    norm=LogNorm(), 
    shading="gouraud",
    alpha=0.5
)

# Plot contours of the potential
ax.contour(
    xs, ys, phi.reshape(xs.shape),
    alpha=0.7,
    colors=CONTOUR_COLOR,
    linewidths=0.5,
    linestyles='--', 
)

# Plot the gradient vector field.
cnorm = LogNorm(f_psi_norms.min(), f_psi_norms.max())
colors = CMAP_VFIELD_PROJ2D(cnorm(f_psi_norms))
ax.quiver(
    xs_vect, ys_vect,
    fx_psi/f_psi_norms, fy_psi/f_psi_norms, 
    angles='xy', scale_units='xy',
    color=CONTOUR_COLOR,
    zorder=3,
)

# Plot the circle
plot_circle_2d(circ_xs, circ_ys, ax)

# ax.set_title("$-\\nabla\\varphi(x',y')$")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

ax.set_xlim(*XLIMS)
ax.set_ylim(*YLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/transformed_gradient.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Transformed potential in 2d with gradient vector field vs transformed field

FIGSIZE = (3.5*sf, 3.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

# Plot the transformed potential
sc = ax.pcolormesh(
    xs, ys, phi.reshape(xs.shape),
    cmap=CMAP_SURFACE,
    norm=LogNorm(), 
    shading="gouraud",
    alpha=0.5
)

# Plot contours of the potential
ax.contour(
    xs, ys, phi.reshape(xs.shape),
    alpha=0.25,
    colors=CONTOUR_COLOR,
    linewidths=0.5,
    linestyles='--', 
)

# Plot the gradient vector field.
# cnorm = LogNorm(f_psi_norms.min(), f_psi_norms.max())
# colors = CMAP_VFIELD_PROJ2D(cnorm(f_psi_norms))
ax.quiver(
    xs_vect, ys_vect,
    fx_psi/f_psi_norms, fy_psi/f_psi_norms, 
    angles='xy', scale_units='xy',
    color=CONTOUR_COLOR,
    zorder=3,
    alpha=0.5
)

# Plot the transformed vector field.
cnorm = LogNorm(f_trans_norms.min(), f_trans_norms.max())
colors = CMAP_VFIELD_PROJ2D(cnorm(f_trans_norms))
ax.quiver(
    xs_vect, ys_vect,
    fx_trans/f_trans_norms, fy_trans/f_trans_norms, 
    angles='xy',
    color=colors,
    zorder=3,
)

# Plot the circle
plot_circle_2d(circ_xs, circ_ys, ax)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

ax.set_xlim(*XLIMS)
ax.set_ylim(*YLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/projected_field_vs_transformed_gradient.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Saddle plot

FIGSIZE = (5*sf, 5*sf)
fig = plt.figure(figsize=FIGSIZE, layout="constrained")
ax = fig.add_subplot(111, projection='3d')

# Plot the saddle
rgb = CMAP_SURFACE(LogNorm()(phi)).reshape([xs.shape[0], xs.shape[1], -1])
surf = ax.plot_surface(
    xs, ys, zs, 
    alpha=0.7,
    rstride=1, cstride=1, facecolors=rgb,
    linewidth=0, antialiased=False, shade=False,
)

# Plot the transformed circle
plot_circle_3d(circ_xs, circ_ys, circ_zs, ax)

# Plot the embedded vector field
cnorm = LogNorm(f_emb_norms.min(), f_emb_norms.max())
colors = CMAP_VFIELD_SURFACE(cnorm(f_emb_norms))
sc = ax.quiver(
    xs_vect.flatten(), ys_vect.flatten(), zs_vect.flatten(), 
    fx_emb, fy_emb, fz_emb,
    linewidth=0.1,
    length=0.2, 
    normalize=True,
    arrow_length_ratio=0.3,
    pivot='middle',
    colors=colors,
    alpha = np.where(ys_vect.flatten() < 1.5, 1., 0.1),
)

ax.view_init(azim=-75, elev=50)

ax.set_title("")
# ax.set_title(f"$\kappa_1={k1}$, $\kappa_2={k2}$")
# ax.set_xlabel("$x$", labelpad=-15)
# ax.set_ylabel("$y$", labelpad=-15)
# ax.set_zlabel("$z$", labelpad=-15)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/saddle_manifold.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Projected field in 2d

FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

# Plot the projected field
cnorm = LogNorm(f_proj_norms.min(), f_proj_norms.max())
colors = CMAP_VFIELD_PROJ2D(cnorm(f_proj_norms))
ax.quiver(
    xs_vect, ys_vect,
    fx_proj/f_proj_norms, fy_proj/f_proj_norms, 
    angles='xy',
    color=colors,
    zorder=3,
)

# Plot the circle
plot_circle_2d(circ_xs, circ_ys, ax)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

ax.set_xlim(*XLIMS)
ax.set_ylim(*YLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/projected_field_2d.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Projected vector field in 3d

FIGSIZE = (4*sf, 4*sf)

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111, projection='3d')  # 3D axes

# Plot the projected vector field
cnorm = LogNorm(f_proj_norms.min(), f_proj_norms.max())
colors = CMAP_VFIELD_PROJ2D(cnorm(f_proj_norms))
ax.quiver(
    xs_vect.flatten(), ys_vect.flatten(), 0*zs_vect.flatten(), 
    fx_proj, fy_proj, fz_proj, 
    length=0.25, normalize=True,
    linewidth=0.2,
    color=colors,
)

# Plot the circle
plot_circle_3d(circ_xs, circ_ys, 0*circ_zs, ax)

ax.set_xlabel("$x$", labelpad=-15)
ax.set_ylabel("$y$", labelpad=-15)
ax.set_zlabel("$z$", labelpad=-15)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
ax.set_zlim(-1, 1)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/projected_field_3d.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Projected field with comparison to embedded field

FIGSIZE = (3.5*sf, 3.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")


# Plot the projected field
cnorm = LogNorm(f_proj_norms.min(), f_proj_norms.max())
colors = CMAP_VFIELD_PROJ2D(cnorm(f_proj_norms))
ax.quiver(
    xs_vect, ys_vect,
    fx_proj/f_proj_norms, fy_proj/f_proj_norms, 
    angles='xy', scale=5, scale_units='xy',
    color=colors,
    alpha=0.9,
    zorder=3,
)

# Plot a vector showing relative size of the embedded vector
ax.quiver(
    xs_vect, ys_vect,
    fx_proj/f_proj_norms/proj_scaling_factor, fy_proj/f_proj_norms/proj_scaling_factor, 
    angles='xy', scale=5, scale_units='xy',
    width=0.005,
    headwidth=0,
    headlength=0,
    headaxislength=0,
    color='r',
    alpha=0.25,
    zorder=2
)

# Plot the circle
plot_circle_2d(circ_xs, circ_ys, ax)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

ax.set_xlim(*XLIMS)
ax.set_ylim(*YLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/projected_field_vs_embedding.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Transformed potential field via metric ginv

FIGSIZE = (3.5*sf, 3.5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

cnorm = LogNorm(f_trans_norms.min(), f_trans_norms.max())
colors = CMAP_VFIELD_PROJ2D(cnorm(f_trans_norms))
ax.quiver(
    xs_vect, ys_vect,
    fx_trans/f_trans_norms, fy_trans/f_trans_norms, 
    color=colors,
)

# Plot the circle
plot_circle_2d(circ_xs, circ_ys, ax)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

ax.set_xlim(*XLIMS)
ax.set_ylim(*YLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/metric_field.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Inferred vector field and potential

FIGSIZE = (5*sf, 3.75*sf)
SIG_TO_PLOT = [0, 1]

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

# Plot the inferred potential
model.plot_phi(
    signal=SIG_TO_PLOT,
    xrange=XLIMS,
    yrange=YLIMS,
    res=50,
    show=True,
    include_tilt_inset=True,
    include_cbar=False,
    cbar_title="",
    title="",
    # xlabel="$x$", ylabel=None,
    # xticks=False, yticks=False,
    ax=ax,
    tight_layout=False,
)

# Plot the projected vector field, i.e. the expected field
cnorm = LogNorm(f_trans_norms.min(), f_trans_norms.max())
colors = CMAP_VFIELD_EXPECTED(cnorm(f_trans_norms))
ax.quiver(
    xs_vect, ys_vect,
    f_trans[:,0]/f_trans_norms, f_trans[:,1]/f_trans_norms, 
    color=colors,
    # alpha=0.5,
)

# Plot the inferred vector field
diff_in_norms = np.abs((f_inf_norms - f_trans_norms)/f_trans_norms)
print("***", diff_in_norms.min(), diff_in_norms.max())
cnorm = Normalize(vmin=0, vmax=diff_in_norms.max())
colors = CMAP_VFIELD_DIFF(cnorm(diff_in_norms))
sc = ax.quiver(
    xs_vect, ys_vect,
    fx_inf/f_inf_norms, fy_inf/f_inf_norms, 
    color=colors,
    cmap=CMAP_VFIELD_DIFF,
    norm=cnorm,
    # alpha=0.5,
)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(sc, cax=cax)
# cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
# cbar.ax.tick_params(labelsize=cbar_ticklabelsize)

# Plot the circle
plot_circle_2d(circ_xs, circ_ys, ax)

ax.set_xlabel("")
ax.set_ylabel("")
# ax.set_title(f"$\kappa_1={k1}$, $\kappa_2={k2}$")

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi_inferred.pdf")
    plt.close()
