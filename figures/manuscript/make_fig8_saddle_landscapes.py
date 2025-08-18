"""Figure 8 Script (Saddle landscapes)

Generate plots used in Figure 9 of the accompanying manuscript.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize
plt.style.use('figures/manuscript/styles/fig_8.mplstyle')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

from plnn.io import load_model_from_directory
from plnn.models.algebraic_pl import AlgebraicPL
from plnn.pl import CHIR_COLOR, FGF_COLOR

parser = argparse.ArgumentParser()
# parser.add_argument('-o', '--outdir', type=str, required=True)
args = parser.parse_args()

SEED = 42
rng = np.random.default_rng(seed=SEED)
key = jrandom.PRNGKey(rng.integers(2**32))

OUTDIR = f"figures/manuscript/out/fig8_saddle_landscapes"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches


##############################################################################
##############################################################################
##  Configuration and setup

Y0, Y1 = 1, 3

SIG_TO_PLOT = [0, 1]
ULIMS = [-2, 2]
VLIMS = [-2, 2]
XLIMS = [0, 3]
YLIMS = [Y0, Y1]
RES_PLOT = 100
RES_VECT = 20

greys = plt.colormaps.get_cmap('Greys')(np.linspace(0.1, 1.0, 100))
mygreys = LinearSegmentedColormap.from_list("myGreys", greys)
mygreys_r = mygreys.reversed()

CMAP_SURFACE = mygreys_r # plt.colormaps.get_cmap('Greys_r')
CMAP_VFIELD_UVSPACE = mygreys # ListedColormap([1, 1, 1])  # White vectors
# CMAP_VFIELD_PROJ2D = mygreys  # plt.colormaps.get_cmap('Greys')
CMAP_VFIELD_SURFACE = mygreys # plt.colormaps.get_cmap('Greys')
# CMAP_VFIELD_INFERRED = mygreys # plt.colormaps.get_cmap('Greys')
# CMAP_VFIELD_DIFF = plt.colormaps.get_cmap('RdBu')


##############################################################################
##############################################################################
##  Helper functions


XSHIFT, YSHIFT = 1.5, 2

def u_func(x, y):
    return x - XSHIFT

def v_func(x, y):
    return y - YSHIFT

def x_func(u, v):
    return u + XSHIFT

def y_func(u, v):
    return v + YSHIFT

def f0(x, args):
    a0 = args['a0']
    b0 = args['b0']
    t0 = args['t0']
    m0 = args['m0']
    return m0 * x + a0 * jnp.sin(2 * jnp.pi * x / t0) + b0

def f1(x, args):
    a1 = args['a1']
    b1 = args['b1']
    t1 = args['t1']
    m1 = args['m1']
    return m1 * x + a1 * jnp.sin(2 * jnp.pi * x / t1) + b1

def z_func(x, y):
    args = {
        'a0': 0.5, 'a1': 0.75,  # oscillation amplitude
        'b0': 1.5, 'b1': 3.5,   # verticle shift
        't0': 3.0, 't1': 2.0,   # period
        'm0': 1.0, 'm1': 1.0,   # slope
    }
    y0, y1 = Y0, Y1
    z = (y - y1) / (y0 - y1) * f0(x, args) + (y - y0) / (y1 - y0) * f1(x, args)
    return z

func_hx = jax.grad(z_func, 0)
func_hy = jax.grad(z_func, 1)

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

##############################################################################
##############################################################################
##  Computations

# Get the coordinate grid points (x,y,z).
x = np.linspace(*XLIMS, RES_PLOT)
y = np.linspace(*YLIMS, RES_PLOT)
xs, ys = np.meshgrid(x, y)
xys = np.array([xs.flatten(), ys.flatten()]).T
zs = z_func(xs, ys)

# Convert the (x,y,z) coordinate grid points to (u,v) points.
us = u_func(xs, ys)
vs = v_func(xs, ys)
uvs = np.array([us.flatten(), vs.flatten()]).T

# Get embedded vector positions in (x,y,z) coordinates.
x_vect = np.linspace(*XLIMS, RES_VECT, endpoint=True)[1:-1]
y_vect = np.linspace(*YLIMS, RES_VECT, endpoint=True)[1:-1]
xs_vect, ys_vect = np.meshgrid(x_vect, y_vect)
xys_vect = np.array([xs_vect.flatten(), ys_vect.flatten()]).T
zs_vect = z_func(xs_vect, ys_vect)

# Convert the (x,y,z) vector positions to (u,v) vector positions
us_vect = u_func(xs_vect, ys_vect)
vs_vect = v_func(xs_vect, ys_vect)
uvs_vect = np.array([us_vect.flatten(), vs_vect.flatten()]).T

# Compute the value of the potential at (u,v) points and shift to minimum 1.
phi = model_star.phi_with_signal(0, uvs, jnp.array(SIG_TO_PLOT))
phi = phi + 1 - phi.min()

# Compute the gradient of the potential and vector field at the vector positions.
gphi_uv = model_star.grad_phi_with_signal(0, uvs_vect, jnp.array(SIG_TO_PLOT))
f_uv = -gphi_uv[:,:,None]

print(f_uv.shape)

# Compute the jacobian of the embedding
# jac_emb = J_emb(uvs_vect[:,0], uvs_vect[:,1])
# jac_emb = jac_emb.T.swapaxes(1,2)

# Compute the embedded vector field
v_perp = np.array([
    -jax.vmap(func_hx)(xs_vect.flatten(), ys_vect.flatten()), 
    -jax.vmap(func_hy)(xs_vect.flatten(), ys_vect.flatten()), 
    np.ones_like(xs_vect.flatten() + ys_vect.flatten())
]).T
v_perp_norm = np.sqrt(np.sum(v_perp * v_perp, axis=1))
nhat = v_perp / v_perp_norm[:,None]
print(nhat.shape)
projector = np.identity(3) - np.matmul(nhat[:,:,None], nhat[:,None,:])
print(projector.shape)
print(f_uv.shape)
f_uv_ext = np.zeros([f_uv.shape[0], 3, 1])
f_uv_ext[:,0:2,:] = f_uv
f_emb = np.matmul(projector, f_uv_ext).squeeze()

fx_emb, fy_emb, fz_emb = f_emb.T
f_emb_norms = np.sqrt(fx_emb**2 + fy_emb**2 + fz_emb**2)

##############################################################################
##############################################################################
##  Latent space plot u, v

FIGSIZE = (3*sf, 3*sf)
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
    include_tilt_inset=False,
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

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(*ULIMS)
ax.set_ylim(*VLIMS)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/latent_space.pdf")
    plt.close()


##############################################################################
##############################################################################
##  Surface plot

FIGSIZE = (6*sf, 5*sf)
fig = plt.figure(figsize=FIGSIZE, layout="constrained")
ax = fig.add_subplot(111, projection='3d')

# Plot surface
# rgb = CMAP_SURFACE(LogNorm()(phi)).reshape([xs.shape[0], xs.shape[1], -1])
surf = ax.plot_surface(
    xs, ys, zs, 
    alpha=0.5,
    rstride=1, cstride=1, 
    # facecolors=rgb,
    color='grey',
    linewidth=0, antialiased=True, shade=True,
)


# Plot plane
mx, my = 0.5, 0.5
f_plane = lambda x, y: mx * x + my * y - 2
surf = ax.plot_surface(
    xs, ys, f_plane(xs, ys), 
    alpha=0.5, color='grey',
    linewidth=0, antialiased=True, shade=True,
)

# Plot vector field on surface
cnorm = LogNorm(f_emb_norms.min(), f_emb_norms.max())
sc = ax.quiver(
    xs_vect.flatten(), ys_vect.flatten(), zs_vect.flatten(), 
    fx_emb, fy_emb, fz_emb,
    linewidth=0.2,
    length=0.1, 
    normalize=True,
    arrow_length_ratio=0.5,
    pivot='middle',
    zorder=1,
    colors='k',
)

# Plot vector field on plane
cnorm = LogNorm(f_emb_norms.min(), f_emb_norms.max())
sc = ax.quiver(
    xs_vect.flatten(), ys_vect.flatten(), f_plane(xs_vect, ys_vect).flatten(), 
    fx_emb, fy_emb, fz_emb,
    linewidth=0.2,
    length=0.1, 
    normalize=True,
    arrow_length_ratio=0.5,
    pivot='middle',
    zorder=1,
    colors='k',
)

ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.1)
ax.set_ylim(-0.1, ax.get_ylim()[1])
ax.set_zlim(0, 5)
ax.view_init(azim=-80)

# ax.set_title(f"$\kappa_1={k1}$, $\kappa_2={k2}$")
# ax.set_xlabel("$g_1$", labelpad=-15)
# ax.set_ylabel("$g_2$", labelpad=-15)
# ax.set_zlabel("$g_3$", labelpad=-15)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/manifold.pdf")
    plt.close()
