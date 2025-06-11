"""Analysis of transformations

Generates plots used for SI Figure.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figures/supplement/styles/nonlinear_manifolds.mplstyle')
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

from plnn.models.algebraic_pl import AlgebraicPL


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--embedding", type=str, required=True,
                    choices=["embedding1", "embedding2", "embedding3"])
parser.add_argument("-k1", "--kappa1", type=float, required=True)
parser.add_argument("-k2", "--kappa2", type=float, required=True)
parser.add_argument("-ul", "--ulims", type=float, nargs=2, default=None)
parser.add_argument("-vl", "--vlims", type=float, nargs=2, default=None)
parser.add_argument("-p", "--potential", type=str, default=None,
                    choices=["phi1", "phi2", "quadratic"])
args = parser.parse_args()

embedding_key = args.embedding
potential_key = args.potential
include_potential = potential_key is not None

embedding_name = {
    "embedding1": "Embedding 1",
    "embedding2": "Embedding 2",
    "embedding3": "Embedding 3",
}[embedding_key]

k1 = args.kappa1
k2 = args.kappa2

ULIMS_DEFAULT = [-2, 2]
VLIMS_DEFAULT = [-2, 2]
ULIMS = ULIMS_DEFAULT if args.ulims is None else args.ulims
VLIMS = VLIMS_DEFAULT if args.vlims is None else args.vlims

SEED = 42
rng = np.random.default_rng(seed=SEED)
key = jrandom.PRNGKey(rng.integers(2**32))


OUTDIR = f"figures/supplement/out/nonlinear_manifolds/transformations"
SAVEPLOTS = True


##############################################################################
##############################################################################
##  Configuration and setup

SIG_TO_PLOT = [0, 0]

RES_U = 11
RES_V = 11
color_u = 'r'
color_v = 'b'
alpha_u = 0.25
alpha_v = 0.25

greys = plt.colormaps.get_cmap('Greys')(np.linspace(0.1, 1.0, 100))
mygreys = LinearSegmentedColormap.from_list("myGreys", greys)
mygreys_r = mygreys.reversed()

CMAP_SURFACE = mygreys_r # plt.colormaps.get_cmap('Greys_r')


##############################################################################
##############################################################################
##  Load model if specified

if potential_key == "phi1":
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
elif potential_key == "phi2":
    key, subkey = jrandom.split(key, 2)
    model_star, hyperparams = AlgebraicPL.make_model(
        key=subkey,
        dtype=jnp.float64,
        algebraic_phi_id="phi2",
        tilt_weights=[[1, 0],[0, 1]],
        tilt_bias=[0, 0],
        sigma_init=0.1,
        signal_type="sigmoid",
        nsigparams=4,
        dt0=0.05,
    )
elif potential_key == "quadratic":
    a = 1  # TODO: generalize
    b = 1  # TODO: generalize
    key, subkey = jrandom.split(key, 2)
    model_star, hyperparams = AlgebraicPL.make_model(
        key=subkey,
        dtype=jnp.float64,
        algebraic_phi_id="quadratic",
        tilt_weights=[[1, 0],[0, 1]],
        tilt_bias=[0, 0],
        sigma_init=0.1,
        signal_type="sigmoid",
        nsigparams=4,
        dt0=0.05,
        phi_args={'a': a, 'b': b}
    )


##############################################################################
##############################################################################
##  Define embeddings


class Embedding1:

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def u_func(self, x, y):
        return np.arcsinh(np.sqrt(self.k1/2) * x) + \
               np.arcsinh(np.sqrt(self.k2/2) * y)

    def v_func(self, x, y):
        return np.arcsinh(np.sqrt(self.k1/2) * x) - \
               np.arcsinh(np.sqrt(self.k2/2) * y)

    def x_func(self, u, v):
        return np.sqrt(2/self.k1) * np.sinh((u+v)/2)

    def y_func(self, u, v):
        return np.sqrt(2/self.k2) * np.sinh((u-v)/2)

    def z_func(self, x, y):
        return self.k1/2 * x*x - self.k2/2 * y*y

    def jacobian(self, u, v):
        return np.array([
            [np.cosh(0.5*(u + v)) / np.sqrt(2 * self.k1), 
             np.cosh(0.5*(u + v)) / np.sqrt(2 * self.k1),],
            [np.cosh(0.5*(u - v)) / np.sqrt(2 * self.k2), 
            -np.cosh(0.5*(u - v)) / np.sqrt(2 * self.k2),],
            [np.cosh(u) * np.sinh(v), 
             np.sinh(u) * np.cosh(v)],
        ])
    

class Embedding2:

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def u_func(self, x, y):
        a = 1 / np.sqrt(self.k1)
        b = 1 / np.sqrt(self.k2)
        return x/a + y/b

    def v_func(self, x, y):
        a = 1 / np.sqrt(self.k1)
        b = 1 / np.sqrt(self.k2)
        return x/a - y/b

    def x_func(self, u, v):
        return (u + v) / (2 * np.sqrt(self.k1))

    def y_func(self, u, v):
        return (u - v) / (2 * np.sqrt(self.k2))

    def z_func(self, x, y):
        return self.k1/2 * x*x - self.k2/2 * y*y

    def J_emb(self, u, v):
        return np.array([
            [1/(2*np.sqrt(self.k1)) * np.ones_like(u+v),  
             1/(2*np.sqrt(self.k1)) * np.ones_like(u+v)],
            [1/(2*np.sqrt(self.k2)) * np.ones_like(u+v), 
            -1/(2*np.sqrt(self.k2)) * np.ones_like(u+v)],
            [v/2, u/2],
        ])
    

class Embedding3:

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def u_func(self, x, y):
        return np.arcsinh(np.sqrt(self.k1/2) * x)

    def v_func(self, x, y):
        return -np.arcsinh(np.sqrt(self.k2/2) * y)

    def x_func(self, u, v):
        return np.sqrt(2/self.k1) * np.sinh(u)

    def y_func(self, u, v):
        return np.sqrt(2/self.k2) * np.sinh(-v)

    def z_func(self, x, y):
        return self.k1/2 * x*x - self.k2/2 * y*y

    def jacobian(self, u, v):
        return np.array([
            [np.sqrt(2 / self.k1) * np.cosh(u), np.zeros_like(u+v)],
            [np.zeros_like(u+v), -np.sqrt(2 / self.k2) * np.cosh(v)],
            [np.sinh(2*u), -2 * np.sinh(v) * np.cosh(v)],
        ])

J_PI = np.array([[1,0,0],[0,1,0],[0,0,0]])


if embedding_key == "embedding1":
    emb = Embedding1(k1, k2)
elif embedding_key == "embedding2":
    emb = Embedding2(k1, k2)
elif embedding_key == "embedding3":
    emb = Embedding3(k1, k2)
else:
    raise NotImplementedError(f"Embedding '{embedding_key}' not implemented.")

os.makedirs(OUTDIR, exist_ok=True)

##############################################################################
##############################################################################
###  Plot transformed grid lines

FIGSIZE = None
FNAME = f"{embedding_key}_gridlines_{k1}_{k2}"
if include_potential:
    FNAME += f"_{potential_key}"

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=FIGSIZE)
ax1.set_aspect('equal')
ax2.set_aspect('equal')


# Plot gridlines and transformed gridlines
us = np.linspace(*ULIMS, RES_U, endpoint=True)
vs = np.linspace(*VLIMS, RES_V, endpoint=True)

ts = np.linspace(0, 1, 101, endpoint=True)
for i, u in enumerate(us):
    ufunc = lambda t: u * np.ones_like(t)
    vfunc = lambda t: VLIMS[0] * (1 - t) + VLIMS[1] * t
    us_plot = ufunc(ts)
    vs_plot = vfunc(ts)
    xs_plot = emb.x_func(us_plot, vs_plot)
    ys_plot = emb.y_func(us_plot, vs_plot)
    ax1.plot(
        us_plot, vs_plot, 
        color=color_u,
        alpha=alpha_u if i > 0 else 1,
    )
    ax2.plot(
        xs_plot, ys_plot,
        color=color_u,
        alpha=alpha_u if i > 0 else 1,
    )

for i, v in enumerate(vs):
    ufunc = lambda t: ULIMS[0] * (1 - t) + ULIMS[1] * t
    vfunc = lambda t: v * np.ones_like(t)
    us_plot = ufunc(ts)
    vs_plot = vfunc(ts)
    xs_plot = emb.x_func(us_plot, vs_plot)
    ys_plot = emb.y_func(us_plot, vs_plot)
    ax1.plot(
        us_plot, vs_plot, 
        color=color_v,
        alpha=alpha_v if i > 0 else 1,
    )
    ax2.plot(
        xs_plot, ys_plot,
        color=color_v,
        alpha=alpha_v if i > 0 else 1,
    )


# Plot potential if specified
if include_potential:
    model_star.plot_phi(
        signal=SIG_TO_PLOT,
        xrange=ULIMS,
        yrange=VLIMS,
        res=50,
        ax=ax1,
        title="",
        show=True,
        tight_layout=False,
        include_cbar=False,
        include_tilt_inset=False,
    )

    # Determine max value of phi within (u,v) domain
    u = np.linspace(*ax1.get_xlim(), 50)
    v = np.linspace(*ax1.get_ylim(), 50)
    us, vs = np.meshgrid(u, v)
    uvs = np.array([us.flatten(), vs.flatten()]).T
    phi = model_star.phi_with_signal(0, uvs, jnp.array(SIG_TO_PLOT))
    phi_max = phi.max()

    xlims = ax2.get_xlim()
    ylims = ax2.get_ylim()
    x = np.linspace(*xlims, 50)
    y = np.linspace(*ylims, 50)
    xs, ys = np.meshgrid(x, y)
    us = emb.u_func(xs, ys)
    vs = emb.v_func(xs, ys)
    uvs = np.array([us.flatten(), vs.flatten()]).T
    phi = model_star.phi_with_signal(0, uvs, jnp.array(SIG_TO_PLOT))
    phi = 1 + phi - phi.min()
    phi = np.where(phi <= phi_max, phi, np.nan)

    # Plot the transformed potential
    sc = ax2.pcolormesh(
        xs, ys, phi.reshape(xs.shape),
        cmap=CMAP_SURFACE,
        norm=LogNorm(), 
        shading="gouraud",
    )

# Format axes
ax1.set_xlabel(f"$u$")
ax1.set_ylabel(f"$v$")
ax1.set_xlim(ULIMS[0] - 0.01, ULIMS[1] + 0.01)
ax1.set_ylim(VLIMS[0] - 0.01, VLIMS[1] + 0.01)

ax2.set_xlabel(f"$x$")
ax2.set_ylabel(f"$y$")

# fig.suptitle(embedding_name)
ax2.set_title(f"$\\kappa_1={k1:.3g}$, $\\kappa_2={k2:.3g}$")


if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FNAME}.pdf", bbox_inches="tight")
    plt.close()
