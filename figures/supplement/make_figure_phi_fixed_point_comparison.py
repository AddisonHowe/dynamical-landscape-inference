"""Figure phi-fixed-point-comparison

Generate plots used in Figures phi[n]-fixed-point-comparison of the supplement.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figures/supplement/styles/fig_standard.mplstyle')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.optimize import minimize

import equinox as eqx

from plnn.io import load_model_from_directory, load_model_training_metadata
from plnn.models import AlgebraicPL
from plnn.loss_functions import cdist


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, 
                    help="Name of trained model directory w/o prefix.")
parser.add_argument('-p', '--phi', type=str, required=True, 
                    choices=['phi1', 'phi2'], 
                    help="Name of output directory. Defaults to model name")
parser.add_argument('-s', '--signal', type=float, nargs=2, required=True, 
                    help="2D signal value")
parser.add_argument('-o', '--outdir', type=str, default=None, 
                    help="Name of output directory. Defaults to model name")
parser.add_argument('--truesigma', type=float, required=True)
parser.add_argument('--logloss', default=True, 
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--startidx', default=0, type=int)
args = parser.parse_args()


modeldir = args.input  # trained model directory
modelname = modeldir[0:-16]
outdir = modeldir if args.outdir is None else args.outdir

logloss = args.logloss
startidx = args.startidx
truesigma = args.truesigma
phi_true = args.phi
signal = args.signal

SAVEPLOTS = True

MODELDIRBASE = "data/trained_models/plnn_synbindec"
OUTDIRBASE = "figures/supplement/out/fig_phi_fixed_point_comparison"
SEED = 5747283
rng = np.random.default_rng(seed=SEED)

ARR_SCALE = 2
ARR_HEAD_L = 0.1
ARR_HEAD_W = 0.1
COL_TRUE = 'k'
COL_INF = 'r'

OUTDIR = f"{OUTDIRBASE}/{outdir}/"

os.makedirs(OUTDIR, exist_ok=True)

sf = 1/2.54  # scale factor from [cm] to inches

##############################################################################
##############################################################################
##  Load model and training information

MODELDIR = f"{MODELDIRBASE}/{modeldir}"

model, _, _, _, _ = load_model_from_directory(MODELDIR)
logged_args, training_info = load_model_training_metadata(MODELDIR)

# tilt_weights = model.get_parameters()['tilt.w'][0]
# tilt_bias = model.get_parameters()['tilt.b'][0]
# noise_parameter = model.get_sigma()

# if tilt_bias is None:
#     tilt_bias = np.zeros(tilt_weights.shape[0])
# else:
#     tilt_bias = tilt_bias[0]

# def signal_to_tilts(signal):
#     return np.dot(tilt_weights, signal) + tilt_bias

# def tilts_to_signals(tilts):
#     assert tilts.shape[0] == 2
#     y = tilts - tilt_bias[:,None]
#     return np.linalg.solve(tilt_weights, y)


model_star, _ = AlgebraicPL.make_model(
    key=jrandom.PRNGKey(0),
    dtype=jnp.float64,
    algebraic_phi_id=phi_true,
    tilt_weights=[[1, 0],[0, 1]],
    tilt_bias=[0, 0],
    sigma_init=truesigma,
    signal_type="sigmoid",
    nsigparams=4,
)


##############################################################################
##############################################################################
##  Compute...

def get_phi(model, signal):
    def phi(x):
        return model.eval_phi_with_signal(0, x, signal)
    return phi

xrange = [-2, 2]
yrange = [-2, 2]

NSTARTS = 1000
starting_points = np.zeros([NSTARTS, 2])
starting_points[:,0] = np.random.uniform(*xrange, NSTARTS)
starting_points[:,1] = np.random.uniform(*yrange, NSTARTS)

phi = get_phi(model, np.array(signal))
phi_star = get_phi(model_star, np.array(signal))

sigma = model.get_sigma()
sigma_star = model_star.get_sigma()

jacobian = jax.jacfwd(phi)  # opposite of dynamics F
hess = jax.hessian(phi)  # opposite of jacobian J_{F}
jacobian_star = jax.jacfwd(phi_star)
hess_star = jax.hessian(phi_star)


minima = []
vals = []
for x0 in starting_points:
    minres = eqx.filter_jit(minimize)(
        phi, x0,
        method='BFGS',
    )
    if minres.success:
        minima.append(minres.x)
        vals.append(minres.fun)
minima = np.array(minima)
vals = np.array(vals)


minima_star = []
vals_star = []
for x0 in starting_points:
    minres = eqx.filter_jit(minimize)(
        phi_star, x0,
        method='BFGS',
    )
    if minres.success:
        minima_star.append(minres.x)
        vals_star.append(minres.fun)
minima_star = np.array(minima_star)
vals_star = np.array(vals_star)


tol = 1e-4
    

def get_unique_points(xs, tol):
    dists = cdist(xs, xs)
    idxs = jnp.triu_indices(len(dists))
    dists = dists.at[idxs].set(jnp.inf)
    unique_xs = jnp.all(dists > tol, axis=1)
    return xs[unique_xs, :]


def filter_unique_points(
        xs,
        tol,
        block_size=100,
):
    npoints = len(xs)
    nblocks = (npoints // block_size) + (npoints % block_size != 0)
    unique_xs = []
    for i in range(nblocks):
        startidx = i * block_size
        stopidx = (i + 1) * block_size
        xsubset = xs[startidx:stopidx]
        unique_xs.append(get_unique_points(xsubset, tol))
    unique_xs = np.vstack(unique_xs)
    if len(unique_xs) == npoints:
        # No reduction occurred
        # print("no reduction")
        return unique_xs
    else:
        # print("Recursion!")
        return filter_unique_points(unique_xs, tol, block_size)
    

mins = filter_unique_points(minima, tol=1e-4)
mins_star = filter_unique_points(minima_star, tol=1e-4)
mins = mins[mins[:, 0].argsort()]
mins_star = mins_star[mins_star[:, 0].argsort()]

##############################################################################
##############################################################################
##  Compute linearization around minima

js = []
covs = []
eval_list = []
evec_list = []
sval_list = []
for m in mins:
    j = hess(m)
    cov = 0.5 * sigma**2 * jnp.linalg.inv(j)
    evals, evecs = np.linalg.eigh(j)  # TODO: J is correct, not cov?
    svals = np.sqrt(0.5)*sigma/np.sqrt(evals)
    # Store computed values
    js.append(j)
    covs.append(cov)
    eval_list.append(evals)
    evec_list.append(evecs)
    sval_list.append(svals)

js_star = []
covs_star = []
eval_list_star = []
evec_list_star = []
sval_list_star = []
for m in mins_star:
    j = hess_star(m)
    cov = 0.5 * sigma_star**2 * jnp.linalg.inv(j)
    evals, evecs = np.linalg.eigh(j)
    svals = np.sqrt(0.5)*sigma_star/np.sqrt(evals)
    # Store computed values
    js_star.append(j)
    covs_star.append(cov)
    eval_list_star.append(evals)
    evec_list_star.append(evecs)
    sval_list_star.append(svals)
    print("m\n", m)
    print("j\n", j)
    print("cov\n", cov)


# Compute the maximum of the ground truth eigenvectors
j_star_max_norm = np.max([np.linalg.norm(j, axis=0).max() for j in js_star])
# e_star_max_norm = np.max(sval_list_star)
# print(e_star_max_norm)

ujs = []
ujs_star = []
for i in range(len(mins)):
    j = js[i]
    j_star = js_star[i]
    
    uj = j / j_star_max_norm
    uj_star = j_star / j_star_max_norm
    ujs.append(uj)
    ujs_star.append(uj_star)


##############################################################################
##############################################################################
##  Plot computed minima

fig, ax = plt.subplots(1, 1, layout='tight')
ax.set_aspect('equal')

ax.plot(
    mins_star[:,0], mins_star[:,1], 'o',
    color=COL_TRUE, 
    label='ground truth',
)

ax.plot(
    mins[:,0], mins[:,1], '.',
    color=COL_INF,  
    label='inferred',
)

for i in range(len(mins)):
    m = mins[i]
    m_star = mins_star[i]
    evec = evec_list[i] * sval_list[i]
    evec_star = evec_list_star[i] * sval_list_star[i]
    for vidx in range(2):
        v = ARR_SCALE * evec[:,vidx]
        v_star = ARR_SCALE * evec_star[:,vidx]
        print("*", np.linalg.norm(v_star), sval_list_star[i][vidx])
        
        ax.arrow(
            *m_star, *v_star, 
            head_width=ARR_HEAD_W, head_length=ARR_HEAD_L, 
            fc=COL_TRUE, ec=COL_TRUE, alpha=0.5,
        )
        ax.arrow(
            *m, *v, 
            head_width=ARR_HEAD_W, head_length=ARR_HEAD_L, 
            fc=COL_INF, ec=COL_INF, alpha=0.5,
        )

# ax.set_xlim(*xrange)
# ax.set_ylim(*yrange)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig(f"{OUTDIR}/minima_plot.pdf")
plt.close()

##############################################################################
##############################################################################
print("Done!")