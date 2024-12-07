"""Figure phi-fixed-point-history

Generate plots used in Figures phi[n]-fixed-point-history of the supplement.
"""

import argparse
import os, pathlib
import time
import tqdm as tqdm
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
OUTDIRBASE = "figures/supplement/out/fig_phi_fixed_point_history"
SEED = 57283674
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
_, training_info = load_model_training_metadata(MODELDIR, load_all=True)

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

xrange = [-2, 2]
yrange = [-2, 2]

NSTARTS = 20

starting_points = np.zeros([NSTARTS*NSTARTS, 2])
x = np.linspace(*xrange, NSTARTS)
y = np.linspace(*yrange, NSTARTS)
xs, ys = np.meshgrid(x, y)
starting_points[:,0] = xs.flatten()
starting_points[:,1] = ys.flatten()

# starting_points[:,0] = np.random.uniform(*xrange, NSTARTS)
# starting_points[:,1] = np.random.uniform(*yrange, NSTARTS)



tol = 1e-4


##############################################################################
##############################################################################
##  Helper functions

def get_phi(model, signal):
    def phi(x):
        return model.eval_phi_with_signal(0, x, signal)
    return phi

def get_unique_points(xs, tol):
    dists = cdist(xs, xs)
    idxs = jnp.triu_indices(len(dists))
    dists = dists.at[idxs].set(jnp.inf)
    unique_xs = jnp.all(dists > tol, axis=1)
    return xs[unique_xs, :]

def filter_unique_points(xs, tol, block_size=100):
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
        return unique_xs
    else:
        # Recurse
        return filter_unique_points(unique_xs, tol, block_size)


##############################################################################
##############################################################################
##  True model computations

phi_star = get_phi(model_star, np.array(signal))
sigma_star = model_star.get_sigma()
hess_star = eqx.filter_jit(jax.hessian(phi_star))

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

mins_star = filter_unique_points(minima_star, tol=1e-4)
mins_star = mins_star[mins_star[:, 0].argsort()]

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

# Compute the maximum of the ground truth eigenvectors
j_star_max_norm = np.max([np.linalg.norm(j, axis=0).max() for j in js_star])

##############################################################################
##############################################################################
##  Inferred model computations



def do_minimize(x0, phi):
    return minimize(phi, x0, method='BFGS')

vectorized_minimize = eqx.filter_jit(jax.vmap(do_minimize, (0, None)))
    
@eqx.filter_jit
def make_computation(m, sigma, hess):
    j = hess(m)
    cov = 0.5 * sigma**2 * jnp.linalg.inv(j)
    evals, evecs = jnp.linalg.eigh(j)
    svals = jnp.sqrt(0.5)*sigma/jnp.sqrt(evals)
    return j, cov, evals, evecs, svals

results = {
    'epochs': [],
    'mu_errors': [],
    'svec_errors': [],
}
loss_hist = training_info['loss_hist_valid']
best_loss = np.inf
times_compute = []
times_reduce = []
SHORTSTOP = 100000
for i in tqdm.trange(min(len(loss_hist), SHORTSTOP), disable=False):
    vloss = loss_hist[i]
    fpath = f"{MODELDIR}/states/{modelname}_{i+1}.pth"
    if pathlib.Path(fpath).is_file():
        # print("Loading model...")
        model, _, idx, name, fpath = load_model_from_directory(
            MODELDIR, idx=i+1
        )
        loss_better = vloss < best_loss
        # NOTE: Might cause issue if model saved periodically, regardless of vloss
        assert loss_better, "Loss did not decrease!"
        # print(f"Loaded model {fpath}", f"{vloss:.5g} < {best_loss:.5g}", loss_better)
        best_loss = vloss
        results['epochs'].append(idx)
    else:
        continue
    
    phi = get_phi(model, np.array(signal))
    sigma = model.get_sigma()
    hess = jax.hessian(phi)  # opposite of jacobian J_{F}
    
    # print("Computing fixed points...")
    time0 = time.time()
    
    # Version 1 (Not Vectorized)
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
    
    # Version 2 (Vectorized)
    # minres_all = vectorized_minimize(starting_points, phi)
    # minima = minres_all.x[minres_all.success,:]
    # vals = minres_all.fun[minres_all.success]

    time1 = time.time()
    times_compute.append(time1 - time0)

    # Find unique fixed points
    # print("Reducing...")
    time0 = time.time()

    mins = filter_unique_points(minima, tol=1e-4)
    mins = mins[mins[:, 0].argsort()]

    time1 = time.time()
    times_reduce.append(time1 - time0)

    # Compute linearization around minima
    # print("computing linearization...")
    js = []
    covs = []
    eval_list = []
    evec_list = []
    sval_list = []
    for m in mins:
        j, cov, evals, evecs, svals = make_computation(m, sigma, hess)
        # Store computed values
        js.append(j)
        covs.append(cov)
        eval_list.append(evals)
        evec_list.append(evecs)
        sval_list.append(svals)
    
    # Difference between means
    mean_dists = cdist(mins_star, mins)
    nearest_idx_to_true_mins = mean_dists.argmin(axis=1)
    results['mu_errors'].append(mean_dists.min(axis=0))
    
    # Difference between covariance structure
    svec_errors = []
    for k, m in enumerate(mins_star):
        nearest_min = nearest_idx_to_true_mins[k]
        svecs_star = evec_list_star[k] * sval_list_star[k]
        svecs = evec_list[nearest_min] * sval_list[nearest_min]
        svecs_star *= (1 - 2*(svecs_star[0,:] < 0))
        svecs *= (1 - 2*(svecs[0,:] < 0))
        vdists = cdist(svecs_star.T, svecs.T)
        svec_errors.append(vdists.min(axis=1).sum())

    results['svec_errors'].append(np.array(svec_errors))
    
print(f"Average time to compute fixed points:", np.median(times_compute))
print(f"Average time to reduce fixed points :", np.median(times_reduce))

##############################################################################
##############################################################################
##  Plot computed minima

colorset = ['k', 'grey', 'b']

epochs = results['epochs']
mu_errors = results['mu_errors']
svec_errors = results['svec_errors']

fig, ax = plt.subplots(1, 1)
for i in range(len(mu_errors)):
    idx = epochs[i]
    errs = mu_errors[i]
    for j in range(len(errs)):
        ax.plot(
            idx, errs[j], '.', 
            color=colorset[j % len(colorset)], 
            markersize=3,
        )
plt.savefig(f"{OUTDIR}/minima_error_evolution.pdf")
plt.close()

fig, ax = plt.subplots(1, 1)
for i in range(len(mu_errors)):
    idx = epochs[i]
    errs = np.log10(mu_errors[i])
    for j in range(len(errs)):
        ax.plot(
            idx, errs[j], '.', 
            color=colorset[j % len(colorset)], 
            markersize=3,
        )
plt.savefig(f"{OUTDIR}/minima_error_evolution_logscale.pdf")
plt.close()


fig, ax = plt.subplots(1, 1)
for i in range(len(svec_errors)):
    idx = epochs[i]
    errs = svec_errors[i]
    for j in range(len(errs)):
        ax.plot(
            idx, errs[j], '.', 
            color=colorset[j % len(colorset)], 
            markersize=3,
        )
plt.savefig(f"{OUTDIR}/svec_error_evolution.pdf")
plt.close()

fig, ax = plt.subplots(1, 1)
for i in range(len(svec_errors)):
    idx = epochs[i]
    errs = np.log10(svec_errors[i])
    for j in range(len(errs)):
        ax.plot(
            idx, errs[j], '.', 
            color=colorset[j % len(colorset)], 
            markersize=3,
        )
plt.savefig(f"{OUTDIR}/svec_error_evolution_logscale.pdf")
plt.close()



# ax.plot(
#     mins_star[:,0], mins_star[:,1], 'o',
#     color=COL_TRUE, 
#     label='ground truth',
# )

# ax.plot(
#     mins[:,0], mins[:,1], '.',
#     color=COL_INF,  
#     label='inferred',
# )

# for i in range(len(mins)):
#     m = mins[i]
#     m_star = mins_star[i]
#     evec = evec_list[i] * sval_list[i]
#     evec_star = evec_list_star[i] * sval_list_star[i]
#     for vidx in range(2):
#         v = ARR_SCALE * evec[:,vidx]
#         v_star = ARR_SCALE * evec_star[:,vidx]
#         print("*", np.linalg.norm(v_star), sval_list_star[i][vidx])
        
#         ax.arrow(
#             *m_star, *v_star, 
#             head_width=ARR_HEAD_W, head_length=ARR_HEAD_L, 
#             fc=COL_TRUE, ec=COL_TRUE, alpha=0.5,
#         )
#         ax.arrow(
#             *m, *v, 
#             head_width=ARR_HEAD_W, head_length=ARR_HEAD_L, 
#             fc=COL_INF, ec=COL_INF, alpha=0.5,
#         )

# ax.set_xlim(*xrange)
# ax.set_ylim(*yrange)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')



##############################################################################
##############################################################################
print("Done!")