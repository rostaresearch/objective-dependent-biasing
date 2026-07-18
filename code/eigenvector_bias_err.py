"""
Error bars for the coordinate-free biasing comparison.

For each bias family we repeat the optimisation over many independent
trials with different random starting coefficients, and record the best
speedup found per trial.  The spread across trials tells us whether the
family-to-family differences are real or just optimiser noise (e.g. the
14-parameter xy-polynomial getting stuck in a worse local optimum than
the 3-parameter v2/committor fits).
"""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import minimize
import analytic_lib as L
from per_state_ceiling import optimise_per_state_analytic
from mfpt_per_state import optimise_mfpt_per_state

U_MAX = 3.0
NX, NY = 20, 10
N_TRIALS = 10          # independent trials per family
N_STARTS = 4           # random starts per trial (parametrised families)
NM_MAXITER = 1200      # Nelder-Mead iteration cap

K0, coords, F = L.grid_2d_generator(nx=NX, ny=NY, barrier_height=4.0, bottleneck=True)
n = K0.shape[0]
pi0 = L.stationary_distribution_from_K(K0)
gap0 = L.spectral_gap_K(K0)
A = list(range(0, NY)); B = list(range((NX-1)*NY, n))
mfpt0 = L.mfpt_K(K0, A, B)
v2 = L.right_eigvec2_K(K0)
q  = L.committor_K(K0, A, B)
x, y = coords[:,0], coords[:,1]

def clip(b): return np.clip(b, -U_MAX, U_MAX)
def gamma_speedup(b): return L.spectral_gap_K(L.tilt_generator(K0, clip(b)))/gap0
def mfpt_speedup(b):  return mfpt0/L.mfpt_K(L.tilt_generator(K0, clip(b)), A, B)

def poly_powers(z, kmax): return np.column_stack([np.asarray(z,float)**k for k in range(1,kmax+1)])
def xy_features(deg=4):
    cols=[]
    for dx in range(deg+1):
        for dy in range(deg+1-dx):
            if dx==0 and dy==0: continue
            cols.append((x**dx)*(y**dy))
    return np.column_stack(cols)

def best_speedup(feats, objective, seed):
    """best over N_STARTS random starts; returns the best speedup found."""
    m = feats.shape[1]; rng = np.random.default_rng(seed)
    sd = np.maximum(np.std(feats,0),1e-9)
    def negobj(c):
        b = feats @ c
        return -(gamma_speedup(b) if objective=="gamma" else mfpt_speedup(b))
    best = -np.inf
    for s in range(N_STARTS):
        c0 = (np.zeros(m) if s==0 else rng.standard_normal(m)*(2.0/sd))
        res = minimize(negobj, c0, method="Nelder-Mead",
                       options=dict(maxiter=NM_MAXITER, xatol=1e-4, fatol=1e-6))
        best = max(best, -res.fun)
    return best

def trials_param(feats, objective):
    return [best_speedup(feats, objective, seed=1000+t) for t in range(N_TRIALS)]

def trials_perstate(objective):
    out=[]
    for t in range(N_TRIALS):
        rng=np.random.default_rng(2000+t)
        warm = rng.standard_normal(n)*1.0
        if objective=="gamma":
            r=optimise_per_state_analytic(K0, U_max=U_MAX, warm_u=warm, maxiter=1500)
            out.append(r['gamma']/gap0)
        else:
            r=optimise_mfpt_per_state(K0, A, B, U_max=U_MAX, warm_u=warm, maxiter=1500, verbose=False)
            out.append(mfpt0/r['mfpt'])
    return out

print(f"n={n} gap0={gap0:.3e} mfpt0={mfpt0:.3e}; {N_TRIALS} trials x {N_STARTS} starts")
res={}
print("xy-poly gamma...");     res['xy_gamma']        = trials_param(xy_features(4), "gamma")
print("v2-poly gamma...");     res['v2_gamma']        = trials_param(poly_powers(v2,3), "gamma")
print("per-state gamma...");   res['ps_gamma']        = trials_perstate("gamma")
print("xy-poly mfpt...");      res['xy_mfpt']         = trials_param(xy_features(4), "mfpt")
print("committor mfpt...");    res['committor_mfpt']  = trials_param(poly_powers(q-0.5,3), "mfpt")
print("per-state mfpt...");    res['ps_mfpt']         = trials_perstate("mfpt")

print("\n%-18s %8s %7s %8s %8s" % ("family/obj","mean","std","min","max"))
summary={}
for k,v in res.items():
    v=np.array(v); summary[k]=dict(mean=float(v.mean()),std=float(v.std()),
                                   min=float(v.min()),max=float(v.max()),vals=v.tolist())
    print("%-18s %8.2f %7.3f %8.2f %8.2f" % (k, v.mean(), v.std(), v.min(), v.max()))
json.dump(summary, open("eigenvector_bias_err.json","w"), indent=2)
print("\nsaved eigenvector_bias_err.json")
