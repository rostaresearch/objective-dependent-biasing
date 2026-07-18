"""
Coordinate-free biasing on a kinetic network.

Question (E. Rosta): on a real MSM we have only K (and pi), not the
underlying x,y.  What intrinsic coordinates can we bias along, and how
do they compare to a bias built from the physical x,y coordinates?

Intrinsic, K-only reaction coordinates tested here:
  * slow right eigenvector v2          -> natural coord for spectral gap
  * committor q (A->B)                 -> optimal 1-D coord for MFPT(A->B)
  * log pi (flattening)                -> the free-energy bias

Compared against:
  * physical (x,y) polynomial bias     -> "knows the embedding"
  * per-state analytic optimum         -> coordinate-free CEILING

All bias families share the SAME box budget |b_i| <= U_max and the same
network, so speedups are directly comparable.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

import analytic_lib as L
from per_state_ceiling import optimise_per_state_analytic
from mfpt_per_state import optimise_mfpt_per_state

U_MAX = 3.0
NX, NY = 20, 10

# ----------------------------------------------------------------------
K0, coords, F = L.grid_2d_generator(nx=NX, ny=NY, barrier_height=4.0,
                                     bottleneck=True)
n = K0.shape[0]
pi0 = L.stationary_distribution_from_K(K0)
gap0 = L.spectral_gap_K(K0)

# A = x=-1 edge (ix=0), B = x=+1 edge (ix=NX-1); index = ix*NY+iy
A = list(range(0, NY))
B = list(range((NX - 1) * NY, n))
mfpt0 = L.mfpt_K(K0, A, B)

# intrinsic coordinates from K alone
v2 = L.right_eigvec2_K(K0)                       # mean 0, std 1
q  = L.committor_K(K0, A, B)                      # in [0,1]
x  = coords[:, 0]; y = coords[:, 1]               # physical (for comparison only)

print(f"n={n}  gap0={gap0:.4e}  mfpt0={mfpt0:.4e}  |A|={len(A)} |B|={len(B)}")

# ----------------------------------------------------------------------
def clip_bias(b):
    return np.clip(b, -U_MAX, U_MAX)

def gamma_of(b):
    return L.spectral_gap_K(L.tilt_generator(K0, clip_bias(b)))

def mfpt_of(b):
    return L.mfpt_K(L.tilt_generator(K0, clip_bias(b)), A, B)

def optimise_features(feats, objective, restarts=6, seed0=0):
    """Bias = clip(feats @ c). Optimise c (Nelder-Mead) for the objective."""
    m = feats.shape[1]
    rng = np.random.default_rng(seed0)
    def obj(c):
        b = feats @ c
        return -gamma_of(b) if objective == "gamma" else mfpt_of(b)
    best = None
    seeds = [np.zeros(m)]
    for s in range(restarts):
        seeds.append(rng.standard_normal(m) * (2.0 / np.maximum(np.std(feats, 0), 1e-9)))
    for c0 in seeds:
        res = minimize(obj, c0, method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    b = clip_bias(feats @ best.x)
    return b

def poly_powers(z, kmax):
    z = np.asarray(z, float)
    return np.column_stack([z**k for k in range(1, kmax + 1)])

def xy_features(deg=4):
    cols = []
    for dx in range(deg + 1):
        for dy in range(deg + 1 - dx):
            if dx == 0 and dy == 0:
                continue
            cols.append((x**dx) * (y**dy))
    return np.column_stack(cols)

# ----------------------------------------------------------------------
results = {}

# ---- flattening (log pi), coordinate-free baseline ----
b_flat = clip_bias(np.log(np.clip(pi0, 1e-300, None)))
results["flatten (log pi)"] = (gamma_of(b_flat) / gap0, mfpt0 / mfpt_of(b_flat))

# ---- physical x,y polynomial (knows geometry) ----
print("optimising xy-poly (gamma)...");  bxy_g = optimise_features(xy_features(4), "gamma")
print("optimising xy-poly (mfpt)...");   bxy_m = optimise_features(xy_features(4), "mfpt")
results["xy-poly (deg4)  [needs x,y]"] = (gamma_of(bxy_g)/gap0, mfpt0/mfpt_of(bxy_m))

# ---- intrinsic v2 polynomial (coordinate-free) ----
print("optimising v2-poly (gamma)...");  bv_g = optimise_features(poly_powers(v2,3), "gamma")
print("optimising v2-poly (mfpt)...");   bv_m = optimise_features(poly_powers(v2,3), "mfpt")
results["v2-poly  [K-only]"] = (gamma_of(bv_g)/gap0, mfpt0/mfpt_of(bv_m))

# ---- intrinsic committor polynomial (coordinate-free) ----
print("optimising committor-poly (gamma)..."); bq_g = optimise_features(poly_powers(q-0.5,3), "gamma")
print("optimising committor-poly (mfpt)...");  bq_m = optimise_features(poly_powers(q-0.5,3), "mfpt")
results["committor-poly  [K-only]"] = (gamma_of(bq_g)/gap0, mfpt0/mfpt_of(bq_m))

# ---- per-state analytic optima (coordinate-free CEILING) ----
print("per-state gamma ceiling...");  ps_g = optimise_per_state_analytic(K0, U_max=U_MAX, maxiter=2000)
print("per-state mfpt ceiling...");   ps_m = optimise_mfpt_per_state(K0, A, B, U_max=U_MAX, maxiter=2000, verbose=False)
results["per-state optimum (CEILING)"] = (ps_g['gamma']/gap0, mfpt0/ps_m['mfpt'])

# ----------------------------------------------------------------------
print("\n" + "="*64)
print(f"{'bias family':32s} {'gamma speedup':>14s} {'MFPT speedup':>14s}")
print("-"*64)
for k, (sg, sm) in results.items():
    print(f"{k:32s} {sg:>14.1f} {sm:>14.1f}")
print("="*64)
print("v2-poly and committor-poly use ONLY K (no x,y).")
