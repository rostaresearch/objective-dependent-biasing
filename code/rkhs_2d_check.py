"""Reproduce the 9-well 2D landscape from
   $DHAM2D/2D_opt_K.py
and apply our analytic per-state spectral and MFPT optimisers for
comparison with the random-Gaussian-bias optimisation done there.

Conventions used here:
  - Free energy F in dimensionless k_B T units (we divide Z by kBT)
  - Rate matrix K0 row-generator:
        K0[i, j] = exp(-(F_j - F_i) / 2)  for adjacent i,j (4-connected),
        diagonal = -row sum.
  - Bias u (and U_max) likewise in k_B T units.

Reports gamma and MFPT speedups for U_max = 3 k_B T (~ 1.8 kcal/mol) and
U_max = 6 k_B T (~ 3.6 kcal/mol) to span a typical biasing budget.
"""
from __future__ import annotations
import numpy as np
from scipy.io import savemat

import analytic_lib as L
from per_state_ceiling import optimise_per_state_analytic
from mfpt_per_state import optimise_mfpt_per_state

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PATH, 'data')
FIGURES = os.path.join(PATH, 'figures')
kBT = 0.5981   # kcal/mol -- matches the DHAM2D config

# ----------------------------------------------------------------------
#  1.  Reproduce the multi-well + saddle landscape (kcal/mol)
# ----------------------------------------------------------------------
N = 20
xs = np.linspace(0.0, 2.0 * np.pi, N)
ys = np.linspace(0.0, 2.0 * np.pi, N)
X, Y = np.meshgrid(xs, ys)   # X[i,j] = xs[j], Y[i,j] = ys[i]

amp = 6.0
A_i = np.array([0.9, 0.3, 0.5, 1.0, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp
x0_i = np.array([1.12, 1.0, 3.0, 4.15, 4.0, 5.27, 5.5, 6.0, 1.0])
y0_i = np.array([1.34, 2.25, 2.31, 3.62, 5.0, 4.14, 4.5, 1.52, 5.0])
sigma_x_i = np.array([0.5, 0.3, 0.4, 2.0, 0.9, 1.0, 0.3, 0.5, 0.5])
sigma_y_i = np.array([0.5, 0.3, 1.0, 0.8, 0.2, 0.3, 1.0, 0.6, 0.7])

A_j = np.array([0.3]) * amp
x0_j = np.array([np.pi])
y0_j = np.array([np.pi])
sigma_x_j = np.array([3.0])
sigma_y_j = np.array([0.3])

Z = np.zeros_like(X) + amp * 4.184    # uniform high baseline (kcal/mol)
for w in range(len(A_i)):
    Z -= A_i[w] * np.exp(
        -(X - x0_i[w]) ** 2 / (2.0 * sigma_x_i[w] ** 2)
        -(Y - y0_i[w]) ** 2 / (2.0 * sigma_y_i[w] ** 2)
    )
for w in range(len(A_j)):
    Z += A_j[w] * np.exp(
        -(X - x0_j[w]) ** 2 / (2.0 * sigma_x_j[w] ** 2)
        -(Y - y0_j[w]) ** 2 / (2.0 * sigma_y_j[w] ** 2)
    )

# Soft boundary walls
k_steep = 5.0
max_barrier = 100.0
offset = 0.7
Z += max_barrier * (1.0 / (1.0 + np.exp( k_steep * (X - (-offset)))))
Z += max_barrier * (1.0 / (1.0 + np.exp(-k_steep * (X - (2.0 * np.pi + offset)))))
Z += max_barrier * (1.0 / (1.0 + np.exp( k_steep * (Y - (-offset)))))
Z += max_barrier * (1.0 / (1.0 + np.exp(-k_steep * (Y - (2.0 * np.pi + offset)))))
Z -= Z.min()      # shift so the lowest point is at 0

print(f"Landscape (kcal/mol): min={Z.min():.3f}, max={Z.max():.3f}, "
      f"range={(Z.max()-Z.min()):.3f}")
print(f"Landscape (k_B T):    range={(Z.max()-Z.min())/kBT:.3f}")

F_dim = Z / kBT                         # dimensionless free energy
F_flat = F_dim.ravel(order='C')

# ----------------------------------------------------------------------
#  2.  Build K0 in row-generator convention, 4-connected on the grid
# ----------------------------------------------------------------------
n = N * N
K0 = np.zeros((n, n))
def idx(ix, iy): return ix * N + iy        # row * N + col
for ix in range(N):
    for iy in range(N):
        i = idx(ix, iy)
        for jx, jy in ((ix - 1, iy), (ix + 1, iy),
                        (ix, iy - 1), (ix, iy + 1)):
            if 0 <= jx < N and 0 <= jy < N:
                j = idx(jx, jy)
                K0[i, j] = np.exp(-(F_flat[j] - F_flat[i]) / 2.0)
np.fill_diagonal(K0, 0)
np.fill_diagonal(K0, -K0.sum(axis=1))
L.check_row_generator(K0)
print("K0 built; row-generator check OK.")

# ----------------------------------------------------------------------
#  3.  Start (5.0, 4.0) -> End (1.0, 1.5).  Map physical coords to states.
# ----------------------------------------------------------------------
def coord_to_state(x_phys: float, y_phys: float) -> int:
    j = int(np.argmin(np.abs(xs - x_phys)))    # column index in X
    i = int(np.argmin(np.abs(ys - y_phys)))    # row index in Y
    return idx(i, j)                            # row*N + col, matches Z.ravel('C')

start_idx = coord_to_state(5.0, 4.0)
end_idx   = coord_to_state(1.0, 1.5)
print(f"start state idx={start_idx}, F={F_flat[start_idx]:.3f} k_BT, "
      f"Z={Z.ravel()[start_idx]:.3f} kcal/mol")
print(f"end   state idx={end_idx}, F={F_flat[end_idx]:.3f} k_BT, "
      f"Z={Z.ravel()[end_idx]:.3f} kcal/mol")

# ----------------------------------------------------------------------
#  4.  Baseline
# ----------------------------------------------------------------------
pi0 = L.stationary_distribution_from_K(K0)
gap0 = L.spectral_gap_K(K0)
mfpt0 = L.mfpt_K(K0, [start_idx], [end_idx])
print(f"\nUnbiased: gamma0={gap0:.4e}, MFPT0(start->end)={mfpt0:.4e}")

# ----------------------------------------------------------------------
#  5.  Per-state optima at two budgets
# ----------------------------------------------------------------------
U_max_list_kT = [3.0, 6.0]
results = []
for U_max in U_max_list_kT:
    print(f"\n=== U_max = {U_max} k_B T (={U_max*kBT:.2f} kcal/mol) ===")

    print("  Spectral-gap per-state opt...")
    rs = optimise_per_state_analytic(K0, U_max=U_max, maxiter=2000)
    Kb_s = L.tilt_generator(K0, rs['u'])
    gap_s = L.spectral_gap_K(Kb_s)
    mfpt_s = L.mfpt_K(Kb_s, [start_idx], [end_idx])
    print(f"    gamma speedup = {gap_s/gap0:.2f}x, MFPT speedup = {mfpt0/mfpt_s:.2f}x")

    print("  MFPT per-state opt...")
    rm = optimise_mfpt_per_state(K0, [start_idx], [end_idx],
                                  U_max=U_max, maxiter=2000, verbose=True)
    Kb_m = L.tilt_generator(K0, rm['u'])
    gap_m = L.spectral_gap_K(Kb_m)
    mfpt_m = rm['mfpt']
    print(f"    gamma speedup = {gap_m/gap0:.2f}x, MFPT speedup = {mfpt0/mfpt_m:.2f}x")

    results.append(dict(
        U_max=float(U_max), U_max_kcal=float(U_max*kBT),
        u_spec=rs['u'].reshape(N, N),
        u_mfpt=rm['u'].reshape(N, N),
        gap_spec=float(gap_s), gap_mfpt=float(gap_m),
        mfpt_spec=float(mfpt_s), mfpt_mfpt=float(mfpt_m),
        speedup_gap_spec=float(gap_s/gap0),
        speedup_mfpt_spec=float(mfpt0/mfpt_s),
        speedup_gap_mfpt=float(gap_m/gap0),
        speedup_mfpt_mfpt=float(mfpt0/mfpt_m),
    ))

# ----------------------------------------------------------------------
#  6.  Save
# ----------------------------------------------------------------------
out = dict(
    N=N, xs=xs, ys=ys, X=X, Y=Y,
    Z=Z, F=F_dim, kBT=kBT,
    start_xy=np.array([5.0, 4.0]),
    end_xy=np.array([1.0, 1.5]),
    gap0=float(gap0), mfpt0=float(mfpt0),
    U_max_list_kT=np.array(U_max_list_kT),
)
for k, r in enumerate(results, start=1):
    out[f'u_spec_{k}'] = r['u_spec']
    out[f'u_mfpt_{k}'] = r['u_mfpt']
    out[f'speedup_gap_spec_{k}'] = r['speedup_gap_spec']
    out[f'speedup_mfpt_spec_{k}'] = r['speedup_mfpt_spec']
    out[f'speedup_gap_mfpt_{k}'] = r['speedup_gap_mfpt']
    out[f'speedup_mfpt_mfpt_{k}'] = r['speedup_mfpt_mfpt']
savemat(f'{DATA}/rkhs_2d_data.mat', out)
print('\nSaved rkhs_2d_data.mat')
