"""Regenerate the 2D-grid per-state spectral and MFPT optima at higher
resolution (nx=40, ny=20), and bundle into grid_2d_hires.mat for the
MATLAB landscape + bias-profile figures.

The sweep line plots can stay at n=200 (those are line plots, not
heatmaps, and the qualitative cross-objective collapse is robust to the
resolution).
"""
import numpy as np
from scipy.io import savemat

import analytic_lib as L
from per_state_ceiling import optimise_per_state_analytic
from mfpt_per_state import optimise_mfpt_per_state

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nx, ny = 40, 20
barrier_height = 4.0
U_max = 3.0

print(f"Building 2D grid nx={nx} ny={ny} (n={nx*ny}) barrier_height={barrier_height}")
K0, coords, F = L.grid_2d_generator(nx=nx, ny=ny,
                                       barrier_height=barrier_height,
                                       bottleneck=True)
n = K0.shape[0]
pi0 = L.stationary_distribution_from_K(K0)
gap0 = L.spectral_gap_K(K0)

xs1 = coords[:, 0].reshape(nx, ny)[:, 0]
ys1 = coords[:, 1].reshape(nx, ny)[0, :]
Fgrid = F.reshape(nx, ny)
pi0_grid = pi0.reshape(nx, ny)

print(f"  n={n}, gap0={gap0:.4e}")

# A = states near x=-1 (bottom row), B = states near x=+1 (top row)
A_set = list(range(0, ny))                  # bottom row (x = -1)
B_set = list(range((nx - 1) * ny, n))       # top row (x = +1)
mfpt0 = L.mfpt_K(K0, A_set, B_set)
print(f"  basins: |A|={len(A_set)} |B|={len(B_set)}  MFPT0={mfpt0:.3e}")

print("\nPer-state spectral-gap ceiling (n=800)...")
ps_spec = optimise_per_state_analytic(K0, U_max=U_max, maxiter=2000)
gap_ps_spec = ps_spec['gamma']
print(f"  speedup = {gap_ps_spec/gap0:.2f}x")

print("\nPer-state MFPT minimum (n=800)...")
ps_mfpt = optimise_mfpt_per_state(K0, A_set, B_set, U_max=U_max,
                                     maxiter=2000, verbose=True)
gap_ps_mfpt = L.spectral_gap_K(L.tilt_generator(K0, ps_mfpt['u']))
mfpt_ps_mfpt = ps_mfpt['mfpt']
print(f"  MFPT speedup = {mfpt0/mfpt_ps_mfpt:.2f}x")

savemat(f'{PATH}/grid_2d_hires.mat', dict(
    nx=nx, ny=ny,
    xs=xs1, ys=ys1,
    F=Fgrid, pi0=pi0_grid,
    gap0=float(gap0),
    mfpt0=float(mfpt0),
    U_max=float(U_max),
    u_ps_spec=ps_spec['u'].reshape(nx, ny),
    u_ps_mfpt=ps_mfpt['u'].reshape(nx, ny),
    gap_ps_spec=float(gap_ps_spec),
    gap_ps_mfpt=float(gap_ps_mfpt),
    mfpt_ps_mfpt=float(mfpt_ps_mfpt),
    speedup_ps_spec=float(gap_ps_spec / gap0),
    speedup_ps_mfpt=float(mfpt0 / mfpt_ps_mfpt),
))
print(f'\nSaved grid_2d_hires.mat')
