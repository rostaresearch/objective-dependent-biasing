"""Bundle the existing 2D-grid analysis results into a single MAT file
   so the MATLAB plotter can load them in one go."""
import json
import numpy as np
from scipy.io import savemat
import analytic_lib as L

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PATH, 'data')
FIGURES = os.path.join(PATH, 'figures')
# Re-build the 2D grid to recover (xs, ys) coordinates + F
nx, ny = 20, 10
K0, coords, F = L.grid_2d_generator(nx=nx, ny=ny, barrier_height=4.0,
                                       bottleneck=True)
pi0 = L.stationary_distribution_from_K(K0)
gap0 = L.spectral_gap_K(K0)

xs1 = coords[:, 0].reshape(nx, ny)[:, 0]
ys1 = coords[:, 1].reshape(nx, ny)[0, :]
Fgrid = F.reshape(nx, ny)
pi0_grid = pi0.reshape(nx, ny)

psc = json.load(open(f'{DATA}/per_state_ceiling.json'))
mpst = json.load(open(f'{DATA}/mfpt_per_state.json'))
sweep = json.load(open(f'{DATA}/sweep_results.json'))

# Helper to extract a column from a list-of-dicts
def col(rows, k):
    return np.asarray([r[k] for r in rows])

out = dict(
    nx=nx, ny=ny,
    xs=xs1, ys=ys1,
    F=Fgrid, pi0=pi0_grid,
    gap0=float(gap0),
    U_max=float(psc['U_max']),

    # Per-state spectral ceiling: polynomial vs per-state bias
    u_poly_spec=np.asarray(psc['u_polynomial']).reshape(nx, ny),
    u_ps_spec=np.asarray(psc['u_per_state']).reshape(nx, ny),
    gap_poly=float(psc['gap_polynomial']),
    gap_ps=float(psc['gap_per_state']),
    speedup_poly_spec=float(psc['speedup_polynomial']),
    speedup_ps_spec=float(psc['speedup_per_state']),

    # Per-state MFPT optimum
    u_ps_mfpt=np.asarray(mpst['u_per_state']).reshape(nx, ny),
    speedup_ps_mfpt=float(mpst['mfpt_speedup_per_state']),

    # Budget sweep
    budget_U=col(sweep['budget'], 'U_max'),
    budget_gap_flat=col(sweep['budget'], 'gap_speedup_flat'),
    budget_gap_poly=col(sweep['budget'], 'gap_speedup_poly_spec'),
    budget_gap_ps=col(sweep['budget'], 'gap_speedup_per_state'),
    budget_gap_mfpt=col(sweep['budget'], 'gap_speedup_mfpt_opt'),
    budget_mfpt_flat=col(sweep['budget'], 'mfpt_speedup_flat'),
    budget_mfpt_poly=col(sweep['budget'], 'mfpt_speedup_poly_spec'),
    budget_mfpt_ps=col(sweep['budget'], 'mfpt_speedup_per_state'),
    budget_mfpt_mfpt=col(sweep['budget'], 'mfpt_speedup_mfpt_opt'),

    # Regime sweep (vs barrier height / FE range)
    regime_FE=col(sweep['barrier'], 'FE_range'),
    regime_gap_flat=col(sweep['barrier'], 'gap_speedup_flat'),
    regime_gap_poly=col(sweep['barrier'], 'gap_speedup_poly_spec'),
    regime_gap_ps=col(sweep['barrier'], 'gap_speedup_per_state'),
    regime_gap_mfpt=col(sweep['barrier'], 'gap_speedup_mfpt_opt'),
    regime_mfpt_flat=col(sweep['barrier'], 'mfpt_speedup_flat'),
    regime_mfpt_poly=col(sweep['barrier'], 'mfpt_speedup_poly_spec'),
    regime_mfpt_ps=col(sweep['barrier'], 'mfpt_speedup_per_state'),
    regime_mfpt_mfpt=col(sweep['barrier'], 'mfpt_speedup_mfpt_opt'),
)
savemat(f'{DATA}/grid_2d_data.mat', out)
print('Saved grid_2d_data.mat')
print(f'  nx={nx} ny={ny} gap0={gap0:.3e}  U_max={psc["U_max"]}')
print(f'  budget points: {len(out["budget_U"])}')
print(f'  regime points: {len(out["regime_FE"])}')
