"""
Two complementary sweeps on the 2D grid network:

  Sweep A (regime): vary the barrier height of the 2D landscape from
                    1 to 8 kT, keep |u| <= 3 kT.  Show the regime
                    transition for each of the four biases (flatten,
                    polynomial spectral, per-state spectral, polynomial MFPT).

  Sweep B (budget): fix barrier 4 kT (FE range ~12 kT), vary symmetric
                    bias budget U_max from 0.5 to 8 kT.  Show how each
                    objective's speedup scales with available bias amplitude.

Both produce a single 4-curve figure (one per panel) showing the
spectral-gap speedup and MFPT speedup as a function of the swept parameter.
"""
from __future__ import annotations
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analytic_lib as L
import optimal_spectral_bias as O
import per_state_ceiling as PSC
import mfpt_per_state as MPS

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PATH, 'data')
FIGURES = os.path.join(PATH, 'figures')
def _optimise_per_state_quick(K0, U_max, warm_u, kind, A_set=None, B_set=None,
                                maxiter=200):
    """One restart only (warm-started from poly), short maxiter."""
    n = K0.shape[0]
    bounds = [(-U_max, U_max)] * n
    if kind == 'spectral':
        def fg(u):
            try:
                g, gr = PSC.gap_and_grad(u, K0)
            except Exception:
                return 1e12, np.zeros(n)
            return -float(g), -gr
    else:  # 'mfpt'
        def fg(u):
            try:
                g, gr = MPS.mfpt_and_grad(u, K0, A_set, B_set)
            except Exception:
                return 1e12, np.zeros(n)
            return float(g), gr
    from scipy.optimize import minimize
    u0 = np.clip(warm_u, -U_max, U_max)
    res = minimize(fg, u0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=maxiter, gtol=1e-8, ftol=1e-10))
    return res.x


def one_point(K0, coords, A_set, B_set, U_max, edges):
    """For one (K0, U_max) configuration, compute the spectrum of speedups."""
    pi0 = L.stationary_distribution_from_K(K0)
    gap0 = L.spectral_gap_K(K0)
    mfpt0 = L.mfpt_K(K0, A_set, B_set)

    out = dict(gap0=float(gap0), mfpt0=float(mfpt0))

    # 1. Budget-clipped flattening (one-shot)
    u_flat = O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=U_max)
    Kb_flat = L.tilt_generator(K0, u_flat)
    out['gap_flat'] = float(L.spectral_gap_K(Kb_flat))
    out['mfpt_flat'] = float(L.mfpt_K(Kb_flat, A_set, B_set))

    # 2. Spectral-optimal polynomial (smooth basis); 1 restart, fewer NM iters
    #    since the per-state stage warm-starts from this anyway.
    poly = O.optimise_polynomial(K0, coords, U_max=U_max, max_order=4,
                                  alpha_smooth=0.001, edges=edges,
                                  n_restarts=1, maxiter=1200)
    Kb_poly = L.tilt_generator(K0, poly['u'])
    out['gap_poly_spec'] = float(L.spectral_gap_K(Kb_poly))
    out['mfpt_poly_spec'] = float(L.mfpt_K(Kb_poly, A_set, B_set))

    # 3. Per-state spectral ceiling (analytic gradient, warm-started, single run)
    u_ps = _optimise_per_state_quick(K0, U_max, warm_u=poly['u'],
                                      kind='spectral', maxiter=300)
    Kb_ps = L.tilt_generator(K0, u_ps)
    out['gap_per_state'] = float(L.spectral_gap_K(Kb_ps))
    out['mfpt_per_state'] = float(L.mfpt_K(Kb_ps, A_set, B_set))

    # 4. Per-state MFPT-optimal (analytic gradient, single run)
    u_mps = _optimise_per_state_quick(K0, U_max, warm_u=poly['u'],
                                       kind='mfpt', A_set=A_set, B_set=B_set,
                                       maxiter=300)
    Kb_mps = L.tilt_generator(K0, u_mps)
    out['gap_mfpt_opt'] = float(L.spectral_gap_K(Kb_mps))
    out['mfpt_mfpt_opt'] = float(L.mfpt_K(Kb_mps, A_set, B_set))

    # Speedup ratios for convenience
    for tag in ('flat', 'poly_spec', 'per_state', 'mfpt_opt'):
        out[f'gap_speedup_{tag}']  = out[f'gap_{tag}']  / gap0
        out[f'mfpt_speedup_{tag}'] = mfpt0 / out[f'mfpt_{tag}']
    return out


# ---------------------------------------------------------------------
#  Sweep A: barrier-height (regime transition)
# ---------------------------------------------------------------------

def sweep_barrier():
    print("\n=== Sweep A: barrier-height regime transition ===")
    barrier_heights = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0]
    U_max = 3.0
    rows = []
    for bh in barrier_heights:
        t0 = time.time()
        K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=bh,
                                              bottleneck=True)
        A_set = np.where(coords[:, 0] < -0.85)[0]
        B_set = np.where(coords[:, 0] >  0.85)[0]
        edges = O.nearest_neighbour_edges(K0)
        pi0 = L.stationary_distribution_from_K(K0)
        F_range = float((-np.log(pi0)).max() - (-np.log(pi0)).min())
        d = one_point(K0, coords, A_set, B_set, U_max, edges)
        d['barrier_height'] = float(bh)
        d['FE_range'] = F_range
        rows.append(d)
        print(f"  bh={bh:4.1f}  FE={F_range:5.2f}  gap0={d['gap0']:.2e}  "
              f"flat={d['gap_speedup_flat']:6.1f}x  "
              f"poly={d['gap_speedup_poly_spec']:6.1f}x  "
              f"ps={d['gap_speedup_per_state']:6.1f}x  "
              f"mfpt_opt={d['gap_speedup_mfpt_opt']:6.1f}x"
              f"  | mfpt: flat={d['mfpt_speedup_flat']:5.1f}x  "
              f"poly={d['mfpt_speedup_poly_spec']:5.1f}x  "
              f"ps={d['mfpt_speedup_per_state']:5.1f}x  "
              f"mfpt_opt={d['mfpt_speedup_mfpt_opt']:6.1f}x"
              f"  [{time.time()-t0:.1f}s]")
    return rows


# ---------------------------------------------------------------------
#  Sweep B: budget U_max
# ---------------------------------------------------------------------

def sweep_budget():
    print("\n=== Sweep B: bias budget U_max ===")
    budgets = [0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0]
    rows = []
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0,
                                          bottleneck=True)
    A_set = np.where(coords[:, 0] < -0.85)[0]
    B_set = np.where(coords[:, 0] >  0.85)[0]
    edges = O.nearest_neighbour_edges(K0)
    for U in budgets:
        t0 = time.time()
        d = one_point(K0, coords, A_set, B_set, U, edges)
        d['U_max'] = float(U)
        rows.append(d)
        print(f"  U={U:4.1f}  "
              f"flat={d['gap_speedup_flat']:6.1f}x  "
              f"poly={d['gap_speedup_poly_spec']:6.1f}x  "
              f"ps={d['gap_speedup_per_state']:6.1f}x  "
              f"mfpt_opt={d['gap_speedup_mfpt_opt']:6.1f}x"
              f"  | mfpt: flat={d['mfpt_speedup_flat']:5.1f}x  "
              f"poly={d['mfpt_speedup_poly_spec']:5.1f}x  "
              f"ps={d['mfpt_speedup_per_state']:5.1f}x  "
              f"mfpt_opt={d['mfpt_speedup_mfpt_opt']:6.1f}x"
              f"  [{time.time()-t0:.1f}s]")
    return rows


# ---------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------

def plot_sweep(rows, x_key, x_label, title, outpath):
    xs = np.array([r[x_key] for r in rows])
    biases = [('flat',       'budget-clipped flatten',   '#1f77b4', 'o'),
              ('poly_spec',  r'spectral $\gamma^*$ (polynomial)', '#2ca02c', 's'),
              ('per_state',  r'spectral $\gamma^*$ (per-state)',  '#000000', '*'),
              ('mfpt_opt',   r'MFPT$^*$ (per-state)',        '#d62728', 'd')]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for tag, label, color, marker in biases:
        gap_sp = [r[f'gap_speedup_{tag}']  for r in rows]
        mfpt_sp = [r[f'mfpt_speedup_{tag}'] for r in rows]
        axes[0].plot(xs, gap_sp, marker=marker, ms=7 if marker == '*' else 5,
                      color=color, label=label, lw=1.4)
        axes[1].plot(xs, mfpt_sp, marker=marker, ms=7 if marker == '*' else 5,
                      color=color, label=label, lw=1.4)
    for ax, ylab, sub in [(axes[0], r'$\gamma^b/\gamma_0$',
                           '(a) spectral-gap speedup'),
                          (axes[1], r'$\mathrm{MFPT}_0/\mathrm{MFPT}^b$',
                           '(b) MFPT(A$\\to$B) speedup')]:
        ax.set_yscale('log')
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylab)
        ax.set_title(sub)
        ax.grid(True, ls=':', alpha=0.4)
        ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.7)
    axes[0].legend(fontsize=8.5, loc='upper left', frameon=False, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.savefig(outpath, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    t_start = time.time()
    barrier_rows = sweep_barrier()
    budget_rows = sweep_budget()

    with open(f'{DATA}/sweep_results.json', 'w') as f:
        json.dump(dict(barrier=barrier_rows, budget=budget_rows), f, indent=2)
    print(f"\nTotal time: {time.time()-t_start:.0f}s.  Saved sweep_results.json")

    plot_sweep(barrier_rows, 'barrier_height',
               r'barrier height parameter [$k_BT$]',
               r'Regime sweep: 2D grid, varying barrier height '
               r'(at $U_{\max}=3\,k_BT$)',
               f'{FIGURES}/fig_regime_sweep.png')
    plot_sweep(budget_rows, 'U_max',
               r'symmetric bias budget $U_{\max}$ [$k_BT$]',
               r'Budget sweep: 2D grid, fixed barrier (4 $k_BT$ amplitude, '
               r'FE range $\approx 12\,k_BT$)',
               f'{FIGURES}/fig_budget_sweep.png')


if __name__ == '__main__':
    main()
