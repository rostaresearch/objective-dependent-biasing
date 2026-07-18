"""
One-shot bias that minimises MFPT(A -> B).

Distinct from:
  - the spectral objective gamma(K^b) -> max  (global equilibration);
  - umbrella-sampling estimator variance E_dagger or E_AB
                                          (sampling efficiency for an observable).

Here we solve
    b* = argmin_b  MFPT(K^(b), A, B),
subject to   |b_i| <= U_max   and   sum_i b_i = 0
(same symmetric, zero-mean budget as the spectral comparison).

We compare four one-shot biases analytically:
  (i)   no bias               -- the unbiased reference,
  (ii)  budget-clipped flatten u = clip(log pi),
  (iii) spectral-gamma optimum (deg-4 polynomial, from optimal_spectral_bias.py),
  (iv)  MFPT-optimal poly      (deg-4 polynomial, new here).
For each, we record both the MFPT(A->B) and the spectral gap gamma, so we can
ask whether the MFPT-optimal bias also helps relaxation -- and vice versa.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import analytic_lib as L
import optimal_spectral_bias as O  # reuse polynomial basis, projection, smoothness

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def mfpt_obj(u, K0, A, B):
    """Returns MFPT(A->B); returns large if biased operator pathological."""
    if not np.all(np.isfinite(u)):
        return 1e12
    try:
        Kb = L.tilt_generator(K0, u)
        m = L.mfpt_K(Kb, A, B)
        return float(m) if np.isfinite(m) and m > 0 else 1e12
    except Exception:
        return 1e12


def optimise_polynomial_mfpt(K0, coords, A, B, U_max=3.0, max_order=4,
                             alpha_smooth=0.001, edges=None,
                             n_restarts=4, seed=0,
                             seed_with_flatten=True):
    Fb = O.basis_poly(coords, max_order=max_order)
    n_feat = Fb.shape[1]
    rng = np.random.default_rng(seed)
    pi0 = L.stationary_distribution_from_K(K0)

    def cost(theta):
        u = Fb @ theta
        u = O.project_symmetric(u, U_max=U_max)
        c = mfpt_obj(u, K0, A, B)
        if alpha_smooth > 0 and edges is not None:
            c += alpha_smooth * O.smoothness_penalty(u, edges)
        return c

    starts = [np.zeros(n_feat)]
    if seed_with_flatten:
        u_flat = O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=U_max)
        theta_flat, *_ = np.linalg.lstsq(Fb, u_flat, rcond=None)
        starts.append(theta_flat)
    for _ in range(n_restarts - len(starts)):
        starts.append(rng.normal(scale=0.6, size=n_feat))

    best = None
    for theta0 in starts:
        res = minimize(cost, theta0, method='Nelder-Mead',
                       options=dict(maxiter=3500, xatol=1e-6, fatol=1e-7))
        if best is None or res.fun < best.fun:
            best = res
    u_best = O.project_symmetric(Fb @ best.x, U_max=U_max)
    Kb = L.tilt_generator(K0, u_best)
    return dict(u=u_best, theta=best.x,
                mfpt=L.mfpt_K(Kb, A, B),
                gap=L.spectral_gap_K(Kb))


def main():
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0,
                                          bottleneck=True)
    n = K0.shape[0]
    pi0 = L.stationary_distribution_from_K(K0)
    A = np.where(coords[:, 0] < -0.85)[0]
    B = np.where(coords[:, 0] >  0.85)[0]
    edges = O.nearest_neighbour_edges(K0)
    U_max = 3.0

    # Reference numbers
    mfpt_0 = L.mfpt_K(K0, A, B)
    gap_0  = L.spectral_gap_K(K0)
    print(f"Network n={n}, FE range={(-np.log(pi0)).max()-(-np.log(pi0)).min():.2f}")
    print(f"  unbiased  MFPT(A->B) = {mfpt_0:.4g},  gamma = {gap_0:.4e}")

    # Budget-clipped flattening
    u_flat = O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=U_max)
    Kb_flat = L.tilt_generator(K0, u_flat)
    mfpt_flat = L.mfpt_K(Kb_flat, A, B)
    gap_flat  = L.spectral_gap_K(Kb_flat)
    print(f"  flatten   MFPT = {mfpt_flat:.4g} ({mfpt_0/mfpt_flat:.2f}x), "
          f"gamma = {gap_flat:.4e} ({gap_flat/gap_0:.2f}x)")

    # Spectral-optimal polynomial (reuse from spectral study)
    print("  optimising spectral-gamma polynomial...", flush=True)
    spec = O.optimise_polynomial(K0, coords, U_max=U_max, max_order=4,
                                  alpha_smooth=0.001, edges=edges, n_restarts=3)
    Kb_spec = L.tilt_generator(K0, spec['u'])
    mfpt_spec = L.mfpt_K(Kb_spec, A, B)
    print(f"  spectral  MFPT = {mfpt_spec:.4g} ({mfpt_0/mfpt_spec:.2f}x), "
          f"gamma = {spec['gap']:.4e} ({spec['gap']/gap_0:.2f}x)")

    # MFPT-optimal polynomial (new)
    print("  optimising MFPT polynomial...", flush=True)
    mfpt_opt = optimise_polynomial_mfpt(K0, coords, A, B, U_max=U_max,
                                         max_order=4, alpha_smooth=0.001,
                                         edges=edges, n_restarts=4)
    print(f"  MFPT-opt  MFPT = {mfpt_opt['mfpt']:.4g} ({mfpt_0/mfpt_opt['mfpt']:.2f}x), "
          f"gamma = {mfpt_opt['gap']:.4e} ({mfpt_opt['gap']/gap_0:.2f}x)")

    # Cross-table: each bias evaluated under each metric
    biases = dict(unbiased=np.zeros(n), flatten=u_flat,
                  spectral=spec['u'], mfpt_opt=mfpt_opt['u'])
    table = {}
    for name, u in biases.items():
        Kb = L.tilt_generator(K0, u)
        m = L.mfpt_K(Kb, A, B)
        g = L.spectral_gap_K(Kb)
        table[name] = dict(mfpt=float(m), gap=float(g),
                           mfpt_speedup=float(mfpt_0 / m),
                           gap_speedup=float(g / gap_0))
    save = dict(U_max=U_max, mfpt_0=float(mfpt_0), gap_0=float(gap_0),
                u_flat=u_flat.tolist(),
                u_spectral=spec['u'].tolist(),
                u_mfpt=mfpt_opt['u'].tolist(),
                table=table)
    with open(f'{PATH}/mfpt_results.json', 'w') as f:
        json.dump(save, f, indent=2)
    print('  Saved mfpt_results.json')

    # ---------- Figure ----------
    nx, ny = 20, 10
    xg = coords[:, 0].reshape(nx, ny)
    yg = coords[:, 1].reshape(nx, ny)
    F0 = -np.log(np.clip(pi0, 1e-300, None)); F0 -= F0.min()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)

    # (a) F
    ax = axes[0, 0]
    c = ax.contourf(xg, yg, F0.reshape(nx, ny), levels=18, cmap='viridis')
    ax.set_title(f'(a) Unbiased  F\nMFPT={mfpt_0:.2e},  $\\gamma_0$={gap_0:.2e}')
    plt.colorbar(c, ax=ax, label=r'$F\;[k_BT]$')

    # (b) flatten
    ax = axes[0, 1]
    c = ax.contourf(xg, yg, u_flat.reshape(nx, ny), levels=18,
                    cmap='coolwarm', vmin=-U_max, vmax=U_max)
    ax.set_title(f'(b) Budget-clipped flatten\n'
                 f'MFPT spd={mfpt_0/mfpt_flat:.1f}x,  '
                 fr'$\gamma$ spd={gap_flat/gap_0:.1f}x')
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    # (c) Spectral-optimal polynomial
    ax = axes[0, 2]
    c = ax.contourf(xg, yg, spec['u'].reshape(nx, ny), levels=18,
                    cmap='coolwarm', vmin=-U_max, vmax=U_max)
    ax.set_title(f"(c) Spectral-$\\gamma$ optimum\n"
                 f"MFPT spd={mfpt_0/mfpt_spec:.1f}x,  "
                 fr"$\gamma$ spd={spec['gap']/gap_0:.1f}x")
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    # (d) MFPT-optimal polynomial
    ax = axes[1, 0]
    c = ax.contourf(xg, yg, mfpt_opt['u'].reshape(nx, ny), levels=18,
                    cmap='coolwarm', vmin=-U_max, vmax=U_max)
    ax.set_title(f"(d) MFPT-optimal bias\n"
                 f"MFPT spd={mfpt_0/mfpt_opt['mfpt']:.1f}x,  "
                 fr"$\gamma$ spd={mfpt_opt['gap']/gap_0:.1f}x")
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    # (e) Difference: MFPT-opt minus spectral-opt
    diff = (mfpt_opt['u'] - spec['u']).reshape(nx, ny)
    ax = axes[1, 1]
    c = ax.contourf(xg, yg, diff, levels=18, cmap='RdBu_r',
                    vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    ax.set_title(r'(e) $b^*_{\rm MFPT} - b^*_{\rm spectral}$' +
                 '\n(where the two optima disagree)')
    plt.colorbar(c, ax=ax, label=r'$\Delta u\;[k_BT]$')

    # (f) bar chart cross-evaluation
    ax = axes[1, 2]
    names = ['unbiased', 'flatten', 'spectral', 'mfpt_opt']
    nice  = ['unbiased', 'flatten', r'spectral $\gamma^*$', r'MFPT$^*$']
    mfpts  = [table[n]['mfpt_speedup'] for n in names]
    gaps   = [table[n]['gap_speedup'] for n in names]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w/2, mfpts, w, color='#d62728', label='MFPT speedup')
    ax.bar(x + w/2, gaps,  w, color='#1f77b4', label=r'$\gamma$ speedup')
    ax.set_xticks(x); ax.set_xticklabels(nice, rotation=15, fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel('speedup vs unbiased (log scale)')
    ax.set_title('(f) Cross-objective evaluation')
    ax.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax.grid(True, axis='y', ls=':', alpha=0.4)
    ax.legend(fontsize=8)
    for i, m, g in zip(x, mfpts, gaps):
        ax.text(i - w/2, m * 1.1, f'{m:.1f}', ha='center', fontsize=7,
                color='#d62728')
        ax.text(i + w/2, g * 1.1, f'{g:.1f}', ha='center', fontsize=7,
                color='#1f77b4')

    fig.suptitle(r'Three different one-shot bias objectives on the 2D grid '
                 r'($|u|\leq 3\,k_BT$, symmetric, zero-mean).  '
                 r'The MFPT-optimal bias and the spectral-$\gamma$ optimal '
                 r'bias are distinct.', fontsize=10.5)
    fig.savefig(f'{PATH}/fig_mfpt_vs_spectral.png',
                dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_mfpt_vs_spectral.png')


if __name__ == '__main__':
    main()
