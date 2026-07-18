"""
What bias profile maximises the spectral gap of the biased operator?

We optimise u_i (a real-valued field over the n states of the 2D grid network)
to MAXIMISE |lambda_2(K^b)|, i.e. MINIMISE the (negative) second eigenvalue.

This is the analytic ceiling for any *single-shot* bias scheme aiming at
fastest global equilibration (no multiple windows, no DHAM, no reweighting).
US protocols can never beat this if they are required to evaluate observables
via the slowest non-stationary mode.

Two parameterisations:
  (a) Polynomial basis in (x, y) up to order 4  (~15 coefficients): smooth,
      regularised, fast to optimise, visually interpretable.
  (b) Full per-state u_i (n parameters): theoretical ceiling, slower.

Outputs:
  fig_optimal_bias.png   side-by-side panels of
                            (i)   free-energy F (unbiased landscape)
                            (ii)  oracle flatten bias u = +kT log pi
                                  (the bias that uniformises pi)
                            (iii) optimum bias (polynomial)
                            (iv)  optimum bias (per-state, ceiling)
                            (v)   resulting biased free energy F_b
                            (vi)  scaling: |lambda_2| vs u-budget
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import analytic_lib as L


import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def gap_objective(u, K0, beta=1.0):
    """Returns -|lambda_2| (so that minimize gives maximum |lambda_2|).
       Robust to bad u (returns large value)."""
    if not np.all(np.isfinite(u)):
        return 0.0
    try:
        Kb = L.tilt_generator(K0, beta * u)
    except Exception:
        return 0.0
    try:
        gap = L.spectral_gap_K(Kb)
        if gap <= 0 or not np.isfinite(gap):
            return 0.0
        return -gap
    except Exception:
        return 0.0


def basis_poly(coords, max_order=4):
    """Polynomial basis in (x, y) up to total degree max_order."""
    x, y = coords[:, 0], coords[:, 1]
    feats = [np.ones_like(x)]
    for o in range(1, max_order + 1):
        for i in range(o + 1):
            feats.append(x ** (o - i) * y ** i)
    F = np.column_stack(feats)
    # standardise columns (skip the constant)
    F[:, 1:] = (F[:, 1:] - F[:, 1:].mean(axis=0)) / (F[:, 1:].std(axis=0) + 1e-10)
    return F


def project_to_budget(u, U_max=4.0):
    rng = u.max() - u.min()
    if rng > 2 * U_max:
        u = u * (2 * U_max / rng)
    return u - u.min()  # shift so min(u) = 0; max(u) <= 2 U_max


def optimise_polynomial(K0, coords, U_max=4.0, max_order=4,
                        seed=0) -> dict:
    """Optimise polynomial coefficients."""
    F = basis_poly(coords, max_order=max_order)
    n_feat = F.shape[1]
    rng = np.random.default_rng(seed)

    def fun(theta):
        u = F @ theta
        u = project_to_budget(u, U_max=U_max)
        return gap_objective(u, K0)

    best = None
    for trial in range(4):
        theta0 = rng.normal(scale=0.5, size=n_feat) if trial > 0 else np.zeros(n_feat)
        res = minimize(fun, theta0, method='Nelder-Mead',
                       options=dict(maxiter=3000, xatol=1e-5, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    u_best = project_to_budget(F @ best.x, U_max=U_max)
    return dict(u=u_best, theta=best.x, gap=-best.fun, basis=F, U_max=U_max)


def optimise_per_state(K0, U_max=4.0, n_starts=2, seed=0, maxiter=600) -> dict:
    """Free per-state optimisation.  Slow for large n; use coarse grid only."""
    n = K0.shape[0]
    rng = np.random.default_rng(seed)
    bounds = [(0.0, 2 * U_max)] * n

    best = None
    starts = [np.zeros(n)] + [rng.uniform(0, 2 * U_max, n) for _ in range(n_starts - 1)]
    # also seed with oracle-flatten projected to budget
    pi0 = L.stationary_distribution_from_K(K0)
    starts.append(project_to_budget(np.log(np.clip(pi0, 1e-300, None)), U_max=U_max))
    for u0 in starts:
        res = minimize(gap_objective, u0, args=(K0,),
                       method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=maxiter, gtol=1e-7))
        if best is None or res.fun < best.fun:
            best = res
    return dict(u=best.x, gap=-best.fun, U_max=U_max)


def main():
    # build a moderate grid for speed (n=20x10 = 200) with deep barrier
    K0, coords, F_target = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0,
                                                bottleneck=True)
    n = K0.shape[0]
    pi0 = L.stationary_distribution_from_K(K0)
    gap0 = L.spectral_gap_K(K0)
    print(f"Network n={n},  FE range = {(-np.log(pi0)).max() - (-np.log(pi0)).min():.2f}")
    print(f"Unbiased spectral gap = {gap0:.4e},  tau_0 = {1/gap0:.2f}")

    # Oracle flatten: u = +log pi (sign corrected)
    u_oracle = L.oracle_flatten_bias(pi0)
    u_oracle = project_to_budget(u_oracle, U_max=4.0)
    gap_oracle = L.spectral_gap_K(L.tilt_generator(K0, u_oracle))
    print(f"Oracle flatten (u-budget=4):  gap = {gap_oracle:.4e},  speedup = {gap_oracle/gap0:.2f}x")

    # Polynomial basis optimum
    print('\nOptimising polynomial (degree 4)...')
    poly = optimise_polynomial(K0, coords, U_max=4.0, max_order=4, seed=0)
    print(f"  Polynomial optimum: gap = {poly['gap']:.4e},  speedup = {poly['gap']/gap0:.2f}x")

    # Per-state optimum (theoretical ceiling)
    print('\nOptimising per-state (n parameters)... slow, please wait')
    per_state = optimise_per_state(K0, U_max=4.0, n_starts=3, maxiter=400)
    print(f"  Per-state optimum: gap = {per_state['gap']:.4e},  speedup = {per_state['gap']/gap0:.2f}x")

    # u-budget scaling
    print('\nu-budget scaling...')
    budgets = np.linspace(0.5, 8.0, 16)
    gap_oracle_sweep = []
    gap_poly_sweep = []
    gap_perstate_sweep = []
    for U in budgets:
        u_o = project_to_budget(L.oracle_flatten_bias(pi0), U_max=U)
        gap_oracle_sweep.append(L.spectral_gap_K(L.tilt_generator(K0, u_o)))
        ps = optimise_polynomial(K0, coords, U_max=U, max_order=4, seed=0)
        gap_poly_sweep.append(ps['gap'])
    gap_oracle_sweep = np.array(gap_oracle_sweep)
    gap_poly_sweep = np.array(gap_poly_sweep)

    # ---- Plot ----
    nx, ny = 20, 10
    x = coords[:, 0].reshape(nx, ny)
    y = coords[:, 1].reshape(nx, ny)
    F0 = -np.log(np.clip(pi0, 1e-300, None))
    F0 -= F0.min()

    fig = plt.figure(figsize=(13, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    c = ax.contourf(x, y, F0.reshape(nx, ny), levels=20, cmap='viridis')
    ax.set_title(f'(a) Unbiased free energy F  (gap={gap0:.2e})')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(c, ax=ax, label=r'$F\,[k_BT]$')

    ax = fig.add_subplot(gs[0, 1])
    c = ax.contourf(x, y, u_oracle.reshape(nx, ny), levels=20, cmap='coolwarm')
    ax.set_title(f'(b) Oracle flatten bias  u = +log π  '
                 f'(gap={gap_oracle:.2e}, {gap_oracle/gap0:.1f}×)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(c, ax=ax, label=r'$u\,[k_BT]$')

    ax = fig.add_subplot(gs[0, 2])
    c = ax.contourf(x, y, poly['u'].reshape(nx, ny), levels=20, cmap='coolwarm')
    ax.set_title(f"(c) Optimal polynomial (deg 4)  "
                 f"(gap={poly['gap']:.2e}, {poly['gap']/gap0:.1f}×)")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(c, ax=ax, label=r'$u\,[k_BT]$')

    ax = fig.add_subplot(gs[1, 0])
    c = ax.contourf(x, y, per_state['u'].reshape(nx, ny), levels=20, cmap='coolwarm')
    ax.set_title(f"(d) Optimal per-state bias (ceiling)  "
                 f"(gap={per_state['gap']:.2e}, {per_state['gap']/gap0:.1f}×)")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(c, ax=ax, label=r'$u\,[k_BT]$')

    # Resulting biased FE under per-state opt
    pi_b = L.biased_pi_from_reference(pi0, per_state['u'])
    F_b = -np.log(np.clip(pi_b, 1e-300, None))
    F_b -= F_b.min()
    ax = fig.add_subplot(gs[1, 1])
    c = ax.contourf(x, y, F_b.reshape(nx, ny), levels=20, cmap='viridis')
    ax.set_title(r'(e) Biased free energy $F^b = F - u$ from per-state optimum')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(c, ax=ax, label=r'$F^b\,[k_BT]$')

    # u-budget scaling
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(budgets, gap_oracle_sweep / gap0, 'o-', color='C0',
            label=r'oracle flatten ($u=+\log\pi$, clipped)')
    ax.plot(budgets, gap_poly_sweep / gap0, 's-', color='C3',
            label='optimal polynomial (deg 4)')
    ax.axhline(per_state['gap'] / gap0, color='k', ls=':', lw=1,
               label=f'per-state ceiling at U=4 ({per_state["gap"]/gap0:.0f}×)')
    ax.set_xlabel(r'$u$-budget $\max(u) - \min(u)$ $[k_BT]$')
    ax.set_ylabel(r'spectral-gap speedup $|\lambda_2^b|/|\lambda_2^0|$')
    ax.set_title('(f) Scaling with u-budget')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, ls=':', alpha=0.4)

    fig.suptitle('Optimal single-shot bias for fastest equilibration on the 2D grid '
                 '(n=200, FE range 8 kBT, deep barrier)', fontsize=11)
    fig.savefig(f'{PATH}/fig_optimal_bias.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved fig_optimal_bias.png')


if __name__ == '__main__':
    main()
