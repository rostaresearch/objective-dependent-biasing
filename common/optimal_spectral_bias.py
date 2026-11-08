"""
Optimal single-shot spectral preconditioning of a kinetic network.

Problem:
    Given a row-generator K0 with stationary pi0, find a bias profile u_i
    (one-shot, a single applied potential) that maximises the spectral gap
    of the biased dynamics K^b, subject to a *symmetric* bias budget

        |u_i| <= U_max         (or equivalently  max u - min u <= 2 U_max,
                                 with zero-mean gauge fixed).

This is NOT umbrella sampling and NOT estimator-variance optimisation; it
is global preconditioning of the relaxation dynamics.  US and barrier/rate
estimation are handled separately.

We compare:
  (a) free energy F (unbiased)
  (b) budget-constrained flattening bias  u = +kT log pi, clipped + centred
  (c) smooth polynomial optimum (degree 4) with explicit nearest-neighbour
      gradient penalty
  (d) F^b = F + u for the polynomial optimum   (visualises the actual
      remaining landscape)
  (e) per-state graph-theoretic ceiling with smoothness regularisation
  (f) reactive-current magnitude under the polynomial optimum
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
DATA = os.path.join(PATH, 'data')
FIGURES = os.path.join(PATH, 'figures')
# ---------- bias projection: symmetric, zero-mean ----------------------------

def project_symmetric(u: np.ndarray, U_max: float = 4.0) -> np.ndarray:
    """Centre then clip to [-U_max, U_max]."""
    u = u - u.mean()
    return np.clip(u, -U_max, U_max)


# ---------- smoothness penalty -----------------------------------------------

def nearest_neighbour_edges(K0: np.ndarray, threshold: float = 1e-12):
    """Pairs (i,j), i<j, where K0[i,j] > threshold (connected edges)."""
    n = K0.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if K0[i, j] > threshold:
                pairs.append((i, j))
    return np.array(pairs)


def smoothness_penalty(u: np.ndarray, edges: np.ndarray) -> float:
    """Mean squared difference over nearest-neighbour edges (normalised so
       the penalty does not scale with system size or edge count)."""
    du = u[edges[:, 0]] - u[edges[:, 1]]
    return float(np.dot(du, du) / max(len(edges), 1))


# ---------- objective --------------------------------------------------------

def gap_obj(u: np.ndarray, K0: np.ndarray) -> float:
    if not np.all(np.isfinite(u)):
        return 0.0
    try:
        Kb = L.tilt_generator(K0, u)
        g = L.spectral_gap_K(Kb)
        if g <= 0 or not np.isfinite(g):
            return 0.0
        return -g
    except Exception:
        return 0.0


# ---------- polynomial basis -------------------------------------------------

def basis_poly(coords: np.ndarray, max_order: int = 4) -> np.ndarray:
    x, y = coords[:, 0], coords[:, 1]
    feats = [np.ones_like(x)]
    for o in range(1, max_order + 1):
        for i in range(o + 1):
            feats.append(x ** (o - i) * y ** i)
    F = np.column_stack(feats)
    F[:, 1:] = (F[:, 1:] - F[:, 1:].mean(axis=0)) / (F[:, 1:].std(axis=0) + 1e-12)
    return F


def optimise_polynomial(K0: np.ndarray, coords: np.ndarray,
                        U_max: float = 4.0, max_order: int = 4,
                        alpha_smooth: float = 0.0, edges=None,
                        n_restarts: int = 4, seed: int = 0,
                        seed_with_flatten: bool = True,
                        maxiter: int = 4000) -> dict:
    Fb = basis_poly(coords, max_order=max_order)
    n_feat = Fb.shape[1]
    rng = np.random.default_rng(seed)
    pi0 = L.stationary_distribution_from_K(K0)

    def cost(theta):
        u = Fb @ theta
        u = project_symmetric(u, U_max=U_max)
        c = gap_obj(u, K0)
        if alpha_smooth > 0 and edges is not None:
            c += alpha_smooth * smoothness_penalty(u, edges)
        return c

    # Seed: zeros + least-squares fit to budget-clipped flattening direction
    starts = [np.zeros(n_feat)]
    if seed_with_flatten:
        u_flat = project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=U_max)
        theta_flat, *_ = np.linalg.lstsq(Fb, u_flat, rcond=None)
        starts.append(theta_flat)
    for _ in range(n_restarts - len(starts)):
        starts.append(rng.normal(scale=0.6, size=n_feat))

    best = None
    for theta0 in starts:
        res = minimize(cost, theta0, method='Nelder-Mead',
                       options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-7))
        if best is None or res.fun < best.fun:
            best = res
    u_best = project_symmetric(Fb @ best.x, U_max=U_max)
    gap = L.spectral_gap_K(L.tilt_generator(K0, u_best))
    return dict(u=u_best, theta=best.x, gap=float(gap), U_max=U_max)


def optimise_per_state(K0: np.ndarray, U_max: float = 4.0,
                       alpha_smooth: float = 0.01, edges=None,
                       n_starts: int = 3, maxiter: int = 600, seed: int = 0,
                       warm_u: np.ndarray | None = None) -> dict:
    n = K0.shape[0]
    rng = np.random.default_rng(seed)
    bounds = [(-U_max, U_max)] * n

    def cost(u):
        u = u - u.mean()
        c = gap_obj(u, K0)
        if alpha_smooth > 0 and edges is not None:
            c += alpha_smooth * smoothness_penalty(u, edges)
        return c

    pi0 = L.stationary_distribution_from_K(K0)
    starts = []
    # 1. budget-flattening
    starts.append(project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=U_max))
    # 2. warm start from polynomial optimum if provided
    if warm_u is not None:
        starts.append(warm_u.copy())
    # 3. random starts
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(rng.uniform(-U_max / 2, U_max / 2, n))

    best = None
    for u0 in starts:
        res = minimize(cost, u0, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=maxiter, gtol=1e-8))
        if best is None or res.fun < best.fun:
            best = res
    u_best = project_symmetric(best.x, U_max=U_max)
    gap = L.spectral_gap_K(L.tilt_generator(K0, u_best))
    return dict(u=u_best, gap=float(gap), U_max=U_max)


# ---------- reactive current under a generator -------------------------------

def reactive_current_magnitudes(K: np.ndarray, pi: np.ndarray,
                                A, B) -> np.ndarray:
    """Per-state out-going reactive current magnitude under TPT."""
    q = L.committor_K(K, A, B)
    n = K.shape[0]
    flux_out = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            f_ij = pi[i] * K[i, j] * q[j] * (1.0 - q[i])
            f_ji = pi[j] * K[j, i] * q[i] * (1.0 - q[j])
            net = max(f_ij - f_ji, 0.0)
            flux_out[i] += net
    return flux_out


# ---------- main -------------------------------------------------------------

def main():
    # 2D grid: barrier 4 kT, bottleneck on, n = 20x10 = 200 for speed
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0,
                                          bottleneck=True)
    n = K0.shape[0]
    pi0 = L.stationary_distribution_from_K(K0)
    F0 = -np.log(np.clip(pi0, 1e-300, None)); F0 -= F0.min()
    gap0 = L.spectral_gap_K(K0)
    edges = nearest_neighbour_edges(K0)
    print(f"Network n={n},  FE range = {F0.max():.2f} kT,  nedges = {len(edges)}")
    print(f"  unbiased gap = {gap0:.4e},  tau_0 = {1/gap0:.2f}")

    U_max = 3.0   # symmetric: bias in [-3, +3] kT
    print(f"  bias budget: |u_i| <= {U_max} kT  (symmetric, zero-mean)")

    # (b) Budget-constrained flattening bias  u = +log pi, clipped + centred
    u_flat = project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=U_max)
    gap_flat = L.spectral_gap_K(L.tilt_generator(K0, u_flat))
    print(f"  budget-clipped flattening: gap={gap_flat:.4e}, speedup={gap_flat/gap0:.2f}x")

    # (c) Polynomial optimum, with smoothness penalty alpha=0.005
    poly = optimise_polynomial(K0, coords, U_max=U_max, max_order=4,
                                alpha_smooth=0.001, edges=edges, n_restarts=3)
    print(f"  polynomial (deg 4, sym budget, alpha=0.005): gap={poly['gap']:.4e}, "
          f"speedup={poly['gap']/gap0:.2f}x", flush=True)

    # (e) Per-state ceiling with smoothness alpha=0.1 (stronger to keep physical)
    print('  per-state optimisation (graph-theoretic ceiling, alpha_smooth=0.01)...', flush=True)
    # Unconstrained graph-theoretic ceiling (no smoothness penalty)
    ps = optimise_per_state(K0, U_max=U_max, alpha_smooth=0.0, edges=edges,
                            n_starts=3, maxiter=400, warm_u=poly['u'])
    print(f"  per-state ceiling: gap={ps['gap']:.4e}, speedup={ps['gap']/gap0:.2f}x",
          flush=True)

    # Reactive currents
    A = np.where(coords[:, 0] < -0.85)[0]
    B = np.where(coords[:, 0] >  0.85)[0]
    flux_unb  = reactive_current_magnitudes(K0, pi0, A, B)
    pi_poly = L.biased_pi_from_reference(pi0, poly['u'])
    flux_poly = reactive_current_magnitudes(
        L.tilt_generator(K0, poly['u']), pi_poly, A, B)

    # Budget sweep (fewer points + 2 restarts to keep runtime sane)
    print('  budget sweep...', flush=True)
    Bs = np.array([0.5, 1.5, 3.0, 4.5, 6.0])
    g_flat, g_poly, g_ps = [], [], []
    for Bv in Bs:
        u = project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=Bv)
        g_flat.append(L.spectral_gap_K(L.tilt_generator(K0, u)) / gap0)
        p = optimise_polynomial(K0, coords, U_max=Bv, max_order=4,
                                alpha_smooth=0.001, edges=edges, n_restarts=2)
        g_poly.append(p['gap'] / gap0)
        print(f"    budget {Bv:.1f}: flat={g_flat[-1]:.2f}x  poly={g_poly[-1]:.2f}x",
              flush=True)
    g_ps = [(U_max, ps['gap'] / gap0)]

    # Save numerical results before plotting (in case mpl fails)
    import json
    save = dict(
        n=n, FE_range=float(F0.max()), gap0=float(gap0), tau0=float(1/gap0),
        U_max=float(U_max),
        u_flat=u_flat.tolist(), gap_flat=float(gap_flat),
        speedup_flat=float(gap_flat/gap0),
        u_poly=poly['u'].tolist(), gap_poly=float(poly['gap']),
        speedup_poly=float(poly['gap']/gap0),
        u_ps=ps['u'].tolist(), gap_ps=float(ps['gap']),
        speedup_ps=float(ps['gap']/gap0),
        budgets=Bs.tolist(), speedup_flat_sweep=g_flat, speedup_poly_sweep=g_poly,
        speedup_ps_sweep=[(b, v) for b, v in g_ps],
        flux_unb=flux_unb.tolist(), flux_poly=flux_poly.tolist(),
    )
    with open(f'{DATA}/spectral_results.json', 'w') as f:
        json.dump(save, f, indent=2)
    print(f'Saved spectral_results.json', flush=True)

    # ---------- Figure ----------
    nx, ny = 20, 10
    xg = coords[:, 0].reshape(nx, ny)
    yg = coords[:, 1].reshape(nx, ny)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), constrained_layout=True)

    # (a) F
    ax = axes[0, 0]
    c = ax.contourf(xg, yg, F0.reshape(nx, ny), levels=18, cmap='viridis')
    ax.set_title(f'(a) Unbiased free energy F\n'
                 f'spec gap = {gap0:.2e}, tau_0 = {1/gap0:.0f}')
    plt.colorbar(c, ax=ax, label=r'$F\;[k_BT]$')

    # (b) Budget-clipped flattening
    ax = axes[0, 1]
    c = ax.contourf(xg, yg, u_flat.reshape(nx, ny), levels=18, cmap='coolwarm',
                    vmin=-U_max, vmax=U_max)
    ax.set_title(f'(b) Budget-clipped flattening\n'
                 fr'$u = \mathrm{{clip}}_{{[-{U_max},{U_max}]}}(\log\pi)$,'
                 f' speedup = {gap_flat/gap0:.1f}x')
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    # (c) Polynomial optimum
    ax = axes[0, 2]
    c = ax.contourf(xg, yg, poly['u'].reshape(nx, ny), levels=18, cmap='coolwarm',
                    vmin=-U_max, vmax=U_max)
    ax.set_title(f'(c) Smooth polynomial optimum (deg 4)\n'
                 fr'$|u|\leq{U_max}$, $\alpha_{{\rm smooth}}=0.005$,'
                 f' speedup = {poly["gap"]/gap0:.1f}x')
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    # (d) Biased FE under polynomial optimum.
    # pi^(b) ∝ pi^(0) e^{-u}  =>  F^(b) = F^(0) + u  (F - u would deepen, not fill).
    F_poly = F0 + poly['u']; F_poly -= F_poly.min()
    ax = axes[1, 0]
    c = ax.contourf(xg, yg, F_poly.reshape(nx, ny), levels=18, cmap='viridis')
    ax.set_title(r'(d) Biased free energy $F^{b} = F + u$ (polynomial opt)')
    plt.colorbar(c, ax=ax, label=r'$F^b\;[k_BT]$')

    # (e) Per-state (graph-theoretic ceiling)
    ax = axes[1, 1]
    c = ax.contourf(xg, yg, ps['u'].reshape(nx, ny), levels=18, cmap='coolwarm',
                    vmin=-U_max, vmax=U_max)
    ax.set_title(f'(e) Per-state ceiling (smoothed)\n'
                 fr'$n$ params, $\alpha_{{\rm smooth}}=0.1$,'
                 f' speedup = {ps["gap"]/gap0:.1f}x')
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    # (f) Reactive current under polynomial vs unbiased
    ax = axes[1, 2]
    # log of ratio so we see WHERE current changed
    ratio = (flux_poly + 1e-30) / (flux_unb + 1e-30)
    log_ratio = np.log10(ratio).reshape(nx, ny)
    c = ax.contourf(xg, yg, log_ratio, levels=18, cmap='RdBu_r',
                    vmin=-2, vmax=2)
    ax.set_title(r'(f) Reactive current ratio'
                 '\n' r'$\log_{10}(\Phi^{\rm poly}/\Phi^{\rm unb})$ per state')
    plt.colorbar(c, ax=ax, label=r'$\log_{10}$ ratio')

    fig.suptitle('Optimal one-shot spectral preconditioning bias on the 2D grid '
                 r'$|u|\leq 3\,k_BT$ (symmetric budget, zero-mean), '
                 'edge-smoothness regularised', fontsize=11)
    fig.savefig(f'{FIGURES}/fig_spectral_bias.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_spectral_bias.png')

    # ---------- Second figure: budget scaling ----------
    fig2, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(Bs, g_flat, 'o-', color='C0', label='budget-clipped flattening')
    ax.plot(Bs, g_poly, 's-', color='C3', label='smooth polynomial optimum')
    if g_ps:
        ax.plot([b for b, _ in g_ps], [v for _, v in g_ps],
                'k*', ms=12, label='per-state ceiling (graph-theoretic)')
    ax.set_xlabel(r'symmetric bias budget $U_{\max}$ $[k_BT]$')
    ax.set_ylabel(r'spectral-gap speedup  $\gamma_b/\gamma_0$')
    ax.set_yscale('log')
    ax.set_title('Spectral-gap speedup vs symmetric bias budget')
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(fontsize=9)
    fig2.savefig(f'{FIGURES}/fig_spectral_budget.png', dpi=170, bbox_inches='tight')
    plt.close(fig2)
    print('Saved fig_spectral_budget.png')


if __name__ == '__main__':
    main()
