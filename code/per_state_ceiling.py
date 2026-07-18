"""
Per-state spectral-gap ceiling with analytic eigenvalue-perturbation gradient.

Replaces the finite-difference L-BFGS-B (which numerically stalled at
~55 speedup on the 2D grid) with the exact gradient of gamma = -lambda_2
with respect to each u_i:

    d gamma / d u_i  =  -  ( w_2^T  (dK^b/du_i)  v_2 )  /  ( w_2^T  v_2 )

where v_2 (right), w_2 (left) are the eigenpair of K^b for lambda_2.

The action  (dK^b/du_i) v_2  is sparse and closed form (derivation above);
total gradient cost is O(n^2) per evaluation plus one O(n^3) eig.
"""
from __future__ import annotations
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import analytic_lib as L
import optimal_spectral_bias as O


import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ----------------------------------------------------------------------
#  Spectral gap + analytic gradient
# ----------------------------------------------------------------------

def _sort_eigs(K: np.ndarray):
    """Right eigenvectors, sorted descending by Re(lambda).  Returns
       (lams, V_right).  lambda_1 = 0; lambda_2 is the spectral gap (negative)."""
    w, V = np.linalg.eig(K)
    order = np.argsort(-np.real(w))
    return w[order], V[:, order]


def gap_and_grad(u: np.ndarray, K0: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (gamma, dgamma/du_i).  gamma = -lambda_2(K^b)."""
    n = K0.shape[0]
    Kb = L.tilt_generator(K0, u)
    # Right eigenpair for lambda_2
    w_r, V_r = _sort_eigs(Kb)
    lam2 = float(np.real(w_r[1]))
    v2 = np.real(V_r[:, 1])
    # Left eigenpair for lambda_2  (eigvecs of Kb.T)
    w_l, V_l = _sort_eigs(Kb.T)
    # Match: left lam2 should equal right lam2 (for a real spectrum)
    # find the column whose eigenvalue is closest to lam2
    idx = int(np.argmin(np.abs(np.real(w_l) - lam2)))
    w2 = np.real(V_l[:, idx])
    # Biorthonormalisation
    denom = float(np.dot(w2, v2))
    if abs(denom) < 1e-14:
        # numerically degenerate; fall back to finite diff would be needed,
        # but for typical DB-preserving Kb this won't trigger
        raise ValueError("degenerate eigenpair")
    # gamma and its gradient
    gamma = -lam2

    # (M_i v2)[p]:
    #   p == i:  lam2 * v2[i] / 2
    #   p != i:  Kb[p, i] * (v2[p] - v2[i]) / 2
    #
    # Compute (M_i v2) for all i in one matrix op.
    # Let A[p, i] = Kb[p, i] * (v2[p] - v2[i]) for p != i,
    # and          lam2 * v2[i]                for p == i.
    A = Kb * (v2[:, None] - v2[None, :])      # rows p, cols i
    np.fill_diagonal(A, lam2 * v2)            # p == i case
    A = 0.5 * A
    # numerator[i] = sum_p w2[p] * A[p, i]
    numerator = w2 @ A                         # shape (n,)

    dgamma_du = -numerator / denom
    return gamma, dgamma_du


def optimise_per_state_analytic(K0: np.ndarray, U_max: float = 3.0,
                                warm_u: np.ndarray = None,
                                maxiter: int = 2000) -> dict:
    n = K0.shape[0]
    pi0 = L.stationary_distribution_from_K(K0)
    bounds = [(-U_max, U_max)] * n

    def fg(u):
        try:
            g, grad = gap_and_grad(u, K0)
        except Exception:
            return 1e12, np.zeros(n)
        return -float(g), -grad   # minimise -gamma

    starts = []
    # 1. polynomial warm start
    if warm_u is not None:
        starts.append(O.project_symmetric(warm_u, U_max=U_max))
    # 2. budget-clipped flatten
    starts.append(O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)),
                                       U_max=U_max))
    # 3. zeros
    starts.append(np.zeros(n))

    best = None
    for u0 in starts:
        t0 = time.time()
        res = minimize(fg, u0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=maxiter, gtol=1e-9, ftol=1e-11))
        dt = time.time() - t0
        print(f"    start ‖u0‖={np.linalg.norm(u0):.2f}  ->  gamma={-res.fun:.5e}"
              f"  iters={res.nit}  time={dt:.1f}s")
        if best is None or res.fun < best.fun:
            best = res
    # The biased generator K^b is invariant under constant shifts of u, so the
    # absolute mean is a gauge degree of freedom that L-BFGS-B leaves
    # under-determined.  Report u as the *optimizer's* solution (which already
    # satisfies |u_i| <= U_max) without forcing zero-mean (which would only be
    # valid by re-scaling, not a simple shift).
    u_best = best.x.copy()
    Kb = L.tilt_generator(K0, u_best)
    return dict(u=u_best, gamma=float(L.spectral_gap_K(Kb)),
                gamma_from_opt=float(-best.fun))


# ----------------------------------------------------------------------
#  Validate the analytic gradient against finite differences
# ----------------------------------------------------------------------

def verify_gradient(K0, u, eps=1e-4):
    g0, grad = gap_and_grad(u, K0)
    rng = np.random.default_rng(0)
    n = len(u)
    fd = np.zeros(n)
    # sample a few components for speed
    idxs = rng.choice(n, size=min(8, n), replace=False)
    for i in idxs:
        up = u.copy(); up[i] += eps
        um = u.copy(); um[i] -= eps
        gp = -L.spectral_gap_K(L.tilt_generator(K0, up))
        gm = -L.spectral_gap_K(L.tilt_generator(K0, um))
        fd[i] = -(gp - gm) / (2 * eps)
    err = np.max(np.abs(fd[idxs] - grad[idxs]) /
                 (np.max(np.abs(grad[idxs])) + 1e-12))
    return g0, err, idxs, grad[idxs], fd[idxs]


def main():
    # Build the same 2D grid as the spectral preconditioning study
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0,
                                          bottleneck=True)
    n = K0.shape[0]
    pi0 = L.stationary_distribution_from_K(K0)
    gap0 = L.spectral_gap_K(K0)
    edges = O.nearest_neighbour_edges(K0)
    U_max = 3.0
    print(f"Network n={n},  gap0 = {gap0:.4e},  U_max = {U_max}")

    # 1. Gradient verification
    print("\nVerifying analytic gradient vs finite differences...")
    u_test = O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)),
                                  U_max=U_max)
    g0, err, idxs, gana, gfd = verify_gradient(K0, u_test)
    print(f"  gamma at flatten   = {g0:.5e}")
    print(f"  max relative error = {err:.3e}")
    for i, ga, gf in zip(idxs, gana, gfd):
        print(f"    i={i:3d}  analytic={ga:+.5e}  fd={gf:+.5e}  diff={ga-gf:+.2e}")
    assert err < 1e-3, f"Gradient mismatch: max rel err = {err}"

    # 2. Run polynomial first (for warm start)
    print("\nOptimising polynomial (deg 4) for warm start...")
    spec_poly = O.optimise_polynomial(K0, coords, U_max=U_max, max_order=4,
                                       alpha_smooth=0.001, edges=edges,
                                       n_restarts=3)
    print(f"  polynomial gap     = {spec_poly['gap']:.5e}  "
          f"({spec_poly['gap']/gap0:.2f}x)")

    # 3. Per-state with analytic gradient
    print("\nOptimising per-state with analytic gradient...")
    ps = optimise_per_state_analytic(K0, U_max=U_max,
                                       warm_u=spec_poly['u'], maxiter=3000)
    print(f"  per-state final gamma = {ps['gamma']:.5e}  "
          f"({ps['gamma']/gap0:.2f}x)")

    # Save
    save = dict(
        n=n, gap0=float(gap0), U_max=U_max,
        gap_polynomial=float(spec_poly['gap']),
        speedup_polynomial=float(spec_poly['gap']/gap0),
        gap_per_state=float(ps['gamma']),
        speedup_per_state=float(ps['gamma']/gap0),
        u_polynomial=spec_poly['u'].tolist(),
        u_per_state=ps['u'].tolist(),
    )
    with open(f'{PATH}/per_state_ceiling.json', 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved per_state_ceiling.json")

    # ---------- Figure ----------
    nx, ny = 20, 10
    xg = coords[:, 0].reshape(nx, ny)
    yg = coords[:, 1].reshape(nx, ny)
    F0 = -np.log(np.clip(pi0, 1e-300, None)); F0 -= F0.min()
    u_flat = O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)),
                                  U_max=U_max)
    gap_flat = L.spectral_gap_K(L.tilt_generator(K0, u_flat))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    c = ax.contourf(xg, yg, u_flat.reshape(nx, ny), levels=18,
                    cmap='coolwarm', vmin=-U_max, vmax=U_max)
    ax.set_title(f'(a) Budget-clipped flatten\n'
                 fr'$\gamma/\gamma_0 = {gap_flat/gap0:.1f}\times$')
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    ax = axes[0, 1]
    c = ax.contourf(xg, yg, spec_poly['u'].reshape(nx, ny), levels=18,
                    cmap='coolwarm', vmin=-U_max, vmax=U_max)
    ax.set_title(f'(b) Smooth polynomial (deg 4)\n'
                 fr'$\gamma/\gamma_0 = {spec_poly["gap"]/gap0:.1f}\times$')
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    ax = axes[1, 0]
    c = ax.contourf(xg, yg, ps['u'].reshape(nx, ny), levels=18,
                    cmap='coolwarm', vmin=-U_max, vmax=U_max)
    ax.set_title(f'(c) Per-state ceiling (analytic gradient)\n'
                 fr'$\gamma/\gamma_0 = {ps["gamma"]/gap0:.1f}\times$')
    plt.colorbar(c, ax=ax, label=r'$u\;[k_BT]$')

    # Difference: per-state minus polynomial
    diff = (ps['u'] - spec_poly['u']).reshape(nx, ny)
    ax = axes[1, 1]
    c = ax.contourf(xg, yg, diff, levels=18, cmap='RdBu_r',
                    vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    ax.set_title(r'(d) Per-state minus polynomial')
    plt.colorbar(c, ax=ax, label=r'$\Delta u\;[k_BT]$')

    fig.suptitle('Per-state spectral-gap ceiling with analytic gradient '
                 fr'($|u|\leq {U_max}\,k_BT$, 2D grid, $n={n}$)', fontsize=11)
    fig.savefig(f'{PATH}/fig_per_state_ceiling.png',
                dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_per_state_ceiling.png')


if __name__ == '__main__':
    main()
