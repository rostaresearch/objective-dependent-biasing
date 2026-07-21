"""
Per-state MFPT ceiling via analytic adjoint gradient.

Solves
   b* = argmin_b  MFPT(K^(b), A -> B)
        subject to |u_i| <= U_max,
with the full per-state freedom, using the adjoint-state gradient

   d g / d u_i  =  - (1/2) [ lambda_i (K^b m)_i
                            + (K^{b.T} (lambda * m))_i
                            - m_i (K^{b.T} lambda)_i ]

where  M_NN m_N = -1  and  M_NN^T lambda_N = a  (a = 1/|A| on A, 0 else).
Embeds m_N, lambda_N to length-n with zeros on the target B.
"""
from __future__ import annotations
import os
import time
import numpy as np
from scipy.optimize import minimize
import analytic_lib as L

# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mfpt_and_grad(u: np.ndarray, K0: np.ndarray,
                  A_set, B_set) -> tuple[float, np.ndarray]:
    """Return (mfpt_AB, gradient).  All states must be 0..n-1.
       A_set and B_set are arrays of state indices (disjoint)."""
    n = K0.shape[0]
    Kb = L.tilt_generator(K0, u)

    # Restricted system M_NN m_N = -1
    N_mask = np.ones(n, dtype=bool)
    N_mask[B_set] = False
    N_idx = np.where(N_mask)[0]
    M_NN = Kb[np.ix_(N_idx, N_idx)]
    m_N = np.linalg.solve(M_NN, -np.ones(len(N_idx)))

    # Adjoint:  M_NN^T lambda_N = a
    a = np.zeros(len(N_idx))
    for i_global in A_set:
        # find local position of A_set[i_global] in N_idx
        local = np.searchsorted(N_idx, i_global)
        a[local] = 1.0 / len(A_set)
    lam_N = np.linalg.solve(M_NN.T, a)

    # MFPT(A -> B)
    full_m = np.zeros(n)
    full_m[N_idx] = m_N
    g = float(a @ m_N)   # = (1/|A|) sum_{i in A} m_N[i]  (since A subset N)

    # Embed to full-n
    full_lam = np.zeros(n)
    full_lam[N_idx] = lam_N

    # Three vector quantities, each O(n^2)
    alpha = Kb @ full_m              # (K^b m)_i
    beta  = Kb.T @ (full_lam * full_m)  # (K^{b.T} (lam * m))_i
    gamma = Kb.T @ full_lam          # (K^{b.T} lam)_i

    # Gradient: dg/du_i = -(1/2) [lam_i alpha_i + beta_i - m_i gamma_i]
    grad = -0.5 * (full_lam * alpha + beta - full_m * gamma)
    return g, grad


def optimise_mfpt_per_state(K0, A_set, B_set, U_max=3.0,
                            warm_u=None, maxiter=2000, verbose=True):
    n = K0.shape[0]
    bounds = [(-U_max, U_max)] * n
    pi0 = L.stationary_distribution_from_K(K0)

    def fg(u):
        try:
            g, gr = mfpt_and_grad(u, K0, A_set, B_set)
        except Exception:
            return 1e12, np.zeros(n)
        return float(g), gr

    starts = []
    if warm_u is not None:
        starts.append(np.clip(warm_u, -U_max, U_max))
    # also seed with budget-clipped flattening
    import optimal_spectral_bias as O
    starts.append(O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)),
                                       U_max=U_max))
    starts.append(np.zeros(n))

    best = None
    for u0 in starts:
        t0 = time.time()
        res = minimize(fg, u0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=maxiter, gtol=1e-9, ftol=1e-11))
        dt = time.time() - t0
        if verbose:
            print(f"    start ‖u0‖={np.linalg.norm(u0):.2f}  ->  mfpt={res.fun:.5g}"
                  f"  iters={res.nit}  time={dt:.1f}s")
        if best is None or res.fun < best.fun:
            best = res
    u_best = best.x.copy()
    Kb = L.tilt_generator(K0, u_best)
    return dict(u=u_best,
                mfpt=float(L.mfpt_K(Kb, A_set, B_set)),
                gap=float(L.spectral_gap_K(Kb)))


# ----------------------------------------------------------------------
#  Demo / test
# ----------------------------------------------------------------------
def _verify_gradient(K0, u, A_set, B_set, eps=1e-4):
    g0, grad = mfpt_and_grad(u, K0, A_set, B_set)
    rng = np.random.default_rng(0)
    n = len(u)
    idxs = rng.choice(n, size=8, replace=False)
    fd = np.zeros(8)
    for k, i in enumerate(idxs):
        up = u.copy(); up[i] += eps
        um = u.copy(); um[i] -= eps
        gp = L.mfpt_K(L.tilt_generator(K0, up), A_set, B_set)
        gm = L.mfpt_K(L.tilt_generator(K0, um), A_set, B_set)
        fd[k] = (gp - gm) / (2 * eps)
    err = np.max(np.abs(fd - grad[idxs]) /
                 (np.max(np.abs(grad[idxs])) + 1e-12))
    return g0, err, idxs, grad[idxs], fd


def main():
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0,
                                          bottleneck=True)
    n = K0.shape[0]
    pi0 = L.stationary_distribution_from_K(K0)
    A_set = np.where(coords[:, 0] < -0.85)[0]
    B_set = np.where(coords[:, 0] >  0.85)[0]
    U_max = 3.0

    mfpt_0 = L.mfpt_K(K0, A_set, B_set)
    gap_0  = L.spectral_gap_K(K0)
    print(f"Network n={n},  unbiased MFPT={mfpt_0:.4g},  gap={gap_0:.4e}")

    # 1. Gradient verification
    print("\nVerifying analytic gradient vs finite differences...")
    import optimal_spectral_bias as O
    u_test = O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)),
                                  U_max=U_max)
    g0, err, idxs, gana, gfd = _verify_gradient(K0, u_test, A_set, B_set)
    print(f"  mfpt at flatten    = {g0:.5g}")
    print(f"  max relative error = {err:.3e}")
    for i, ga, gf in zip(idxs, gana, gfd):
        print(f"    i={i:3d}  analytic={ga:+.5e}  fd={gf:+.5e}")

    # 2. Polynomial warm start (reuse from MFPT poly study)
    print("\nUsing polynomial warm start from mfpt_results.json...")
    import json
    try:
        with open(os.path.join(PATH, 'mfpt_results.json')) as f:
            warm = np.asarray(json.load(f)['u_mfpt'])
    except Exception:
        warm = None
        print("  (no warm start found)")

    # 3. Per-state MFPT
    print("\nOptimising per-state MFPT (analytic gradient)...")
    res = optimise_mfpt_per_state(K0, A_set, B_set, U_max=U_max,
                                    warm_u=warm, maxiter=3000)
    print(f"\n  per-state final MFPT = {res['mfpt']:.5g} "
          f"({mfpt_0/res['mfpt']:.2f}x speedup)")
    print(f"  per-state final gap  = {res['gap']:.5e} "
          f"({res['gap']/gap_0:.2f}x speedup)")

    # Save
    save = dict(
        n=n, mfpt_0=float(mfpt_0), gap_0=float(gap_0), U_max=U_max,
        u_per_state=res['u'].tolist(),
        mfpt_per_state=float(res['mfpt']),
        gap_per_state=float(res['gap']),
        mfpt_speedup_per_state=float(mfpt_0 / res['mfpt']),
        gap_speedup_per_state=float(res['gap'] / gap_0),
    )
    with open(os.path.join(PATH, 'mfpt_per_state.json'),
              'w') as f:
        json.dump(save, f, indent=2)
    print('  Saved mfpt_per_state.json')


if __name__ == '__main__':
    main()
