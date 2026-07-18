"""Option B: pure-box polynomial (u = clip(p)) for BOTH objectives.

Needed if we report the B-vs-C ladder. B is the review's exact ask: the common feasible
set is the box |b_i| <= U (equivalently the shift-invariant max(b)-min(b) <= 2U), reached by
clipping, so the search space is clip(W) with W = {deg-4 polys} INCLUDING the constant term.
The shipped code searched clip(V), V = zero-mean polys, discarding the pre-clip offset --
a strict subset, worth 27%.

B is not smooth (it saturates), which is precisely why it is informative next to C:
same 15 coefficients, only difference is whether the bias may saturate the budget.

Warm-started from the per-state optima fit into the polynomial space, then analytic-gradient
BFGS. The clip's gradient is masked outside the box (a valid subgradient).
"""
from __future__ import annotations
import sys, json, time
import numpy as np
sys.path.insert(0, r'C:\Users\edina\Dropbox\MSM_Roundtable_2026')
from scipy.optimize import minimize
from scipy.io import loadmat
import analytic_lib as L
import optimal_spectral_bias as O
from per_state_ceiling import gap_and_grad
from mfpt_per_state import mfpt_and_grad

PATH = r'C:\Users\edina\Dropbox\MSM_Roundtable_2026'
ALPHA = 0.001


def pen_and_grad(u, edges):
    a, b = edges[:, 0], edges[:, 1]
    du = u[a] - u[b]
    ne = max(len(edges), 1)
    g = np.zeros_like(u)
    np.add.at(g, a, 2.0 * du / ne)
    np.add.at(g, b, -2.0 * du / ne)
    return float(np.dot(du, du) / ne), g


def make_fg(kind, K0, Fb, U, edges, A=None, B=None):
    nf = Fb.shape[1]

    def fg(theta):
        p = Fb @ theta
        u = np.clip(p, -U, U)
        mask = (np.abs(p) < U).astype(float)      # clip kills the gradient outside
        try:
            if kind == 'gap':
                val, dval = gap_and_grad(u, K0); f, df = -val, -dval
            else:
                val, dval = mfpt_and_grad(u, K0, A, B); f, df = val, dval
        except Exception:
            return 1e12, np.zeros(nf)
        if not np.isfinite(f):
            return 1e12, np.zeros(nf)
        pn, dpn = pen_and_grad(u, edges)
        return float(f + ALPHA * pn), Fb.T @ ((df + ALPHA * dpn) * mask)
    return fg


def seeds(Fb, U, pi0):
    ks = [np.clip(np.log(np.clip(pi0, 1e-300, None)) -
                  np.log(np.clip(pi0, 1e-300, None)).mean(), -U, U)]
    d = loadmat(f'{PATH}/grid_2d_data.mat')
    for key in ('u_ps_spec', 'u_poly_spec', 'u_ps_mfpt'):
        if key in d:
            ks.append(np.asarray(d[key], float).ravel(order='C'))
    S = []
    for u in ks:
        th, *_ = np.linalg.lstsq(Fb, u, rcond=None)
        S.append(th)
        S.append(th * 1.6)          # push harder into saturation
    rng = np.random.default_rng(0)
    for _ in range(4):
        S.append(rng.normal(scale=1.2, size=Fb.shape[1]))
    return S


def run(kind, K0, Fb, U, edges, pi0, A=None, B=None):
    fg = make_fg(kind, K0, Fb, U, edges, A, B)
    best, vals = None, []
    for t0 in seeds(Fb, U, pi0):
        r = minimize(fg, t0, jac=True, method='BFGS',
                     options=dict(maxiter=600, gtol=1e-10))
        vals.append(r.fun)
        if best is None or r.fun < best.fun:
            best = r
    return best, np.array(vals)


def main():
    t0 = time.time()
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0, bottleneck=True)
    pi0 = L.stationary_distribution_from_K(K0)
    gap0 = L.spectral_gap_K(K0)
    edges = O.nearest_neighbour_edges(K0)
    Fb = O.basis_poly(coords, max_order=4)
    U = 3.0
    A = np.where(coords[:, 0] < -0.85)[0]
    B = np.where(coords[:, 0] > 0.85)[0]
    mfpt0 = L.mfpt_K(K0, A, B)
    out = dict(U_max=U, gap0=float(gap0), mfpt0=float(mfpt0), alpha_smooth=ALPHA,
               constraint='option B: pure box, u = clip(deg-4 poly)')

    bg, vg = run('gap', K0, Fb, U, edges, pi0)
    ug = np.clip(Fb @ bg.x, -U, U)
    sp_g = L.spectral_gap_K(L.tilt_generator(K0, ug)) / gap0
    print(f'SPECTRAL B: S_gamma={sp_g:.4f}x  #sat={int(np.sum(np.isclose(np.abs(ug),U,atol=1e-9)))}/200  '
          f'seeds@best={int(np.sum(vg <= vg.min()*0.99))}/{len(vg)}   [expect ~87.9]', flush=True)
    out['spectral'] = dict(speedup=float(sp_g), u=ug.tolist(), theta=bg.x.tolist())

    bm, vm = run('mfpt', K0, Fb, U, edges, pi0, A, B)
    um = np.clip(Fb @ bm.x, -U, U)
    Kbm = L.tilt_generator(K0, um)
    sp_m = mfpt0 / L.mfpt_K(Kbm, A, B)
    print(f'MFPT     B: S_MFPT={sp_m:.4f}x  #sat={int(np.sum(np.isclose(np.abs(um),U,atol=1e-9)))}/200  '
          f'seeds@best={int(np.sum(vm <= vm.min()*1.01))}/{len(vm)}   [NEW]', flush=True)
    out['mfpt'] = dict(speedup=float(sp_m), gap_speedup=float(L.spectral_gap_K(Kbm)/gap0),
                       u=um.tolist(), theta=bm.x.tolist())

    with open(f'{PATH}/polynomial_optionB.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n--- the ladder (n=200, U=3) ---')
    print(f'  spectral: C(smooth) 65.8  |  B(clipped) {sp_g:.1f}  |  per-state 91.30')
    print(f'  MFPT    : C(smooth) 59.3  |  B(clipped) {sp_m:.1f}  |  per-state 85.14')
    print(f'saved polynomial_optionB.json  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
