"""Option C, lean & robust: BFGS from strong physical warm-starts, no Powell.

The warm starts (per-state ceiling and pure-box polynomial, fit into the degree-4 space) are
already near-optimal, so analytic-gradient BFGS converges in tens of iterations. We take the
best over a handful of physically-motivated seeds plus a few random directions. Each u is a
genuine degree-4 polynomial (never clipped), midrange-centred and rescaled to spread<=2U.
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


def u_of(theta, Fb, U):
    p = Fb @ theta
    a, b = int(np.argmax(p)), int(np.argmin(p))
    s = p[a] - p[b]
    if s < 1e-10:
        return np.zeros_like(p), None
    mid = 0.5 * (p[a] + p[b])
    return 2.0 * U * (p - mid) / s, (a, b, s, mid, p)


def du_dtheta(Fb, U, aux):
    a, b, s, mid, p = aux
    Fa, Fbb = Fb[a], Fb[b]
    J = (2.0 * U / s) * (Fb - 0.5 * (Fa + Fbb)[None, :])
    J -= (2.0 * U / s ** 2) * np.outer(p - mid, Fa - Fbb)
    return J


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
        u, aux = u_of(theta, Fb, U)
        if aux is None:
            return 1e12, np.zeros(nf)
        try:
            if kind == 'gap':
                val, dval = gap_and_grad(u, K0); f, df = -val, -dval
            else:
                val, dval = mfpt_and_grad(u, K0, A, B); f, df = val, dval
        except Exception:
            return 1e12, np.zeros(nf)
        if not np.isfinite(f):
            return 1e12, np.zeros(nf)
        p, dp = pen_and_grad(u, edges)
        return float(f + ALPHA * p), du_dtheta(Fb, U, aux).T @ (df + ALPHA * dp)
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
    rng = np.random.default_rng(0)
    for _ in range(4):
        S.append(rng.normal(scale=1.0, size=Fb.shape[1]))
    return S


def optimise(kind, K0, Fb, U, edges, pi0, A=None, B=None):
    fg = make_fg(kind, K0, Fb, U, edges, A, B)
    best, vals = None, []
    for t0 in seeds(Fb, U, pi0):
        r = minimize(fg, t0, jac=True, method='BFGS',
                     options=dict(maxiter=500, gtol=1e-10))
        vals.append(r.fun)
        if best is None or r.fun < best.fun:
            best = r
    return best, np.array(vals), fg


def verify(fg, theta):
    f0, g = fg(theta)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(theta), 5, replace=False)
    e = 1e-6
    errs = [abs((fg(theta + e*(np.arange(len(theta)) == i))[0] -
                 fg(theta - e*(np.arange(len(theta)) == i))[0]) / (2*e) - g[i]) /
            (abs(g[i]) + 1e-12) for i in idx]
    return max(errs)


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
               constraint='option C: smooth deg-4 poly, spread<=2U, BFGS from warm starts')

    bg, vg, fgg = optimise('gap', K0, Fb, U, edges, pi0)
    ug, _ = u_of(bg.x, Fb, U)
    sp_g = L.spectral_gap_K(L.tilt_generator(K0, ug)) / gap0
    print(f'SPECTRAL  S_gamma={sp_g:.4f}x   FDerr={verify(fgg, bg.x):.1e}  '
          f'seeds@best={int(np.sum(vg <= vg.min()*0.99))}/{len(vg)}  '
          f'#clip={int(np.sum(np.isclose(np.abs(ug),U,atol=1e-9)))}/200', flush=True)
    out['spectral'] = dict(speedup=float(sp_g), u=ug.tolist(), theta=bg.x.tolist(),
                           mfpt_of_bias=float(L.mfpt_K(L.tilt_generator(K0, ug), A, B)),
                           mfpt_speedup_of_bias=float(mfpt0/L.mfpt_K(L.tilt_generator(K0, ug), A, B)))

    bm, vm, fgm = optimise('mfpt', K0, Fb, U, edges, pi0, A, B)
    um, _ = u_of(bm.x, Fb, U)
    Kbm = L.tilt_generator(K0, um)
    sp_m = mfpt0 / L.mfpt_K(Kbm, A, B)
    print(f'MFPT      S_MFPT={sp_m:.4f}x   FDerr={verify(fgm, bm.x):.1e}  '
          f'seeds@best={int(np.sum(vm <= vm.min()*1.01))}/{len(vm)}  '
          f'#clip={int(np.sum(np.isclose(np.abs(um),U,atol=1e-9)))}/200', flush=True)
    out['mfpt'] = dict(speedup=float(sp_m), gap_speedup=float(L.spectral_gap_K(Kbm)/gap0),
                       u=um.tolist(), theta=bm.x.tolist())

    out['ceilings'] = dict(gamma=91.30, mfpt=85.14)
    out['shipped_clipbased'] = dict(spectral_poly=68.98, mfpt_poly=82.13)
    with open(f'{PATH}/polynomial_optionC_lean.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nspectral: 68.98 -> {sp_g:.2f}  ({100*sp_g/91.30:.1f}% of ceiling)')
    print(f'mfpt:     82.13 -> {sp_m:.2f}  ({100*sp_m/85.14:.1f}% of ceiling)')
    print(f'saved polynomial_optionC_lean.json   [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
