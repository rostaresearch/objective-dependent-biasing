"""Consolidate ONE best-found option-C optimum at (barrier 4.0, U=3.0).

Why: three runs of the SAME problem disagreed -- headline 65.806, budget sweep 65.806,
regime sweep 66.807 -- because the regime sweep's continuation chain (bh=1 -> 1.5 -> 2 ->
3 -> 4) entered a better basin than any of the fixed seed banks. The landscape is rugged,
so "best found" only means anything if every reported number comes from the same, largest
seed bank. Otherwise Fig. 3 and the regime figure disagree at the same point.

This reproduces the continuation chain, captures its theta, pools it with every other seed
we have, and writes the single winner (value + theta + u) that all figures and text must use.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
sys.path.insert(0, r'C:\Users\edina\Dropbox\MSM_Roundtable_2026')
from scipy.optimize import minimize
from scipy.io import loadmat
import analytic_lib as L
import optimal_spectral_bias as O
from rerun_polynomial_optionC_lean import u_of, make_fg

PATH = r'C:\Users\edina\Dropbox\MSM_Roundtable_2026'
U = 3.0


def run(kind, K0, Fb, Uv, edges, A, B, seeds, maxiter=400, powell=False):
    fg = make_fg(kind, K0, Fb, Uv, edges, A, B)
    best = None
    for t0 in seeds:
        t0 = np.asarray(t0, float)
        n = np.linalg.norm(t0)
        if n > 0:
            t0 = t0 / n
        r = minimize(fg, t0, jac=True, method='BFGS',
                     options=dict(maxiter=maxiter, gtol=1e-11))
        if powell:
            r2 = minimize(lambda th: fg(th)[0], r.x, method='Powell',
                          options=dict(maxfev=800, xtol=1e-7, ftol=1e-10))
            if r2.fun < r.fun:
                r = r2
        if best is None or r.fun < best.fun:
            best = r
    return best


def main():
    t0 = time.time()
    head = json.load(open(f'{PATH}/polynomial_optionC_robust.json'))
    lean = json.load(open(f'{PATH}/polynomial_optionC_lean.json'))
    d = loadmat(f'{PATH}/grid_2d_data.mat')

    # --- reproduce the regime continuation chain, capturing theta at each barrier ---
    print('reproducing the regime continuation chain to recover its basin...')
    chain = {'gap': [], 'mfpt': []}
    pg = np.asarray(head['spectral']['theta'], float)
    pm = np.asarray(head['mfpt']['theta'], float)
    for bh in (1.0, 1.5, 2.0, 3.0, 4.0):
        Kb, cb, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=bh, bottleneck=True)
        eb = O.nearest_neighbour_edges(Kb)
        Fbb = O.basis_poly(cb, max_order=4)
        Ab = np.where(cb[:, 0] < -0.85)[0]
        Bb = np.where(cb[:, 0] > 0.85)[0]
        pi0b = L.stationary_distribution_from_K(Kb)
        lp = np.log(np.clip(pi0b, 1e-300, None))
        thf, *_ = np.linalg.lstsq(Fbb, np.clip(lp - lp.mean(), -U, U), rcond=None)
        bg = run('gap', Kb, Fbb, U, eb, Ab, Bb, [pg, thf, -thf])
        pg = bg.x; chain['gap'].append(pg.copy())
        bm = run('mfpt', Kb, Fbb, U, eb, Ab, Bb, [pm, thf, -thf])
        pm = bm.x; chain['mfpt'].append(pm.copy())
        print(f'  bh={bh}: chain advanced', flush=True)

    # --- final: pool EVERY seed we have at barrier 4.0 / U=3.0 ---
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0, bottleneck=True)
    gap0 = L.spectral_gap_K(K0)
    edges = O.nearest_neighbour_edges(K0)
    Fb = O.basis_poly(coords, max_order=4)
    A = np.where(coords[:, 0] < -0.85)[0]
    B = np.where(coords[:, 0] > 0.85)[0]
    mfpt0 = L.mfpt_K(K0, A, B)
    pi0 = L.stationary_distribution_from_K(K0)
    lp = np.log(np.clip(pi0, 1e-300, None))
    th_flat, *_ = np.linalg.lstsq(Fb, np.clip(lp - lp.mean(), -U, U), rcond=None)
    warm = []
    for k in ('u_ps_spec', 'u_poly_spec', 'u_ps_mfpt'):
        if k in d:
            t, *_ = np.linalg.lstsq(Fb, np.asarray(d[k], float).ravel(order='C'), rcond=None)
            warm.append(t)

    print('\nfinal pooled optimisation at barrier 4.0, U=3.0')
    seeds_g = ([np.asarray(head['spectral']['theta'], float),
                np.asarray(lean['spectral']['theta'], float),
                th_flat, -th_flat] + warm + chain['gap'])
    bg = run('gap', K0, Fb, U, edges, A, B, seeds_g, maxiter=800, powell=True)
    ug, _ = u_of(bg.x, Fb, U)
    sp_g = L.spectral_gap_K(L.tilt_generator(K0, ug)) / gap0

    seeds_m = ([np.asarray(head['mfpt']['theta'], float),
                np.asarray(lean['mfpt']['theta'], float),
                th_flat, -th_flat] + warm + chain['mfpt'])
    bm = run('mfpt', K0, Fb, U, edges, A, B, seeds_m, maxiter=800, powell=True)
    um, _ = u_of(bm.x, Fb, U)
    sp_m = mfpt0 / L.mfpt_K(L.tilt_generator(K0, um), A, B)

    print(f'\n  CONSOLIDATED spectral S_gamma = {sp_g:.4f}x   '
          f'(prior best 66.807; headline was 65.806)')
    print(f'  CONSOLIDATED MFPT    S_MFPT   = {sp_m:.4f}x   '
          f'(prior best 60.577; headline was 59.307)')
    print(f'  clipped states: {int(np.sum(np.isclose(np.abs(ug), U, atol=1e-9)))}/200 '
          f'(spectral), max|b|={np.abs(ug).max():.4f}')
    print(f'  % of per-state benchmark: spectral {100*sp_g/91.2957:.1f}%, '
          f'MFPT {100*sp_m/85.1388:.1f}%')

    json.dump(dict(spectral=dict(speedup=float(sp_g), u=ug.tolist(),
                                 theta=bg.x.tolist()),
                   mfpt=dict(speedup=float(sp_m), u=um.tolist(),
                             theta=bm.x.tolist()),
                   note='consolidated best-found option C at barrier 4.0, U=3.0; '
                        'pooled seed bank incl. the regime continuation chain'),
              open(f'{PATH}/polynomial_optionC_FINAL.json', 'w'), indent=2)
    print(f'\nsaved polynomial_optionC_FINAL.json  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
