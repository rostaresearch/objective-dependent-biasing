"""Step 3 (and the optimizations for step 4): per-state optima under |b_i| <= U.

Jobs (run as separate processes, single-threaded BLAS):
  python step3_optimize.py faithful_spectral U   imported per_state_ceiling.optimise_per_state_analytic
  python step3_optimize.py fast_spectral         symmetric-route engine (C_lib.SymGap), multi-start
                                                 L-BFGS-B plus soft-min continuation, all U and K2
  python step3_optimize.py mfpt                  imported mfpt_per_state.optimise_mfpt_per_state
                                                 plus random starts and a TNC run, all cases

Outputs go to runs/<job>.npz and runs/<job>.json.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

import C_lib as C
from C_lib import L, PSC, MPS

RUNS = C.HERE / 'runs'
RUNS.mkdir(exist_ok=True)
OPTS = dict(maxiter=3000, gtol=1e-9, ftol=1e-11)     # as in per_state_ceiling / mfpt_per_state
N_RANDOM = 6


def generators():
    T = C.load_T()
    K = C.generator_from_T(T)
    # Generator-construction sensitivity (same model, not a second lag estimate):
    # K2 = (T^2 - I)/(2 tau) has the stationary distribution of T but a different jump chain.
    K2 = C.generator_from_T(T @ T, tau=2 * C.TAU)
    return dict(K=K, K2=K2)


def ab_sets():
    s1 = json.loads((C.HERE / 'step1_T_check.json').read_text())
    return dict(primary=(np.array(s1['AB_primary']['A']), np.array(s1['AB_primary']['B'])),
                alternative=(np.array(s1['AB_alternative']['A']), np.array(s1['AB_alternative']['B'])))


def flatten_start(K0, U):
    pi0 = C.stationary(K0)
    return PSC.O.project_symmetric(np.log(np.clip(pi0, 1e-300, None)), U_max=U)


def run_lbfgsb(fg, u0, U, method='L-BFGS-B'):
    n = len(u0)
    t0 = time.time()
    opts = dict(OPTS) if method == 'L-BFGS-B' else dict(maxfun=20000, ftol=1e-12, gtol=1e-10)
    res = minimize(fg, u0, jac=True, method=method, bounds=[(-U, U)] * n, options=opts)
    return dict(fun=float(res.fun), x=res.x.copy(), nit=int(getattr(res, 'nit', -1)),
                nfev=int(res.nfev), message=str(res.message), time_s=time.time() - t0)


def save(name, results: dict):
    arrays = {}
    meta = {}
    for key, r in results.items():
        arrays[key] = r['x']
        meta[key] = {k: v for k, v in r.items() if k != 'x'}
    np.savez(RUNS / f'{name}.npz', **arrays)
    C.dump(RUNS / f'{name}.json', meta)


# ---------------------------------------------------------------- jobs

def job_faithful_spectral(U):
    K = generators()['K']
    t0 = time.time()
    out = PSC.optimise_per_state_analytic(K, U_max=U, warm_u=None, maxiter=OPTS['maxiter'])
    r = dict(x=out['u'], gamma=out['gamma'], gamma_from_opt=out['gamma_from_opt'],
             time_s=time.time() - t0)
    save(f'faithful_spectral_U{U:g}', {'best': r})
    print('done', U, r['gamma'], flush=True)


def spectral_multistart(K0, U, label, prev=None):
    pi0 = C.stationary(K0)
    eng = C.SymGap(K0, pi0)

    def fg(u):
        g, gr = eng.gap_grad(u)
        return -g, -gr

    starts = {'flatten': flatten_start(K0, U), 'zeros': np.zeros(K0.shape[0])}
    rng = np.random.default_rng(int(1000 * U) + (0 if label == 'K' else 7))
    for k in range(N_RANDOM):
        starts[f'random{k}'] = rng.uniform(-U, U, K0.shape[0])
    if prev is not None:
        starts['scaled_lower_U'] = np.clip(prev, -U, U)
    res = {}
    for name, u0 in starts.items():
        r = run_lbfgsb(fg, u0, U)
        r['gamma'] = -r['fun']
        res[name] = r
        print(f'  {label} U={U} {name:16s} gamma={r["gamma"]:.6e} nit={r["nit"]} '
              f'{r["time_s"]:.0f}s {r["message"][:40]}', flush=True)
    # Soft-min continuation from the best L-BFGS-B point and from the flatten start
    gap0 = L.spectral_gap_K(K0)
    best_name = max(res, key=lambda k: res[k]['gamma'])
    for tag, u0 in (('softmin_from_best', res[best_name]['x']), ('softmin_from_flatten', starts['flatten'])):
        u = u0.copy()
        for beta in (20 / gap0, 100 / gap0, 500 / gap0, 2500 / gap0):
            r = run_lbfgsb(lambda v: tuple(-np.asarray(x) for x in eng.softmin_grad(v, beta)), u, U)
            u = r['x']
        # polish on the exact gap
        r = run_lbfgsb(fg, u, U)
        r['gamma'] = -r['fun']
        r['gamma_before_polish'] = eng.gap_grad(u)[0]
        res[tag] = r
        print(f'  {label} U={U} {tag:16s} gamma={r["gamma"]:.6e} (pre-polish {r["gamma_before_polish"]:.6e})',
              flush=True)
    return res


def job_fast_spectral():
    G = generators()
    prev = None
    for U in (1.0, 2.0, 3.0):
        res = spectral_multistart(G['K'], U, 'K', prev)
        save(f'fast_spectral_K_U{U:g}', res)
        best = max(res, key=lambda k: res[k]['gamma'])
        prev = res[best]['x'] * (U + 1) / U
    res = spectral_multistart(G['K2'], 2.0, 'K2')
    save('fast_spectral_K2_U2', res)


def mfpt_multistart(K0, A, B, U, label):
    def fg(u):
        try:
            g, gr = MPS.mfpt_and_grad(u, K0, A, B)
        except Exception:
            return 1e12, np.zeros(len(u))
        return float(g), gr

    res = {}
    q0 = L.committor_K(K0, A, B)
    t0 = time.time()
    out = MPS.optimise_mfpt_per_state(K0, A, B, U_max=U, warm_u=U * (1 - 2 * q0), maxiter=OPTS['maxiter'])
    res['imported_optimizer'] = dict(x=out['u'], fun=out['mfpt'], time_s=time.time() - t0,
                                     message='mfpt_per_state.optimise_mfpt_per_state, starts: '
                                             'committor-shaped U(1-2q0), flatten, zeros')
    rng = np.random.default_rng(int(100 * U) + len(A) + 3 * len(B))
    for k in range(N_RANDOM):
        r = run_lbfgsb(fg, rng.uniform(-U, U, K0.shape[0]), U)
        res[f'random{k}'] = r
    r = run_lbfgsb(fg, np.zeros(K0.shape[0]), U, method='TNC')
    res['TNC_from_zeros'] = r
    best = min(res, key=lambda k: res[k]['fun'])
    r = run_lbfgsb(fg, res[best]['x'], U, method='TNC')
    res['TNC_from_best'] = r
    for name, r in res.items():
        print(f'  {label} U={U} {name:20s} mfpt={r["fun"]:.6g}', flush=True)
    return res


def job_mfpt():
    G = generators()
    S = ab_sets()
    A, B = S['primary']
    cases = [('K_AB_primary', G['K'], A, B, U) for U in (1.0, 2.0, 3.0)]
    cases += [('K_BA_primary', G['K'], B, A, 2.0),
              ('K_AB_alternative', G['K'], *S['alternative'], 2.0),
              ('K2_AB_primary', G['K2'], A, B, 2.0)]
    for label, K0, a, b, U in cases:
        res = mfpt_multistart(K0, a, b, U, label)
        save(f'mfpt_{label}_U{U:g}', res)


if __name__ == '__main__':
    job = sys.argv[1]
    if job == 'faithful_spectral':
        job_faithful_spectral(float(sys.argv[2]))
    elif job == 'fast_spectral':
        job_fast_spectral()
    elif job == 'mfpt':
        job_mfpt()
    else:
        raise SystemExit(f'unknown job {job}')
