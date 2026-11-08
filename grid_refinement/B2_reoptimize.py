"""Grid refinement, step 2: re-optimise the per-state optima on a finer grid.

    python B2_reoptimize.py <system> <grid> <U> <objective> [n_random]

system    : bottleneck (grid '20x10', '40x20', '80x40') or nine_well (grid '20', '40')
objective : gap  (maximise gamma = -lambda_2 of K^b)
            mfpt (minimise mean_{i in A} MFPT_i(A->B) of K^b)
Constraint: pure box |b_i| <= U, as in per_state_ceiling.optimise_per_state_analytic
and mfpt_per_state.optimise_mfpt_per_state (L-BFGS-B, analytic gradients).

Starts: the coarse saved optimum of the same objective interpolated onto this grid,
the coarse optimum of the other objective, budget-clipped flattening and zeros (the
starts of the original optimisers), the saved n=800 optimum where it exists, the
interpolated re-optimised 40x20 optimum for the 80x40 grid, and n_random uniform
random starts.

Gap objective: stage 1 is L-BFGS-B on the exact gamma with the Hellmann-Feynman
gradient.  When lambda_2 and lambda_3 are close the gap is not differentiable, so for the
best stage-1 run stage 2 continues on the smooth soft-min of the slowest
nonzero rates, s_t = -(1/t) log sum_k exp(-t mu_k) <= mu_2 = gamma, with t increased
step by step; the exact gamma is re-evaluated at the end and the better of the two
stages is kept.  Stationarity at the final point is tested with the generalised
(Clarke) gradient over the eigenspace of the near-degenerate cluster.
"""
import sys
import time
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

import B_lib as BL

# optional key=value arguments: starts=a,b (subset of starts), maxiter=, ftol=, part= (output suffix)
opts = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
pos = [a for a in sys.argv[1:] if '=' not in a]
system, grid, U, objective = pos[0], pos[1], float(pos[2]), pos[3]
n_random = int(pos[4]) if len(pos) > 4 else 3
MAXITER = int(opts.get('maxiter', 20000))
FTOL = float(opts.get('ftol', 1e-13))
tag = f'{system}_{grid}_U{U:g}_{objective}' + (f'_part{opts["part"]}' if 'part' in opts else '')
print('job', tag, flush=True)

if system == 'bottleneck':
    nx, ny = map(int, grid.split('x'))
    s = BL.bottleneck_system(nx, ny)
    d = loadmat(BL.DATA/'grid_2d_data.mat')
    assert U == 3.0
    coarse = dict(gap=d['u_ps_spec'], mfpt=d['u_ps_mfpt'])
    xs_c, ys_c, axes = d['xs'].ravel(), d['ys'].ravel(), 'ij'
else:
    s = BL.nine_well_system(int(grid))
    d = loadmat(BL.DATA/'rkhs_2d_data.mat')
    k = {3.0: 1, 6.0: 2}[U]
    coarse = dict(gap=d[f'u_spec_{k}'], mfpt=d[f'u_mfpt_{k}'])
    xs_c, ys_c, axes = d['xs'].ravel(), d['ys'].ravel(), 'xy'
other = 'mfpt' if objective == 'gap' else 'gap'

starts = [('interp_coarse', BL.interpolate_bias(coarse[objective], xs_c, ys_c, s, axes)),
          ('interp_coarse_other', BL.interpolate_bias(coarse[other], xs_c, ys_c, s, axes))]
if system == 'bottleneck' and grid == '40x20':
    h = loadmat(BL.DATA/'grid_2d_hires.mat')
    starts.append(('saved_n800', h['u_ps_spec' if objective == 'gap' else 'u_ps_mfpt'].ravel()))
if system == 'bottleneck' and grid == '80x40':
    prev = BL.OUT/f'B2_bottleneck_40x20_U3_{objective}.npz'
    if prev.exists():
        p = np.load(prev)
        s40 = BL.bottleneck_system(40, 20)
        starts.insert(0, ('interp_reopt_40x20',
                          BL.interpolate_bias(p['best'].reshape(40, 20), s40.xs, s40.ys, s, 'ij')))
if system == 'nine_well' and grid == '80':
    prev = BL.OUT/f'B2_nine_well_40_U{U:g}_{objective}.npz'
    p = np.load(prev)
    s40 = BL.nine_well_system(40)
    starts.insert(0, ('interp_reopt_40',
                      BL.interpolate_bias(p['best'].reshape(40, 40), s40.xs, s40.ys, s, 'xy')))
starts += [('flatten', BL.flatten_start(s, U)), ('zeros', np.zeros(s.n))]
starts += BL.random_starts(s.n, U, n_random, seed=20261010 + int(U))
if 'starts' in opts:
    keep = opts['starts'].split(',')
    starts = [st for st in starts if st[0] in keep]
    assert len(starts) == len(keep), 'unknown start name'

solver = BL._MFPTSolver(s)


def fg_mfpt(b):
    try:
        return BL.mfpt_and_grad_fast(b, s, solver)
    except Exception as e:                          # same fallback as the originals
        print('   eval failure', e, flush=True)
        return 1e300, np.zeros(s.n)


def fg_gap(b):
    try:
        g, G = BL.gap_and_grad_fast(b, s)
        return -g, -G
    except Exception as e:
        print('   eval failure', e, flush=True)
        return 1e300, np.zeros(s.n)


def softmin_fg(t, kk=7):
    def f(b):
        Kb = BL.tilt(s.K0, b)
        vals, vecs = BL.top_eigs(Kb, s.logpi0 - b, k=kk)
        mu = -vals[1:]
        z = -t*(mu - mu[0])
        w = np.exp(z); Z = w.sum(); w /= Z
        st = mu[0] - np.log(Z)/t
        G = np.zeros(s.n)
        KT = Kb.T.tocsr()
        for wk, ph in zip(w, vecs[:, 1:].T):
            if wk > 1e-14:
                G += wk*(-0.5*np.asarray(KT @ (ph*ph)).ravel())
        return -st, -G
    return f


def clarke_check(b, rel_tol=1e-3, kk=7):
    """Smallest box-projected generalised gradient of lambda_2 over the eigenspace
    of the cluster {lambda_k : |lambda_k - lambda_2| <= rel_tol |lambda_2|}."""
    Kb = BL.tilt(s.K0, b)
    vals, vecs = BL.top_eigs(Kb, s.logpi0 - b, k=kk)
    lam = vals[1:]
    C = np.flatnonzero(np.abs(lam - lam[0]) <= rel_tol*abs(lam[0]))
    Phi = vecs[:, 1:][:, C]
    KT = Kb.T.tocsr()
    m = len(C)
    # basis gradients for the products phi_k phi_l
    Gkl = {(a, c): 0.5*np.asarray(KT @ (Phi[:, a]*Phi[:, c])).ravel() for a in range(m) for c in range(a, m)}

    def G_of(Y):
        G = np.zeros(s.n)
        for (a, c), v in Gkl.items():
            G += (Y[a, c] if a == c else 2*Y[a, c])*v
        return G

    def resid(theta):
        Lm = np.zeros((m, m)); Lm[np.tril_indices(m)] = theta
        Y = Lm @ Lm.T; Y /= np.trace(Y)
        return BL.projected_gradient(b, G_of(Y), U), Y
    single = BL.projected_gradient(b, G_of(np.diag(np.r_[1.0, np.zeros(m - 1)])), U)
    best = (np.abs(single).max(), None)
    if m > 1:
        rng = np.random.default_rng(0)
        for trial in range(8 + 4*m):
            th0 = rng.normal(size=m*(m + 1)//2) if trial else np.eye(m)[np.tril_indices(m)]/np.sqrt(m)
            r = minimize(lambda th: np.sum(resid(th)[0]**2), th0, method='Nelder-Mead',
                         options=dict(maxiter=4000, xatol=1e-12, fatol=1e-30))
            rr, Y = resid(r.x)
            if np.abs(rr).max() < best[0]:
                best = (np.abs(rr).max(), Y)
    gam = -lam[0]
    return dict(cluster_size=int(m), lams=vals.tolist(),
                rel_sep_l2_l3=float((lam[0] - lam[1])/abs(lam[0])), lam3_over_lam2=float(lam[1]/lam[0]),
                single_eigvec_proj_grad_rel=float(np.abs(single).max()/gam),
                clarke_proj_grad_rel=float(best[0]/gam),
                clarke_Y=None if best[1] is None else best[1].tolist())


t_job = time.time()
if objective == 'mfpt':
    runs = BL.optimise(fg_mfpt, s.n, U, starts, label=tag, maxiter=MAXITER, ftol=FTOL)
    for r in runs:
        r['value'] = r['fun']
else:
    runs = BL.optimise(fg_gap, s.n, U, starts, label=tag, maxiter=MAXITER, ftol=FTOL)
    for r in runs:
        r['stage1_gamma'] = -r['fun']
        x = r['x']
        vals, _ = BL.top_eigs(BL.tilt(s.K0, x), s.logpi0 - x, k=4)
        r['stage1_rel_sep'] = float((vals[1] - vals[2])/abs(vals[1]))   # (lambda_2 - lambda_3)/|lambda_2| >= 0
    best1 = max(runs, key=lambda r: r['stage1_gamma'])
    for r in runs:
        r['value'] = r['stage1_gamma']
        if r is best1:                      # polish the best stage-1 run only
            x = r['x'].copy()
            t0 = time.time()
            for c in [100, 1000, 10000]:
                gam = BL.gap_value(x, s)
                res = minimize(softmin_fg(c/gam), x, jac=True, method='L-BFGS-B', bounds=[(-U, U)]*s.n,
                               options=dict(maxiter=3000, gtol=1e-12, ftol=1e-14, maxcor=30))
                x = res.x
            g2 = BL.gap_value(x, s)
            r['stage2_gamma'] = g2
            r['stage2_seconds'] = time.time() - t0
            print(f'   [{tag}] {r["start"]:>12s} stage1 gamma={r["stage1_gamma"]:.10g} '
                  f'stage2 gamma={g2:.10g} ({time.time()-t0:.0f}s)', flush=True)
            if g2 > r['stage1_gamma']:
                r['x'] = x; r['value'] = g2
best = (min if objective == 'mfpt' else max)(runs, key=lambda r: r['value'])
check = clarke_check(best['x']) if objective == 'gap' else None
if objective == 'mfpt':
    v, G = BL.mfpt_and_grad_fast(best['x'], s)
    check = dict(proj_grad_rel=float(np.abs(BL.projected_gradient(best['x'], G, U)).max()/v))
vals = [r['value'] for r in runs]
summary = dict(job=tag, maxiter=MAXITER, ftol=FTOL, system=system, grid=grid, n=s.n, U=U, objective=objective,
               best_start=best['start'], best_value=best['value'],
               value_spread_rel=float((max(vals) - min(vals))/abs(best['value'])),
               stationarity=check, seconds=time.time() - t_job,
               runs=[{k: v for k, v in r.items() if k != 'x'} for r in runs])
BL.dump(summary, BL.OUT/f'B2_{tag}.json')
np.savez(BL.OUT/f'B2_{tag}.npz', best=best['x'], **{f'run_{r["start"]}': r['x'] for r in runs})
print('done', tag, 'best', best['start'], best['value'], check, f'{time.time()-t_job:.0f}s', flush=True)
