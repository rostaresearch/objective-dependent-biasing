"""Collect the Langevin runs, compute MFPTs with confidence intervals, compare with the network.

Estimators
  * Runs in which every started walker reached B: sample mean of the first-passage times, 95%
    percentile bootstrap interval over walkers (20,000 resamples).
  * Runs stopped before every walker finished (the unbiased reference): exponential maximum-
    likelihood estimate with right censoring, T = (sum of completed times + elapsed times of the
    walkers still running) / (number completed). 95% interval from 2E/T ~ chi^2(2n). This is exact
    for an exponential first-passage-time distribution; the coefficient of variation of the
    completed times and the network's predicted distribution are reported to check that assumption.
  * Headline value per case: the runs at the different time steps pooled (inverse-variance weighted
    mean for complete runs; total exposure / total transitions for censored runs). The time-step
    bias at dt <= 4e-4 is ~1-2% (MFPT-U6 sweep with 5000 walkers per dt; exact 1D calculation in
    em_bias_1d.py), below the statistical resolution of the U = 3 runs.
  * dt trend (diagnostic): weighted straight-line fit of the MFPT against dt, bootstrapped.
Network reference: SQRA on the aligned cell-centred grids (network_aligned.json), Richardson-
extrapolated in h, cross-checked by the Scharfetter-Gummel discretization; the audit's 120 x 120
vertex grid (network_predictions.json) is also listed.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from common import HERE

RUNS = HERE / 'runs'                    # runs on the laptop
RUNS_MARKOV = HERE / 'runs_markov'      # runs on markov.phys.ucl.ac.uk, copied back (different seeds)
RNG = np.random.default_rng(20261011)
NBOOT = 20000
CASES = ['unbiased', 'MFPT_U3', 'spectral_U3', 'MFPT_U6', 'spectral_U6']


def load_runs():
    groups = {}
    files = [(f, 'laptop') for f in sorted(RUNS.glob('*.npz'))] +             [(f, 'markov') for f in sorted(RUNS_MARKOV.glob('*.npz'))]
    for f, host in files:
        z = np.load(f)
        meta = json.loads(str(z['meta']))
        dt = float(z['dt'])
        running = z['state'] == 1
        groups.setdefault((meta['case'], meta['interp'], dt), []).append(dict(
            file=f.name, path=str(f), host=host, seed=meta['seed'], fpt=z['steps'] * dt, cens=z['nst'][running] * dt,
            final=bool(z['final']), wall_hours=float(z['wall']) / 3600, nwalk=meta['nwalk'],
            lane_steps=float(z['steps'].sum() + z['nst'][running].sum())))
    return groups


def summarize(runs):
    fpt = np.concatenate([r['fpt'] for r in runs])
    cens = np.concatenate([r['cens'] for r in runs])
    n = len(fpt)
    out = dict(seeds=[r['seed'] for r in runs], files=[r['file'] for r in runs], hosts=[r['host'] for r in runs],
               n_by_host={h: int(sum(len(r['fpt']) for r in runs if r['host'] == h)) for h in set(r['host'] for r in runs)},
               n_completed=n,
               n_censored=int(len(cens)), wall_hours=float(sum(r['wall_hours'] for r in runs)),
               complete=bool(all(r['final'] for r in runs) and len(cens) == 0))
    if n == 0:
        out['mfpt'] = None
        return out, None
    out['cv_completed'] = float(fpt.std(ddof=1) / fpt.mean()) if n > 1 else None
    out['median_over_mean_completed'] = float(np.median(fpt) / fpt.mean())
    if out['complete']:
        boots = RNG.choice(fpt, size=(NBOOT, n), replace=True).mean(axis=1)
        out.update(estimator='sample mean, bootstrap over walkers', mfpt=float(fpt.mean()),
                   se=float(fpt.std(ddof=1) / np.sqrt(n)),
                   ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
    else:
        E = fpt.sum() + cens.sum()
        boots = 2 * E / RNG.chisquare(2 * n, size=NBOOT)
        out.update(estimator='exponential MLE with right censoring', mfpt=float(E / n), exposure=float(E),
                   se=float(E / n / np.sqrt(n)),
                   ci95=[float(2 * E / stats.chi2.ppf(.975, 2 * n)), float(2 * E / stats.chi2.ppf(.025, 2 * n))])
        if n > 1:
            out['completed_only_mean'] = float(fpt.mean())
    return out, boots


def richardson(values, hs):
    v = np.asarray(values, float); h = np.asarray(hs, float)
    res = dict(values=v.tolist(), h=h.tolist(), finest=float(v[-1]))
    if len(v) >= 3:
        d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
        if d1 * d2 > 0 and abs(d2) < abs(d1):
            r = h[-2] / h[-1]
            p = np.log(abs(d1 / d2)) / np.log(r)
            res.update(order=float(p), extrapolated=float(v[-1] + d2 / (r ** p - 1)))
        else:
            res.update(order=None, extrapolated=float(v[-1]), note='not monotone; finest grid used')
    res['spread_two_finest'] = float(abs(v[-1] - v[-2]))
    return res


def network_reference():
    al = json.loads((HERE / 'network_aligned.json').read_text())['rows']
    vx = json.loads((HERE / 'network_predictions.json').read_text())['rows']
    fdp = HERE / 'network_fd_check.json'
    fd = json.loads(fdp.read_text())['rows'] if fdp.exists() else []
    ref = {}
    for case in CASES:
        for interp in (['none'] if case == 'unbiased' else ['bilinear', 'bicubic']):
            rows = sorted([r for r in al if r['case'] == case and r['interp'] == interp], key=lambda r: r['n'])
            if not rows:
                continue
            sq = richardson([r['mfpt_sqra'] for r in rows], [r['h'] for r in rows])
            sg = richardson([r['mfpt_sg'] for r in rows], [r['h'] for r in rows])
            v120 = [r for r in vx if r['case'] == case and r['n'] == 120 and r['interp'] == interp]
            fdr = sorted([r for r in fd if r['case'] == case and r['interp'] == interp], key=lambda r: r['n'])
            hs = {r['n']: r['h'] for r in rows}
            fdx = richardson([r['mfpt_fd'] for r in fdr], [hs[r['n']] for r in fdr]) if len(fdr) >= 3 else None
            ref[f'{case}|{interp}'] = dict(central_fd=fdx,
                grids=[r['n'] for r in rows], sqra=sq, sg=sg, continuum=sq['extrapolated'],
                continuum_sg=sg['extrapolated'],
                discretization_uncertainty=float(max(abs(sq['extrapolated'] - sg['extrapolated']),
                                                     abs(sq['extrapolated'] - sq['finest']))),
                vertex_120=v120[0]['mfpt_sqra'] if v120 else None)
    return ref


def dt_fit(points):
    """points: list of (dt, mean, boots). Weighted linear fit mean = T0 + c dt; bootstrap the fit."""
    dts = np.array([p[0] for p in points]); means = np.array([p[1] for p in points])
    B = np.array([p[2] for p in points])            # (k, NBOOT)
    w = 1 / B.var(axis=1)
    X = np.c_[np.ones_like(dts), dts]
    W = np.diag(w)
    P = np.linalg.solve(X.T @ W @ X, X.T @ W)       # (2, k)
    coef = P @ means
    cb = P @ B
    return dict(dts=dts.tolist(), T0=float(coef[0]), slope=float(coef[1]),
                T0_ci95=[float(np.percentile(cb[0], 2.5)), float(np.percentile(cb[0], 97.5))],
                relative_dt_error_at_largest_dt=float(coef[1] * dts.max() / coef[0]),
                relative_dt_error_at_largest_dt_ci95=[float(np.percentile(cb[1] * dts.max() / cb[0], 2.5)),
                                                      float(np.percentile(cb[1] * dts.max() / cb[0], 97.5))]), cb[0]


def ratio_ci(a, b):
    r = a / b
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    groups = load_runs()
    ref = network_reference()
    res = dict(description=__doc__, runs={}, dt_extrapolation={}, comparison={}, speedups={}, ranking={})
    boots = {}
    for (case, interp, dt), runs in sorted(groups.items()):
        s, b = summarize(runs)
        key = f'{case}|{interp}|dt={dt:g}'
        res['runs'][key] = s
        boots[(case, interp, dt)] = b
        print(key, {k: s.get(k) for k in ['n_completed', 'n_censored', 'mfpt', 'ci95', 'cv_completed', 'complete']})
    # dt trend (diagnostic) and pooled estimate over the time steps (headline)
    best = {}
    for case in CASES:
        for interp in ['none', 'bilinear', 'bicubic']:
            dts = sorted(dt for (c, i, dt) in boots if c == case and i == interp and boots[(c, i, dt)] is not None)
            if not dts:
                continue
            k = f'{case}|{interp}'
            rs = [res['runs'][f'{k}|dt={dt:g}'] for dt in dts]
            if len(dts) >= 2 and all(r['complete'] for r in rs):
                fit, _ = dt_fit([(dt, r['mfpt'], boots[(case, interp, dt)]) for dt, r in zip(dts, rs)])
                res['dt_extrapolation'][k] = fit
            if all(r['complete'] for r in rs):
                B = np.array([boots[(case, interp, dt)] for dt in dts])
                w = 1 / B.var(axis=1)
                pooled_b = (w[:, None] * B).sum(axis=0) / w.sum()
                m = float((w * np.array([r['mfpt'] for r in rs])).sum() / w.sum())
                src = 'inverse-variance weighted mean over dt = ' + ', '.join(f'{d:g}' for d in dts)
            else:
                fpt = np.concatenate([np.concatenate([g['fpt'] for g in groups[(case, interp, dt)]]) for dt in dts])
                cen = np.concatenate([np.concatenate([g['cens'] for g in groups[(case, interp, dt)]]) for dt in dts])
                E, n = fpt.sum() + cen.sum(), len(fpt)
                m = float(E / n)
                pooled_b = 2 * E / RNG.chisquare(2 * n, size=NBOOT)
                src = (f'exponential MLE with censoring, pooled over dt = ' + ', '.join(f'{d:g}' for d in dts) +
                       f' ({n} transitions)')
            best[k] = dict(source=src, mfpt=m, boots=pooled_b,
                           ci95=[float(np.percentile(pooled_b, 2.5)), float(np.percentile(pooled_b, 97.5))],
                           smallest_dt=dts[0], smallest_dt_mfpt=rs[0]['mfpt'], smallest_dt_ci95=rs[0]['ci95'],
                           per_dt={f'{dt:g}': dict(mfpt=r['mfpt'], ci95=r['ci95'], n=r['n_completed'],
                                                   censored=r['n_censored'], n_by_host=r['n_by_host'])
                                   for dt, r in zip(dts, rs)})
    # excess scatter between independent complete runs of a case (run_consistency.py): when the run means
    # scatter more than their standard errors allow, widen the interval by the Birge ratio sqrt(chi2/dof)
    rc_path = HERE / 'run_consistency.json'
    rc = json.loads(rc_path.read_text())['across_runs'] if rc_path.exists() else {}
    for k, b in best.items():
        a_ = rc.get(k)
        if a_ and a_['runs'] >= 3 and a_['chi2'] > a_['dof']:
            r_ = float(np.sqrt(a_['chi2'] / a_['dof']))
            b['birge_ratio'] = r_
            b['birge_p_value'] = a_['p_value']
            b['boots_scaled'] = b['mfpt'] + r_ * (b['boots'] - b['mfpt'])
    # comparison with the network
    for k, b in best.items():
        if k not in ref:
            continue
        net = ref[k]['continuum']
        bs = b['boots']
        res['comparison'][k] = dict(
            measured=b['mfpt'], measured_ci95=b['ci95'], measured_source=b['source'],
            per_dt=b['per_dt'],
            network_continuum=net, network_continuum_sg=ref[k]['continuum_sg'],
            network_discretization_uncertainty=ref[k]['discretization_uncertainty'],
            network_vertex_120=ref[k]['vertex_120'],
            measured_over_network=b['mfpt'] / net,
            measured_over_network_ci95=[float(np.percentile(bs, 2.5) / net), float(np.percentile(bs, 97.5) / net)])
        if 'birge_ratio' in b:
            bsc = b['boots_scaled']
            res['comparison'][k].update(
                birge_ratio=b['birge_ratio'], run_scatter_p_value=b['birge_p_value'],
                measured_ci95_scaled=[float(np.percentile(bsc, 2.5)), float(np.percentile(bsc, 97.5))],
                measured_over_network_ci95_scaled=[float(np.percentile(bsc, 2.5) / net), float(np.percentile(bsc, 97.5) / net)])
    # speedups: T0 measured (Langevin) and T0 from the network, against measured biased MFPTs
    u = best.get('unbiased|none')
    T0_net = ref['unbiased|none']['continuum']
    for k, b in best.items():
        if k.startswith('unbiased') or k not in ref:
            continue
        pred = T0_net / ref[k]['continuum']
        row = dict(network=pred,
                   with_network_T0=T0_net / b['mfpt'], with_network_T0_ci95=ratio_ci(np.full(NBOOT, T0_net), b['boots']))
        if u is not None:
            row.update(measured=u['mfpt'] / b['mfpt'], measured_ci95=ratio_ci(u['boots'], b['boots']),
                       unbiased_source=u['source'])
            # same time step for both (dt = 4e-4), where most of the unbiased data are
            ku, kb = (('unbiased', 'none', 4e-4), (k.split('|')[0], k.split('|')[1], 4e-4))
            if ku in boots and kb in boots and boots[ku] is not None and boots[kb] is not None:
                mu_, mb_ = res['runs'][f'unbiased|none|dt=0.0004']['mfpt'], res['runs'][f'{k}|dt=0.0004']['mfpt']
                row.update(measured_same_dt_4e4=mu_ / mb_, measured_same_dt_4e4_ci95=ratio_ci(boots[ku], boots[kb]))
        res['speedups'][k] = row
    for U in ['U3', 'U6']:
        for interp in ['bilinear', 'bicubic']:
            km, ks = f'MFPT_{U}|{interp}', f'spectral_{U}|{interp}'
            if km in best and ks in best:
                res['ranking'][f'{U}|{interp}'] = dict(
                    description='T(spectral bias) / T(MFPT bias) = speedup(MFPT bias) / speedup(spectral bias)',
                    measured=best[ks]['mfpt'] / best[km]['mfpt'], measured_ci95=ratio_ci(best[ks]['boots'], best[km]['boots']),
                    network=ref[ks]['continuum'] / ref[km]['continuum'])
    res['network_reference'] = ref
    (HERE / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


if __name__ == '__main__':
    r = main()
    for k, v in r['comparison'].items():
        print(k, f"measured {v['measured']:.5g} {np.round(v['measured_ci95'], 1)} network {v['network_continuum']:.5g} "
                 f"ratio {v['measured_over_network']:.3f} {np.round(v['measured_over_network_ci95'], 3)}")
    for k, v in r['speedups'].items():
        print('speedup', k, v)
    for k, v in r['ranking'].items():
        print('ranking', k, v)
