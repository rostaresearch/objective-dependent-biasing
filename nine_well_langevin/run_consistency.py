"""Consistency of independent Langevin runs of the same case.

Within each complete run: walkers ordered by id are split into 10 blocks (chi-squared test of the
block means) and the lag-1 correlation of consecutive walker ids is computed; correlated random
streams would show up here. Across runs of the same case (different seeds, time steps, hosts):
chi-squared of the run means about their weighted mean, using each run's standard error
sd/sqrt(n). Time-step effects (<= 2%) are small against these errors and are ignored here.
"""
import json

import numpy as np
from scipy import stats

from analysis import load_runs
from common import HERE

if __name__ == '__main__':
    out = dict(description=__doc__, within_run=[], across_runs={})
    groups = load_runs()
    cases = sorted({(c, i) for (c, i, _) in groups})
    for c, i in cases:
        rows = []
        for (cc, ii, dt), runs in sorted(groups.items()):
            if (cc, ii) != (c, i):
                continue
            for r in runs:
                if len(r['cens']) or not r['final']:
                    continue
                z = np.load(r['path'])
                t = (z['steps'] * dt)[np.argsort(z['ids'])]
                n = len(t)
                if n < 100:
                    continue
                nb = 10
                b = t[:n // nb * nb].reshape(nb, -1).mean(axis=1)
                chi = float(np.sum((b - t.mean()) ** 2) / (t.var() / (n // nb)))
                w = dict(case=c, interp=i, dt=dt, host=r['host'], seed=r['seed'], n=n, mean=float(t.mean()),
                         se=float(t.std(ddof=1) / np.sqrt(n)), block_p=float(stats.chi2.sf(chi, nb - 1)),
                         lag1_r=float(np.corrcoef(t[:-1], t[1:])[0, 1]), lag1_se=float(1 / np.sqrt(n)))
                out['within_run'].append(w)
                rows.append(w)
        if len(rows) >= 2:
            m = np.array([w['mean'] for w in rows]); se = np.array([w['se'] for w in rows])
            wt = 1 / se ** 2
            mu = float((wt * m).sum() / wt.sum())
            chi = float(np.sum(((m - mu) / se) ** 2))
            out['across_runs'][f'{c}|{i}'] = dict(runs=len(rows), weighted_mean=mu, chi2=chi, dof=len(rows) - 1,
                                                  p_value=float(stats.chi2.sf(chi, len(rows) - 1)),
                                                  z_scores=[round(float(v), 2) for v in (m - mu) / se])
            print(c, i, out['across_runs'][f'{c}|{i}'])
    wr = out['within_run']
    out['within_run_summary'] = dict(runs=len(wr), min_block_p=min(w['block_p'] for w in wr),
                                     max_abs_lag1_over_se=max(abs(w['lag1_r']) / w['lag1_se'] for w in wr))
    print(out['within_run_summary'])
    (HERE / 'run_consistency.json').write_text(json.dumps(out, indent=1) + '\n')
