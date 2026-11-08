"""Kolmogorov-Smirnov comparison of each complete Langevin sample with the network's first-passage-time
distribution. The network survival function (network_fpt_distribution.json, aligned grid n = 228) is
rescaled in time so that its mean equals the h -> 0 network MFPT (results.json), because the n = 228
means differ from the continuum by 0.1% (MFPT biases) to 6% (spectral bias, U = 6). Times below
3 / (fastest retained rate) are excluded, where the truncated mode sum is inaccurate.
"""
import json

import numpy as np
from scipy import stats

from analysis import load_runs
from common import HERE
from network_fpt_distribution import survival

if __name__ == '__main__':
    dist = json.loads((HERE / 'network_fpt_distribution.json').read_text())
    res = json.loads((HERE / 'results.json').read_text())
    out = dict(description=__doc__, rows=[])
    for (case, interp, dt), runs in sorted(load_runs().items()):
        k = f'{case}|{interp}'
        if k not in dist or any(len(r['cens']) for r in runs) or not all(r['final'] for r in runs):
            continue
        m = dist[k]
        scale = m['mfpt_direct'] / res['network_reference'][k]['continuum']
        f = np.sort(np.concatenate([r['fpt'] for r in runs]))
        n = len(f)
        ok = f * scale >= 3 / m['rates'][-1]
        S = survival(m, f[ok] * scale)
        emp_hi = 1 - np.arange(n)[ok] / n            # survival just before each point
        emp_lo = 1 - (np.arange(n)[ok] + 1) / n      # and just after
        D = float(max(np.max(np.abs(S - emp_hi)), np.max(np.abs(S - emp_lo))))
        row = dict(case=case, interp=interp, dt=dt, n=n, ks_D=D, p_value=float(stats.kstwo.sf(D, n)),
                   time_rescale=scale, mean_over_network=float(f.mean() / res['network_reference'][k]['continuum']))
        out['rows'].append(row)
        print(row)
    (HERE / 'ks_check.json').write_text(json.dumps(out, indent=1) + '\n')
