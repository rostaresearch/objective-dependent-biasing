"""Small-N check of the C_TS noise band: path sampling against multinomial draws, 5,000 pairs.

Each reactive path contributes exactly one last crossing, so the path-based counts of a set must
be multinomial with the exact law p0. With 200 pairs (part 1) the N = 100 means differed by 0.010;
this rerun with 25 times more pairs tests whether that difference is systematic.
Writes check_small_N.json.
"""
import json

import numpy as np

from common import HERE, systems
from pathstats import tables, sample_sets, seeds_for, cts_rows

P = 5000


def main():
    S = systems()
    out = {}
    for name in ['nine_well', 'bottleneck_grid']:
        s = S[name]
        tab = tables(s, np.zeros(s.n))
        for N in [100, 300]:
            net, last, _ = sample_sets(tab, seeds_for(777, N, n=2*P), N)
            ct = cts_rows(last[0::2], last[1::2])
            mult = np.random.default_rng(N).multinomial(N, s.p0, size=2*P)
            cm = cts_rows(mult[0::2], mult[1::2])
            pooled = last.sum(0); e = pooled.sum()*s.p0; m = e > 5
            out[f'{name}_N{N}'] = dict(pairs=P, paths_mean=float(ct.mean()), paths_se=float(ct.std()/np.sqrt(P)),
                                       multinomial_mean=float(cm.mean()), multinomial_se=float(cm.std()/np.sqrt(P)),
                                       paths_median=float(np.median(ct)), multinomial_median=float(np.median(cm)),
                                       pooled_chi2_per_dof=float((((pooled-e)**2/e)[m]).sum()/(m.sum()-1)))
            print(name, N, out[f'{name}_N{N}'], flush=True)
    (HERE/'check_small_N.json').write_text(json.dumps(out, indent=1)+'\n')


if __name__ == '__main__':
    main()
