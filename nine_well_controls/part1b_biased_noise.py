"""Part 1b: each biased ensemble's own two-sample noise, and a separation test that allows for it.

Paths from a biased network can be more variable than unbiased paths (longer, more back-and-forth
traversals), so a biased-vs-unbiased overlap can fall below the unbiased-vs-unbiased band even when the
exact laws differ little. Here a second set of R biased path sets per N is drawn (new seeds) and paired
with the R biased sets of part 1, giving the biased-vs-biased band. A bias is counted as separated from
noise at N when the 95th percentile of its biased-vs-unbiased overlap lies below the 5th percentile of
both noise bands. The large-N mean of each own-noise band is also predicted from exact path moments.
"""
from __future__ import annotations

import json
import time

import numpy as np

from common import HERE, systems, tilt, law, signed_current, to_jsonable
from pathstats import (tables, sample_sets, seeds_for, norm_signed, omega_rows, cts_rows, path_moments,
                       clt_two_sample_omega, clt_two_sample_cts)
from part1_noise_ceiling import NS, R, SYS_ID, BIAS_ID, band


def main():
    S = systems()
    p1 = json.loads((HERE/'results_part1.json').read_text())
    counts = np.load(HERE/'part1_counts.npz')
    out = dict(description='biased-vs-biased two-sample noise and two-band separation test', systems={})
    t0 = time.time()
    for name, s in S.items():
        res = {}
        for key, b in s.biases.items():
            tab = tables(s, b)
            lb = law(tilt(s.K0, b), s.bias_pi(b), s.A, s.B, s.q0)
            xb = signed_current(lb, s.ea, s.eb)
            mom = path_moments(s, tab, np.sign(xb))
            rb = dict(exact_mean_jumps=mom['mean_jumps'], by_N={})
            for N in NS:
                net2, last2, _ = sample_sets(tab, seeds_for(SYS_ID[name], 100+BIAS_ID[key], N, n=R), N)
                net1 = counts[f'{name}_{key}_N{N}_net']; last1 = counts[f'{name}_{key}_N{N}_last']
                om_bb = omega_rows(norm_signed(net1), norm_signed(net2))
                ct_bb = cts_rows(last1, last2)
                vs = p1['systems'][name]['biased'][key]['by_N'][str(N)]
                nz = p1['systems'][name]['noise'][str(N)]
                row = dict(C_TS_own_noise=band(ct_bb), Omega_J_own_noise=band(om_bb),
                           C_TS_own_noise_clt_mean=1-clt_two_sample_cts(lb['probability'], N),
                           Omega_J_own_noise_clt_mean=1-clt_two_sample_omega(mom, N))
                for m in ['C_TS', 'Omega_J']:
                    row[m+'_separated_both_bands'] = bool(vs[m]['p95'] < min(nz[m]['p05'], row[m+'_own_noise']['p05']))
                rb['by_N'][str(N)] = row
            for m in ['C_TS', 'Omega_J']:
                sep = [N for N in NS if rb['by_N'][str(N)][m+'_separated_both_bands']]
                rb[m+'_first_separated_N_both_bands'] = sep[0] if sep else None
            res[key] = rb
            print(name, key, 'own noise N=1e4: C_TS', round(rb['by_N']['10000']['C_TS_own_noise']['median'], 4),
                  'Omega', round(rb['by_N']['10000']['Omega_J_own_noise']['median'], 4),
                  'first separated (both bands):', rb['C_TS_first_separated_N_both_bands'],
                  rb['Omega_J_first_separated_N_both_bands'], f'{time.time()-t0:.0f}s', flush=True)
        out['systems'][name] = res
    (HERE/'results_part1b.json').write_text(json.dumps(to_jsonable(out), indent=1)+'\n')


if __name__ == '__main__':
    main()
