"""Part 1: sampling-noise ceiling for C_TS and Omega_J, and when each bias rises above it.

For the unbiased nine-well and bottleneck networks: 2R independent sets of N reactive
A->B paths per N; C_TS and Omega_J between set 2r and set 2r+1 (r = 0..R-1).
For each saved optimized bias: R sets of N biased paths, each compared with the
independent unbiased set 2r. Model-level (network) results, not MD.

Independent routes for the noise band:
  C_TS:    multinomial draws from the exact last-crossing law, and the closed-form
           large-N mean sum_e sqrt(p_e(1-p_e)/(pi N)).
  Omega_J: exact per-path moments of the net traversal counts (linear solves, no
           sampling) and the delta-method large-N mean.
"""
from __future__ import annotations

import json
import time

import numpy as np

from common import HERE, systems, metrics, signed_current, to_jsonable
from pathstats import (tables, sample_sets, seeds_for, norm_signed, omega_rows, cts_rows,
                       path_moments, clt_two_sample_omega, clt_two_sample_cts)

NS = [100, 300, 1000, 3000, 10000, 30000, 100000]
R = 200                                   # pairs per N (noise) and biased sets per N
SYS_ID = {'nine_well': 1, 'bottleneck_grid': 2}
BIAS_ID = {'MFPT_U3': 1, 'spectral_U3': 2, 'MFPT_U6': 3, 'spectral_U6': 4, 'polynomial_spectral_U3': 5}


def band(v):
    v = np.asarray(v)
    return dict(median=float(np.median(v)), p05=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), mean=float(np.mean(v)))


def main():
    S = systems()
    out = dict(description='two-sample sampling noise of C_TS and Omega_J; model-level (network) paths',
               N_values=NS, pairs_per_N=R, systems={})
    raw = {}
    for name, s in S.items():
        t_start = time.time()
        res = dict(n_cut_edges=len(s.p0), n_edges=s.n_edges, noise={}, biased={}, checks={})
        tab0 = tables(s, np.zeros(s.n))
        sigma = np.sign(s.J0); sigma[np.abs(s.J0) < 1e-14] = 0.
        mom0 = path_moments(s, tab0, sigma)
        # check: exact moments reproduce the exact normalized current of law()
        x_mom = mom0['mean']/np.abs(mom0['mean']).sum()
        res['checks']['moment_mean_vs_law_current_maxabs'] = float(np.abs(x_mom - s.J0).max())
        res['checks']['exact_mean_jumps_unbiased'] = mom0['mean_jumps']
        rng = np.random.default_rng(SYS_ID[name])
        unb = {}
        for N in NS:
            net, last, steps = sample_sets(tab0, seeds_for(SYS_ID[name], 0, N, n=2*R), N)
            unb[N] = (net, last)
            raw[f'{name}_unbiased_N{N}_net'] = net.astype(np.int32)
            raw[f'{name}_unbiased_N{N}_last'] = last.astype(np.int32)
            xs = norm_signed(net)
            om = omega_rows(xs[0::2], xs[1::2])
            ct = cts_rows(last[0::2], last[1::2])
            # independent route for C_TS: multinomial histograms from the exact law
            mult = rng.multinomial(N, s.p0, size=(2*R))
            ct_mult = cts_rows(mult[0::2], mult[1::2])
            res['noise'][str(N)] = dict(
                C_TS=band(ct), Omega_J=band(om),
                C_TS_multinomial=band(ct_mult),
                C_TS_clt_mean=1-clt_two_sample_cts(s.p0, N),
                Omega_J_clt_mean=1-clt_two_sample_omega(mom0, N),
                C_TS_vs_exact=band(cts_rows(last, np.broadcast_to(s.p0, last.shape))),
                Omega_J_vs_exact=band(omega_rows(xs, s.J0[None, :])),
                mean_jumps=float(steps.sum()/(2*R*N))+1)  # +1: launch step
            print(name, 'unbiased', N, f"C_TS {res['noise'][str(N)]['C_TS']['median']:.5f}",
                  f"Omega {res['noise'][str(N)]['Omega_J']['median']:.5f}", f'{time.time()-t_start:.0f}s', flush=True)
        for key, b in s.biases.items():
            ex = metrics(s, b)
            tab = tables(s, b)
            rb = dict(exact=dict(C_TS=ex['C_TS'], Omega_J=ex['Omega_J'], mfpt_speedup=ex['mfpt_speedup']), by_N={})
            for N in NS:
                net, last, steps = sample_sets(tab, seeds_for(SYS_ID[name], BIAS_ID[key], N, n=R), N)
                raw[f'{name}_{key}_N{N}_net'] = net.astype(np.int32)
                raw[f'{name}_{key}_N{N}_last'] = last.astype(np.int32)
                net0, last0 = unb[N]
                om = omega_rows(norm_signed(net), norm_signed(net0[0::2]))
                ct = cts_rows(last, last0[0::2])
                nb = res['noise'][str(N)]
                rb['by_N'][str(N)] = dict(C_TS=band(ct), Omega_J=band(om), mean_jumps=float(steps.sum()/(R*N))+1,
                                          C_TS_separated=bool(np.percentile(ct, 95) < nb['C_TS']['p05']),
                                          Omega_J_separated=bool(np.percentile(om, 95) < nb['Omega_J']['p05']))
            for m in ['C_TS', 'Omega_J']:
                sep = [N for N in NS if rb['by_N'][str(N)][m+'_separated']]
                rb[m+'_first_separated_N'] = sep[0] if sep else None
            res['biased'][key] = rb
            print(name, key, 'exact', round(ex['C_TS'], 4), round(ex['Omega_J'], 4),
                  'N=1e5 median', round(rb['by_N'][str(NS[-1])]['C_TS']['median'], 4),
                  round(rb['by_N'][str(NS[-1])]['Omega_J']['median'], 4),
                  'first separated', rb['C_TS_first_separated_N'], rb['Omega_J_first_separated_N'],
                  f'{time.time()-t_start:.0f}s', flush=True)
        res['seconds'] = time.time()-t_start
        out['systems'][name] = res
    # reproducibility check: identical seeds give identical counts
    s = S['nine_well']; tab0 = tables(s, np.zeros(s.n))
    a = sample_sets(tab0, seeds_for(SYS_ID['nine_well'], 0, 100, n=2*R), 100)
    out['checks'] = dict(reproducible_same_seeds=bool(np.array_equal(a[0], raw['nine_well_unbiased_N100_net'])
                                                      and np.array_equal(a[1], raw['nine_well_unbiased_N100_last'])))
    (HERE/'results_part1.json').write_text(json.dumps(to_jsonable(out), indent=1)+'\n')
    np.savez_compressed(HERE/'part1_counts.npz', **raw)


if __name__ == '__main__':
    main()
