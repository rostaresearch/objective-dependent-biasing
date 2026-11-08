"""Part 4: statistical cost of a bias, as the reweighting effective sample size.

Samples drawn from the biased stationary distribution pib are reweighted to the unbiased pi0 with
w_i = pi0_i/pib_i (so E_pib[w] = 1). The normalized Kish effective sample size is
    ESS/N = (E_pib w)^2 / E_pib[w^2] = 1/sum_i pib_i w_i^2 = 1/(1 + chi^2(pi0 || pib)).
It assumes independent samples; time correlation in a real simulation lowers it further.
Reported per bias: ESS/N, max_i w_i, Var_pib(log w). The values for the random biases, the
alpha-scaling curves and the controls are computed in parts 2 and 3 with the same function
(common.ess_stats); this script lists the optimized and control biases and runs the checks.

Checks:
  1. ESS/N = 1 at b = 0, and at alpha = 0 on every alpha-scaling curve (part 2 output).
  2. Gauge invariance: b and b + c give the same ESS/N.
  3. Second implementation: analytic_lib.reweighting_ess (repo code) for every optimized bias.
  4. Direct Monte Carlo: n i.i.d. draws from pib, Kish ESS (sum w)^2/(n sum w^2), 40 repeats,
     for one moderate-cost and one high-cost bias.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import HERE, ROOT, systems, ess_stats, to_jsonable

sys.path.insert(0, str(ROOT/'common'))
from analytic_lib import reweighting_ess   # noqa: E402


def mc_kish(pi0, pib, n, reps, rng):
    w_all = pi0/pib
    vals = []
    for _ in range(reps):
        idx = rng.choice(len(pib), size=n, p=pib)
        w = w_all[idx]
        vals.append(float(w.sum()**2/(n*(w*w).sum())))
    return np.array(vals)


def main():
    S = systems()
    res = dict(definition='ESS/N = 1/sum_i pib_i w_i^2 with w = pi0/pib (Kish, independent samples)',
               optimized={}, controls={}, checks={})
    for name, s in S.items():
        e0 = ess_stats(s.pi0, s.bias_pi(np.zeros(s.n)))
        res['checks'][f'{name}_ess_at_b0'] = e0['ess_frac']
        for key, b in s.biases.items():
            e = ess_stats(s.pi0, s.bias_pi(b))
            e_shift = ess_stats(s.pi0, s.bias_pi(b + 7.3))
            e_repo = reweighting_ess(s.pi0, s.bias_pi(b), 1.0)
            e['repo_reweighting_ess'] = e_repo
            e['relative_difference_vs_repo'] = abs(e_repo - e['ess_frac'])/e['ess_frac']
            e['gauge_shift_difference'] = abs(e_shift['ess_frac'] - e['ess_frac'])
            e['log10_max_w'] = float(np.log10(e['max_w']))
            res['optimized'][f'{name}/{key}'] = e
            print(name, key, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)
    # alpha = 0 checks from part 2
    p2 = json.loads((HERE/'results_part2.json').read_text())
    a0 = [cfg['alpha_curves'][k]['ess_frac'][0] for cfg in p2['configs'].values() for k in cfg['alpha_curves']]
    res['checks']['alpha0_ess_max_deviation_from_1'] = float(np.max(np.abs(np.array(a0)-1)))
    # controls from part 3
    p3 = json.loads((HERE/'results_part3.json').read_text())
    for name, row in p3['systems'].items():
        for key, c in row['controls'].items():
            res['controls'][f'{name}/{key}'] = {k: c['exact'][k] for k in ['ess_frac', 'max_w', 'var_logw', 'mfpt_speedup']}
    # direct Monte Carlo
    rng = np.random.default_rng(4)
    mc = {}
    for name, key, n in [('nine_well', 'spectral_U3', 100000), ('nine_well', 'MFPT_U6', 1000000),
                         ('bottleneck_grid', 'MFPT_U3', 100000)]:
        s = S[name]; pib = s.bias_pi(s.biases[key])
        v = mc_kish(s.pi0, pib, n, 40, rng)
        exact = ess_stats(s.pi0, pib)['ess_frac']
        mc[f'{name}/{key}'] = dict(n_samples=n, repeats=40, exact=exact, mc_median=float(np.median(v)),
                                   mc_p05=float(np.percentile(v, 5)), mc_p95=float(np.percentile(v, 95)),
                                   mc_mean=float(v.mean()), mc_sem=float(v.std(ddof=1)/np.sqrt(len(v))))
        print('MC', name, key, mc[f'{name}/{key}'], flush=True)
    res['checks']['monte_carlo'] = mc
    (HERE/'results_part4.json').write_text(json.dumps(to_jsonable(res), indent=1)+'\n')


if __name__ == '__main__':
    main()
