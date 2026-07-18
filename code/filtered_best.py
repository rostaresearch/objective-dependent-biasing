"""
Post-hoc validity filter on the umbrella study results.

Per E. Rosta's review: an "honest best" protocol must satisfy
   coverage_failure_pdagger < 0.01
   coverage_failure_rate    < 0.01
   min neighbor overlap     >= 0.05         (when W > 1)
   W >= 2                                    (single-window cases trivially
                                              have no overlap and can give
                                              pathologically high speedup
                                              by ignoring everything outside
                                              the bias well)

Reports both the unconstrained mathematical optimum and the valid-overlap
best, side by side.  Saves filtered_results.json for the bar chart.
"""
from __future__ import annotations
import json
import numpy as np

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OV_MIN = 0.05
COVFAIL_MAX = 0.01


def is_valid(r: dict) -> bool:
    if r.get('W', 1) < 2:
        return False
    if r.get('coverage_failure_pdagger', 0.0) > COVFAIL_MAX:
        return False
    if r.get('coverage_failure_rate', 0.0) > COVFAIL_MAX:
        return False
    ov = r.get('median_overlap', float('nan'))
    if ov != ov:    # nan
        return False
    if ov < OV_MIN:
        return False
    return True


def best(rows, key, valid_only=False):
    best_per_cv = {}
    for r in rows:
        if 'error' in r:
            continue
        if valid_only and not is_valid(r):
            continue
        cv = r['cv']
        if cv not in best_per_cv or r[key] > best_per_cv[cv][key]:
            best_per_cv[cv] = r
    return best_per_cv


def main():
    with open(f'{PATH}/barrier_study.json') as f:
        R = json.load(f)

    out = {'filter_settings': dict(
        coverage_failure_max=COVFAIL_MAX, overlap_min=OV_MIN, W_min=2)}

    for net in ['grid', 'pentapeptide']:
        rows = [r for r in R[net]['rows'] if 'error' not in r]
        ub_barr = best(rows, 'barrier_speedup', valid_only=False)
        ub_rate = best(rows, 'rate_speedup',    valid_only=False)
        v_barr  = best(rows, 'barrier_speedup', valid_only=True)
        v_rate  = best(rows, 'rate_speedup',    valid_only=True)
        out[net] = dict(
            label=R[net]['label'],
            oracle=R[net]['oracle'],
            p_dagger=R[net]['p_dagger'],
            mfpt_unbiased=R[net]['mfpt_unbiased'],
            spectral_gap=R[net]['spectral_gap'],
            best_barrier_unconstrained=ub_barr,
            best_rate_unconstrained=ub_rate,
            best_barrier_valid=v_barr,
            best_rate_valid=v_rate,
        )

        print(f"\n=== {R[net]['label']} ===")
        print(f"  Oracle flatten:  barrier {R[net]['oracle']['barrier_speedup']:.3g}x"
              f"   rate {R[net]['oracle']['rate_speedup']:.3g}x")
        print(f"\n  Validity filter:  cov_fail < {COVFAIL_MAX},  overlap >= {OV_MIN},  W >= 2")
        cvs = sorted(set(r['cv'] for r in rows))
        hdr = f"  {'CV':10s} | {'unconstrained barrier':<28s} {'valid-best barrier':<28s} | {'unconstrained rate':<28s} {'valid-best rate':<28s}"
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for cv in cvs:
            uB = ub_barr.get(cv, {}); vB = v_barr.get(cv, {})
            uR = ub_rate.get(cv, {}); vR = v_rate.get(cv, {})
            def fmt(d, key='barrier_speedup'):
                if not d: return '       --'
                ov = d.get('median_overlap', float('nan'))
                ov_s = f'O={ov:.2f}' if ov == ov else 'O=  -'
                return f"{d[key]:9.2e} (W={d['W']:2d},k={d['kappa']:5.1f},{ov_s})"
            print(f"  {cv:10s} | {fmt(uB,'barrier_speedup'):<28s} {fmt(vB,'barrier_speedup'):<28s} "
                  f"| {fmt(uR,'rate_speedup'):<28s} {fmt(vR,'rate_speedup'):<28s}")

    with open(f'{PATH}/filtered_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('\nSaved filtered_results.json')


if __name__ == '__main__':
    main()
