"""
Post-hoc validity screen on the umbrella study results.

Each objective is screened on ITS OWN coverage metric (this is the fix for the
earlier bug that conflated the two objectives by requiring both coverage
failures to be small simultaneously):

   barrier objective  ->  coverage_failure_pdagger < 0.01   (transition-state band)
   rate    objective  ->  coverage_failure_rate    < 0.01   (reactive tube)

plus, for both:
   min neighbour overlap  >= 0.05
   W >= 3                 (single- and two-window protocols are degenerate
                           "mirages": they can post a huge coverage gain by
                           ignoring everything outside one bias well)

For each network / CV / objective we report:
  * best_valid    : best protocol passing that objective's screen (or null)
  * best_found    : best protocol found at W>=3 (may fail the screen); carries
                    its own coverage failure Phi so failing bars can be drawn
                    faded and annotated instead of as a bare "none"
  * w1_optimum    : the unconstrained W=1 mathematical optimum (reported in the
                    companion table, not plotted)

Saves filtered_results.json for the bar chart and the tables.
"""
from __future__ import annotations
import os
import json
import numpy as np

PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root; override via MSM_ROOT
DATA = os.path.join(PATH, 'data')       # deposited inputs/outputs live under data/
FIGURES = os.path.join(PATH, 'figures') # figures live under figures/
OV_MIN, COVFAIL_MAX, W_MIN = 0.05, 0.01, 3
SCORE  = {'barrier': 'barrier_speedup',           'rate': 'rate_speedup'}
COVKEY = {'barrier': 'coverage_failure_pdagger',  'rate': 'coverage_failure_rate'}


def _finite(r, key):
    v = r.get(key)
    return v is not None and np.isfinite(v)

def _overlap_ok(r):
    ov = r.get('median_overlap')
    return ov is not None and np.isfinite(ov) and ov >= OV_MIN

def candidates(rows):
    """W>=3 protocols with a valid neighbour overlap."""
    return [r for r in rows if r.get('W', 1) >= W_MIN and _overlap_ok(r)]

def best_valid(rows, obj):
    c = [r for r in candidates(rows)
         if _finite(r, SCORE[obj]) and r.get(COVKEY[obj], 1.0) < COVFAIL_MAX]
    return max(c, key=lambda r: r[SCORE[obj]]) if c else None

def best_found(rows, obj):
    c = [r for r in candidates(rows) if _finite(r, SCORE[obj])]
    return max(c, key=lambda r: r[SCORE[obj]]) if c else None

def w1_optimum(rows, obj):
    c = [r for r in rows if r.get('W') == 1 and _finite(r, SCORE[obj])]
    return max(c, key=lambda r: r[SCORE[obj]]) if c else None


def _slim(r, obj):
    if r is None:
        return None
    return dict(score=r[SCORE[obj]], W=r['W'], kappa=r['kappa'],
                overlap=r.get('median_overlap'),
                coverage_failure=r.get(COVKEY[obj]))


def main():
    with open(f'{DATA}/barrier_study.json') as f:
        R = json.load(f)

    out = {'screen': dict(coverage_failure_max=COVFAIL_MAX,
                          overlap_min=OV_MIN, W_min=W_MIN,
                          note='each objective screened on its own coverage metric')}

    for net in ['grid', 'pentapeptide']:
        rows = [r for r in R[net]['rows'] if 'error' not in r]
        cvs = [cv for cv in dict.fromkeys(r['cv'] for r in rows)]
        per_cv = {}
        for cv in cvs:
            rs = [r for r in rows if r['cv'] == cv]
            per_cv[cv] = {obj: dict(valid=_slim(best_valid(rs, obj), obj),
                                    best_found=_slim(best_found(rs, obj), obj),
                                    w1=_slim(w1_optimum(rs, obj), obj))
                          for obj in ('barrier', 'rate')}
        out[net] = dict(label=R[net]['label'],
                        oracle=R[net]['oracle'], cvs=cvs, per_cv=per_cv)

        # console summary
        print(f"\n=== {R[net]['label']} ===  (screen: Phi<{COVFAIL_MAX}, O>={OV_MIN}, W>={W_MIN})")
        for cv in cvs:
            b, r_ = per_cv[cv]['barrier'], per_cv[cv]['rate']
            def show(o):
                v = o['valid']; f = o['best_found']
                if v:
                    return f"valid {v['score']:.3g} (W{v['W']},k{v['kappa']:.0f})"
                return f"FAILS  best {f['score']:.3g} @Phi={f['coverage_failure']:.3f}" if f else "--"
            print(f"  {cv:10s} | barrier: {show(b):<34s} | rate: {show(r_)}")

    with open(f'{DATA}/filtered_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('\nSaved filtered_results.json')


if __name__ == '__main__':
    main()
