"""
OBSOLETE / SUPERSEDED -- not part of the canonical pipeline.

The canonical generator for the published umbrella-coverage figure (fig2_notitle)
is ``fig7_coverage_gain.py``, which reads the current ``filtered_results.json``
schema (per_cv[cv][objective]['valid'/'best_found']) and writes to figures/.
This script uses an earlier results schema and is retained only for reference.

Final headline figure for the contribution.
Shows unconstrained best (light) overlaid with valid-overlap best (solid)
for both barrier and rate objectives.  Greyed bars indicate no valid protocol
in the swept (W, kappa) grid.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def fmt_label(d):
    if not d: return ''
    ov = d.get('median_overlap', float('nan'))
    if ov != ov:
        return f"W={d['W']},κ={d['kappa']:.0f}"
    return f"W={d['W']},κ={d['kappa']:.0f},O={ov:.2f}"


def main():
    with open(f'{PATH}/filtered_results.json') as f:
        F = json.load(f)
    nets = [('grid', '2D grid (n=800, deep barrier)'),
            ('pentapeptide', 'Pentapeptide MSM (n=250)')]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    for ax, (key, label) in zip(axes, nets):
        R = F[key]
        ub_b, ub_r = R['best_barrier_unconstrained'], R['best_rate_unconstrained']
        v_b,  v_r  = R['best_barrier_valid'],         R['best_rate_valid']
        cvs = sorted(ub_b.keys())
        xs = np.arange(len(cvs))
        w = 0.36

        ub_barr  = [ub_b.get(cv, {}).get('barrier_speedup', np.nan) for cv in cvs]
        ub_rate  = [ub_r.get(cv, {}).get('rate_speedup',    np.nan) for cv in cvs]
        v_barr   = [v_b.get(cv, {}).get('barrier_speedup', np.nan) for cv in cvs]
        v_rate   = [v_r.get(cv, {}).get('rate_speedup',    np.nan) for cv in cvs]

        # Unconstrained (light, with hatching to mark "possibly invalid")
        ax.bar(xs - w/2, ub_barr, w, color='#cfe4ff', edgecolor='#1f77b4',
               hatch='//', label='unconstrained barrier')
        ax.bar(xs + w/2, ub_rate, w, color='#ffd6d6', edgecolor='#d62728',
               hatch='\\\\', label='unconstrained rate')
        # Valid (solid)
        for i, cv in enumerate(cvs):
            if not np.isnan(v_barr[i]):
                ax.bar(xs[i] - w/2, v_barr[i], w, color='#1f77b4', edgecolor='k',
                       label='valid barrier' if i == 0 else None)
            if not np.isnan(v_rate[i]):
                ax.bar(xs[i] + w/2, v_rate[i], w, color='#d62728', edgecolor='k',
                       label='valid rate' if i == 0 else None)
            # annotation
            if not np.isnan(ub_barr[i]):
                ax.annotate(fmt_label(ub_b.get(cv, {})),
                            (xs[i] - w/2, ub_barr[i]),
                            xytext=(0, 4), textcoords='offset points',
                            fontsize=6.5, ha='center', color='#1f77b4', rotation=10)
            if not np.isnan(ub_rate[i]):
                ax.annotate(fmt_label(ub_r.get(cv, {})),
                            (xs[i] + w/2, ub_rate[i]),
                            xytext=(0, 4), textcoords='offset points',
                            fontsize=6.5, ha='center', color='#d62728', rotation=10)

        # Oracle reference lines
        oracle_b = R['oracle']['barrier_speedup']
        oracle_r = R['oracle']['rate_speedup']
        if 0 < oracle_b < 1e10:
            ax.axhline(oracle_b, color='#1f77b4', ls=':', lw=1.2, alpha=0.7,
                       label=f'oracle flatten (barrier)')
        if 0 < oracle_r < 1e10:
            ax.axhline(oracle_r, color='#d62728', ls=':', lw=1.2, alpha=0.7,
                       label=f'oracle flatten (rate)')

        ax.set_xticks(xs); ax.set_xticklabels(cvs, rotation=10)
        ax.set_yscale('log')
        ax.set_ylabel('speedup vs unbiased (log scale)')
        ax.set_title(label, fontsize=11)
        ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.grid(True, axis='y', ls=':', alpha=0.4)
        ax.legend(fontsize=7, loc='upper left', ncol=2)

    fig.suptitle("Analytic umbrella speedup per CV.   "
                 "Hatched/light = mathematical optimum (may have low overlap or coverage failure).   "
                 "Solid = valid-protocol best (overlap >= 0.05, coverage failure < 0.01, W>=2).",
                 fontsize=9.5)
    fig.savefig(f'{PATH}/fig_final.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_final.png')


if __name__ == '__main__':
    main()
