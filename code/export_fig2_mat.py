"""Export the data behind manuscript Figure 2 (Analytic coverage speedup per CV)
to fig2_data.mat, so plot_fig2_us.m can rebuild it in MATLAB as an EDITABLE .fig.

Mirrors make_final_figure.py exactly (same fields, same ordering) but emits a .mat
instead of a matplotlib PNG. Follows the project convention used by
export_grid_2d_mat.py -> plot_grid_2d.m.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.io import savemat

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAN = float('nan')


def lab(d):
    if not d:
        return ''
    ov = d.get('median_overlap', NAN)
    if ov != ov:  # nan
        return f"W={d['W']},k={d['kappa']:.0f}"
    return f"W={d['W']},k={d['kappa']:.0f},O={ov:.2f}"


def pack(R):
    ub_b, ub_r = R['best_barrier_unconstrained'], R['best_rate_unconstrained']
    v_b, v_r = R['best_barrier_valid'], R['best_rate_valid']
    cvs = sorted(ub_b.keys())
    g = lambda d, cv, k: d.get(cv, {}).get(k, NAN)
    return dict(
        cvs=np.array(cvs, dtype=object),
        ub_barr=np.array([g(ub_b, cv, 'barrier_speedup') for cv in cvs], float),
        ub_rate=np.array([g(ub_r, cv, 'rate_speedup') for cv in cvs], float),
        v_barr=np.array([g(v_b, cv, 'barrier_speedup') for cv in cvs], float),
        v_rate=np.array([g(v_r, cv, 'rate_speedup') for cv in cvs], float),
        lab_barr=np.array([lab(ub_b.get(cv, {})) for cv in cvs], dtype=object),
        lab_rate=np.array([lab(ub_r.get(cv, {})) for cv in cvs], dtype=object),
        oracle_b=float(R['oracle']['barrier_speedup']),
        oracle_r=float(R['oracle']['rate_speedup']),
    )


def main():
    with open(f'{PATH}/filtered_results.json') as f:
        F = json.load(f)
    out = dict(
        grid=pack(F['grid']),
        penta=pack(F['pentapeptide']),
        title_grid='2D grid (n=800, deep barrier)',
        title_penta='Pentapeptide MSM (n=250)',
    )
    savemat(f'{PATH}/fig2_data.mat', out)
    for k in ('grid', 'penta'):
        d = out[k]
        print(f"{k}: cvs={list(d['cvs'])}")
        print(f"   ub_barr={np.round(d['ub_barr'],3)}")
        print(f"   ub_rate={np.round(d['ub_rate'],3)}")
        print(f"   v_barr ={np.round(d['v_barr'],3)}")
        print(f"   v_rate ={np.round(d['v_rate'],3)}")
        print(f"   oracle b={d['oracle_b']:.3f} r={d['oracle_r']:.3f}")
    print('\nSaved fig2_data.mat')


if __name__ == '__main__':
    main()
