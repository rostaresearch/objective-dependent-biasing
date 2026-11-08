"""Network (SQRA) MFPT predictions in physical time, with an independent discretization check.

For each grid n x n on [0, 2pi]^2 (nodes at linspace(0, 2pi, n), no-flux edges as in the
deposited generator), each bias (unbiased, MFPT/spectral at U = 3 and 6) and each continuous
interpolation of the saved 20x20 bias (bilinear, as in grid_sensitivity.py; bicubic spline as a
sensitivity check):

  * SQRA network, K_ij = (D/h^2) exp(-(U_j - U_i)/2), U = F + b  (generator_grid with k0 = 1,
    then t_phys = t_net h^2 / D). At n = 120 the biased network is built with the audit's tilt()
    and checked against generator_grid(F + b).
  * Scharfetter-Gummel network on the same nodes (independent discretization of the same
    Smoluchowski operator; it must approach the same h -> 0 limit).

Source convention: the A -> B MFPT is the mean over the A cells with equal weight (uniform over A),
exactly as in exact_last_crossing.mfpt. The Boltzmann-weighted (stationary within A) value is also
reported as a sensitivity check.
"""
import json
import time

import numpy as np

from common import (BIAS_KEYS, D_COEF, HERE, L_BOX, audit_mfpt, audit_potential, bias_on_points,
                    generator_grid, generator_sg, grid, mfpt_vector, tilt)

GRIDS = [20, 40, 120, 240, 480]
CASES = ['unbiased'] + list(BIAS_KEYS)
INTERPS = ['bilinear', 'bicubic']


def main():
    out = dict(description=__doc__, D=D_COEF, k0='D/h^2', source='uniform over A cells', rows=[])
    log = open(HERE / 'network_predictions.log', 'w')
    for n in GRIDS:
        x, coords, A, B = grid(n)
        h = x[1] - x[0]
        F = audit_potential(x).ravel()
        for interp in INTERPS:
            for case in CASES:
                if case == 'unbiased' and interp == 'bicubic':
                    continue
                t0 = time.time()
                b = np.zeros(F.size) if case == 'unbiased' else bias_on_points(case, interp, coords[:, 0], coords[:, 1])
                U = F + b
                K = generator_grid(U.reshape(n, n))
                if n == 120 and case != 'unbiased':
                    Kt = tilt(generator_grid(F.reshape(n, n)), b)
                    tilt_err = float(abs(Kt - K).max())
                    assert tilt_err < 1e-12, tilt_err
                t_sqra = mfpt_vector(K, B)
                solver_check = None
                if n <= 120:  # the audit's own solver (CSR spsolve), as a check of mfpt_vector
                    solver_check = float(audit_mfpt(K, A, B) / t_sqra[A].mean() - 1)
                    assert abs(solver_check) < 1e-7, solver_check
                t_sg = mfpt_vector(generator_sg(U.reshape(n, n)), B)
                w = np.exp(-(U[A] - U[A].min())); w /= w.sum()
                row = dict(n=n, h=h, interp=interp if case != 'unbiased' else 'none', case=case,
                           n_A=len(A), n_B=len(B),
                           bias_range=[float(b.min()), float(b.max())],
                           mfpt_sqra=float(t_sqra[A].mean() * h * h / D_COEF),
                           mfpt_sqra_stationaryA=float(w @ t_sqra[A] * h * h / D_COEF),
                           mfpt_sqra_centre_cell=float(t_sqra[A[np.argmin(np.linalg.norm(
                               (np.c_[coords[A, 0], coords[A, 1]] - np.array([4.960409453036515, 3.968327562429212])), axis=1))]] * h * h),
                           mfpt_sg=float(t_sg[A].mean() * h * h / D_COEF),
                           audit_solver_relative_difference=solver_check,
                           seconds=time.time() - t0)
                out['rows'].append(row)
                msg = json.dumps(row)
                print(msg, flush=True); log.write(msg + '\n'); log.flush()
    # speedups
    ref = {(r['n']): r for r in out['rows'] if r['case'] == 'unbiased'}
    for r in out['rows']:
        u = ref[r['n']]
        r['speedup_sqra'] = u['mfpt_sqra'] / r['mfpt_sqra']
        r['speedup_sqra_stationaryA'] = u['mfpt_sqra_stationaryA'] / r['mfpt_sqra_stationaryA']
        r['speedup_sg'] = u['mfpt_sg'] / r['mfpt_sg']
    # Richardson-type extrapolation from the two finest grids, assuming error ~ h^p with p fitted from
    # three grids when monotone, else reported as not extrapolated.
    extrap = {}
    for interp in ['none'] + INTERPS:
        for case in CASES:
            rows = sorted([r for r in out['rows'] if r['case'] == case and r['interp'] == interp and r['n'] >= 120],
                          key=lambda r: r['n'])
            if len(rows) < 3:
                continue
            for key in ['mfpt_sqra', 'mfpt_sg']:
                v = np.array([r[key] for r in rows]); hh = np.array([r['h'] for r in rows])
                d1, d2 = v[1] - v[0], v[2] - v[1]
                res = dict(values=v.tolist(), h=hh.tolist())
                if d1 * d2 > 0 and abs(d2) < abs(d1):
                    p = np.log(abs(d1 / d2)) / np.log(hh[0] / hh[1])
                    res.update(order=float(p), extrapolated=float(v[2] + d2 / ((hh[1] / hh[2]) ** p - 1)))
                extrap[f'{case}|{interp}|{key}'] = res
    out['extrapolation'] = extrap
    (HERE / 'network_predictions.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
