"""Grid refinement: evaluate the nine-well per-state optima re-optimized on 80x80.

Reads the merged B2_nine_well_80_U{3,6}_{gap,mfpt}.json/.npz and writes
B7_ninewell_80.json with, for each U, the same fields as the 40x40 entries of
B3_reoptimized.json (speedups, cross values, Omega_J, C_TS, lambda3/lambda2, all
second-route checks, start-to-start spread, stationarity), plus
  * the re-optimized 40x40 optima interpolated onto 80x80, for comparison;
  * sensitivity-weighted profile differences (80x80 re-optimized minus the 40x40
    re-optimized optimum interpolated), as in B3c_profile_weighted.py.
Python 3.8 compatible (run on markov).
"""
import json
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve

import B_lib as BL
from B_lib import mfpt_mask

OUT = BL.OUT


def job(tag):
    j = json.loads((OUT/('B2_' + tag + '.json')).read_text())
    z = np.load(OUT/('B2_' + tag + '.npz'))
    j['best_x'] = z['best']
    j['all_x'] = {k[4:]: z[k] for k in z.files if k.startswith('run_')}
    return j


def summarise_job(j):
    keep = ['best_start', 'best_value', 'value_spread_rel', 'stationarity', 'seconds']
    out = {k: j[k] for k in keep}
    out['runs'] = [{k: r.get(k) for k in ['start', 'value', 'stage1_gamma', 'stage2_gamma', 'nit', 'nfev',
                                          'message', 'proj_grad_rel', 'stage1_rel_sep']} for r in j['runs']]
    return out


def run_spread(j, mask=None):
    best = j['best_value']
    xs = [j['all_x'][r['start']] for r in j['runs'] if abs(r['value'] - best) <= 1e-8*abs(best)]
    X = np.array(xs)
    sp = X.max(axis=0) - X.min(axis=0)
    if mask is not None:
        sp = sp[mask]
    return dict(n_runs_at_best=len(xs), max_cell_spread=float(sp.max()),
                frac_cells_spread_gt_0p1=float(np.mean(sp > 0.1)))


def weights(s, b, obj):
    Kb = BL.tilt(s.K0, b)
    if obj == 'gap':
        _, vecs = BL.top_eigs(Kb, s.logpi0 - b, k=3)
        w = vecs[:, 1]**2
    else:
        N = np.setdiff1d(np.arange(s.n), s.B)
        M = Kb.tocsr()[N][:, N]
        a = np.zeros(len(N)); a[np.searchsorted(N, s.A)] = 1/len(s.A)
        w = np.zeros(s.n); w[N] = -spsolve(M.T.tocsc(), a)
    return w/w.sum()


def compare(b, ref, w, U):
    d = b - ref
    return dict(weighted_rms_over_U=float(np.sqrt(np.sum(w*d*d))/U),
                weight_in_cells_diff_gt_half_U=float(np.sum(w[np.abs(d) > U/2])),
                unweighted_rms_over_U=float(np.sqrt(np.mean(d*d))/U))


r = loadmat(BL.DATA/'rkhs_2d_data.mat')
xs9 = r['xs'].ravel()
s80 = BL.nine_well_system(80)
s40 = BL.nine_well_system(40)
mk = mfpt_mask(s80)
res = {}
for U, k in [(3, 1), (6, 2)]:
    jm, jg = job('nine_well_80_U%d_mfpt' % U), job('nine_well_80_U%d_gap' % U)
    b40m = np.load(OUT/('B2_nine_well_40_U%d_mfpt.npz' % U))['best']
    b40g = np.load(OUT/('B2_nine_well_40_U%d_gap.npz' % U))['best']
    i40m = BL.interpolate_bias(b40m.reshape(40, 40), s40.xs, s40.ys, s80, 'xy')
    i40g = BL.interpolate_bias(b40g.reshape(40, 40), s40.xs, s40.ys, s80, 'xy')
    biases = {'MFPT_opt_U%d' % U: jm['best_x'], 'spectral_opt_U%d' % U: jg['best_x'],
              'MFPT_coarse_interp_U%d' % U: BL.interpolate_bias(r['u_mfpt_%d' % k], xs9, xs9, s80, 'xy'),
              'spectral_coarse_interp_U%d' % U: BL.interpolate_bias(r['u_spec_%d' % k], xs9, xs9, s80, 'xy'),
              'MFPT_reopt40_interp_U%d' % U: i40m, 'spectral_reopt40_interp_U%d' % U: i40g}
    base, rows = BL.evaluate(s80, biases, dense_mfpt_limit=1600)
    res['U%d_80' % U] = dict(
        base=base, protocols=rows, job_mfpt=summarise_job(jm), job_gap=summarise_job(jg),
        spread_mfpt=run_spread(jm, mk), spread_gap=run_spread(jg),
        profile_weighted=dict(
            gap_80_vs_reopt40=compare(jg['best_x'], i40g, weights(s80, jg['best_x'], 'gap'), U),
            mfpt_80_vs_reopt40=compare(jm['best_x'], i40m, weights(s80, jm['best_x'], 'mfpt'), U)))
    print('U', U, 'evaluated', flush=True)
    BL.dump(res, OUT/'B7_ninewell_80.json')
print('saved B7_ninewell_80.json')
