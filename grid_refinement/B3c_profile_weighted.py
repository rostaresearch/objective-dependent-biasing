"""Profile convergence weighted by where each objective is sensitive.

Many cells (high walls, wells off the A->B route, the interior of B) barely affect
the objective, so the optimum is nearly undetermined there and an unweighted RMS
difference mostly measures those cells.  Weights, computed on the finer grid with
the re-optimised bias:
  MFPT optimum     : expected time spent in each cell before reaching B, starting
                     uniformly in A (the occupation density, -lambda of the adjoint);
  spectral optimum : phi_2^2, the squared unit slow eigenvector of the symmetrised
                     generator (pi_i v2_i^2 in the original variables).
Reported: weighted RMS / U, and the weight carried by cells with |db| > U/2.
"""
import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import spsolve

import B_lib as BL


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


def npz(tag):
    return np.load(BL.OUT/f'B2_{tag}.npz')['best']


out = {}
d = loadmat(BL.DATA/'grid_2d_data.mat')
saved_b = dict(gap=d['u_ps_spec'], mfpt=d['u_ps_mfpt'])
xs_c, ys_c = d['xs'].ravel(), d['ys'].ravel()
for obj in ['gap', 'mfpt']:
    for fine, coarse in [('40x20', 'saved20x10'), ('80x40', 'saved20x10'), ('80x40', '40x20')]:
        s = BL.bottleneck_system(*map(int, fine.split('x')))
        b = npz(f'bottleneck_{fine}_U3_{obj}')
        if coarse == 'saved20x10':
            ref = BL.interpolate_bias(saved_b[obj], xs_c, ys_c, s, 'ij')
        else:
            sc = BL.bottleneck_system(*map(int, coarse.split('x')))
            ref = BL.interpolate_bias(npz(f'bottleneck_{coarse}_U3_{obj}').reshape(sc.shape), sc.xs, sc.ys, s, 'ij')
        out[f'bottleneck/{obj}/{fine}_vs_{coarse}'] = compare(b, ref, weights(s, b, obj), 3.0)
        print('bottleneck', obj, fine, coarse, out[f'bottleneck/{obj}/{fine}_vs_{coarse}'], flush=True)

r = loadmat(BL.DATA/'rkhs_2d_data.mat')
xs9 = r['xs'].ravel()
s20 = BL.nine_well_system(20)
s40 = BL.nine_well_system(40)
for U, k in [(3, 1), (6, 2)]:
    for obj, key in [('gap', f'u_spec_{k}'), ('mfpt', f'u_mfpt_{k}')]:
        b20 = npz(f'nine_well_20_U{U}_{obj}')
        b40 = npz(f'nine_well_40_U{U}_{obj}')
        out[f'nine_well_U{U}/{obj}/20x20_reopt_vs_saved20x20'] = compare(
            b20, r[key].ravel(), weights(s20, b20, obj), U)
        out[f'nine_well_U{U}/{obj}/40x40_vs_saved20x20'] = compare(
            b40, BL.interpolate_bias(r[key], xs9, xs9, s40, 'xy'), weights(s40, b40, obj), U)
        out[f'nine_well_U{U}/{obj}/40x40_vs_reopt20x20'] = compare(
            b40, BL.interpolate_bias(b20.reshape(20, 20), xs9, xs9, s40, 'xy'), weights(s40, b40, obj), U)
        for kk in [k2 for k2 in out if k2.startswith(f'nine_well_U{U}/{obj}/')]:
            print(kk, out[kk], flush=True)
BL.dump(out, BL.OUT/'B3c_profile_weighted.json')
print('saved B3c_profile_weighted.json')
