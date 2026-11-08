"""Grid refinement, step 1: grid refinement with the saved coarse per-state optima
interpolated bilinearly onto each finer grid (no re-optimisation).

bottleneck : grids 20x10, 40x20, 80x40, 160x80; saved U=3 optima from
             grid_2d_data.mat: u_ps_spec (per-state spectral-gap optimum) and
             u_ps_mfpt (per-state A->B MFPT optimum).  The *_mfpt_poly arrays
             (MFPT of the spectral-optimal polynomial) are not used.
nine_well  : grids 20, 40, 80, 120 (as grid_sensitivity.py); saved optima from
             rkhs_2d_data.mat: u_spec_1/u_mfpt_1 (U=3), u_spec_2/u_mfpt_2 (U=6).

Usage: python B1_refine_interpolated.py bottleneck|nine_well
"""
import sys
import numpy as np
from scipy.io import loadmat

import B_lib as BL

which = sys.argv[1]
out = dict(description='coarse per-state optima interpolated bilinearly onto finer grids; no re-optimisation',
           source_sha256={}, levels=[])
if which == 'bottleneck':
    src = BL.DATA/'grid_2d_data.mat'
    d = loadmat(src)
    xs_c, ys_c = d['xs'].ravel(), d['ys'].ravel()
    coarse = {'MFPT_opt_U3': d['u_ps_mfpt'], 'spectral_opt_U3': d['u_ps_spec']}
    grids = [(20, 10), (40, 20), (80, 40), (160, 80)]
    make = lambda g: BL.bottleneck_system(*g)
    axes = 'ij'
    out['variables'] = dict(MFPT_opt_U3='grid_2d_data.mat:u_ps_mfpt (20x10, [ix,iy])',
                            spectral_opt_U3='grid_2d_data.mat:u_ps_spec (20x10, [ix,iy])')
    out['barrier_height'] = BL.BOTTLENECK_BARRIER
else:
    src = BL.DATA/'rkhs_2d_data.mat'
    d = loadmat(src)
    xs_c, ys_c = d['xs'].ravel(), d['ys'].ravel()
    coarse = {'MFPT_opt_U3': d['u_mfpt_1'], 'spectral_opt_U3': d['u_spec_1'],
              'MFPT_opt_U6': d['u_mfpt_2'], 'spectral_opt_U6': d['u_spec_2']}
    grids = [20, 40, 80, 120]
    make = BL.nine_well_system
    axes = 'xy'
    out['variables'] = {k: f'rkhs_2d_data.mat:{v} (20x20, [iy,ix])' for k, v in
                        zip(coarse, ['u_mfpt_1', 'u_spec_1', 'u_mfpt_2', 'u_spec_2'])}
out['source_sha256'][BL.rel(src)] = BL.sha(src)

for g in grids:
    s = make(g)
    biases = {k: BL.interpolate_bias(v, xs_c, ys_c, s, axes) for k, v in coarse.items()}
    if s.n == 400 or (which == 'bottleneck' and s.n == 200):
        for k, v in coarse.items():           # identity on the source grid
            assert np.max(abs(biases[k] - (v.ravel() if axes == 'ij' else v.ravel()))) < 1e-12
    base, rows = BL.evaluate(s, biases)
    base['protocols'] = rows
    out['levels'].append(base)
    print(which, s.label, 'done', f'{base["seconds"]:.0f}s', flush=True)
    BL.dump(out, BL.OUT/f'B1_{which}_interpolated.json')
print('saved')
