"""Grid refinement, step 3: re-optimized optima versus the coarse ones.

Reads the B2_*.json/.npz job outputs and writes B3_reoptimized.json with, for each
system / budget / grid:
  * S_MFPT, S_gap, Omega_J, C_TS of the re-optimized MFPT and spectral optima
    (cross-objective values included), with all second-route checks (B_lib.evaluate);
  * the same quantities for the coarse optimum interpolated onto that grid;
  * start-to-start spread, stationarity and lambda_2/lambda_3 separation;
  * profile differences between resolutions;
  * a finite-difference check of both analytic gradients at the re-optimized
    40x20 bottleneck optima, and of the one-sided derivatives at a nine-well
    spectral optimum where lambda_2 = lambda_3.
"""
import json
import numpy as np
from scipy.io import loadmat

import B_lib as BL

OUT = BL.OUT


def job(system, grid, U, obj):
    tag = f'{system}_{grid}_U{U:g}_{obj}'
    p = OUT/f'B2_{tag}.json'
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    z = np.load(OUT/f'B2_{tag}.npz')
    j['best_x'] = z['best']
    j['all_x'] = {k[4:]: z[k] for k in z.files if k.startswith('run_')}
    return j


def mfpt_mask(s):
    """Cells whose bias changes the A->B MFPT: all cells outside B and the B cells
    adjacent to a cell outside B.  Inside B the MFPT does not depend on b."""
    isB = np.zeros(s.n, bool); isB[s.B] = True
    c = s.K0.tocoo()
    off = c.row != c.col
    adj = np.zeros(s.n, bool)
    adj[c.col[off & ~isB[c.row] & isB[c.col]]] = True
    return ~isB | adj


def profile_diff(b, ref, U, mask=None):
    d = b - ref
    if mask is not None:
        d = d[mask]
    return dict(rms_over_U=float(np.sqrt(np.mean(d**2))/U),
                rms_shift_removed_over_U=float(np.std(d)/U),
                max_abs=float(np.abs(d).max()),
                frac_cells_diff_gt_half_U=float(np.mean(np.abs(d) > U/2)))


def run_spread(j, s, U, mask=None):
    """Per-cell spread of b over the runs that reached the best value (rel. 1e-8)."""
    best = j['best_value']
    xs = [j['all_x'][r['start']] for r in j['runs'] if abs(r['value'] - best) <= 1e-8*abs(best)]
    X = np.array(xs)
    sp = X.max(axis=0) - X.min(axis=0)
    if mask is not None:
        sp = sp[mask]
    return dict(n_runs_at_best=len(xs), max_cell_spread=float(sp.max()),
                frac_cells_spread_gt_0p1=float(np.mean(sp > 0.1)))


def summarise_job(j):
    keep = ['best_start', 'best_value', 'value_spread_rel', 'stationarity', 'seconds']
    out = {k: j[k] for k in keep}
    out['runs'] = [{k: r.get(k) for k in ['start', 'value', 'stage1_gamma', 'stage2_gamma', 'nit', 'nfev',
                                          'message', 'proj_grad_rel', 'stage1_rel_sep']} for r in j['runs']]
    return out


res = dict(description='re-optimized per-state optima (pure box |b_i|<=U, L-BFGS-B, analytic gradients) '
                       'and their comparison with the coarse saved optima', bottleneck={}, nine_well={})

# ------------------------------------------------------------------ bottleneck
d = loadmat(BL.DATA/'grid_2d_data.mat')
coarse = dict(mfpt=d['u_ps_mfpt'], gap=d['u_ps_spec'])
xs_c, ys_c = d['xs'].ravel(), d['ys'].ravel()
reopt = {}
for grid in ['20x10', '40x20', '80x40']:
    jm, jg = job('bottleneck', grid, 3.0, 'mfpt'), job('bottleneck', grid, 3.0, 'gap')
    if jm is None or jg is None:
        continue
    s = BL.bottleneck_system(*map(int, grid.split('x')))
    reopt[grid] = dict(s=s, mfpt=jm['best_x'], gap=jg['best_x'])
    biases = {'MFPT_opt_U3': jm['best_x'], 'spectral_opt_U3': jg['best_x'],
              'MFPT_coarse_interp_U3': BL.interpolate_bias(coarse['mfpt'], xs_c, ys_c, s, 'ij'),
              'spectral_coarse_interp_U3': BL.interpolate_bias(coarse['gap'], xs_c, ys_c, s, 'ij')}
    base, rows = BL.evaluate(s, biases, dense_mfpt_limit=1600)
    mk = mfpt_mask(s)
    entry = dict(base=base, protocols=rows, job_mfpt=summarise_job(jm), job_gap=summarise_job(jg),
                 spread_mfpt=run_spread(jm, s, 3.0, mk), spread_gap=run_spread(jg, s, 3.0),
                 n_cells_mfpt_insensitive=int((~mk).sum()),
                 profile_vs_coarse_interp=dict(
                     mfpt=profile_diff(jm['best_x'], biases['MFPT_coarse_interp_U3'], 3.0, mk),
                     gap=profile_diff(jg['best_x'], biases['spectral_coarse_interp_U3'], 3.0)))
    res['bottleneck'][grid] = entry
    print('bottleneck', grid, 'evaluated', flush=True)

# re-optimized optima interpolated onto finer grids (series 2)
series2 = {}
for src in ['40x20', '80x40']:
    if src not in reopt:
        continue
    r = reopt[src]
    for tgt in [(80, 40), (160, 80)]:
        if tgt[0] <= r['s'].nx:
            continue
        s = BL.bottleneck_system(*tgt)
        biases = {f'MFPT_opt_{src}_interp': BL.interpolate_bias(r['mfpt'].reshape(r['s'].shape), r['s'].xs, r['s'].ys, s, 'ij'),
                  f'spectral_opt_{src}_interp': BL.interpolate_bias(r['gap'].reshape(r['s'].shape), r['s'].xs, r['s'].ys, s, 'ij')}
        base, rows = BL.evaluate(s, biases, dense_mfpt_limit=1600)
        series2[f'{src}->{s.label}'] = dict(base=base, protocols=rows)
        if s.label in reopt:
            mk = mfpt_mask(s)
            series2[f'{src}->{s.label}']['profile_reopt_vs_interp'] = dict(
                mfpt=profile_diff(reopt[s.label]['mfpt'], biases[f'MFPT_opt_{src}_interp'], 3.0, mk),
                gap=profile_diff(reopt[s.label]['gap'], biases[f'spectral_opt_{src}_interp'], 3.0))
res['bottleneck_reopt_interpolated'] = series2

# finite-difference check of both gradients at the re-optimized 40x20 optima
if '40x20' in reopt:
    s = reopt['40x20']['s']
    rng = np.random.default_rng(7)
    fdc = {}
    for obj in ['gap', 'mfpt']:
        b = reopt['40x20'][obj]
        free = np.flatnonzero(np.abs(np.abs(b) - 3) > 1e-3)
        bound = np.flatnonzero(np.abs(np.abs(b) - 3) <= 1e-3)
        idx = np.r_[rng.choice(free, min(8, len(free)), replace=False), rng.choice(bound, min(4, len(bound)), replace=False)]
        if obj == 'gap':
            f = lambda v: BL.gap_value(v, s)
            v0, G = BL.gap_and_grad_fast(b, s)
        else:
            f = lambda v: BL.mfpt_value(v, s)
            v0, G = BL.mfpt_and_grad_fast(b, s)
        eps = 1e-5
        fd = []
        for i in idx:
            e = np.zeros(s.n); e[i] = eps
            fd.append((f(b + e) - f(b - e))/(2*eps))
        fd = np.array(fd)
        scale = np.abs(G).max()
        fdc[obj] = dict(cells=idx.tolist(), analytic=G[idx].tolist(), finite_difference=fd.tolist(),
                        max_abs_err_over_max_grad=float(np.abs(fd - G[idx]).max()/scale), eps=eps,
                        note='first 8 cells are interior to the box, last 4 at a bound')
        print('FD check', obj, fdc[obj]['max_abs_err_over_max_grad'], flush=True)
    res['fd_check_bottleneck_40x20'] = fdc

# ------------------------------------------------------------------ nine-well
r = loadmat(BL.DATA/'rkhs_2d_data.mat')
xs9 = r['xs'].ravel()
for U, k in [(3.0, 1), (6.0, 2)]:
    coarse9 = dict(mfpt=r[f'u_mfpt_{k}'], gap=r[f'u_spec_{k}'])
    for grid in ['20', '40']:
        jm, jg = job('nine_well', grid, U, 'mfpt'), job('nine_well', grid, U, 'gap')
        if jm is None or jg is None:
            continue
        s = BL.nine_well_system(int(grid))
        biases = {f'MFPT_opt_U{U:g}': jm['best_x'], f'spectral_opt_U{U:g}': jg['best_x'],
                  f'MFPT_coarse_interp_U{U:g}': BL.interpolate_bias(coarse9['mfpt'], xs9, xs9, s, 'xy'),
                  f'spectral_coarse_interp_U{U:g}': BL.interpolate_bias(coarse9['gap'], xs9, xs9, s, 'xy')}
        base, rows = BL.evaluate(s, biases, dense_mfpt_limit=1600)
        mk = mfpt_mask(s)
        res['nine_well'][f'U{U:g}_{grid}'] = dict(
            base=base, protocols=rows, job_mfpt=summarise_job(jm), job_gap=summarise_job(jg),
            spread_mfpt=run_spread(jm, s, U, mk), spread_gap=run_spread(jg, s, U),
            profile_vs_coarse_interp=dict(
                mfpt=profile_diff(jm['best_x'], biases[f'MFPT_coarse_interp_U{U:g}'], U, mk),
                gap=profile_diff(jg['best_x'], biases[f'spectral_coarse_interp_U{U:g}'], U)))
        print('nine_well', U, grid, 'evaluated', flush=True)

# one-sided derivatives of gamma at a crossing (saved nine-well U=6 spectral optimum)
s = BL.nine_well_system(20)
b = r['u_spec_2'].ravel()
g0 = BL.gap_value(b, s)
_, G = BL.gap_and_grad_fast(b, s)
free = np.flatnonzero(np.abs(np.abs(b) - 6) > 1e-3)
idx = np.random.default_rng(3).choice(free, 8, replace=False)
eps = 1e-6
fwd, bwd = [], []
for i in idx:
    e = np.zeros(s.n); e[i] = eps
    fwd.append((BL.gap_value(b + e, s) - g0)/eps)
    bwd.append((g0 - BL.gap_value(b - e, s))/eps)
res['kink_check_nine_well_20_U6_saved_spectral'] = dict(
    cells=idx.tolist(), HF_gradient=G[idx].tolist(), forward_difference=fwd, backward_difference=bwd,
    max_forward_minus_backward_over_max_grad=float(np.max(np.abs(np.array(fwd) - np.array(bwd)))/np.abs(G).max()))
print('kink check', res['kink_check_nine_well_20_U6_saved_spectral']['max_forward_minus_backward_over_max_grad'])

BL.dump(res, OUT/'B3_reoptimized.json')
print('saved B3_reoptimized.json')
