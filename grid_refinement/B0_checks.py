"""Grid refinement, step 0: provenance and code checks.

1. Which barrier height reproduces the saved bottleneck optima (grid_2d_data.mat)?
2. Reproduce the saved speedups of the n=200 and n=800 optima with the original
   dense analytic_lib routines, including the A/B sets each script used.
3. Sparse generator / tilt equal the dense analytic_lib ones.
4. Fast gradients equal the imported originals (per_state_ceiling.gap_and_grad,
   mfpt_per_state.mfpt_and_grad) and central finite differences.
"""
import time
import numpy as np
from scipy.io import loadmat

import B_lib as BL
from B_lib import L, X, DATA, ROOT, OUT
from per_state_ceiling import gap_and_grad as gap_and_grad_dense
from mfpt_per_state import mfpt_and_grad as mfpt_and_grad_dense

out = {}

# ---- 1-2. provenance of the saved bottleneck optima
d = loadmat(DATA/'grid_2d_data.mat')
dl = loadmat(ROOT/'grid_2d_data.mat')
out['grid_2d_data_repo_vs_local_max_abs_diff'] = {
    k: float(np.max(abs(np.asarray(d[k], float) - np.asarray(dl[k], float))))
    for k in d if not k.startswith('__')}
out['sha256'] = {BL.rel(p): BL.sha(p) for p in [DATA/'grid_2d_data.mat', DATA/'grid_2d_hires.mat',
                                              DATA/'rkhs_2d_data.mat']}
prov = {}
for bh in [4.0, 8.0]:
    K0, coords, F = L.grid_2d_generator(nx=20, ny=10, barrier_height=bh, bottleneck=True)
    A = np.where(coords[:, 0] < -0.85)[0]; B = np.where(coords[:, 0] > 0.85)[0]
    g0 = L.spectral_gap_K(K0); m0 = L.mfpt_K(K0, A, B)
    row = dict(F_max_abs_diff_vs_mat=float(np.max(abs(F.reshape(20, 10) - d['F']))),
               gap0=g0, gap0_mat=float(d['gap0']), mfpt0=m0, n_A=len(A), n_B=len(B))
    for key in ['u_ps_spec', 'u_ps_mfpt']:
        u = d[key].ravel()
        Kb = L.tilt_generator(K0, u)
        row[key] = dict(gap_speedup=L.spectral_gap_K(Kb)/g0, mfpt_speedup=m0/L.mfpt_K(Kb, A, B))
    prov[f'barrier_height_{bh}'] = row
    print('barrier', bh, row, flush=True)
prov['saved_values'] = dict(speedup_ps_spec=float(d['speedup_ps_spec']),
                            speedup_ps_mfpt=float(d['speedup_ps_mfpt']))
out['bottleneck_n200_provenance'] = prov

# n=800 optima (grid_2d_hires.mat, export_grid_2d_hires.py): A,B = single boundary columns
h = loadmat(DATA/'grid_2d_hires.mat')
K0, coords, F = L.grid_2d_generator(nx=40, ny=20, barrier_height=4.0, bottleneck=True)
g0 = L.spectral_gap_K(K0)
A_row = list(range(0, 20)); B_row = list(range(39*20, 800))
s40 = BL.bottleneck_system(40, 20)
hp = dict(F_max_abs_diff_vs_mat=float(np.max(abs(F.reshape(40, 20) - h['F']))),
          gap0=g0, gap0_mat=float(h['gap0']),
          mfpt0_single_column=L.mfpt_K(K0, A_row, B_row), mfpt0_mat=float(h['mfpt0']),
          mfpt0_footprint=L.mfpt_K(K0, s40.A, s40.B), n_A_footprint=len(s40.A),
          saved=dict(speedup_ps_spec=float(h['speedup_ps_spec']), speedup_ps_mfpt=float(h['speedup_ps_mfpt'])))
for key in ['u_ps_spec', 'u_ps_mfpt']:
    u = h[key].ravel(); Kb = L.tilt_generator(K0, u)
    hp[key] = dict(gap_speedup=L.spectral_gap_K(Kb)/g0,
                   mfpt_speedup_single_column=hp['mfpt0_single_column']/L.mfpt_K(Kb, A_row, B_row),
                   mfpt_speedup_footprint=hp['mfpt0_footprint']/L.mfpt_K(Kb, s40.A, s40.B))
print('hires', hp, flush=True)
out['bottleneck_n800_provenance'] = hp

# n=200 footprint A/B equal the x<-0.85 / x>0.85 sets used in the paper and exact_last_crossing
s20 = BL.bottleneck_system(20, 10)
_, c20, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0)
out['bottleneck_n200_AB_equal_paper'] = bool(np.array_equal(s20.A, np.where(c20[:, 0] < -.85)[0]) and
                                             np.array_equal(s20.B, np.where(c20[:, 0] > .85)[0]))

# nine-well 20x20: A,B single cells as in rkhs_2d_check / exact_last_crossing
r = loadmat(DATA/'rkhs_2d_data.mat')
n20 = BL.nine_well_system(20)
out['nine_well_20_F_max_abs_diff'] = float(np.max(abs(n20.F.reshape(20, 20) - r['F'])))
out['nine_well_20_nA_nB'] = [len(n20.A), len(n20.B)]

# ---- 3. sparse generator and tilt equal the dense analytic_lib ones
gen = {}
rng = np.random.default_rng(1)
for nx, ny in [(20, 10), (40, 20), (80, 40)]:
    K0d, cd, Fd = L.grid_2d_generator(nx=nx, ny=ny, barrier_height=4.0, bottleneck=True)
    s = BL.bottleneck_system(nx, ny)
    b = rng.uniform(-3, 3, s.n)
    gen[f'{nx}x{ny}'] = dict(
        coords=float(np.max(abs(cd - s.coords))), F=float(np.max(abs(Fd - s.F))),
        K0_rel=float(np.max(abs(K0d - s.K0.toarray()))/np.max(abs(K0d))),
        tilt_rel=float(np.max(abs(L.tilt_generator(K0d, b) - BL.tilt(s.K0, b).toarray()))/np.max(abs(K0d))))
    print('generator check', nx, ny, gen[f'{nx}x{ny}'], flush=True)
out['sparse_vs_dense_generator'] = gen

# ---- 4. gradients: fast vs imported dense originals vs finite differences
def fd_check(f, b, idx, eps):
    return np.array([(f(np.where(np.arange(len(b)) == i, b + eps, b)) -
                      f(np.where(np.arange(len(b)) == i, b - eps, b)))/(2*eps) for i in idx])


grads = {}
cases = [('bottleneck_20x10', s20, 3.0, d['u_ps_spec'].ravel(), d['u_ps_mfpt'].ravel()),
         ('bottleneck_40x20', s40, 3.0, None, None),
         ('nine_well_20x20', n20, 6.0, r['u_spec_2'].ravel(), r['u_mfpt_2'].ravel())]
for name, s, U, bspec, bmfpt in cases:
    K0d = s.K0.toarray()
    pts = [('random', rng.uniform(-U, U, s.n))]
    if bspec is not None:
        pts += [('saved_spectral_opt', bspec), ('saved_mfpt_opt', bmfpt)]
    for pname, b in pts:
        t = time.time(); gd, Gd = gap_and_grad_dense(b, K0d); td = time.time() - t
        t = time.time(); gf, Gf, ex = BL.gap_and_grad_fast(b, s, return_extra=True); tf = time.time() - t
        idx = rng.choice(s.n, 12, replace=False)
        fd = -fd_check(lambda v: -BL.gap_value(v, s), b, idx, 1e-5)
        row = dict(gap_dense=gd, gap_fast=gf, gap_rel_diff=abs(gd - gf)/gd,
                   gap_analytic_lib=L.spectral_gap_K(L.tilt_generator(K0d, b)),
                   lam3_over_lam2=float(ex['lams'][2]/ex['lams'][1]),
                   grad_fast_vs_dense_rel=float(np.max(abs(Gd - Gf))/np.max(abs(Gd))),
                   grad_fast_vs_fd_rel=float(np.max(abs(fd - Gf[idx]))/np.max(abs(Gf[idx]))),
                   t_dense=td, t_fast=tf)
        t = time.time(); md, Md = mfpt_and_grad_dense(b, K0d, s.A, s.B); td = time.time() - t
        t = time.time(); mf, Mf = BL.mfpt_and_grad_fast(b, s); tf = time.time() - t
        fdm = fd_check(lambda v: BL.mfpt_value(v, s), b, idx, 1e-5)
        row.update(mfpt_dense=md, mfpt_fast=mf, mfpt_rel_diff=abs(md - mf)/md,
                   mfpt_analytic_lib=L.mfpt_K(L.tilt_generator(K0d, b), s.A, s.B),
                   mgrad_fast_vs_dense_rel=float(np.max(abs(Md - Mf))/np.max(abs(Md))),
                   mgrad_fast_vs_fd_rel=float(np.max(abs(fdm - Mf[idx]))/np.max(abs(Mf[idx]))),
                   mt_dense=td, mt_fast=tf)
        grads[f'{name}/{pname}'] = row
        print('grad check', name, pname, {k: (f'{v:.3e}' if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
out['gradient_checks'] = grads
BL.dump(out, OUT/'B0_checks.json')
print('saved B0_checks.json')
