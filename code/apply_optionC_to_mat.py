"""Write the option-C polynomial results into grid_2d_data.mat, for the MATLAB figures.

Only the POLYNOMIAL fields change. The per-state fields (u_ps_spec 91.3x, u_ps_mfpt 85.14x,
budget_gap_ps, budget_mfpt_ps, regime_*_ps, *_mfpt_opt) are untouched: per_state_ceiling.py
and mfpt_per_state.py never projected their answers, so they were already on the pure box
and are unaffected by review item #8.

IMPORTANT semantics (I got this wrong once): the only polynomial curve in this mat is
`poly_spec`, the polynomial SPECTRAL optimum, and it is plotted on BOTH axes --
`*_gap_poly` is its spectral speedup and `*_mfpt_poly` is the MFPT speedup OF THAT SAME
BIAS (the cross-objective collapse curve). `*_mfpt_poly` is NOT the MFPT-optimal
polynomial. See optionC_sweeps_specbias_mfpt.py.

Writes a backup first and asserts the shared point matches the headline.
"""
from __future__ import annotations
import os
import json
import shutil
import numpy as np
from scipy.io import loadmat, savemat

PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # bundle root; override via MSM_ROOT
MAT = f'{PATH}/grid_2d_data.mat'
BAK = f'{PATH}/grid_2d_data_preOptionC.mat'

shutil.copyfile(MAT, BAK)
print(f'backed up -> {BAK}')

d = loadmat(MAT)
nx, ny = int(np.asarray(d['nx']).ravel()[0]), int(np.asarray(d['ny']).ravel()[0])
gap0 = float(np.asarray(d['gap0']).ravel()[0])

head = json.load(open(f'{PATH}/polynomial_optionC_FINAL.json'))
sw = json.load(open(f'{PATH}/optionC_sweeps_CORRECT.json'))

u_C = np.asarray(head['spectral']['u'], float)
sp_C = float(head['spectral']['speedup'])
assert np.abs(u_C).max() <= 3.0 + 1e-9, 'option-C bias escapes the box'

print(f"speedup_poly_spec : {float(np.asarray(d['speedup_poly_spec']).ravel()[0]):.3f}"
      f'  ->  {sp_C:.3f}')
d['u_poly_spec'] = u_C.reshape(nx, ny)
d['speedup_poly_spec'] = np.array([[sp_C]])
d['gap_poly'] = np.array([[sp_C * gap0]])

bU = np.asarray(d['budget_U']).ravel()
assert list(bU) == list(sw['budgets']), f'budget grid mismatch {bU} vs {sw["budgets"]}'
for k, new in (('budget_gap_poly', sw['budget_gap_poly']),
               ('budget_mfpt_poly', sw['budget_mfpt_poly'])):
    print(f'{k:17s}:', np.round(np.asarray(d[k]).ravel(), 2), '->', np.round(new, 2))
    d[k] = np.asarray(new, float).reshape(1, -1)

assert np.allclose(np.asarray(d['regime_FE']).ravel(), sw['regime_FE'], rtol=1e-6)
for k, new in (('regime_gap_poly', sw['regime_gap_poly']),
               ('regime_mfpt_poly', sw['regime_mfpt_poly'])):
    print(f'{k:17s}:', np.round(np.asarray(d[k]).ravel(), 2), '->', np.round(new, 2))
    d[k] = np.asarray(new, float).reshape(1, -1)

i3 = sw['budgets'].index(3.0)
i4 = sw['barriers'].index(4.0)
ok = True
for tag, got in (('budget U=3', sw['budget_gap_poly'][i3]),
                 ('regime bh=4', sw['regime_gap_poly'][i4])):
    if abs(got - sp_C) / sp_C > 0.005:
        print(f'  !! {tag} = {got:.3f} vs headline {sp_C:.3f} -- figure would contradict text')
        ok = False
    else:
        print(f'  OK {tag} = {got:.3f} matches headline {sp_C:.3f}')

# the cross-objective collapse must survive
m = sw['budget_mfpt_poly']
print(f"\ncross-objective collapse check: mfpt_poly at U=8 is {m[-1]:.2f} "
      f"(peak {max(m):.2f}) -> {'collapse present' if m[-1] < max(m) else '!! NO COLLAPSE'}")

savemat(MAT, {k: v for k, v in d.items() if not k.startswith('__')})
print(f'\nwrote {MAT}  (consistent={ok})')
