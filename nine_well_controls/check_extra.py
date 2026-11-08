"""Extra cross-checks quoted in the README (writes results_checks.json).

1. Spectral-gap speedup: symmetrized eigenproblem (common.spectral_gap) against the repository's
   dense nonsymmetric route (analytic_lib.spectral_gap_K) and the values stored in the .mat files.
2. Omega_J: signed undirected-edge form against the directed sparse form of the TS audit, for every
   optimized bias, every control and the first 200 random biases of each kind (same seeds as part 2).
3. Where the unbiased current flows: share of sum|J| on edges that touch a cell adjacent to the cut.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.io import loadmat

from common import HERE, ROOT, systems, metrics, tilt, spectral_gap, to_jsonable
from part2_random_biases import CONFIGS, KINDS, draw

sys.path.insert(0, str(ROOT/'common'))
from analytic_lib import spectral_gap_K   # noqa: E402

DATA = ROOT/'data'


def main():
    S = systems()
    out = dict(gap_speedup={}, omega_forms={}, current_near_cut={})
    nine = loadmat(DATA/'rkhs_2d_data.mat'); grid = loadmat(DATA/'grid_2d_data.mat')
    stored = {('nine_well', 'MFPT_U3'): nine['speedup_gap_mfpt_1'], ('nine_well', 'spectral_U3'): nine['speedup_gap_spec_1'],
              ('nine_well', 'MFPT_U6'): nine['speedup_gap_mfpt_2'], ('nine_well', 'spectral_U6'): nine['speedup_gap_spec_2'],
              ('bottleneck_grid', 'spectral_U3'): grid['speedup_ps_spec'],
              ('bottleneck_grid', 'polynomial_spectral_U3'): grid['speedup_poly_spec']}
    for name, s in S.items():
        g0_dense = spectral_gap_K(s.K0.toarray())
        for key, b in s.biases.items():
            Kb = tilt(s.K0, b)
            sym = spectral_gap(Kb, s.bias_pi(b))/s.gap0
            dense = spectral_gap_K(Kb.toarray())/g0_dense
            row = dict(symmetrized=sym, dense=dense, rel_diff=abs(sym-dense)/dense)
            if (name, key) in stored:
                st = float(np.ravel(stored[(name, key)])[0])
                row['stored'] = st; row['rel_diff_stored'] = abs(sym-st)/st
            out['gap_speedup'][f'{name}/{key}'] = row
    # Omega_J forms
    worst, count = 0., 0
    for name, s in S.items():
        for b in s.biases.values():
            m = metrics(s, b); worst = max(worst, abs(m['Omega_J']-m['Omega_J_sparse'])); count += 1
    ctrl = np.load(HERE/'part3_control_biases.npz')
    for k in ctrl.files:
        name = 'nine_well' if k.startswith('nine_well') else 'bottleneck_grid'
        m = metrics(S[name], ctrl[k]); worst = max(worst, abs(m['Omega_J']-m['Omega_J_sparse'])); count += 1
    for ci, (name, U) in enumerate(CONFIGS):
        for kid, kind in enumerate(KINDS):
            B = draw(S[name], kind, U, np.random.default_rng([20261010, ci, kid]))[:200]
            for b in B:
                m = metrics(S[name], b, with_law=True)
                worst = max(worst, abs(m['Omega_J']-m['Omega_J_sparse'])); count += 1
    out['omega_forms'] = dict(n_biases=count, max_abs_difference=worst)
    # current near the cut
    for name, s in S.items():
        near = np.zeros(s.n, bool); near[s.ref['i']] = True; near[s.ref['j']] = True
        touch = near[s.ea] | near[s.eb]
        out['current_near_cut'][name] = dict(share_of_sum_abs_J=float(np.abs(s.J0)[touch].sum()),
                                             n_edges_touching=int(touch.sum()), n_edges=int(s.n_edges))
    print(json.dumps(to_jsonable(out), indent=1))
    (HERE/'results_checks.json').write_text(json.dumps(to_jsonable(out), indent=1)+'\n')


if __name__ == '__main__':
    main()
