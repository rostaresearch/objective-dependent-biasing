"""The A->B MFPT does not depend on b inside B away from its boundary, so the per-state
MFPT optimum is not unique there.  Omega_J, C_TS and the MFPT do not depend on those
values either, but the spectral gap does.  This script reports, for each bottleneck
grid, the spread of the cross value S_gamma over the MFPT runs that reached the best
MFPT, and the range of S_gamma when the undetermined cells are set to -U, 0 or +U.
(Nine-well: every cell of B touches a cell outside B, so nothing is undetermined.)
"""
import json
import numpy as np

import B_lib as BL
from B_lib import mfpt_mask

out = {}
for grid in ['20x10', '40x20', '80x40']:
    j = json.loads((BL.OUT/f'B2_bottleneck_{grid}_U3_mfpt.json').read_text())
    z = np.load(BL.OUT/f'B2_bottleneck_{grid}_U3_mfpt.npz')
    s = BL.bottleneck_system(*map(int, grid.split('x')))
    mk = mfpt_mask(s)
    g0 = BL.gap_value(np.zeros(s.n), s)
    best = j['best_value']
    runs = [r['start'] for r in j['runs'] if abs(r['value'] - best) <= 1e-8*best]
    sg = {r: BL.gap_value(z[f'run_{r}'], s)/g0 for r in runs}
    fills = {}
    for name, val in [('-U', -3.0), ('0', 0.0), ('+U', 3.0)]:
        b = z['best'].copy(); b[~mk] = val
        assert abs(BL.mfpt_value(b, s)/best - 1) < 1e-10
        fills[name] = BL.gap_value(b, s)/g0
    out[grid] = dict(n_undetermined_cells=int((~mk).sum()), S_gap_over_runs=sg,
                     S_gap_run_range=[min(sg.values()), max(sg.values())],
                     S_gap_with_undetermined_cells_set_to=fills)
    print(grid, out[grid], flush=True)
BL.dump(out, BL.OUT/'B3b_mfpt_cross_spread.json')
