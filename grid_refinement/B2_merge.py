"""Merge B2 part files (a job whose starts were split over several processes) into
the standard B2_<tag>.json / .npz.  For the gap objective the stationarity check of
the part that holds the best run is kept (it was computed for that run).

    python B2_merge.py nine_well_40_U6_mfpt
"""
import json
import sys
import numpy as np

import B_lib as BL

tag = sys.argv[1]
parts = sorted(BL.OUT.glob(f'B2_{tag}_part*.json'))
runs, xs, meta, part_of = [], {}, None, {}
for p in parts:
    j = json.loads(p.read_text())
    meta = meta or j
    z = np.load(p.with_suffix('.npz'))
    runs += j['runs']
    xs.update({k: z[k] for k in z.files if k.startswith('run_')})
    for r in j['runs']:
        part_of[r['start']] = j
obj = meta['objective']
best = (min if obj == 'mfpt' else max)(runs, key=lambda r: r['value'])
bx = xs[f'run_{best["start"]}']
if meta['system'] == 'nine_well':
    s = BL.nine_well_system(int(meta['grid']))
else:
    s = BL.bottleneck_system(*map(int, meta['grid'].split('x')))
if obj == 'mfpt':
    v, G = BL.mfpt_and_grad_fast(bx, s)
    stat = dict(proj_grad_rel=float(np.abs(BL.projected_gradient(bx, G, meta['U'])).max()/v))
else:
    pj = part_of[best['start']]
    assert pj['best_start'] == best['start']
    stat = pj['stationarity']
vals = [r['value'] for r in runs]
summary = dict(job=tag, merged_from=[p.name for p in parts], maxiter=meta['maxiter'], ftol=meta['ftol'],
               system=meta['system'], grid=meta['grid'], n=meta['n'], U=meta['U'], objective=obj,
               best_start=best['start'], best_value=best['value'],
               value_spread_rel=float((max(vals) - min(vals))/abs(best['value'])),
               stationarity=stat, seconds=sum(json.loads(p.read_text())['seconds'] for p in parts), runs=runs)
BL.dump(summary, BL.OUT/f'B2_{tag}.json')
np.savez(BL.OUT/f'B2_{tag}.npz', best=bx, **xs)
print('merged', tag, len(runs), 'runs; best', best['start'], best['value'], summary['stationarity'])
