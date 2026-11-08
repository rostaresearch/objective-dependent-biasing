"""Part 3: positive controls, biases built to move the reactive paths.

(i)  top50: +U on the S-side cells next to the q0 = 1/2 surface that carry the top 50% of the
     unbiased last-crossing probability (cells ranked by their summed last-crossing mass and
     added until the cumulative mass reaches 0.5), 0 elsewhere.
(ii) channel: +U over the region that carries the main unbiased reactive current across the
     surface, chosen from the unbiased current map:
       nine-well: rows 4-8, columns 5-9 of the 20x20 grid (x in [1.65, 2.98], y in [1.32, 2.65]),
                  the corridor from the central wells to the B basin; it holds the S-side cells of
                  73% of the unbiased last crossings;
       bottleneck: the open gap, x-index 6-13 and y-index 4-5 (|x| <= 0.37, |y| = 0.11), the only
                  low-barrier passage through the ridge at x = 0.
(iii) bottleneck only, channel_pushpull: (ii) plus -U on the two flanking rows (y-index 3 and 6,
     same x range), still within |b| <= U. Added because +3 alone leaves the gap the lowest
     passage (see README).
Exact network-model metrics, plus sampled C_TS and Omega_J against the unbiased sets of part 1.
A control counts as separated from noise at N when the 95th percentile of its sampled overlap with
the unbiased sets lies below the 5th percentile of both noise bands: unbiased vs unbiased (part 1)
and control vs control (two independent sets from the control network).
"""
from __future__ import annotations

import json

import numpy as np

from common import HERE, systems, metrics, to_jsonable
from pathstats import tables, sample_sets, seeds_for, norm_signed, omega_rows, cts_rows

NS = [100, 300, 1000, 3000, 10000]          # controls separate from noise at small N
R = 200


def top50_cells(s):
    pc = np.zeros(s.n); np.add.at(pc, s.ref['i'], s.p0)
    order = np.argsort(-pc); cum = np.cumsum(pc[order])
    k = int(np.searchsorted(cum, .5))+1
    return order[:k], pc


def controls(s):
    out = {}
    top, pc = top50_cells(s)
    if s.name == 'nine_well':
        r, c = np.divmod(np.arange(s.n), s.shape[1])
        chan = np.flatnonzero((r >= 4) & (r <= 8) & (c >= 5) & (c <= 9))
        for U in [3., 6.]:
            b = np.zeros(s.n); b[top] = U; out[f'top50_U{U:g}'] = (b, top)
            b = np.zeros(s.n); b[chan] = U; out[f'channel_U{U:g}'] = (b, chan)
    else:
        ix, iy = np.divmod(np.arange(s.n), s.shape[1])
        gap = np.flatnonzero((iy >= 4) & (iy <= 5) & (ix >= 6) & (ix <= 13))
        flank = np.flatnonzero(((iy == 3) | (iy == 6)) & (ix >= 6) & (ix <= 13))
        U = 3.
        b = np.zeros(s.n); b[top] = U; out['top50_U3'] = (b, top)
        b = np.zeros(s.n); b[gap] = U; out['channel_U3'] = (b, gap)
        b = np.zeros(s.n); b[gap] = U; b[flank] = -U; out['channel_pushpull_U3'] = (b, np.r_[gap, flank])
    return out, pc


def main():
    S = systems()
    counts = np.load(HERE/'part1_counts.npz')
    noise = json.loads((HERE/'results_part1.json').read_text())
    res = dict(description='positive controls; exact network-model metrics and sampled overlaps', systems={})
    arrays = {}
    for si, (name, s) in enumerate(S.items()):
        ctrl, pc = controls(s)
        rows = {}
        for ki, (key, (b, cells)) in enumerate(ctrl.items()):
            m = metrics(s, b)
            row = dict(cells=cells.tolist(), cell_xy=s.coords[cells].tolist(),
                       last_crossing_mass_of_cells=float(pc[cells].sum()), exact=m, sampled={})
            tab = tables(s, b)
            for N in NS:
                net, last, _ = sample_sets(tab, seeds_for(10+si, 100+ki, N, n=R), N)
                net2, last2, _ = sample_sets(tab, seeds_for(10+si, 200+ki, N, n=R), N)   # own-noise sets
                net0 = counts[f'{name}_unbiased_N{N}_net'][0::2]
                last0 = counts[f'{name}_unbiased_N{N}_last'][0::2]
                om = omega_rows(norm_signed(net), norm_signed(net0))
                ct = cts_rows(last, last0)
                om_bb = omega_rows(norm_signed(net), norm_signed(net2))
                ct_bb = cts_rows(last, last2)
                q = lambda v: dict(median=float(np.median(v)), p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))
                row['sampled'][str(N)] = dict(C_TS=q(ct), Omega_J=q(om), C_TS_own_noise=q(ct_bb), Omega_J_own_noise=q(om_bb))
            for mname in ['C_TS', 'Omega_J']:
                sep = [N for N in NS if row['sampled'][str(N)][mname]['p95'] <
                       min(noise['systems'][name]['noise'][str(N)][mname]['p05'],
                           row['sampled'][str(N)][mname+'_own_noise']['p05'])]
                row[mname+'_first_separated_N'] = sep[0] if sep else None
            arrays[f'{name}_{key}'] = b
            rows[key] = row
            print(name, key, {k: round(m[k], 4) for k in ['C_TS', 'Omega_J', 'mfpt_speedup', 'gap_speedup', 'ess_frac']},
                  'sampled N=1e4', round(row['sampled']['10000']['C_TS']['median'], 4),
                  round(row['sampled']['10000']['Omega_J']['median'], 4), flush=True)
        res['systems'][name] = dict(controls=rows)
    (HERE/'results_part3.json').write_text(json.dumps(to_jsonable(res), indent=1)+'\n')
    np.savez_compressed(HERE/'part3_control_biases.npz', **arrays)


if __name__ == '__main__':
    main()
