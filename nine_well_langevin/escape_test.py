"""Precise Langevin-versus-network test on sub-problems of the unbiased A -> B transition.

The unbiased A -> B MFPT (~7e5) only allows a few dozen transitions per day. Here the same unbiased
dynamics start in A but stop at intermediate squares on the route to B (half-width 2pi/38, centred
on coarse nodes, so the aligned grids cover them exactly). Their MFPTs are short enough for
thousands of walkers, so a few-percent difference between Langevin and the network would show.

Usage: python escape_test.py network        (network MFPTs, aligned grids 228 and 456, extrapolated)
       python escape_test.py langevin DT N1 N2  (Langevin, numba kernel, out_dir runs_escape;
                                                N1, N2 walkers for the two nearer targets)
"""
import json
import sys

import numpy as np

from common import HERE, L_BOX, audit_potential, generator_grid, mfpt_vector
from network_aligned import aligned_grid

HB = L_BOX / 19
TARGETS = {'node_11_12': (11, 12), 'node_8_9': (8, 9), 'node_6_6': (6, 6)}
OUT = HERE / 'escape_test.json'


def network():
    res = {}
    for name, (I, J) in TARGETS.items():
        c = np.array([I * HB, J * HB])
        vals = []
        for n in [114, 228, 456]:
            x, h, coords, A, B = aligned_grid(n)
            T = np.flatnonzero(np.max(abs(coords - c), axis=1) <= HB / 2)
            K = generator_grid(audit_potential(x))
            vals.append(float(mfpt_vector(K, T)[A].mean() * h * h))
        d1, d2 = vals[1] - vals[0], vals[2] - vals[1]
        p = np.log(abs(d1 / d2)) / np.log(2) if d1 * d2 > 0 and abs(d2) < abs(d1) else None
        ext = vals[2] + d2 / (2 ** p - 1) if p else vals[2]
        res[name] = dict(centre=c.tolist(), half_width=HB / 2, network_n114_228_456=vals, order=p, network_continuum=ext)
        print(name, res[name], flush=True)
    return res


def langevin_runs(dt, nwalks, seed=7):
    import langevin
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, nwalk in nwalks.items():
        I, J = TARGETS[name]
        net = out['network'][name]['network_continuum']
        r = langevin.run('unbiased', 'none', dt, nwalk, seed=seed, tag='_' + name, expected=net, quiet=True,
                         out_dir=HERE / 'runs_escape', target=(I * HB, J * HB, HB / 2, HB / 2))
        z = np.load(HERE / 'runs_escape' / (r['name'] + '.npz'))
        f = z['steps'] * dt
        b = np.random.default_rng(3).choice(f, (20000, len(f))).mean(axis=1)
        row = dict(target=name, dt=dt, n=len(f), mean=float(f.mean()),
                   ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], network=net,
                   ratio=float(f.mean() / net),
                   ratio_ci95=[float(np.percentile(b, 2.5) / net), float(np.percentile(b, 97.5) / net)],
                   cv=float(f.std() / f.mean()))
        out.setdefault('langevin', []).append(row)
        print(row, flush=True)
        OUT.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'network':
        out = json.loads(OUT.read_text()) if OUT.exists() else {}
        out['description'] = __doc__
        out['network'] = network()
        OUT.write_text(json.dumps(out, indent=1) + '\n')
    else:
        import numba
        numba.set_num_threads(3)
        langevin_runs(float(sys.argv[2]), {'node_11_12': int(sys.argv[3]), 'node_8_9': int(sys.argv[4])})
