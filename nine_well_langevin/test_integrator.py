"""Checks of the Langevin integrator.

1. Free diffusion (F = b = 0) in the reflecting box, start on the line x = 4, absorbed in the strip
   x <= 1. Exact MFPT: ((L - 1)^2 - (L - 4)^2) / (2D). Euler-Maruyama with the boundary checked only
   at the end of each step overestimates it; to first order the absorbing boundary moves outward by
   beta sqrt(2 D dt) with beta = -zeta(1/2)/sqrt(2 pi) = 0.5826 (Siegmund's correction), which
   adds (L - 1) beta sqrt(2 D dt) / D.
2. Tabulated gradient of F (4096^2 table, bilinear) against the analytic gradient: the same walkers
   (same random streams) on the MFPT-U6 bias, compared walker by walker.
"""
import json
import shutil

import numpy as np

import langevin
from common import HERE, L_BOX

OUT = HERE / 'tests_runs'
BETA = 0.5825971579390106


def free_test(dt, nwalk, seed):
    exact = ((L_BOX - 1) ** 2 - (L_BOX - 4) ** 2) / 2
    corr = (L_BOX - 1) * BETA * np.sqrt(2 * dt)
    r = langevin.run('free', 'none', dt, nwalk, seed=seed, expected=exact, quiet=True, out_dir=OUT, tag='_test')
    z = np.load(OUT / (r['name'] + '.npz'))
    f = z['steps'] * dt
    se = f.std(ddof=1) / np.sqrt(len(f))
    return dict(dt=dt, n=len(f), mean=float(f.mean()), se=float(se), exact=exact, exact_plus_correction=exact + corr,
                z_vs_exact=float((f.mean() - exact) / se), z_vs_corrected=float((f.mean() - exact - corr) / se))


def table_test(dt, nwalk, seed):
    res = {}
    for pot in ['table', 'analytic']:
        r = langevin.run('MFPT_U6', 'bilinear', dt, nwalk, seed=seed, expected=23.2, quiet=True, out_dir=OUT,
                         tag=f'_{pot}', potential=pot)
        z = np.load(OUT / (r['name'] + '.npz'))
        order = np.argsort(z['ids'])
        res[pot] = z['steps'][order] * dt
    a, t = res['analytic'], res['table']
    d = t - a
    return dict(dt=dt, n=len(a), mean_analytic=float(a.mean()), mean_table=float(t.mean()),
                fraction_identical=float(np.mean(d == 0)), mean_paired_difference=float(d.mean()),
                se_paired_difference=float(d.std(ddof=1) / np.sqrt(len(d))),
                relative_difference=float(d.mean() / a.mean()))


if __name__ == '__main__':
    out = dict(description=__doc__)
    out['free_diffusion'] = [free_test(dt, 40000, 21) for dt in [1.6e-3, 4e-4]]
    print(out['free_diffusion'], flush=True)
    out['table_vs_analytic'] = table_test(4e-4, 2000, 31)
    print(out['table_vs_analytic'], flush=True)
    (HERE / 'test_integrator.json').write_text(json.dumps(out, indent=1) + '\n')
