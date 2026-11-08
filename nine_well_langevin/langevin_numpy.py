"""Independent second implementation of the Langevin first-passage runs (cross-check of langevin.py).

Shares no code with langevin.py or common.py's numba kernels: the force of F is written out again in
vectorised numpy from the formula in grid_sensitivity.potential, the bilinear bias gradient is
written out again, and the noise comes from numpy's PCG64 generator (standard_normal). Same model:
overdamped Euler-Maruyama, D = 1, reflecting walls at 0 and 2pi, start uniform in the A square,
stop at the first step that lands in the B square. All walkers are advanced together; finished
walkers are removed. Walkers still running at t_max = 15 x the network MFPT are kept as censored
(probability ~3e-7 per walker for an exponential distribution).

Usage: python langevin_numpy.py CASE DT NWALK SEED
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.special import expit

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('E_LANGEVIN_ROOT', str(HERE.parent))) / 'data'
L = 2 * np.pi
KT = .5981
AMP = 6.
AM = np.array([.9, .3, .5, 1., .2, .4, .9, .9, .9]) * AMP
X0 = np.array([1.12, 1., 3., 4.15, 4., 5.27, 5.5, 6., 1.])
Y0 = np.array([1.34, 2.25, 2.31, 3.62, 5., 4.14, 4.5, 1.52, 5.])
SX = np.array([.5, .3, .4, 2., .9, 1., .3, .5, .5])
SY = np.array([.5, .3, 1., .8, .2, .3, 1., .6, .7])
CA = np.array([15, 12]) * L / 19          # A square centre (coarse node (15, 12)), half-width L/38
CB = np.array([3, 5]) * L / 19            # B square centre (coarse node (3, 5))
HW = L / 38


def grad_F(x, y):
    """Gradient of F (kBT) at arrays x, y."""
    dx = x[:, None] - X0; dy = y[:, None] - Y0
    g = AM * np.exp(-dx ** 2 / (2 * SX ** 2) - dy ** 2 / (2 * SY ** 2))
    gx = np.sum(g * dx / SX ** 2, axis=1)
    gy = np.sum(g * dy / SY ** 2, axis=1)
    e = .3 * AMP * np.exp(-(x - np.pi) ** 2 / 18 - (y - np.pi) ** 2 / .18)
    gx -= e * (x - np.pi) / 9
    gy -= e * (y - np.pi) / .09
    for v, out in ((x, 'x'), (y, 'y')):
        s1 = expit(-5 * (v + .7)); s2 = expit(5 * (v - (L + .7)))
        w = 100 * (-5 * s1 * (1 - s1) + 5 * s2 * (1 - s2))
        if out == 'x':
            gx += w
        else:
            gy += w
    return gx / KT, gy / KT


def grad_b_bilinear(tab, x, y):
    hb = L / 19
    i = np.clip((x / hb).astype(int), 0, 18); j = np.clip((y / hb).astype(int), 0, 18)
    s = x / hb - i; r = y / hb - j
    b00, b10, b01, b11 = tab[j, i], tab[j, i + 1], tab[j + 1, i], tab[j + 1, i + 1]
    return ((1 - r) * (b10 - b00) + r * (b11 - b01)) / hb, ((1 - s) * (b01 - b00) + s * (b11 - b10)) / hb


def run(case, dt, nwalk, seed, t_max):
    keys = {'MFPT_U3': 'u_mfpt_1', 'spectral_U3': 'u_spec_1', 'MFPT_U6': 'u_mfpt_2', 'spectral_U6': 'u_spec_2'}
    tab = None if case == 'unbiased' else loadmat(DATA / 'rkhs_2d_data.mat')[keys[case]]
    rng = np.random.Generator(np.random.PCG64(seed))
    x = CA[0] + HW * rng.uniform(-1, 1, nwalk); y = CA[1] + HW * rng.uniform(-1, 1, nwalk)
    alive = np.arange(nwalk); fpt = np.full(nwalk, np.nan)
    n, nmax, sq = 0, int(t_max / dt), np.sqrt(2 * dt)
    while len(alive) and n < nmax:
        gx, gy = grad_F(x, y)
        if tab is not None:
            bx, by = grad_b_bilinear(tab, x, y); gx += bx; gy += by
        z = rng.standard_normal((2, len(x)))
        x = x - dt * gx + sq * z[0]; y = y - dt * gy + sq * z[1]
        x = np.where(x < 0, -x, np.where(x > L, 2 * L - x, x))
        y = np.where(y < 0, -y, np.where(y > L, 2 * L - y, y))
        n += 1
        hit = (np.abs(x - CB[0]) <= HW) & (np.abs(y - CB[1]) <= HW)
        if hit.any():
            fpt[alive[hit]] = n * dt
            keep = ~hit; alive, x, y = alive[keep], x[keep], y[keep]
    return fpt, len(alive), n * dt


if __name__ == '__main__':
    case, dt, nwalk, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    expected = {'MFPT_U3': 2021.7, 'MFPT_U6': 23.22, 'spectral_U3': 13017., 'spectral_U6': 140750., 'unbiased': 724460.}[case]
    t0 = time.time()
    fpt, ncens, tc = run(case, dt, nwalk, seed, 15 * expected)
    done = fpt[np.isfinite(fpt)]
    E = done.sum() + ncens * tc
    boots = np.random.default_rng(1).choice(done, (20000, len(done))).mean(axis=1)
    res = dict(case=case, dt=dt, nwalk=nwalk, seed=seed, n_completed=int(len(done)), n_censored=int(ncens),
               mean_completed=float(done.mean()), mle_with_censoring=float(E / len(done)),
               ci95_bootstrap=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               network_continuum=expected, ratio=float(E / len(done) / expected), wall_s=time.time() - t0)
    print(res, flush=True)
    out = HERE / 'langevin_numpy.json'
    allres = json.loads(out.read_text()) if out.exists() else []
    allres.append(res)
    out.write_text(json.dumps(allres, indent=1) + '\n')
    np.save(HERE / 'runs_numpy' / f'{case}_dt{dt:g}_s{seed}.npy', fpt) if (HERE / 'runs_numpy').exists() else None
