"""Size of the Euler-Maruyama time-step error on a barrier-crossing MFPT, computed without sampling.

1D double well U(x) = a (x^2 - 1)^2 (barrier a kBT, curvature 8a at the minima, -4a at the top),
D = 1. Exact continuum MFPT from x0 = -1 to the absorbing point x = c by quadrature,
    T = int_{x0}^{c} dy e^{U(y)} int_{-inf}^{y} dz e^{-U(z)}.
EM chain x' = x - dt U'(x) + sqrt(2 dt) xi, absorbed when x' >= c: its mean absorption time is solved
exactly (up to a fine spatial quadrature of the Gaussian kernel) from (I - P) n = 1, T_EM = dt n(x0).
Run for the curvatures and dt values used in the 2D runs, to see whether EM can account for a
5-35% shortfall at dt = 4e-4.
"""
import json

import numpy as np
from scipy.integrate import quad
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.special import erf

from common import HERE


def exact(a, c, x0=-1.):
    U = lambda x: a * (x * x - 1) ** 2
    inner = lambda y: quad(lambda z: np.exp(-U(z) + U(y)), -3, y, limit=200)[0]
    return quad(inner, x0, c, limit=200)[0]


def em_chain(a, c, dt, x0=-1., h=None):
    s = np.sqrt(2 * dt)
    h = h or s / 12
    x = np.arange(-2.2, c, h) + h / 2          # cell centres; absorbing beyond c
    drift = x - dt * 4 * a * x * (x * x - 1)
    lo = x - h / 2
    # probability of landing in each cell = Gaussian mass over the cell
    w = int(np.ceil(7 * s / h)) + 2
    rows, cols, vals = [], [], []
    for k in range(-w, w + 1):
        j = np.arange(len(x)) + 0
        # target cell index for each source: nearest cell to drift + k*h
        centre = np.round((drift - x[0]) / h).astype(int) + k
        ok = (centre >= 0) & (centre < len(x))
        a1 = (lo[np.clip(centre, 0, len(x) - 1)] - drift) / s
        p = .5 * (erf((a1 + h / s) / np.sqrt(2)) - erf(a1 / np.sqrt(2)))
        rows.append(j[ok]); cols.append(centre[ok]); vals.append(p[ok])
    P = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(x), len(x)))
    n = spsolve((identity(len(x)) - P).tocsc(), np.ones(len(x)))
    return float(dt * np.interp(x0, x, n))


if __name__ == '__main__':
    out = dict(description=__doc__, rows=[])
    for a in [6., 9., 12.]:
        T = exact(a, 0.5)
        for dt in [4e-4, 2e-4, 1e-4]:
            Te = em_chain(a, 0.5, dt)
            row = dict(barrier=a, curvature_min=8 * a, dt=dt, exact=T, em=Te, relative_error=Te / T - 1)
            out['rows'].append(row)
            print(row, flush=True)
    (HERE / 'em_bias_1d.json').write_text(json.dumps(out, indent=1) + '\n')
