"""Third discretization of the continuum A -> B MFPT: central finite differences of the backward
Kolmogorov equation

    D Lap t - D grad(F + b) . grad t = -1  outside B,  t = 0 in B,  no-flux at the outer walls,

on the aligned cell-centred grids of network_aligned.py, with the exact (analytic) gradients of F
and of the interpolated bias at the cell centres. Unlike SQRA and Scharfetter-Gummel this is not a
rate matrix (it uses the force, not potential differences), so it is an independent check of the
h -> 0 value. Source: uniform over the A square.
"""
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from common import (D_COEF, HERE, bias_bicubic, bias_bilinear, bicubic_coefficients, load_bias_table,
                    potential_grad)
from network_aligned import aligned_grid


def grad_U(case, interp, coords):
    g = np.array([potential_grad(x, y)[1:] for x, y in coords])
    if case != 'unbiased':
        tab = load_bias_table(case)
        if interp == 'bilinear':
            g += np.array([bias_bilinear(tab, x, y)[1:] for x, y in coords])
        else:
            coef = bicubic_coefficients(tab)
            g += np.array([bias_bicubic(coef, x, y)[1:] for x, y in coords])
    return g


def fd_mfpt(n, case, interp):
    x, h, coords, A, B = aligned_grid(n)
    g = grad_U(case, interp, coords)
    N = n * n
    idx = np.arange(N).reshape(n, n)       # idx[iy, ix]
    rows, cols, vals = [], [], []
    for axis, comp in ((1, 0), (0, 1)):    # axis 1 = x (columns), axis 0 = y (rows)
        for step in (-1, 1):
            nb = np.roll(idx, -step, axis=axis)
            edge = (np.arange(n) == (0 if step == -1 else n - 1))
            mask = edge[None, :] if axis == 1 else edge[:, None]
            mask = np.broadcast_to(mask, (n, n))
            nb = np.where(mask, idx, nb)    # no-flux wall: ghost value equals the cell value
            coef = D_COEF / h ** 2 - step * D_COEF * g[:, comp] / (2 * h)
            rows.append(idx.ravel()); cols.append(nb.ravel()); vals.append(coef)
            rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(-np.full(N, D_COEF / h ** 2))
    L = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    free = np.setdiff1d(np.arange(N), B)
    t = np.zeros(N)
    t[free] = spsolve(L[free][:, free].tocsc(), -np.ones(len(free)))
    return float(t[A].mean())


if __name__ == '__main__':
    grids = [int(v) for v in sys.argv[1:]] or [114, 228, 456]
    out = dict(description=__doc__, rows=[])
    for case, interp in [('unbiased', 'none'), ('MFPT_U3', 'bilinear'), ('MFPT_U3', 'bicubic'),
                         ('spectral_U3', 'bilinear'), ('MFPT_U6', 'bilinear')]:
        for n in grids:
            v = fd_mfpt(n, case, interp)
            out['rows'].append(dict(n=n, case=case, interp=interp, mfpt_fd=v))
            print(n, case, interp, v, flush=True)
    (HERE / 'network_fd_check.json').write_text(json.dumps(out, indent=1) + '\n')
