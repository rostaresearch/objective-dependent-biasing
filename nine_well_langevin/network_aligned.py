"""Network MFPT predictions on cell-centred grids aligned with the A and B squares.

The vertex grids of grid_sensitivity.py (nodes at linspace(0, 2pi, n)) represent A and B by the
cells whose centres fall inside each square; the covered area then changes irregularly with n, and
some biased MFPTs (spectral) are sensitive to it. Here the domain [0, 2pi]^2 is divided into n x n
cells of side h = 2pi/n with nodes at cell centres. All square edges lie at half-integer multiples
of the 20 x 20 spacing 2pi/19, so for n a multiple of 38 the squares are exact unions of cells and the
no-flux outer faces coincide with the reflecting walls at 0 and 2pi used in the Langevin runs.

SQRA and Scharfetter-Gummel generators as in network_predictions.py; k0 = D/h^2. Source: uniform
over the A cells, which for these grids is exactly the uniform distribution on the A square in the
limit. Solver: SuperLU (spsolve).
"""
import json
import sys
import time

import numpy as np

from common import (BIAS_KEYS, CENTRE_A, CENTRE_B, D_COEF, HALF, HERE, L_BOX, audit_potential, bias_on_points,
                    generator_grid, generator_sg, mfpt_vector)


def aligned_grid(n):
    assert n % 38 == 0
    h = L_BOX / n
    x = (np.arange(n) + .5) * h
    X, Y = np.meshgrid(x, x)
    coords = np.c_[X.ravel(), Y.ravel()]
    A, B = [np.flatnonzero(np.max(abs(coords - c), axis=1) <= HALF) for c in (CENTRE_A, CENTRE_B)]
    assert len(A) == len(B) == (n // 19) ** 2
    return x, h, coords, A, B


def main(grids):
    rows = []
    log = open(HERE / 'network_aligned.log', 'a')
    for n in grids:
        x, h, coords, A, B = aligned_grid(n)
        F = audit_potential(x).ravel()
        for case in ['unbiased'] + list(BIAS_KEYS):
            for interp in (['none'] if case == 'unbiased' else ['bilinear', 'bicubic']):
                t0 = time.time()
                b = np.zeros(F.size) if case == 'unbiased' else bias_on_points(case, interp, coords[:, 0], coords[:, 1])
                U = F + b
                K = generator_grid(U.reshape(n, n))
                row = dict(n=n, h=h, case=case, interp=interp, n_A=len(A), n_B=len(B))
                t = mfpt_vector(K, B)
                row['mfpt_sqra'] = float(t[A].mean() * h * h / D_COEF)
                row['mfpt_sg'] = float(mfpt_vector(generator_sg(U.reshape(n, n)), B)[A].mean() * h * h / D_COEF)
                row['seconds'] = time.time() - t0
                rows.append(row)
                msg = json.dumps(row)
                print(msg, flush=True); log.write(msg + '\n'); log.flush()
    return rows


if __name__ == '__main__':
    grids = [int(v) for v in sys.argv[1:]] or [114, 228, 456]
    rows = main(grids)
    out_path = HERE / 'network_aligned.json'
    old = json.loads(out_path.read_text())['rows'] if out_path.exists() else []
    keep = [r for r in old if r['n'] not in grids]
    out_path.write_text(json.dumps(dict(description=__doc__, D=D_COEF, rows=keep + rows), indent=1) + '\n')
