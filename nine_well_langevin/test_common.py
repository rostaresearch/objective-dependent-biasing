"""Checks of the continuous potential, its gradient and the two bias interpolants."""
import json

import numpy as np

from common import (BIAS_KEYS, COARSE_N, HERE, L_BOX, audit_potential, bias_bicubic, bias_bilinear,
                    bias_on_points, bicubic_coefficients, load_bias_table, potential_grad)


def run():
    rng = np.random.default_rng(1)
    out = {}
    # 1. potential equals the audit potential up to an additive constant (on the audit's own grid)
    x = np.linspace(0, L_BOX, 120)
    Fa = audit_potential(x)
    X, Y = np.meshgrid(x, x)
    Fn = np.array([[potential_grad(a, b)[0] for a, b in zip(rx, ry)] for rx, ry in zip(X, Y)])
    d = (Fn - Fn.min()) - Fa
    out['potential_vs_audit_maxabs'] = float(abs(d - d.mean()).max())
    # 2. analytic gradient against central finite differences of the audit-equivalent value
    pts = rng.uniform(0, L_BOX, size=(2000, 2))
    eps = 1e-6
    err = 0.; gmax = 0.
    for px, py in pts:
        _, gx, gy = potential_grad(px, py)
        fx = (potential_grad(px + eps, py)[0] - potential_grad(px - eps, py)[0]) / (2 * eps)
        fy = (potential_grad(px, py + eps)[0] - potential_grad(px, py - eps)[0]) / (2 * eps)
        err = max(err, abs(gx - fx), abs(gy - fy)); gmax = max(gmax, abs(gx), abs(gy))
    out['potential_gradient_fd_maxabs'] = err
    out['potential_gradient_max'] = gmax
    # 3. bias interpolants
    for name in BIAS_KEYS:
        tab = load_bias_table(name)
        coef = bicubic_coefficients(tab)
        lin = np.array([bias_bilinear(tab, a, b)[0] for a, b in pts])
        cub = np.array([bias_bicubic(coef, a, b)[0] for a, b in pts])
        e_lin = float(abs(lin - bias_on_points(name, 'bilinear', pts[:, 0], pts[:, 1])).max())
        e_cub = float(abs(cub - bias_on_points(name, 'bicubic', pts[:, 0], pts[:, 1])).max())
        # interpolation property at the nodes
        xs = np.linspace(0, L_BOX, COARSE_N)
        node_err = max(abs(bias_bicubic(coef, xs[i], xs[j])[0] - tab[j, i]) for i in range(COARSE_N) for j in range(COARSE_N))
        # gradients against finite differences
        gerr_lin = gerr_cub = 0.
        for px, py in pts[:500]:
            for f, tb in ((bias_bilinear, tab), (bias_bicubic, coef)):
                _, gx, gy = f(tb, px, py)
                fx = (f(tb, px + eps, py)[0] - f(tb, px - eps, py)[0]) / (2 * eps)
                fy = (f(tb, px, py + eps)[0] - f(tb, px, py - eps)[0]) / (2 * eps)
                e = max(abs(gx - fx), abs(gy - fy))
                if f is bias_bilinear:
                    gerr_lin = max(gerr_lin, e)
                else:
                    gerr_cub = max(gerr_cub, e)
        fine = np.linspace(0, L_BOX, 801)
        FX, FY = np.meshgrid(fine, fine)
        cub_fine = bias_on_points(name, 'bicubic', FX.ravel(), FY.ravel())
        out[name] = dict(bilinear_vs_scipy=e_lin, bicubic_vs_scipy=e_cub, bicubic_node_error=float(node_err),
                         bilinear_grad_fd=gerr_lin, bicubic_grad_fd=gerr_cub,
                         bicubic_range_fine=[float(cub_fine.min()), float(cub_fine.max())])
    print(json.dumps(out, indent=1))
    assert out['potential_vs_audit_maxabs'] < 1e-10
    assert out['potential_gradient_fd_maxabs'] < 1e-5 * max(1, out['potential_gradient_max'])
    for name in BIAS_KEYS:
        r = out[name]
        assert r['bilinear_vs_scipy'] < 1e-12 and r['bicubic_vs_scipy'] < 1e-10 and r['bicubic_node_error'] < 1e-10
    (HERE / 'test_common.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    run()
