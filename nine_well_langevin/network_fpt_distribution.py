"""Network prediction of the whole A -> B first-passage-time distribution (not only its mean).

Absorbing generator Q = K restricted to the states outside B. Under detailed balance
Q_s = Pi^{1/2} Q Pi^{-1/2} is symmetric; for SQRA its off-diagonal entries are all k0. The survival
probability from the uniform-over-A start is

    S(t) = sum_k exp(-mu_k t) a_k c_k,  a_k = p0^T Pi^{-1/2} phi_k,  c_k = phi_k^T Pi^{1/2} 1,

using the M slowest modes (shift-invert Lanczos). Truncation only affects times shorter than
1/mu_M; the check sum_k a_k c_k / mu_k against the directly solved MFPT is reported.
Times are physical (k0 = D/h^2).
"""
import json

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from common import BIAS_KEYS, D_COEF, HERE, audit_potential, bias_on_points, generator_grid, mfpt_vector
from network_aligned import aligned_grid

N_GRID = 228   # cell-centred grid aligned with the A and B squares (network_aligned.py)
N_MODES = 60


def survival_modes(n, case, interp='bilinear', m=N_MODES):
    x, h, coords, A, B = aligned_grid(n)
    U = audit_potential(x).ravel()
    if case != 'unbiased':
        U = U + bias_on_points(case, interp, coords[:, 0], coords[:, 1])
    K = generator_grid(U.reshape(n, n))
    keep = np.setdiff1d(np.arange(n * n), B)
    Q = csr_matrix(K[keep][:, keep])
    pi = np.exp(-(U[keep] - U[keep].min()))
    s = np.sqrt(pi)
    Qs = diags(s) @ Q @ diags(1 / s)
    Qs = csr_matrix((Qs + Qs.T) / 2)
    mu, phi = eigsh(-Qs, k=m, sigma=0, which='LM')
    order = np.argsort(mu); mu, phi = mu[order], phi[:, order]
    p0 = np.zeros(len(keep)); p0[np.searchsorted(keep, A)] = 1 / len(A)
    a = (p0 / s) @ phi
    c = phi.T @ s
    w = a * c
    rate = mu * D_COEF / h ** 2
    t_direct = mfpt_vector(K, B)[A].mean() * h ** 2 / D_COEF
    return dict(n=n, case=case, interp=interp, rates=rate.tolist(), weights=w.tolist(),
                weight_sum=float(w.sum()), mfpt_modes=float(np.sum(w / rate)), mfpt_direct=float(t_direct),
                slowest_rate_times_mfpt=float(rate[0] * t_direct), slowest_weight=float(w[0]))


def survival(model, t):
    r = np.array(model['rates']); w = np.array(model['weights'])
    return np.exp(-np.outer(t, r)) @ w


if __name__ == '__main__':
    out = {}
    for case in ['unbiased'] + list(BIAS_KEYS):
        for interp in (['bilinear', 'bicubic'] if case != 'unbiased' else ['none']):
            m = survival_modes(N_GRID, case, interp, m=300 if case == 'MFPT_U6' else N_MODES)
            out[f'{case}|{interp}'] = m
            print(case, interp, {k: m[k] for k in ['weight_sum', 'mfpt_modes', 'mfpt_direct',
                                                    'slowest_rate_times_mfpt', 'slowest_weight']}, flush=True)
    (HERE / 'network_fpt_distribution.json').write_text(json.dumps(out, indent=1) + '\n')
