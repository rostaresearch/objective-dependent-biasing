"""Pooled vs transition-resolved vs no-unbiasing DHAM on the real Ras data
(review Critical 1, Edina's ask B.2).

The shipped estimator applies ONE mean-field weight per edge,
    W_ij = exp(+ (V_state[j] - V_state[i]) / 2kT),
i.e. average-then-exponentiate.  The transition-resolved estimator uses each
observation's ACTUAL bias difference,
    W_ij = E_obs[ exp(+X/2) ],        X = (V_endpoint_j - V_endpoint_i)/kT,
i.e. exponentiate-then-average.  Everything else (symmetrisation, row
normalisation, pseudocount, A/B, committor, current) is held identical, so the
difference isolates the estimator.

Orientation convention: for the undirected edge {i,j} we pool observations
travelled in BOTH directions, orienting each one's bias difference i->j.  This
mirrors the count symmetrisation the shipped estimator already performs.

Third arm: no unbiasing at all (W = 1).  If Omega_J is unchanged across all
three, the diagnostic is not testing the reweighting -- which is the point.
"""
from __future__ import annotations
import os
import sys, json
import numpy as np

# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PATH, 'data')
FIGURES = os.path.join(PATH, 'figures')

sys.path.insert(0, PATH)
from mechanism_audit_ras_proper import regularise_if_disconnected, KBT
from mechanism_audit_highd_n20 import (
    stationary_from_K, committor_K, positive_net_current, tilt_generator)
from ras_pi_jensen_Rij import build_with_transitions, N_CV, N_F

OUT = os.path.join(DATA, 'ras_pi_resolved_dham.json')


def edge_weights(tr_i, tr_j, tr_X, N, mode):
    """W[i,j] for the undirected edge {i,j}, pooling both travel directions
    with the bias difference oriented i->j."""
    W = np.ones((N, N))
    if mode == 'none':
        return W
    # orient every observation as (lo -> hi) then fill both directions
    key = np.minimum(tr_i, tr_j).astype(np.int64) * N + np.maximum(tr_i, tr_j)
    sign = np.where(tr_i <= tr_j, 1.0, -1.0)      # X oriented lo->hi
    Xo = tr_X * sign
    order = np.argsort(key, kind='stable')
    key_s, Xo_s = key[order], Xo[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    for g in np.split(np.arange(len(key_s)), bounds):
        lo, hi = divmod(int(key_s[g[0]]), N)
        X = Xo_s[g]
        W[lo, hi] = float(np.mean(np.exp(+X / 2)))   # exponentiate-then-average
        W[hi, lo] = float(np.mean(np.exp(-X / 2)))
    return W


def diagnostics(C, V_state, A, B, W):
    C_used, was_reg = regularise_if_disconnected(C, A, B, n_cv=N_CV, n_f=N_F,
                                                 alpha=1e-3)
    Csym = 0.5 * (C_used + C_used.T)
    Cu = Csym * W
    row = Cu.sum(axis=1, keepdims=True)
    M0 = np.zeros_like(Cu); ok = row[:, 0] > 0
    M0[ok] = Cu[ok] / row[ok]
    K0 = M0 - np.eye(M0.shape[0])
    b = V_state / KBT
    Kb = tilt_generator(K0, b)
    pi0 = stationary_from_K(K0); pib = stationary_from_K(Kb)
    q0 = committor_K(K0, A, B, eps=1e-12); qb = committor_K(Kb, A, B, eps=1e-12)
    Jh0 = positive_net_current(K0, pi0, q0); Jhb = positive_net_current(Kb, pib, qb)
    return dict(Omega_J=float(np.minimum(Jh0, Jhb).sum()),
                D_edge=0.5 * float((Jh0 * np.abs(b[None, :] - b[:, None])).sum()),
                regularised=bool(was_reg)), Jh0


def main():
    d = np.load(os.path.join(DATA, 'pi_features.npz'),
                allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    fi = names.index('Pi_Q61')
    B_ = build_with_transitions(d['cv'], d['vbias'], d['feat'], fi)
    N, V, A, B = B_['N'], B_['V_state'], B_['A'], B_['B']
    print(f"Pi_Q61, {N} states, |A|={len(A)}, |B|={len(B)}, "
          f"{len(B_['tr_X'])} transitions\n")

    dV = (V[None, :] - V[:, None]) / KBT
    W_pooled = np.exp(+0.5 * dV)                              # mean-field
    W_resolved = edge_weights(B_['tr_i'], B_['tr_j'], B_['tr_X'], N, 'resolved')
    W_none = np.ones((N, N))

    res, Jrefs = {}, {}
    print(f"{'estimator':26s} {'Omega_J':>9s} {'D_edge':>9s}")
    for tag, W in (('pooled mean-field (shipped)', W_pooled),
                   ('transition-resolved', W_resolved),
                   ('no unbiasing', W_none)):
        r, J = diagnostics(B_['C'], V, A, B, W)
        res[tag] = r; Jrefs[tag] = J
        print(f"{tag:26s} {r['Omega_J']:9.4f} {r['D_edge']:9.4f}")

    # How much does the reference network itself move between estimators?
    Jp, Jr = Jrefs['pooled mean-field (shipped)'], Jrefs['transition-resolved']
    Jn = Jrefs['no unbiasing']
    print(f"\nreference reactive-current overlap between estimators:")
    print(f"  pooled vs transition-resolved : {float(np.minimum(Jp, Jr).sum()):.4f}")
    print(f"  pooled vs no-unbiasing        : {float(np.minimum(Jp, Jn).sum()):.4f}")

    sp = [res[k]['Omega_J'] for k in res]
    print(f"\nOmega_J spread across estimators = {max(sp) - min(sp):.4f} "
          f"(range {min(sp):.4f} - {max(sp):.4f})")
    print("  Statistical (block-bootstrap over 62 runs) 90% CI width = 0.0154")
    with open(OUT, 'w') as f:
        json.dump({k: v for k, v in res.items()}, f, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == '__main__':
    main()
