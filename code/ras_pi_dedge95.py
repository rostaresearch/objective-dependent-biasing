"""Weighted 95th-percentile edge distortion for the Ras audit (review #13/#14).

The manuscript states that a small current-weighted MEAN can coexist with large tail
distortions, and that we therefore report the weighted 95th percentile alongside it. That
sentence needs a number for Ras, not just for the model systems in Fig. 10c.

Definition (matching pathway_distortion.py:112-115): the 95th percentile of the per-edge
distortion d_ij = |b_j - b_i| / 2 under the normalised reference-current weight
Jhat0_ij / sum(Jhat0), taken over edges carrying positive reference current.

Reported for both bias representations, block-bootstrapped over whole runs so the interval
is comparable to the mean.
"""
from __future__ import annotations
import os
import sys, json
import numpy as np

P = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # bundle root; override via MSM_ROOT
sys.path.insert(0, P)
from mechanism_audit_ras_proper import regularise_if_disconnected, dham_unbias, KBT
from mechanism_audit_highd_n20 import (stationary_from_K, committor_K,
                                       positive_net_current)
from ras_pi_audit import NPZ
import ras_pi_dedge_resolved_boot as R

N_BOOT = 200


def wpct(vals, w, p=95.0):
    """Weighted percentile of vals under weights w."""
    o = np.argsort(vals)
    v, ww = vals[o], w[o]
    c = np.cumsum(ww) / ww.sum()
    return float(v[np.searchsorted(c, p / 100.0)])


def dedge_stats(C, V_state, EX, seen, A, B):
    C_used, _ = regularise_if_disconnected(C, A, B, n_cv=R.NCV, n_f=R.NF, alpha=1e-3)
    M0 = dham_unbias(C_used, V_state)
    K0 = M0 - np.eye(M0.shape[0])
    b = V_state / KBT
    pi0 = stationary_from_K(K0)
    q0 = committor_K(K0, A, B, eps=1e-12)
    J0 = positive_net_current(K0, pi0, q0)
    d_mf = 0.5 * np.abs(b[None, :] - b[:, None])
    d_rs = 0.5 * np.where(seen, np.abs(EX), np.abs(b[None, :] - b[:, None]))
    m = J0 > 0
    w = J0[m]
    return (float(np.sum(J0 * d_mf)), wpct(d_mf[m], w),
            float(np.sum(J0 * d_rs)), wpct(d_rs[m], w))


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    fi = names.index('Pi_Q61')
    cvs, vbs, feats = d['cv'], d['vbias'], d['feat']
    runs = np.array([str(r) for r in d['run']])
    cv_all = np.concatenate([np.asarray(c, float) for c in cvs])
    f_all = np.concatenate([np.asarray(f, float)[:, fi] for f in feats])
    e_cv = np.linspace(cv_all.min() - 1e-6, cv_all.max() + 1e-6, R.NCV + 1)
    e_f = np.linspace(np.nanmin(f_all) - 1e-6, np.nanmax(f_all) + 1e-6, R.NF + 1)
    uruns = sorted(set(runs)); by_run = {r: np.where(runs == r)[0] for r in uruns}

    C, V, EX, seen, A, B = R.build_from(range(len(cvs)), cvs, vbs, feats, fi, e_cv, e_f)
    p = dedge_stats(C, V, EX, seen, A, B)
    print('point estimate:')
    print(f'  mean-field : mean {p[0]:.4f}   95th pct {p[1]:.4f}   ratio {p[1]/p[0]:.2f}')
    print(f'  resolved   : mean {p[2]:.4f}   95th pct {p[3]:.4f}   ratio {p[3]/p[2]:.2f}\n')

    rng = np.random.default_rng(0)
    acc = [[], [], [], []]
    for _ in range(N_BOOT):
        pick = rng.choice(uruns, size=len(uruns), replace=True)
        idx = np.concatenate([by_run[r] for r in pick])
        try:
            C, V, EX, seen, A, B = R.build_from(idx, cvs, vbs, feats, fi, e_cv, e_f)
            if not A or not B:
                continue
            s = dedge_stats(C, V, EX, seen, A, B)
        except Exception:
            continue
        for k in range(4):
            acc[k].append(s[k])
    lab = ['mean-field mean', 'mean-field p95', 'resolved mean', 'resolved p95']
    out = {}
    print(f'block bootstrap over {len(uruns)} runs (n={len(acc[0])}/{N_BOOT}):')
    for k in range(4):
        v = np.array(acc[k])
        out[lab[k]] = dict(median=float(np.median(v)), p5=float(np.percentile(v, 5)),
                           p95=float(np.percentile(v, 95)))
        print(f'  {lab[k]:18s} {np.median(v):.4f} '
              f'[{np.percentile(v,5):.4f}, {np.percentile(v,95):.4f}]')
    json.dump(dict(point=dict(zip(lab, p)), block_run=out),
              open(f'{P}/ras_pi_dedge95.json', 'w'), indent=2)
    print('\nsaved ras_pi_dedge95.json')


if __name__ == '__main__':
    main()
