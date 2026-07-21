"""Compute the bias-scaling (Pareto) sweep for the Ras Pi-unbinding audit and
export everything the MATLAB figure scripts need -> ras_pi_fig_data.mat

Scaling sweep: take the ACTUALLY APPLIED bias b = V_state/kBT and scale it,
b(alpha) = alpha * b. For each alpha report the reactive-current overlap
Omega_J(alpha), the edge distortion D_edge(alpha), and the model MFPT speedup
MFPT(K0)/MFPT(K^{alpha b}) for A->B. alpha=1 is the applied bias.

This is the acceleration-vs-mechanism-conservation tradeoff curve, and it also
serves as a sanity check on the headline Omega_J: overlap must degrade as the
bias is scaled up.
"""
from __future__ import annotations
import sys, json
import numpy as np
from scipy.io import savemat

import os
# Bundle root: override with the MSM_ROOT environment variable.
P = os.environ.get('MSM_ROOT',
     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, P)
sys.path.insert(0, os.environ.get('DHAM_HIGHD', P))

from mechanism_audit_ras_proper import dham_unbias, KBT
from mechanism_audit_highd_n20 import (stationary_from_K, committor_K,
                                       positive_net_current, tilt_generator)
from ras_pi_audit import build, NPZ

ALPHAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])


def mfpt_AB(K, A, B, pi, visited):
    """pi-weighted absorbing MFPT A->B on the visited subgraph.

    analytic_lib.mfpt_K cannot be used: the DHAM-reweighted K has all-zero rows
    for unvisited microstates, so its row-generator check fails. We restrict to
    visited states, which is where all the counts (and all the current) live.
    """
    vis = np.where(visited)[0]
    Bs = set(int(b) for b in B)
    N = [i for i in vis if i not in Bs]
    if not N:
        return np.nan
    idx = {s: k for k, s in enumerate(N)}
    M = K[np.ix_(N, N)]
    try:
        m = np.linalg.solve(M, -np.ones(len(N)))
    except np.linalg.LinAlgError:
        return np.nan
    if not np.all(np.isfinite(m)) or np.any(m < 0):
        return np.nan
    w = np.array([pi[i] for i in A if i in idx], dtype=float)
    mv = np.array([m[idx[i]] for i in A if i in idx], dtype=float)
    if w.sum() <= 0 or mv.size == 0:
        return np.nan
    return float((w * mv).sum() / w.sum())


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    fi = names.index('Pi_Q61')
    C, V_state, A, B, cnt = build(d['cv'], d['vbias'], d['feat'], fi, 12, 8, B_min=8.0)

    M0 = dham_unbias(C, V_state)
    K0 = M0 - np.eye(M0.shape[0])
    b0 = V_state / KBT
    pi0 = stationary_from_K(K0)
    q0 = committor_K(K0, A, B, eps=1e-12)
    Jh0 = positive_net_current(K0, pi0, q0)
    visited = cnt > 0
    mfpt0 = mfpt_AB(K0, A, B, pi0, visited)

    rows = []
    print(f"{'alpha':>6s} {'Omega_J':>9s} {'D_edge':>8s} {'S_MFPT':>10s}")
    for a in ALPHAS:
        b = a * b0
        Kb = tilt_generator(K0, b)
        pib = stationary_from_K(Kb)
        qb = committor_K(Kb, A, B, eps=1e-12)
        Jhb = positive_net_current(Kb, pib, qb)
        oj = float(np.minimum(Jh0, Jhb).sum())
        de = 0.5 * float((Jh0 * np.abs(b[None, :] - b[:, None])).sum())
        try:
            s_mfpt = float(mfpt0 / mfpt_AB(Kb, A, B, pib, visited))
        except Exception:
            s_mfpt = np.nan
        rows.append((float(a), oj, de, s_mfpt))
        print(f"{a:6.2f} {oj:9.4f} {de:8.4f} {s_mfpt:10.3f}")

    rows = np.array(rows)
    J = json.load(open(f'{P}/ras_pi_audit.json'))
    can, sweep, asw = J['canonical'], J['grid_feature_sweep'], J['alpha_sweep']
    # Canonical bars must show BLOCK-bootstrap CIs (resampling whole runs), not the
    # i.i.d.-transition CIs, which are ~13-19x too narrow. See ras_pi_block_boot.py.
    BB = json.load(open(f'{P}/ras_pi_blockboot.json'))['block_run']
    can = dict(can, Omega_J=BB['Omega_J'], D_edge=BB['D_edge'])

    out = dict(
        # canonical bars
        canon_names=np.array(['\\Omega_J', 'D_{edge}'], dtype=object),
        canon_med=np.array([can['Omega_J']['median'], can['D_edge']['median']]),
        canon_lo=np.array([can['Omega_J']['p5'], can['D_edge']['p5']]),
        canon_hi=np.array([can['Omega_J']['p95'], can['D_edge']['p95']]),
        canon_A=can['A'], canon_B=can['B'], canon_nreg=can['n_regularised'],
        # grid x feature
        gf_labels=np.array([r['label'] for r in sweep], dtype=object),
        gf_oj=np.array([r['Omega_J']['median'] for r in sweep]),
        gf_oj_lo=np.array([r['Omega_J']['p5'] for r in sweep]),
        gf_oj_hi=np.array([r['Omega_J']['p95'] for r in sweep]),
        gf_de=np.array([r['D_edge']['median'] for r in sweep]),
        gf_de_lo=np.array([r['D_edge']['p5'] for r in sweep]),
        gf_de_hi=np.array([r['D_edge']['p95'] for r in sweep]),
        # alpha sweep
        al_alpha=np.array([r['alpha'] for r in asw]),
        al_oj=np.array([r['Omega_J']['median'] for r in asw]),
        al_de=np.array([r['D_edge']['median'] for r in asw]),
        # scaling / Pareto
        sc_alpha=rows[:, 0], sc_oj=rows[:, 1], sc_de=rows[:, 2], sc_smfpt=rows[:, 3],
    )
    savemat(f'{P}/ras_pi_fig_data.mat', out)
    print(f"\nApplied bias (alpha=1): Omega_J={rows[ALPHAS==1,1][0]:.4f}  "
          f"S_MFPT={rows[ALPHAS==1,3][0]:.2f}")
    print('saved ras_pi_fig_data.mat')


if __name__ == '__main__':
    main()
