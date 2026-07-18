"""Block-bootstrap the propagation-resolved D_edge on the real Ras data.

D_edge = E_Jhat0[ d_ij ],  d_ij = |b_j - b_i| / 2 = |log(K^b_ij / K^0_ij)|.

The shipped value uses the MEAN-FIELD bias b = V_state/kT, i.e. one mean bias per microstate
pooled over 617 propagations that each carried a DIFFERENT bias. That averaging cancels the
applied tilt along the reactive channel (measured: bin-mean tilt -0.003 kT vs the actual
per-transition tilt -0.321 kT). So the shipped D_edge understates the distortion produced by
the bias Balint actually applied.

Here we recompute the SAME functional with the SAME reference current Jhat0, changing only
the bias used to measure the distortion:

    D_edge_meanfield = 0.5 * sum_ij Jhat0_ij |V_state_j - V_state_i| / kT     [shipped]
    D_edge_resolved  = 0.5 * sum_ij Jhat0_ij |E_obs[X_ij]|                    [this file]

where X is each observed transition's true bias difference. Holding Jhat0 fixed isolates the
bias term, which is the thing in question.

Resamples whole RUNS (62) with replacement -- the same unit as ras_pi_block_boot.py -- so the
interval is directly comparable to the shipped D_edge = 0.063 [0.052, 0.074].
"""
from __future__ import annotations
import sys, json
import numpy as np

P = r'C:\Users\edina\Dropbox\MSM_Roundtable_2026'
sys.path.insert(0, P)
from mechanism_audit_ras_proper import (regularise_if_disconnected, dham_unbias, KBT)
from mechanism_audit_highd_n20 import (stationary_from_K, committor_K,
                                       positive_net_current)
from ras_pi_audit import NPZ

N_BOOT = 200
FEAT = 'Pi_Q61'
NCV, NF = 12, 8
A_MAX, B_MIN, LAG = 5.0, 8.0, 1


def build_from(idx, cvs, vbs, feats, fi, e_cv, e_f):
    """Counts, mean-field V_state, A/B, AND the per-edge mean of the true bias
    difference E_obs[X_ij], from a chosen list of propagation indices."""
    N = NCV * NF
    C = np.zeros((N, N)); V_state = np.zeros(N); cnt = np.zeros(N)
    sumX = np.zeros((N, N)); nX = np.zeros((N, N))
    for j in idx:
        cv = np.asarray(cvs[j], float)
        f = np.asarray(feats[j], float)[:, fi]
        V = np.asarray(vbs[j], float)
        ic = np.clip(np.digitize(cv, e_cv) - 1, 0, NCV - 1)
        ifv = np.clip(np.digitize(f, e_f) - 1, 0, NF - 1)
        s = ic * NF + ifv
        if len(s) > LAG:
            a, b = s[:-LAG], s[LAG:]
            np.add.at(C, (a, b), 1.0)
            X = (V[LAG:] - V[:-LAG]) / KBT          # true per-transition bias diff
            np.add.at(sumX, (a, b), X)
            np.add.at(nX, (a, b), 1.0)
            # orient the reverse direction too, so the edge mean is symmetric
            np.add.at(sumX, (b, a), -X)
            np.add.at(nX, (b, a), 1.0)
        np.add.at(V_state, s, V); np.add.at(cnt, s, 1.0)
    C = 0.5 * (C + C.T)
    ok = cnt > 0; V_state[ok] /= cnt[ok]
    EX = np.zeros((N, N)); m = nX > 0
    EX[m] = sumX[m] / nX[m]
    cvc = 0.5 * (e_cv[:-1] + e_cv[1:]); state_cv = np.repeat(cvc, NF)
    A = [i for i in range(N) if state_cv[i] < A_MAX and cnt[i] > 0]
    B = [i for i in range(N) if state_cv[i] > B_MIN and cnt[i] > 0]
    return C, V_state, EX, m, A, B


def both_dedge(C, V_state, EX, seen, A, B):
    """Return (D_edge_meanfield, D_edge_resolved, Omega_J) sharing one Jhat0."""
    C_used, _ = regularise_if_disconnected(C, A, B, n_cv=NCV, n_f=NF, alpha=1e-3)
    M0 = dham_unbias(C_used, V_state)
    K0 = M0 - np.eye(M0.shape[0])
    b = V_state / KBT
    pi0 = stationary_from_K(K0)
    q0 = committor_K(K0, A, B, eps=1e-12)
    J0 = positive_net_current(K0, pi0, q0)
    db_mf = np.abs(b[None, :] - b[:, None])
    # resolved: use the measured per-edge mean bias difference where observed,
    # else fall back to the mean field (rare; only pseudocount edges)
    db_rs = np.where(seen, np.abs(EX), db_mf)
    D_mf = 0.5 * float((J0 * db_mf).sum())
    D_rs = 0.5 * float((J0 * db_rs).sum())
    return D_mf, D_rs, J0


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    fi = names.index(FEAT)
    cvs, vbs, feats = d['cv'], d['vbias'], d['feat']
    runs = np.array([str(r) for r in d['run']])

    cv_all = np.concatenate([np.asarray(c, float) for c in cvs])
    f_all = np.concatenate([np.asarray(f, float)[:, fi] for f in feats])
    e_cv = np.linspace(cv_all.min() - 1e-6, cv_all.max() + 1e-6, NCV + 1)
    e_f = np.linspace(np.nanmin(f_all) - 1e-6, np.nanmax(f_all) + 1e-6, NF + 1)
    uruns = sorted(set(runs)); by_run = {r: np.where(runs == r)[0] for r in uruns}
    n_prop = len(cvs)
    print(f'{n_prop} propagations, {len(uruns)} runs, feature {FEAT}\n')

    # point estimate on all data
    C, V, EX, seen, A, B = build_from(range(n_prop), cvs, vbs, feats, fi, e_cv, e_f)
    D_mf, D_rs, _ = both_dedge(C, V, EX, seen, A, B)
    print(f'point estimate (all data):')
    print(f'  D_edge mean-field (shipped) = {D_mf:.4f}   [expect 0.0579]')
    print(f'  D_edge propagation-resolved = {D_rs:.4f}   ratio {D_rs/D_mf:.2f}x')
    print(f'  rate factors: e^D = {np.exp(D_mf):.4f}  vs  {np.exp(D_rs):.4f}\n')

    rng = np.random.default_rng(0)
    MF, RS = [], []
    for k in range(N_BOOT):
        pick = rng.choice(uruns, size=len(uruns), replace=True)
        idx = np.concatenate([by_run[r] for r in pick])
        try:
            C, V, EX, seen, A, B = build_from(idx, cvs, vbs, feats, fi, e_cv, e_f)
            if not A or not B:
                continue
            a, b, _ = both_dedge(C, V, EX, seen, A, B)
        except Exception:
            continue
        MF.append(a); RS.append(b)
    MF, RS = np.array(MF), np.array(RS)

    def q(v):
        return dict(median=float(np.median(v)), p5=float(np.percentile(v, 5)),
                    p95=float(np.percentile(v, 95)), n=int(len(v)))

    print(f'block bootstrap over {len(uruns)} runs (n={len(MF)}/{N_BOOT} valid):')
    for tag, v in (('mean-field (shipped)', MF), ('propagation-resolved', RS)):
        s = q(v)
        print(f"  D_edge {tag:22s} {s['median']:.4f} "
              f"[{s['p5']:.4f}, {s['p95']:.4f}]  width {s['p95']-s['p5']:.4f}")
    ratio = RS / np.maximum(MF, 1e-12)
    print(f"\n  ratio resolved/mean-field: median {np.median(ratio):.2f}x "
          f"[{np.percentile(ratio,5):.2f}, {np.percentile(ratio,95):.2f}]")

    out = dict(point=dict(D_edge_meanfield=D_mf, D_edge_resolved=D_rs,
                          ratio=D_rs / D_mf),
               block_run=dict(D_edge_meanfield=q(MF), D_edge_resolved=q(RS),
                              ratio=dict(median=float(np.median(ratio)),
                                         p5=float(np.percentile(ratio, 5)),
                                         p95=float(np.percentile(ratio, 95)))),
               note='Jhat0 held fixed (pooled mean-field reference); only the bias used to '
                    'measure the distortion differs. Resamples whole runs, as in '
                    'ras_pi_block_boot.py, so intervals are directly comparable.')
    json.dump(out, open(f'{P}/ras_pi_dedge_resolved_boot.json', 'w'), indent=2)
    print('\nsaved ras_pi_dedge_resolved_boot.json')


if __name__ == '__main__':
    main()
