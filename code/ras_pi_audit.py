"""Mechanism audit on the Ras phosphate-unbinding MFPT runs (1D_PMg).

Replaces the orchestrator (GTP->GDP conformational) data with the actual
Pi-unbinding data the manuscript describes:
  CV        = P-Mg distance (Angstrom), from tj/RL_MFPT/1D_PMg/distance/
  bias V(s) = sum_j 4.184*a_j*exp(-(s-b_j)^2/(2 c_j^2))  kJ/mol, from params/
  A = {CV < 5.0} (phosphate near pocket), B = {CV > 8.0} (fully released;
      8 A is the release criterion used in the source simulations)

Reuses compute_diagnostics/bootstrap_audit from mechanism_audit_ras_proper.py
unchanged, so DHAM sign, strict boundaries, pseudocount and bootstrap are
identical to the shipped audit. Only the input data differ.
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
from mechanism_audit_ras_proper import compute_diagnostics, bootstrap_audit, KBT

NPZ = os.path.join(DATA, 'pi_features.npz')


def build(cvs, vbs, feats, feat_idx, n_cv, n_f, A_max=5.0, B_min=8.0, lag=1):
    """Mirror build_counts_2d, but with strict per-propagation boundaries."""
    cv_all = np.concatenate([np.asarray(c, float) for c in cvs])
    f_all = np.concatenate([np.asarray(f, float)[:, feat_idx] for f in feats])
    V_all = np.concatenate([np.asarray(v, float) for v in vbs])
    good = np.isfinite(f_all)
    e_cv = np.linspace(cv_all.min() - 1e-6, cv_all.max() + 1e-6, n_cv + 1)
    e_f = np.linspace(np.nanmin(f_all) - 1e-6, np.nanmax(f_all) + 1e-6, n_f + 1)
    ic = np.clip(np.digitize(cv_all, e_cv) - 1, 0, n_cv - 1)
    ifv = np.clip(np.digitize(np.where(good, f_all, e_f[0]), e_f) - 1, 0, n_f - 1)
    s = ic * n_f + ifv
    N = n_cv * n_f

    C = np.zeros((N, N))
    cur = 0
    for c in cvs:                       # strict boundaries: never cross a propagation
        n = len(c); st, en = cur, cur + n; cur += n
        a = s[st:en - lag]; b_ = s[st + lag:en]
        np.add.at(C, (a, b_), 1.0)
    C = 0.5 * (C + C.T)

    V_state = np.zeros(N); cnt = np.zeros(N)
    np.add.at(V_state, s, V_all); np.add.at(cnt, s, 1.0)
    ok = cnt > 0; V_state[ok] /= cnt[ok]

    cv_centers = 0.5 * (e_cv[:-1] + e_cv[1:])
    state_cv = np.repeat(cv_centers, n_f)
    A = [i for i in range(N) if state_cv[i] < A_max and cnt[i] > 0]
    B = [i for i in range(N) if state_cv[i] > B_min and cnt[i] > 0]
    return C, V_state, A, B, cnt


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    cvs, vbs, feats = d['cv'], d['vbias'], d['feat']
    nfr = sum(len(np.asarray(c, float)) for c in cvs)
    print(f"loaded {len(cvs)} propagations, {nfr} frames, features={names}\n")

    out = {}
    for B_min in (8.0, 7.5):
        print(f"================ B = {{CV > {B_min}}} ================")
        print(f"{'feature':14s} {'|A|':>4s} {'|B|':>4s} {'Omega_J':>9s} {'D_edge':>8s}  {'reg':>4s}")
        for fi, fname in enumerate(names):
            C, V_state, A, B, cnt = build(cvs, vbs, feats, fi, 12, 8, B_min=B_min)
            if not A or not B:
                print(f"{fname:14s} {len(A):4d} {len(B):4d}   -- empty A or B --")
                continue
            r = compute_diagnostics(C, V_state, A, B, n_cv=12, n_f=8,
                                    regularise=True, alpha_pseudo=1e-3)
            if r is None:
                print(f"{fname:14s} {len(A):4d} {len(B):4d}   -- None --"); continue
            print(f"{fname:14s} {len(A):4d} {len(B):4d} {r['Omega_J']:9.4f} "
                  f"{r['D_edge']:8.4f}  {str(r['regularised']):>4s}")
            out[f"{fname}_B{B_min}"] = dict(A=len(A), B=len(B), **{
                k: (float(v) if not isinstance(v, bool) else v) for k, v in r.items()})
        print()

    with open(os.path.join(DATA, 'ras_pi_audit_point.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("saved ras_pi_audit_point.json")


if __name__ == '__main__':
    main()
