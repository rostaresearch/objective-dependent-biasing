"""Block bootstrap for the Ras Pi-unbinding audit.

The shipped bootstrap_audit() resamples the ~192k transitions as if i.i.d. They
are not: they are 617 contiguous propagations drawn from 62 independent runs, so
that CI is far too narrow. Here we resample the correct unit -- whole runs (and,
for comparison, whole propagations) with replacement -- and rebuild the count
matrix from scratch each time.

Also reports D_edge vs the A_max cutoff, since D_edge is sensitive to it.
"""
from __future__ import annotations
import sys, json
import numpy as np

import os
# Bundle root: override with the MSM_ROOT environment variable.
P = os.environ.get('MSM_ROOT',
     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, P)
from mechanism_audit_ras_proper import compute_diagnostics
from ras_pi_audit import NPZ

N_BOOT = 200
FEAT = 'Pi_Q61'
NCV, NF = 12, 8


def build_from(idx, cvs, vbs, feats, fi, n_cv, n_f, e_cv, e_f, A_max, B_min, lag=1):
    """Build C, V_state, A, B from a chosen list of propagation indices."""
    N = n_cv * n_f
    C = np.zeros((N, N)); V_state = np.zeros(N); cnt = np.zeros(N)
    for j in idx:
        cv = np.asarray(cvs[j], float)
        f = np.asarray(feats[j], float)[:, fi]
        V = np.asarray(vbs[j], float)
        ic = np.clip(np.digitize(cv, e_cv) - 1, 0, n_cv - 1)
        ifv = np.clip(np.digitize(f, e_f) - 1, 0, n_f - 1)
        s = ic * n_f + ifv
        if len(s) > lag:
            np.add.at(C, (s[:-lag], s[lag:]), 1.0)   # strict: never cross a prop
        np.add.at(V_state, s, V); np.add.at(cnt, s, 1.0)
    C = 0.5 * (C + C.T)
    ok = cnt > 0; V_state[ok] /= cnt[ok]
    cvc = 0.5 * (e_cv[:-1] + e_cv[1:]); state_cv = np.repeat(cvc, n_f)
    A = [i for i in range(N) if state_cv[i] < A_max and cnt[i] > 0]
    B = [i for i in range(N) if state_cv[i] > B_min and cnt[i] > 0]
    return C, V_state, A, B


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    fi = names.index(FEAT)
    cvs, vbs, feats, runs = d['cv'], d['vbias'], d['feat'], np.array([str(r) for r in d['run']])

    # fixed global bin edges (so every resample uses the same grid)
    cv_all = np.concatenate([np.asarray(c, float) for c in cvs])
    f_all = np.concatenate([np.asarray(f, float)[:, fi] for f in feats])
    e_cv = np.linspace(cv_all.min() - 1e-6, cv_all.max() + 1e-6, NCV + 1)
    e_f = np.linspace(np.nanmin(f_all) - 1e-6, np.nanmax(f_all) + 1e-6, NF + 1)

    n_prop = len(cvs); uruns = sorted(set(runs))
    by_run = {r: np.where(runs == r)[0] for r in uruns}
    print(f"{n_prop} propagations, {len(uruns)} runs\n")

    out = {}
    for unit in ("run", "prop"):
        rng = np.random.default_rng(0); OJ, DE = [], []
        for _ in range(N_BOOT):
            if unit == "run":
                pick = rng.choice(uruns, size=len(uruns), replace=True)
                idx = np.concatenate([by_run[r] for r in pick])
            else:
                idx = rng.choice(n_prop, size=n_prop, replace=True)
            C, V_state, A, B = build_from(idx, cvs, vbs, feats, fi, NCV, NF,
                                          e_cv, e_f, 5.0, 8.0)
            if not A or not B: continue
            r = compute_diagnostics(C, V_state, A, B, n_cv=NCV, n_f=NF)
            if r is None or r['Omega_J'] < 1e-6: continue
            OJ.append(r['Omega_J']); DE.append(r['D_edge'])
        OJ, DE = np.array(OJ), np.array(DE)
        f = lambda v: dict(median=float(np.median(v)), p5=float(np.percentile(v,5)),
                           p95=float(np.percentile(v,95)), n=int(len(v)))
        out[f"block_{unit}"] = dict(Omega_J=f(OJ), D_edge=f(DE))
        print(f"=== block bootstrap over {unit}s  (n={len(OJ)}/{N_BOOT}) ===")
        print(f"  Omega_J {np.median(OJ):.4f} [{np.percentile(OJ,5):.4f}, {np.percentile(OJ,95):.4f}]"
              f"   width {np.percentile(OJ,95)-np.percentile(OJ,5):.4f}")
        print(f"  D_edge  {np.median(DE):.4f} [{np.percentile(DE,5):.4f}, {np.percentile(DE,95):.4f}]"
              f"   width {np.percentile(DE,95)-np.percentile(DE,5):.4f}\n")

    # D_edge / Omega_J vs the A cutoff (D_edge is sensitive to it)
    print("=== sensitivity to the A cutoff (all data) ===")
    print(f"{'A_max':>6} {'|A|':>4} {'Omega_J':>9} {'D_edge':>8}")
    rows = []
    for A_max in (3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
        C, V_state, A, B = build_from(range(n_prop), cvs, vbs, feats, fi, NCV, NF,
                                      e_cv, e_f, A_max, 8.0)
        r = compute_diagnostics(C, V_state, A, B, n_cv=NCV, n_f=NF)
        print(f"{A_max:6.1f} {len(A):4d} {r['Omega_J']:9.4f} {r['D_edge']:8.4f}")
        rows.append(dict(A_max=A_max, nA=len(A), Omega_J=r['Omega_J'], D_edge=r['D_edge']))
    out['A_cutoff_sweep'] = rows

    json.dump(out, open(f'{P}/ras_pi_blockboot.json','w'), indent=2, default=float)
    print("\nsaved ras_pi_blockboot.json")


if __name__ == '__main__':
    main()
