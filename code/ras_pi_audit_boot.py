"""Bootstrap CIs for the Ras Pi-unbinding mechanism audit (the 1D_PMg runs).

Uses bootstrap_audit/compute_diagnostics from mechanism_audit_ras_proper.py
unchanged (same DHAM sign, pseudocount rule, 200 multinomial resamples), so the
methodology matches the shipped audit exactly; only the data differ.

Canonical projection: 12x8 grid, A={CV<5.0 A}, B={CV>8.0 A}, alpha=1e-3.
Also sweeps 3 grids x 3 orthogonal features for robustness, and alpha over
three orders of magnitude.
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
from mechanism_audit_ras_proper import bootstrap_audit, compute_diagnostics
from ras_pi_audit import build, NPZ

CANON_FEAT = 'Pi_Q61'          # most orthogonal to the CV (r = -0.09)
SWEEP_FEATS = ['Pi_Q61', 'Pi_wat_4.5', 'Mg_O_coord']
GRIDS = [(10, 7), (12, 8), (14, 9)]


def summarise(vals):
    return dict(median=float(np.median(vals)), p5=float(np.percentile(vals, 5)),
                p95=float(np.percentile(vals, 95)), n=int(len(vals)))


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    cvs, vbs, feats = d['cv'], d['vbias'], d['feat']
    out = {}

    # ---------------- canonical, with bootstrap ----------------
    fi = names.index(CANON_FEAT)
    C, V_state, A, B, cnt = build(cvs, vbs, feats, fi, 12, 8, B_min=8.0)
    pt = compute_diagnostics(C, V_state, A, B, n_cv=12, n_f=8)
    boot, n_reg = bootstrap_audit(C, V_state, A, B, n_cv=12, n_f=8,
                                  n_boot=200, seed=0, alpha_pseudo=1e-3)
    oj, de = summarise(boot['Omega_J']), summarise(boot['D_edge'])
    print(f"=== CANONICAL: 12x8 / {CANON_FEAT}, A={len(A)} B={len(B)} ===")
    print(f"  point   Omega_J={pt['Omega_J']:.4f}  D_edge={pt['D_edge']:.4f}  reg={pt['regularised']}")
    print(f"  Omega_J median={oj['median']:.4f} [{oj['p5']:.4f}, {oj['p95']:.4f}]  n={oj['n']}")
    print(f"  D_edge  median={de['median']:.4f} [{de['p5']:.4f}, {de['p95']:.4f}]  n={de['n']}")
    print(f"  bootstrap resamples needing pseudocount: {n_reg}/200\n")
    out['canonical'] = dict(feature=CANON_FEAT, grid=[12, 8], A=len(A), B=len(B),
                            point=pt, Omega_J=oj, D_edge=de,
                            n_regularised=int(n_reg), n_boot=200)

    # ---------------- grid x feature sweep ----------------
    print("=== GRID x FEATURE SWEEP (medians over 200 bootstrap) ===")
    print(f"{'label':26s} {'A':>3s} {'B':>3s} {'Omega_J median [5,95]':>28s} {'D_edge median [5,95]':>26s}")
    rows = []
    for (ncv, nf) in GRIDS:
        for fname in SWEEP_FEATS:
            j = names.index(fname)
            C, V_state, A, B, cnt = build(cvs, vbs, feats, j, ncv, nf, B_min=8.0)
            if not A or not B:
                print(f"{ncv}x{nf} / {fname:14s}  -- empty A/B --"); continue
            bt, nr = bootstrap_audit(C, V_state, A, B, n_cv=ncv, n_f=nf,
                                     n_boot=200, seed=0, alpha_pseudo=1e-3)
            o, e = summarise(bt['Omega_J']), summarise(bt['D_edge'])
            lab = f"{ncv}x{nf} / {fname}"
            print(f"{lab:26s} {len(A):3d} {len(B):3d}  "
                  f"{o['median']:.3f} [{o['p5']:.3f}, {o['p95']:.3f}]        "
                  f"{e['median']:.3f} [{e['p5']:.3f}, {e['p95']:.3f}]")
            rows.append(dict(label=lab, n_cv=ncv, n_f=nf, feature=fname,
                             A=len(A), B=len(B), Omega_J=o, D_edge=e,
                             n_regularised=int(nr)))
    out['grid_feature_sweep'] = rows
    ojs = [r['Omega_J']['median'] for r in rows]
    des = [r['D_edge']['median'] for r in rows]
    print(f"\n  Omega_J median range: {min(ojs):.3f} - {max(ojs):.3f}")
    print(f"  D_edge  median range: {min(des):.3f} - {max(des):.3f}\n")

    # ---------------- alpha sweep ----------------
    print("=== ALPHA SWEEP (canonical projection) ===")
    fi = names.index(CANON_FEAT)
    C, V_state, A, B, cnt = build(cvs, vbs, feats, fi, 12, 8, B_min=8.0)
    arows = []
    for a in (1e-4, 1e-3, 1e-2):
        bt, nr = bootstrap_audit(C, V_state, A, B, n_cv=12, n_f=8,
                                 n_boot=200, seed=0, alpha_pseudo=a)
        o, e = summarise(bt['Omega_J']), summarise(bt['D_edge'])
        print(f"  alpha={a:<7g} Omega_J={o['median']:.4f}  D_edge={e['median']:.4f}  reg={nr}/200")
        arows.append(dict(alpha=a, Omega_J=o, D_edge=e, n_regularised=int(nr)))
    out['alpha_sweep'] = arows
    do = (max(r['Omega_J']['median'] for r in arows) - min(r['Omega_J']['median'] for r in arows))
    dd = (max(r['D_edge']['median'] for r in arows) - min(r['D_edge']['median'] for r in arows))
    m_o = np.median([r['Omega_J']['median'] for r in arows])
    m_d = np.median([r['D_edge']['median'] for r in arows])
    print(f"  drift: Omega_J {100*do/m_o:.3f}%   D_edge {100*dd/m_d:.3f}%")

    with open(os.path.join(DATA, 'ras_pi_audit.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved ras_pi_audit.json")


if __name__ == '__main__':
    main()
