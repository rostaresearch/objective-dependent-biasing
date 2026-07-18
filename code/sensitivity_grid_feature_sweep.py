"""Grid + feature sensitivity sweep for the Ras audit.

Grids:    10x7, 12x8, 14x9 (the canonical is 12x8).
Features: Y64_swing (canonical), Sw1-Sw2_couple, alpha5_align
          (chosen for orthogonality: intra-loop side chain, inter-Switch
           coupling, allosteric helix alignment).

For each (grid, feature) cell, report:
   |A|, |B|, n_visited / n_total,
   Omega_J (point, median, 5%, 95%),
   D_edge  (point, median, 5%, 95%),
   n_valid / n_attempted bootstraps.

Stable diagnostics across BOTH axes => the result is not driven by a
particular grid resolution or a cherry-picked feature.

Outputs:
   grid_feature_sweep.json
   fig_grid_feature_sweep_matlab.{pdf,png}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mechanism_audit_ras_proper import (
    build_counts_2d, bootstrap_audit, compute_diagnostics,
    NPZ,
)


# (n_cv, n_f) candidates
GRIDS = [(10, 7), (12, 8), (14, 9)]

# (feat_idx, short_name) candidates.  features_meta order from the npz:
#  0 R68-E37_salt, 1 T35-Sw1_retract, 2 Sw1-Sw2_couple,
#  3 Sw2_helix_unwind, 4 E37_chi1_flip, 5 Y64_swing (canonical),
#  6 Y32_Y40_aromat, 7 alpha5_align, 8 Y71_chi_flip
FEATURES = [
    (5, "Y64_swing"),        # canonical: Switch II side-chain swing
    (2, "Sw1-Sw2_couple"),   # inter-Switch coupling (different region)
    (7, "alpha5_align"),     # allosteric alpha5 helix alignment
]

CV_A_MAX = 5.0
CV_B_MIN = 8.0
N_BOOT   = 200
ALPHA    = 1e-3


def run_one(trajs, n_cv, n_f, feat_idx, label):
    build = build_counts_2d(trajs, n_cv=n_cv, n_f=n_f, lag=1,
                              allow_cross_traj=False, feat_idx=feat_idx)
    state_cv = build["state_cv"]
    A_set = [s for s in range(build["N"])
             if state_cv[s] < CV_A_MAX and build["C_all"][s].sum() > 0]
    B_set = [s for s in range(build["N"])
             if state_cv[s] > CV_B_MIN and build["C_all"][s].sum() > 0]
    n_vis = int((build["C_all"].sum(axis=1) > 0).sum())
    if len(A_set) == 0 or len(B_set) == 0:
        return dict(label=label, n_cv=n_cv, n_f=n_f, feat_idx=feat_idx,
                    n_visited=n_vis, n_total=build["N"],
                    A_size=len(A_set), B_size=len(B_set),
                    failed="empty A or B")
    point = compute_diagnostics(build["C_all"], build["V_state"],
                                 A_set, B_set,
                                 n_cv=build["n_cv"], n_f=build["n_f"],
                                 regularise=True, alpha_pseudo=ALPHA)
    boot, n_reg = bootstrap_audit(build["C_all"],
                                    build["V_state"], A_set, B_set,
                                    n_cv=build["n_cv"], n_f=build["n_f"],
                                    n_boot=N_BOOT, seed=0,
                                    alpha_pseudo=ALPHA)
    n_valid = len(boot["Omega_J"])
    if n_valid == 0:
        return dict(label=label, n_cv=n_cv, n_f=n_f, feat_idx=feat_idx,
                    n_visited=n_vis, n_total=build["N"],
                    A_size=len(A_set), B_size=len(B_set),
                    failed="no valid bootstrap")
    out = dict(label=label, n_cv=n_cv, n_f=n_f, feat_idx=feat_idx,
               n_visited=n_vis, n_total=build["N"],
               A_size=len(A_set), B_size=len(B_set),
               n_valid=int(n_valid), n_reg=int(n_reg))
    for k in ("Omega_J", "D_edge"):
        arr = boot[k]
        p5, p50, p95 = (float(np.percentile(arr, p)) for p in (5, 50, 95))
        out[k] = dict(point=float(point[k]), median=p50, p5=p5, p95=p95)
    return out


def main():
    print("=" * 80)
    print("Grid + feature sensitivity sweep")
    print("=" * 80)
    d     = np.load(NPZ, allow_pickle=True)
    trajs = d["data"]

    rows = []
    for (n_cv, n_f) in GRIDS:
        for (fi, fname) in FEATURES:
            label = f"{n_cv}x{n_f} / {fname}"
            print(f"\n  running  {label} ...")
            r = run_one(trajs, n_cv, n_f, fi, label)
            rows.append(r)
            if "failed" in r:
                print(f"    FAILED: {r['failed']}  "
                      f"(visited {r['n_visited']}/{r['n_total']}, "
                      f"|A|={r['A_size']}, |B|={r['B_size']})")
            else:
                print(f"    visited {r['n_visited']}/{r['n_total']}, "
                      f"|A|={r['A_size']}, |B|={r['B_size']}, "
                      f"valid {r['n_valid']}/{N_BOOT}")
                print(f"    Omega_J median = {r['Omega_J']['median']:.3f} "
                      f"[{r['Omega_J']['p5']:.3f}, "
                      f"{r['Omega_J']['p95']:.3f}]")
                print(f"    D_edge  median = {r['D_edge']['median']:.3f} "
                      f"[{r['D_edge']['p5']:.3f}, "
                      f"{r['D_edge']['p95']:.3f}]")

    # Full table
    out = dict(_grids=[list(g) for g in GRIDS],
               _features=[list(f) for f in FEATURES],
               _alpha=ALPHA, _CV_A_max=CV_A_MAX, _CV_B_min=CV_B_MIN,
               _n_boot=N_BOOT, rows=rows)
    with open(HERE / "grid_feature_sweep.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n  wrote {HERE / 'grid_feature_sweep.json'}")

    # Stability summary
    print()
    print("  -- Stability summary -----------------------------------------")
    ok = [r for r in rows if "failed" not in r]
    if ok:
        Oj = [r["Omega_J"]["median"] for r in ok]
        De = [r["D_edge"]["median"]  for r in ok]
        print(f"    Omega_J  range across {len(ok)} cells: "
              f"{min(Oj):.3f} - {max(Oj):.3f}  "
              f"(drift = {100*(max(Oj)-min(Oj))/np.mean(Oj):.1f}%)")
        print(f"    D_edge   range across {len(ok)} cells: "
              f"{min(De):.3f} - {max(De):.3f}  "
              f"(drift = {100*(max(De)-min(De))/np.mean(De):.1f}%)")


if __name__ == "__main__":
    main()
