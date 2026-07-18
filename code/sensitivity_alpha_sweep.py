"""Pseudocount sensitivity sweep for the Ras audit.

Sweeps alpha in {1e-4, 1e-3, 1e-2} (three orders of magnitude) and
reports Omega_J and D_edge as (point, median, 90% CI) per alpha.

If the diagnostics are stable across this range, the pseudocount is a
genuine numerical regulariser; if they drift, the result is driven by
the regulariser, not the data.

Outputs:
    alpha_sweep.json                       full table
    fig_alpha_sweep_matlab.{pdf,png}       MATLAB sensitivity figure
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
    NPZ, KBT,
)


def main():
    ALPHAS = [1e-4, 1e-3, 1e-2]
    CV_A_MAX, CV_B_MIN = 5.0, 8.0

    print("=" * 72)
    print("Pseudocount sensitivity sweep (alpha in {1e-4, 1e-3, 1e-2})")
    print("=" * 72)

    d     = np.load(NPZ, allow_pickle=True)
    trajs = d["data"]
    build = build_counts_2d(trajs, n_cv=12, n_f=8, lag=1,
                              allow_cross_traj=False)
    state_cv = build["state_cv"]
    A_set = [s for s in range(build["N"])
             if state_cv[s] < CV_A_MAX and build["C_all"][s].sum() > 0]
    B_set = [s for s in range(build["N"])
             if state_cv[s] > CV_B_MIN and build["C_all"][s].sum() > 0]
    print(f"  |A| = {len(A_set)}, |B| = {len(B_set)}, "
          f"visited {(build['C_all'].sum(1) > 0).sum()}/{build['N']}")

    table = {"_alphas": ALPHAS, "_A_def": f"cv < {CV_A_MAX}",
             "_B_def": f"cv > {CV_B_MIN}",
             "_allow_cross_traj": False,
             "_dham_sign": "PLUS", "rows": []}

    n_boot = 200
    print(f"\n  bootstrap n_boot = {n_boot} per alpha\n")
    hdr = ("  alpha     Omega_J point  median  [p5,    p95 ]   "
           "D_edge point  median  [p5,    p95 ]   n_reg")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for alpha in ALPHAS:
        point = compute_diagnostics(build["C_all"], build["V_state"],
                                     A_set, B_set,
                                     n_cv=build["n_cv"], n_f=build["n_f"],
                                     regularise=True, alpha_pseudo=alpha)
        boot, n_reg = bootstrap_audit(build["C_all"],
                                        build["V_state"], A_set, B_set,
                                        n_cv=build["n_cv"],
                                        n_f=build["n_f"],
                                        n_boot=n_boot, seed=0,
                                        alpha_pseudo=alpha)
        n_valid = len(boot["Omega_J"])
        row = {"alpha": float(alpha), "n_valid": int(n_valid),
               "n_reg": int(n_reg)}
        for key in ("Omega_J", "D_edge"):
            arr = boot[key]
            p5, p50, p95 = (float(np.percentile(arr, p)) for p in (5, 50, 95))
            row[key] = {"point":  float(point[key]),
                        "median": p50, "p5": p5, "p95": p95}
        table["rows"].append(row)
        print(f"  {alpha:.0e}   {row['Omega_J']['point']:.3f}        "
              f"{row['Omega_J']['median']:.3f}   "
              f"[{row['Omega_J']['p5']:.3f}, {row['Omega_J']['p95']:.3f}]   "
              f"{row['D_edge']['point']:.3f}        "
              f"{row['D_edge']['median']:.3f}   "
              f"[{row['D_edge']['p5']:.3f}, {row['D_edge']['p95']:.3f}]   "
              f"{row['n_reg']}/{row['n_valid']}")

    with open(HERE / "alpha_sweep.json", "w") as fp:
        json.dump(table, fp, indent=2)
    print(f"\n  wrote {HERE / 'alpha_sweep.json'}")

    # Compact stability verdict
    Oj_meds = [r["Omega_J"]["median"] for r in table["rows"]]
    De_meds = [r["D_edge"]["median"]  for r in table["rows"]]
    Oj_rel  = (max(Oj_meds) - min(Oj_meds)) / np.mean(Oj_meds)
    De_rel  = (max(De_meds) - min(De_meds)) / np.mean(De_meds)
    print()
    print("  Stability across 3 orders of magnitude in alpha:")
    print(f"    Omega_J  : median drift = {Oj_rel * 100:.2f}%  "
          f"(range {min(Oj_meds):.3f} - {max(Oj_meds):.3f})")
    print(f"    D_edge   : median drift = {De_rel * 100:.2f}%  "
          f"(range {min(De_meds):.3f} - {max(De_meds):.3f})")


if __name__ == "__main__":
    main()
