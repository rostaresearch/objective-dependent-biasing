"""Mechanism-conservation audit on REAL Ras phosphate-unbinding data.

Loads Balint's master-CV orchestrator trajectories
($RAS_RAW/
 ras_orchestrator_mastercv_bell.npz)
which contain 24 trajectories spanning the GTP -> GDP transition along a
1-D master CV.  Each frame additionally carries 9 protein-distance
features that resolve the high-D conformational state space at each CV
value.

This script demonstrates the key point: even when the bias acts on a
SINGLE collective variable, the mechanism can vary because the system
is high-D.  We discretise on a 2-D microstate grid
    (master CV) x (one orthogonal feature)
and observe that microstates at the same CV value have different
connectivity (different paths).  The audit picks this up.

Outputs:
    ras_mechanism_2d_histograms.png  -- visited microstates + bias profile
    ras_mechanism_table.json         -- audit numbers for the manuscript
    fig_mechanism_ras.png            -- 3-panel audit figure
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RAS_RAW", "ras_lda_fm")).joinpath(""
            "ras_orchestrator_mastercv_bell.npz")
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("DHAM_HIGHD", "."))
# Use the regularised audit from the high-D module (handles sparse MSMs)
from mechanism_audit_highd_n20 import (
    stationary_from_K as stationary_from_generator,
    committor_K as committor,
    positive_net_current,
    tube_density, mechanism_audit, mfpt_K as mfpt_from_K,
    tilt_generator,
)

# choose a feature that is approximately orthogonal to the master CV.
# 0: R68-E37_salt    -- ranges 12-15 A, changes during transition (good!)
# 1: T35-Sw1_retract -- narrow
# 5: Y64_swing       -- 14-16 A, large dynamic range
# We use feature index 5 (Y64_swing) which captures the Switch-II side-chain
# motion and is orthogonal to the master CV.
FEAT_IDX = 5


def load_trajectories(npz=DATA):
    d = np.load(npz, allow_pickle=True)
    return d["data"], d["features_meta"]


def pool_into_2d_grid(trajs, feat_idx=FEAT_IDX,
                       n_cv=10, n_f=7, lag=1, allow_cross_traj=True):
    """Histogram pooled (cv, feature) trajectories and build the count
    matrix at the given lag. Coarser grid + (optionally) allowing
    transitions across consecutive iterations gives a connected MSM
    suitable for the audit demonstration."""
    cv_all   = np.concatenate([t["cv"]              for t in trajs])
    feat_all = np.concatenate([t["d_feat"][:, feat_idx] for t in trajs])
    print(f"  pooled {len(cv_all)} frames; cv in [{cv_all.min():.2f}, {cv_all.max():.2f}]; "
          f"feat in [{feat_all.min():.2f}, {feat_all.max():.2f}]")
    e_cv = np.linspace(cv_all.min() - 1e-6, cv_all.max() + 1e-6, n_cv + 1)
    e_f  = np.linspace(feat_all.min() - 1e-6, feat_all.max() + 1e-6, n_f + 1)
    cv_centers = 0.5 * (e_cv[:-1] + e_cv[1:])
    f_centers  = 0.5 * (e_f[:-1]  + e_f[1:])
    state_cv   = np.repeat(cv_centers, n_f)
    state_f    = np.tile(f_centers, n_cv)
    N = n_cv * n_f

    C = np.zeros((N, N))
    if allow_cross_traj:
        # treat the orchestrator's concatenated stream as one trajectory;
        # this gives connectivity at the iteration boundaries (a single
        # cross-iter transition per boundary, negligible bias).
        ic   = np.clip(np.digitize(cv_all,   e_cv) - 1, 0, n_cv - 1)
        ifv  = np.clip(np.digitize(feat_all, e_f)  - 1, 0, n_f  - 1)
        s    = ic * n_f + ifv
        for k in range(len(s) - lag):
            C[s[k], s[k + lag]] += 1.0
    else:
        for t in trajs:
            ic  = np.clip(np.digitize(t["cv"],              e_cv) - 1, 0, n_cv - 1)
            ifv = np.clip(np.digitize(t["d_feat"][:, feat_idx], e_f) - 1, 0, n_f - 1)
            s   = ic * n_f + ifv
            for k in range(len(s) - lag):
                C[s[k], s[k + lag]] += 1.0
    C = 0.5 * (C + C.T)
    return C, e_cv, e_f, state_cv, state_f, N, n_cv, n_f, cv_all, feat_all


def K_from_counts(C, lag):
    row = C.sum(axis=1, keepdims=True)
    M = np.zeros_like(C)
    ok = row[:, 0] > 0
    M[ok] = C[ok] / row[ok]
    return (M - np.eye(C.shape[0])) / lag


def bias_on_grid(slope_per_cv, state_cv):
    """Linear pull bias V(cv) = -slope * cv  (kT units)."""
    return -slope_per_cv * state_cv


def main():
    print("=" * 72)
    print("Mechanism audit on REAL Ras phosphate-unbinding data")
    print("=" * 72)
    trajs, features_meta = load_trajectories()
    feat_name = features_meta[FEAT_IDX, 0]
    print(f"using orthogonal feature [{FEAT_IDX}]: {feat_name}")

    # 1) build 2-D MSM from pooled frames
    lag = 1
    C, e_cv, e_f, state_cv, state_f, N, n_cv, n_f, cv_all, feat_all = \
        pool_into_2d_grid(trajs, feat_idx=FEAT_IDX, n_cv=12, n_f=8, lag=lag,
                          allow_cross_traj=True)
    print(f"  grid {n_cv} x {n_f} = {N} microstates; "
          f"visited: {int((C.sum(axis=1) > 0).sum())}/{N}")

    K0 = K_from_counts(C, lag)

    # 2) define source/target: GTP-like states (cv near 0) and GDP-like (cv near 9)
    A_set = [s for s in range(N) if state_cv[s] < 1.0 and C[s].sum() > 0]
    B_set = [s for s in range(N) if state_cv[s] > 8.0 and C[s].sum() > 0]
    print(f"  A (GTP-like, cv<1) has {len(A_set)} visited microstates")
    print(f"  B (GDP-like, cv>8) has {len(B_set)} visited microstates")

    # 3) sweep a linear-pull bias and run the audit
    slopes = np.linspace(0.0, 3.0, 13)         # bias slope in kT per CV unit
    rows = {key: [] for key in
            ["slope", "speedup", "Omega_J", "Omega_rho", "D_edge", "R_unsup"]}

    pi0 = stationary_from_generator(K0)
    tau0 = mfpt_from_K(K0, A_set, B_set, pi=pi0)
    print(f"  unbiased MFPT(A->B) on this Ras MSM = {tau0:.3e} frames")

    for a in slopes:
        b = bias_on_grid(a, state_cv)
        m = mechanism_audit(K0, b, A_set, B_set, C=C, C0=10.0)
        rows["slope"].append(a)
        rows["speedup"].append(tau0 / m["tau_AB"] if m["tau_AB"] > 0 else np.nan)
        rows["Omega_J"].append(m["Omega_J"])
        rows["Omega_rho"].append(m["Omega_rho"])
        rows["D_edge"].append(m["D_edge"])
        rows["R_unsup"].append(m["R_unsup"])
    rows = {k: np.array(v) for k, v in rows.items()}

    # 4) report at slope = 1, 2 kT/cv-unit
    print()
    print(f"  {'slope':<6} {'speedup':>10} {'Omega_J':>9} {'Omega_rho':>10} "
          f"{'D_edge':>9} {'R_unsup':>9}")
    for i, a in enumerate(slopes):
        if a in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            print(f"  {a:<6.2f} {rows['speedup'][i]:>10.3e} "
                  f"{rows['Omega_J'][i]:>9.3f} {rows['Omega_rho'][i]:>10.3f} "
                  f"{rows['D_edge'][i]:>9.3f} {rows['R_unsup'][i]:>9.3f}")

    # 5) figures
    # (a) Visited microstates + sample bias profile
    fig = plt.figure(figsize=(11, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    H = C.sum(axis=1).reshape(n_cv, n_f)
    im = ax1.pcolormesh(e_cv, e_f, np.log10(H.T + 1.0), cmap="magma")
    ax1.set_xlabel("master CV")
    ax1.set_ylabel(f"feature: {feat_name}")
    ax1.set_title("(a) log$_{10}$(visits per microstate)")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    # mark A and B regions
    for s in A_set:
        ic, iif = s // n_f, s % n_f
        ax1.add_patch(plt.Rectangle((e_cv[ic], e_f[iif]),
                                     e_cv[ic+1]-e_cv[ic], e_f[iif+1]-e_f[iif],
                                     fill=False, edgecolor="red", lw=1.2))
    for s in B_set:
        ic, iif = s // n_f, s % n_f
        ax1.add_patch(plt.Rectangle((e_cv[ic], e_f[iif]),
                                     e_cv[ic+1]-e_cv[ic], e_f[iif+1]-e_f[iif],
                                     fill=False, edgecolor="cyan", lw=1.2))

    # (b) audit metrics vs bias slope
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(slopes, rows["Omega_J"  ], "-o", color="#7a3d99", label=r"$\Omega_J$",   lw=1.6)
    ax2.plot(slopes, rows["Omega_rho"], "-s", color="#2a4d7a", label=r"$\Omega_\rho$", lw=1.6)
    ax2.plot(slopes, rows["R_unsup"  ], "-^", color="#1f6b47", label=r"$R_{\rm unsup}$", lw=1.6)
    ax2.set_xlabel("bias slope (kT per CV unit)")
    ax2.set_ylabel("overlap / fraction")
    ax2.set_title("(b) mechanism overlap vs bias")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # (c) speedup vs Omega_J  (Pareto)
    ax3 = fig.add_subplot(1, 3, 3)
    valid = (rows["speedup"] > 0) & np.isfinite(rows["speedup"])
    ax3.plot(rows["Omega_J"][valid], rows["speedup"][valid], "-o",
             color="#a05a2c", lw=1.6, ms=5)
    for i in np.where(valid)[0]:
        if slopes[i] in (0.0, 1.0, 2.0, 3.0):
            ax3.annotate(f"slope={slopes[i]:.1f}",
                         (rows["Omega_J"][i], rows["speedup"][i]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax3.set_yscale("log")
    ax3.set_xlabel(r"current overlap $\Omega_J$")
    ax3.set_ylabel(r"MFPT speedup ($\tau_0 / \tau_b$)")
    ax3.set_title(r"(c) Pareto: $\Omega_J$ vs MFPT speedup")
    ax3.invert_xaxis(); ax3.grid(alpha=0.3)

    fig.suptitle(f"[PARETO DEMO -- NOT the main audit] Mechanism-conservation "
                 f"audit on Ras phosphate-unbinding data\n"
                 f"2-D MSM in (master CV, {feat_name}); A = GTP-like (CV<1), "
                 f"B = GDP-like (CV>8)\n"
                 f"Bias here is a synthetic slope sweep (hypothetical), NOT "
                 f"the orchestrator's actually-applied bias.\n"
                 f"See fig_mechanism_ras_proper.pdf for the actually-applied "
                 f"bias + bootstrap audit.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(HERE / "fig_mechanism_ras.pdf")
    fig.savefig(HERE / "fig_mechanism_ras.png", dpi=180)
    print(f"\nwrote {HERE / 'fig_mechanism_ras.pdf'}")
    print(f"wrote {HERE / 'fig_mechanism_ras.png'}")

    out = {
        "feature_index":      int(FEAT_IDX),
        "feature_name":       str(feat_name),
        "n_microstates":      int(N),
        "n_visited":          int((C.sum(axis=1) > 0).sum()),
        "n_A_visited":        len(A_set),
        "n_B_visited":        len(B_set),
        "lag_frames":         lag,
        "tau0_unbiased":      float(tau0),
        "slope_kT_per_cv":    [float(s) for s in slopes],
        "speedup":            [float(x) for x in rows["speedup"  ]],
        "Omega_J":            [float(x) for x in rows["Omega_J"  ]],
        "Omega_rho":          [float(x) for x in rows["Omega_rho"]],
        "D_edge":             [float(x) for x in rows["D_edge"   ]],
        "R_unsup":            [float(x) for x in rows["R_unsup"  ]],
    }
    with open(HERE / "ras_mechanism_table.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {HERE / 'ras_mechanism_table.json'}")


if __name__ == "__main__":
    main()
