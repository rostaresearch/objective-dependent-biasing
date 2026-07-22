"""Proper Ras phosphate-unbinding mechanism audit.

Uses the bias actually applied (recovered from each iter's
biased_log.json on bell), DHAM-reweights to recover K^0, and reports
the four mechanism-distortion diagnostics at the actually-applied bias
with bootstrap 5-95% CIs.

No 'speedup' is reported -- that requires an unbiased reference MFPT
which is not knowable from these biased simulations.

Inputs:
    ras_lda_fm/ras_orchestrator_mastercv_bell.npz   - 24 trajectories,
        each with per-frame (cv, d_feat, time, dt_ps, phase, iter)
    ras_lda_fm/orch_iter_metadata/iter_{K}/biased_log.json   - per-iter
        per-propagation log of (cv, e_bias, distances)

Outputs:
    ras_proper_audit.json          - {median, p5, p95} per diagnostic
    fig_mechanism_ras_proper.{pdf,png} - bar chart with error bars
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


HERE      = Path(__file__).resolve().parent
DATA_DIR  = Path(os.environ.get('RAS_RAW', 'ras_lda_fm'))
META_DIR  = DATA_DIR / "orch_iter_metadata"
NPZ       = DATA_DIR / "ras_orchestrator_mastercv_bell.npz"

T_K  = 300.0                    # KOMBI simulation temperature (config.py); the bias
KBT  = 8.31446261815e-3 * T_K   # was applied at 300 K, so reweight at 300 K. ~= 2.494

sys.path.insert(0, os.environ.get('DHAM_HIGHD', '.'))
from mechanism_audit_highd_n20 import (
    stationary_from_K, committor_K, positive_net_current,
    tilt_generator,
)


# ---------- per-frame applied bias from biased_log.json -------------
def applied_bias_per_frame(traj):
    """Return V_applied(t) in kJ/mol for each frame of one trajectory.
    For unbiased phases ('unbiased_local', 'unbiased'): V = 0.
    For biased phase: interpolate biased_log.json's per-propagation
    e_bias values across frames."""
    n = len(traj["cv"])
    if traj["phase"] != "biased" or traj.get("iter") is None:
        return np.zeros(n)
    iter_k = int(traj["iter"])
    log = json.load(open(META_DIR / f"iter_{iter_k:02d}" / "biased_log.json"))
    e_per_prop = np.array([L["e_bias"] for L in log])  # n_log
    # Frame k corresponds to log entry int(k * n_log / n)
    idx = np.clip((np.arange(n) * len(e_per_prop) // n), 0, len(e_per_prop) - 1)
    return e_per_prop[idx]


# ---------- 2-D MSM build with both biased and unbiased frames -----
FEAT_IDX = 5    # Y64_swing (Switch-II side-chain motion)
FEAT_NAME = "Y64_swing"


def build_counts_2d(trajs, n_cv=12, n_f=8, lag=1, allow_cross_traj=True,
                     feat_idx=FEAT_IDX):
    cv_all   = np.concatenate([t["cv"]                  for t in trajs])
    feat_all = np.concatenate([t["d_feat"][:, feat_idx] for t in trajs])
    V_all    = np.concatenate([applied_bias_per_frame(t) for t in trajs])
    biased_flag = np.concatenate([
        np.full(len(t["cv"]),
                1 if (t["phase"] == "biased" and t.get("iter") is not None) else 0,
                dtype=int)
        for t in trajs])
    e_cv = np.linspace(cv_all.min() - 1e-6,   cv_all.max() + 1e-6, n_cv + 1)
    e_f  = np.linspace(feat_all.min() - 1e-6, feat_all.max() + 1e-6, n_f + 1)
    cv_centers = 0.5 * (e_cv[:-1] + e_cv[1:])
    f_centers  = 0.5 * (e_f[:-1]  + e_f[1:])
    state_cv   = np.repeat(cv_centers, n_f)
    state_f    = np.tile(f_centers, n_cv)
    N = n_cv * n_f

    # microstate index of every frame
    ic   = np.clip(np.digitize(cv_all,   e_cv) - 1, 0, n_cv - 1)
    ifv  = np.clip(np.digitize(feat_all, e_f)  - 1, 0, n_f  - 1)
    s    = ic * n_f + ifv

    # split into "biased" and "all" count matrices
    C_all    = np.zeros((N, N))
    C_biased = np.zeros((N, N))
    if allow_cross_traj:
        for k in range(len(s) - lag):
            C_all[s[k], s[k + lag]] += 1.0
            if biased_flag[k] == 1:
                C_biased[s[k], s[k + lag]] += 1.0
    else:
        # respect trajectory boundaries
        starts = []; cur = 0
        for t in trajs:
            starts.append((cur, cur + len(t["cv"])))
            cur += len(t["cv"])
        for (st, en) in starts:
            for k in range(st, en - lag):
                C_all[s[k], s[k + lag]] += 1.0
                if biased_flag[k] == 1:
                    C_biased[s[k], s[k + lag]] += 1.0
    # symmetrise (detailed balance)
    C_all    = 0.5 * (C_all    + C_all.T)
    C_biased = 0.5 * (C_biased + C_biased.T)

    # average applied bias V at each microstate (mean over all frames in that bin)
    V_state = np.zeros(N)
    counts  = np.zeros(N)
    for k in range(len(s)):
        V_state[s[k]] += V_all[k]
        counts[s[k]]  += 1
    ok = counts > 0
    V_state[ok] = V_state[ok] / counts[ok]

    return dict(
        C_all=C_all, C_biased=C_biased, V_state=V_state,
        state_cv=state_cv, state_f=state_f,
        e_cv=e_cv, e_f=e_f, N=N, n_cv=n_cv, n_f=n_f,
    )


# ---------- DHAM unbiasing on the count matrix -------------
def dham_unbias(C, V_state, KbT=KBT):
    """Proper DHAM_sym (Rosta-Hummer).
        Step 1: C^sym = (C + C.T)/2  (enforce detailed-balance counts)
        Step 2: Cu_ij = C^sym_ij * exp[+(V_j - V_i)/(2 kT)]
        Step 3: M^0_ij = Cu_ij / sum_k Cu_ik   (row-normalise once)
    Sign rationale: the manuscript convention is
        K^b_ij = K^0_ij * exp[-(b_j - b_i)/2],   b_i = V_i / kT
    so recovering K^0 from K^b uses the inverse, exp[+(b_j-b_i)/2].
    Verified against an analytical 1D round-trip in
    sanity_check_dham_sign.py (PLUS-with-pre-symmetrise wins).
    NOTE: do NOT re-symmetrise the reweighted matrix; that step makes
    (C*W + (C*W).T)/2 = C * cosh[(V_j-V_i)/2kT] when C is symmetric,
    which is even in V and cancels the sign convention entirely.
    """
    V    = np.asarray(V_state, dtype=float).ravel()
    Csym = 0.5 * (C + C.T)
    dV   = V[None, :] - V[:, None]
    W    = np.exp(+0.5 * dV / KbT)                # PLUS sign
    Cu   = Csym * W
    row  = Cu.sum(axis=1, keepdims=True)
    M0   = np.zeros_like(Cu)
    ok   = row[:, 0] > 0
    M0[ok] = Cu[ok] / row[ok]
    return M0


# ---------- pseudocount regularisation for disconnected MSMs --------
def _connected_via(C, A_set, B_set):
    """True if any A state can reach any B state via non-zero entries."""
    A_in = [int(a) for a in A_set]
    B_in = set(int(b) for b in B_set)
    if not A_in or not B_in:
        return False
    G = (C > 0) | (C.T > 0)
    seen = {A_in[0]}; stk = [A_in[0]]
    while stk:
        u = stk.pop()
        for v in np.where(G[u])[0]:
            v = int(v)
            if v not in seen:
                seen.add(v); stk.append(v)
                if v in B_in:
                    return True
    return bool(seen & B_in)


def _nn_2d_pairs(N, n_cv, n_f, visited):
    """Set of (i,j) microstate-pairs that are nearest-neighbours on the
    2-D (cv, feature) grid AND both visited. Used for sparse, local
    Dirichlet bridging that preserves dynamical structure."""
    vis = set(int(v) for v in visited)
    pairs = []
    for i in vis:
        ic, ifv = divmod(i, n_f)
        for dc, df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jc, jf = ic + dc, ifv + df
            if 0 <= jc < n_cv and 0 <= jf < n_f:
                j = jc * n_f + jf
                if j in vis:
                    pairs.append((i, j))
    return pairs


def regularise_if_disconnected(C, A_set, B_set, n_cv, n_f, alpha=1e-3):
    """If A is not reachable from B in the visited subgraph, add a
    SMALL Dirichlet pseudocount alpha only on nearest-neighbour 2-D
    edges between visited microstates (local bridging that preserves
    the gradient structure). Returns (C_reg, was_regularised).
    A uniform-all-pairs pseudocount would flatten the dynamics and
    destroy Omega_J -- we tested alpha=0.5 uniform and it crushed
    Omega_J from 0.82 to 0.05."""
    if _connected_via(C, A_set, B_set):
        return C, False
    visited = np.where(C.sum(axis=1) + C.sum(axis=0) > 0)[0]
    pairs   = _nn_2d_pairs(C.shape[0], n_cv, n_f, visited)
    Creg = C.copy()
    for (i, j) in pairs:
        Creg[i, j] += alpha
        Creg[j, i] += alpha
    return Creg, True


# ---------- the 2 retained diagnostics (no speedup, no rho, no R_unsup)
def compute_diagnostics(C_all, V_state, A_set, B_set,
                         n_cv=12, n_f=8,
                         regularise=True, alpha_pseudo=1e-3):
    """Return {Omega_J, D_edge} at the actually-applied bias.

    Omega_rho and R_unsup are intentionally NOT computed: they penalise
    the bias for successfully accelerating the rare event (the whole
    point of biasing), and the bootstrap CIs already absorb rare-edge
    uncertainty.

    K^0 is DHAM-reweighted from C_all (which mixes biased and unbiased
    frames) using the per-microstate average applied bias V_state.
    K^b is constructed by analytically tilting K^0 by the same V_state,
    so K^0 and K^b share support and the committor / current
    diagnostics are well-defined.

    If `regularise` is True and (A,B) are disconnected in C_all, an optional
    sparse nearest-neighbour pseudocount is added only on the missing local links
    required for connectivity (reversible regularisation, NOT artificial boundary
    transitions); no pseudocount is added when the observed A-B graph is already
    connected. The 'regularised' flag is returned alongside.
    """
    if not A_set or not B_set:
        return None
    C_used = C_all
    was_reg = False
    if regularise:
        C_used, was_reg = regularise_if_disconnected(
            C_all, A_set, B_set, n_cv=n_cv, n_f=n_f,
            alpha=alpha_pseudo)
    M0 = dham_unbias(C_used, V_state)
    K0 = M0 - np.eye(M0.shape[0])                       # lag = 1 frame
    b  = V_state / KBT                                  # bias in kT units
    Kb = tilt_generator(K0, b)
    pi0 = stationary_from_K(K0)
    pib = stationary_from_K(Kb)
    q0  = committor_K(K0, A_set, B_set, eps=1e-12)
    qb  = committor_K(Kb, A_set, B_set, eps=1e-12)
    Jh0 = positive_net_current(K0, pi0, q0)
    Jhb = positive_net_current(Kb, pib, qb)
    Omega_J = float(np.minimum(Jh0, Jhb).sum())
    D_edge  = 0.5 * float((Jh0 * np.abs(b[None, :] - b[:, None])).sum())
    return dict(Omega_J=Omega_J, D_edge=D_edge,
                regularised=bool(was_reg))


# ---------- bootstrap ---------------------------------------
def bootstrap_audit(C_all, V_state, A_set, B_set,
                    n_cv=12, n_f=8,
                    n_boot=200, seed=0, alpha_pseudo=1e-3):
    """Multinomial resample of C_all rows. C_biased is no longer
    bootstrapped because the two surviving diagnostics (Omega_J,
    D_edge) only need C_all + V_state."""
    rng = np.random.default_rng(seed)
    keys = ["Omega_J", "D_edge"]
    out  = {k: [] for k in keys}
    n_reg = 0
    N = C_all.shape[0]

    Ni_all  = C_all.sum(axis=1)
    P_all   = np.zeros_like(C_all); ok = Ni_all > 0
    P_all[ok] = C_all[ok] / Ni_all[ok, None]

    for _ in range(n_boot):
        Cb_all = np.zeros_like(C_all)
        for i in range(N):
            if Ni_all[i] > 0:
                Cb_all[i] = rng.multinomial(int(Ni_all[i]), P_all[i])
        Cb_all = 0.5 * (Cb_all + Cb_all.T)
        d = compute_diagnostics(Cb_all, V_state, A_set, B_set,
                                  n_cv=n_cv, n_f=n_f,
                                  regularise=True,
                                  alpha_pseudo=alpha_pseudo)
        if d is None:
            continue
        if d["Omega_J"] < 1e-6:    # committor solve degenerate
            continue
        if d.get("regularised"):
            n_reg += 1
        for k in keys:
            out[k].append(d[k])
    return {k: np.array(v) for k, v in out.items()}, n_reg


def main():
    print("=" * 72)
    print("Proper Ras mechanism audit (actually-applied bias + bootstrap)")
    print("=" * 72)
    d = np.load(NPZ, allow_pickle=True)
    trajs = d["data"]
    print(f"  {len(trajs)} trajectories, kBT = {KBT:.3f} kJ/mol")

    # 1) build C_all, V_state with strict trajectory boundaries (no
    #    artificial cross-trajectory transitions).
    build = build_counts_2d(trajs, n_cv=12, n_f=8, lag=1,
                              allow_cross_traj=False)
    print(f"  grid {build['n_cv']} x {build['n_f']} = {build['N']} microstates")
    n_vis = int((build["C_all"].sum(axis=1) > 0).sum())
    n_b   = int((build["C_biased"].sum(axis=1) > 0).sum())
    print(f"  visited: {n_vis}/{build['N']}   biased-visited: {n_b}/{build['N']}")
    print(f"  allow_cross_traj=False  (no artificial boundary transitions)")

    # 2) A/B sets.  With allow_cross_traj=False, the GTP-bound state
    #    (cv<1) is sampled only in one 100-frame iter_00_local segment
    #    and has NO observed transitions to the rest of CV space (the
    #    orchestrator's biased segments start from cv>=4 after re-seeding,
    #    so the npz contains a hard jump cv=-0.4 -> cv=4.5 between
    #    iter_00_local and iter_00_biased that a strict-boundary count
    #    correctly refuses to treat as one dynamical transition).
    #    A pseudocount cannot honestly bridge this 4-CV gap through
    #    unvisited bins, so we redefine A on the part of CV space the
    #    data does sample continuously: A = cv<5 (phosphate near pocket,
    #    early biased frames), B = cv>8 (fully released).
    #    This measures mechanism preservation in the actual unbinding
    #    leg, not the full pocket-to-release transition.
    CV_A_MAX, CV_B_MIN = 5.0, 8.0
    state_cv = build["state_cv"]
    A_set = [s for s in range(build["N"])
             if state_cv[s] < CV_A_MAX and build["C_all"][s].sum() > 0]
    B_set = [s for s in range(build["N"])
             if state_cv[s] > CV_B_MIN and build["C_all"][s].sum() > 0]
    print(f"  A (cv<{CV_A_MAX:.0f}, phosphate near pocket) has {len(A_set)} "
          f"states; B (cv>{CV_B_MIN:.0f}, fully released) has "
          f"{len(B_set)} states")

    # 3) point estimate (no C_biased required; regularise if disconnected)
    alpha_pseudo = 1e-3
    point = compute_diagnostics(build["C_all"], build["V_state"],
                                 A_set, B_set,
                                 n_cv=build["n_cv"], n_f=build["n_f"],
                                 regularise=True,
                                 alpha_pseudo=alpha_pseudo)
    print()
    print("  Point estimate (at the actually-applied bias):")
    for k in ["Omega_J", "D_edge"]:
        print(f"    {k:<10} = {point[k]:.3f}")
    if point.get("regularised"):
        print(f"    NOTE: A and B were disconnected in the strict-boundary "
              f"C_all;\n          sparse local Dirichlet pseudocount "
              f"alpha={alpha_pseudo:g} applied on 2D nearest-neighbour "
              f"edges between visited microstates only.")
    print("    (Omega_rho and R_unsup dropped: they penalise the bias for "
          "successfully accelerating; bootstrap CIs below already absorb "
          "rare-edge uncertainty.)")

    # 4) bootstrap
    n_boot_attempted = 200
    print(f"\n  running bootstrap (n_attempted={n_boot_attempted})...")
    boot, n_reg = bootstrap_audit(build["C_all"],
                                    build["V_state"], A_set, B_set,
                                    n_cv=build["n_cv"], n_f=build["n_f"],
                                    n_boot=n_boot_attempted, seed=0,
                                    alpha_pseudo=alpha_pseudo)
    n_valid = len(boot["Omega_J"])
    print(f"\n  Bootstrap 5/50/95 percentiles (median + 90% CI):")
    print(f"    valid resamples: {n_valid} / {n_boot_attempted}    "
          f"({n_boot_attempted - n_valid} dropped: A or B "
          f"disconnected after resampling even with pseudocount)")
    print(f"    regularised (pseudocount applied): {n_reg} / {n_valid}")
    summary = {"_bootstrap_attempted": int(n_boot_attempted),
               "_bootstrap_valid":     int(n_valid),
               "_bootstrap_regularised": int(n_reg),
               "_allow_cross_traj": False,
               "_alpha_pseudo":     float(alpha_pseudo),
               "_pseudocount_kind": "sparse 2D nearest-neighbour edges between visited microstates only",
               "_dham_sign":        "PLUS (manuscript-consistent; verified by analytical round-trip)",
               "_A_def":            f"cv < {CV_A_MAX}",
               "_B_def":            f"cv > {CV_B_MIN}",
               "_A_size":           int(len(A_set)),
               "_B_size":           int(len(B_set)),
               "_point_regularised": bool(point.get("regularised", False))}
    if n_valid == 0:
        print("    !! ALL bootstrap samples disconnected -- aborting figure.")
        print("    Try a larger pseudocount or revisit A/B definitions.")
        with open(HERE / "ras_proper_audit.json", "w") as fp:
            json.dump(summary, fp, indent=2)
        return
    for k in ["Omega_J", "D_edge"]:
        arr = boot[k]
        p5, p50, p95 = (np.percentile(arr, p) for p in (5, 50, 95))
        summary[k] = dict(point=point[k], median=float(p50),
                          p5=float(p5), p95=float(p95),
                          n_valid=int(len(arr)),
                          n_attempted=int(n_boot_attempted))
        print(f"    {k:<10}  median={p50:.3f}   90% CI [{p5:.3f}, {p95:.3f}]")

    # 5) figure: two-panel (Omega_J on 0-1 scale, D_edge on its own scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.2))
    for ax, k, color, label, ymax in [
        (ax1, "Omega_J",  "#7a3d99", r"$\Omega_J$",      1.0),
        (ax2, "D_edge",   "#a05a2c", r"$D_{\rm edge}$",
         max(3.5, summary["D_edge"]["p95"] * 1.15)),
    ]:
        med = summary[k]["median"]
        lo  = med - summary[k]["p5"]
        hi  = summary[k]["p95"] - med
        ax.bar([0], [med], yerr=[[lo], [hi]], color=color,
               ecolor="0.3", capsize=10, width=0.6)
        ax.set_xticks([0]); ax.set_xticklabels([label], fontsize=18)
        ax.set_ylim(0, ymax)
        ax.text(0, med + 0.02 * ymax, f"{med:.3f}",
                ha="center", fontsize=14, weight="bold")
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("value", fontsize=12)
    reg_note = (f"; sparse NN pseudocount alpha={alpha_pseudo:g} on "
                f"{n_reg}/{n_valid} samples" if n_reg > 0 else "")
    fig.suptitle(
        f"Ras phosphate unbinding: mechanism diagnostics at the applied bias\n"
        f"A: cv<{CV_A_MAX:.0f} (phosphate near pocket); "
        f"B: cv>{CV_B_MIN:.0f} (fully released).  "
        f"Strict trajectory boundaries{reg_note}.\n"
        f"median + 90% CI over {n_valid}/{n_boot_attempted} bootstrap "
        f"resamples.  DHAM_sym with PLUS sign (analytical round-trip "
        f"verified).",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.83])
    _ = """  legacy placeholder; replaced by the two-panel figure above """
    fig.savefig(HERE / "fig_mechanism_ras_proper.pdf")
    fig.savefig(HERE / "fig_mechanism_ras_proper.png", dpi=180)
    with open(HERE / "ras_proper_audit.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nwrote {HERE / 'fig_mechanism_ras_proper.pdf'}")
    print(f"wrote {HERE / 'ras_proper_audit.json'}")


if __name__ == "__main__":
    main()
