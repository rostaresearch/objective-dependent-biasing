"""Mechanism-conservation audit on the 1D asymmetric double-well, reusing the
saved spectral / MFPT-optimal per-state biases in `asymmetric_1d.json`.

Computes the four diagnostics of the JCTC v2 manuscript ("Implications for
protein simulations"):
    Omega_J(b)      eq 13 -- normalised positive-net-current overlap
    Omega_rho(b)    eq 15 -- transition-tube-density overlap
    D_edge(b)       eq 16 -- current-weighted edge-rate distortion
    R_unsup(b, C)   eq 18 -- biased current routed through count-poor edges

Sweeps a scaling b(alpha) = alpha * b*, alpha in [0, 1.2], for both the
spectral and MFPT optima, and produces fig_mechanism_asym1d.pdf -- a 4-panel
figure parallel to fig_mechanism_main_v2.pdf (which is on the 9-well 2D
landscape).

Run:
    cd $MSM_ROOT
    python mechanism_audit_asym1d.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


# ---------- generator + audit primitives ------------------------------------

def asym_1d_generator(F, base_rate=1.0):
    """Detailed-balance tridiagonal generator on a 1D grid with potential F[i]
    (in k_B T). Off-diagonals K[i, i+/-1] = base_rate * exp(-0.5 (F[j]-F[i])),
    diagonals make rows sum to zero. (Matches asymmetric_1d.asym_1d_generator.)"""
    n = len(F)
    K = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                K[i, j] = base_rate * np.exp(-0.5 * (F[j] - F[i]))
    np.fill_diagonal(K, -K.sum(axis=1))
    return K


def tilt_generator(K0, u):
    """K^b_ij = K0_ij * exp(-(u_j - u_i)/2); detailed-balance preserved by
    construction. Matches analytic_lib.tilt_generator -- the convention is
    that high u at A and low u at B accelerates A->B (system flows down u).
    Diagonals reset to make rows sum to 0."""
    du = (u[None, :] - u[:, None]) * 0.5
    Kb = K0 * np.exp(-du)
    np.fill_diagonal(Kb, 0.0)
    np.fill_diagonal(Kb, -Kb.sum(axis=1))
    return Kb


def stationary_1d_tridiag(K):
    """Detailed-balance stationary for a 1D nearest-neighbour generator,
    in log-space to survive deep barriers.
        pi_i / pi_{i-1} = K[i-1, i] / K[i, i-1]
    """
    n = K.shape[0]
    log_pi = np.zeros(n)
    for i in range(1, n):
        log_pi[i] = log_pi[i - 1] + np.log(K[i - 1, i]) - np.log(K[i, i - 1])
    log_pi -= log_pi.max()                # avoid overflow in exp
    pi = np.exp(log_pi); pi /= pi.sum()
    return pi


def stationary_from_generator(K):
    n = K.shape[0]
    # If tridiagonal use the closed-form (numerically stable);
    # otherwise fall back to constrained least-squares.
    off = np.abs(K) - np.diag(np.diag(K))
    off = np.abs(off)
    # check tridiag pattern
    is_tri = True
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and off[i, j] > 1e-30:
                is_tri = False; break
        if not is_tri: break
    if is_tri:
        return stationary_1d_tridiag(K)
    A = np.vstack([K.T, np.ones(n)])
    b = np.zeros(n + 1); b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.maximum(pi, 0.0) / np.sum(np.maximum(pi, 0.0))


def committor_1d_tridiag(K, iA, iB):
    """Closed-form forward committor for 1D nearest-neighbour K:
        q(i) = sum_{k=iA}^{i-1} 1/(pi_k * K[k, k+1])
             / sum_{k=iA}^{iB-1} 1/(pi_k * K[k, k+1])
    Boundary: q(iA)=0, q(iB)=1. Works in log space."""
    n = K.shape[0]
    pi = stationary_1d_tridiag(K)
    log_step = -(np.log(pi[:-1]) + np.log(K[np.arange(n - 1), np.arange(1, n)]))
    # cumulative sum in log: log(sum_k exp(log_step_k))
    log_cum = np.full(n, -np.inf)
    log_cum[iA] = -np.inf
    cur = -np.inf
    for k in range(iA, n - 1):
        cur = np.logaddexp(cur, log_step[k]) if cur != -np.inf else log_step[k]
        log_cum[k + 1] = cur
    log_total = log_cum[iB]
    # outside [iA, iB] clamp
    q = np.zeros(n)
    for i in range(n):
        if i <= iA:
            q[i] = 0.0
        elif i >= iB:
            q[i] = 1.0
        else:
            q[i] = float(np.exp(log_cum[i] - log_total))
    return np.clip(q, 0.0, 1.0)


def committor(K, A_set, B_set):
    """Forward committor q with q_A = 0, q_B = 1.
    Uses 1D closed form when the system is tridiagonal."""
    n = K.shape[0]
    iA, iB = min(A_set), max(B_set)
    if iA < iB and len(A_set) == 1 and len(B_set) == 1:
        # check tridiagonal
        is_tri = True
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(K[i, j]) > 1e-30:
                    is_tri = False; break
            if not is_tri: break
        if is_tri:
            return committor_1d_tridiag(K, iA, iB)
    R = np.array([i for i in range(n) if i not in A_set and i not in B_set])
    KRR = K[np.ix_(R, R)]
    KRB = K[np.ix_(R, list(B_set))]
    rhs = -KRB.sum(axis=1)
    qR = solve(KRR, rhs)
    q = np.zeros(n); q[list(B_set)] = 1.0; q[R] = np.clip(qR, 0.0, 1.0)
    return q


def mfpt_1d_tridiag(K, iA, iB):
    """Closed-form MFPT(iA -> iB) for 1D nearest-neighbour K with reflecting
    left boundary at 0 (since pi is normalised). Standard formula:
        tau(i -> i+1) = (1/W_{i,i+1}) * sum_{j=0}^{i} pi_j/pi_i
    Sum over i in [iA, iB-1] to get tau(iA -> iB). Log-stable."""
    n = K.shape[0]
    pi = stationary_1d_tridiag(K)
    log_pi = np.log(pi)
    # For each i in [iA, iB-1], compute log(sum_{j<=i} pi_j / pi_i) = log(P(<=i)) - log(pi_i)
    log_cum_pi = np.full(n, -np.inf)
    cur = -np.inf
    for j in range(n):
        cur = np.logaddexp(cur, log_pi[j]) if cur != -np.inf else log_pi[j]
        log_cum_pi[j] = cur
    # Aggregate the per-step MFPTs in log space (positive numbers).
    log_terms = []
    for i in range(iA, iB):
        log_t = log_cum_pi[i] - log_pi[i] - np.log(K[i, i + 1])
        log_terms.append(log_t)
    # sum_terms = exp(log_terms). Numerically stable:
    M = max(log_terms)
    return float(np.exp(M) * sum(np.exp(t - M) for t in log_terms))


def mfpt_from_K(K, A_set, B_set, pi=None):
    iA, iB = min(A_set), max(B_set)
    if iA < iB and len(A_set) == 1 and len(B_set) == 1:
        is_tri = True
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                if abs(i - j) > 1 and abs(K[i, j]) > 1e-30:
                    is_tri = False; break
            if not is_tri: break
        if is_tri:
            return mfpt_1d_tridiag(K, iA, iB)
    R = np.array([i for i in range(K.shape[0]) if i not in B_set])
    KRR = K[np.ix_(R, R)]
    h = solve(KRR, -np.ones(len(R)))
    A_in_R = [int(np.where(R == a)[0][0]) for a in A_set]
    if pi is None or len(A_set) == 1:
        return float(np.mean(h[A_in_R]))
    return float(np.average(h[A_in_R], weights=pi[list(A_set)]))


def positive_net_current(K, pi, q):
    n = K.shape[0]
    F = pi[:, None] * K * (1.0 - q)[:, None] * q[None, :]    # gross f_ij
    np.fill_diagonal(F, 0.0)
    Jp = np.maximum(F - F.T, 0.0)
    Z = Jp.sum() + 1e-300
    return Jp / Z


def tube_density(pi, q):
    rho = pi * q * (1.0 - q)
    return rho / (rho.sum() + 1e-300)


def mechanism_audit(K0, u, A_set, B_set, C=None, C0=10.0):
    pi0 = stationary_from_generator(K0)
    q0  = committor(K0, A_set, B_set)
    Jh0 = positive_net_current(K0, pi0, q0)
    rh0 = tube_density(pi0, q0)

    Kb  = tilt_generator(K0, u)
    pib = stationary_from_generator(Kb)
    qb  = committor(Kb, A_set, B_set)
    Jhb = positive_net_current(Kb, pib, qb)
    rhb = tube_density(pib, qb)

    Omega_J   = float(np.minimum(Jh0, Jhb).sum())
    Omega_rho = float(np.minimum(rh0, rhb).sum())
    D_edge    = 0.5 * float((Jh0 * np.abs(u[None, :] - u[:, None])).sum())

    R_unsup = np.nan
    if C is not None:
        support_w = 1.0 - C / (C + C0)
        R_unsup = float((Jhb * support_w).sum())

    tau_AB = mfpt_from_K(Kb, A_set, B_set, pi=pib)
    return dict(Omega_J=Omega_J, Omega_rho=Omega_rho, D_edge=D_edge,
                R_unsup=R_unsup, tau_AB=tau_AB)


# ---------- run on the saved 1D asym data ----------------------------------

def main():
    data = json.load(open(HERE / "asymmetric_1d.json"))
    F   = np.array(data["F"])
    x   = np.array(data["x"])
    iA  = int(data["iA"]); iB = int(data["iB"])
    A_set, B_set = {iA}, {iB}

    K0 = asym_1d_generator(F)
    pi0 = stationary_from_generator(K0)
    tau0 = mfpt_from_K(K0, A_set, B_set, pi=pi0)
    print(f"unbiased MFPT(A->B) = {tau0:.4e}    pi0(A)={pi0[iA]:.3e}  pi0(B)={pi0[iB]:.3e}")

    # Build a synthetic "unbiased count matrix" with realistic statistics:
    # short stationary trajectory equivalent of N=1e5 transitions sampled
    # from pi0 with the unbiased transition rates. For an analytical example
    # this is just for the R_unsup audit demonstration.
    N_total = 1_000_000
    P0 = np.eye(len(F)) + 1e-3 * K0           # transition prob at small dt
    np.fill_diagonal(P0, 0.0); np.fill_diagonal(P0, 1.0 - P0.sum(axis=1))
    C = pi0[:, None] * P0 * N_total
    # symmetrise (DHAM_sym-style)
    C = 0.5 * (C + C.T)

    # ---- sweep alpha for both the spectral and MFPT optima at each U_max ----
    alphas = np.linspace(0.0, 1.2, 25)
    sweep = {}
    for res in data["results"]:
        U_max = float(res["U_max"])
        for tag, key_u in [("spectral", "u_spectral"), ("mfpt", "u_mfpt")]:
            u_star = np.array(res[key_u])
            audit_rows = []
            for a in alphas:
                u = a * u_star
                m = mechanism_audit(K0, u, A_set, B_set, C=C, C0=10.0)
                m["alpha"] = a
                audit_rows.append(m)
            sweep[(U_max, tag)] = audit_rows
            # print at alpha=1
            full = audit_rows[-1] if alphas[-1] == 1.0 else audit_rows[
                int(np.argmin(np.abs(alphas - 1.0)))]
            speedup = tau0 / full["tau_AB"]
            print(f"  U_max={U_max:.1f}  {tag:8s}  alpha=1   "
                  f"MFPT speedup={speedup:.3e}   "
                  f"Omega_J={full['Omega_J']:.3f}  "
                  f"Omega_rho={full['Omega_rho']:.3f}  "
                  f"D_edge={full['D_edge']:.3f}  "
                  f"R_unsup={full['R_unsup']:.3f}")

    # ---- 4-panel figure mimicking fig_mechanism_main_v2 -----------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    colors = {"spectral": "#1f77b4", "mfpt": "#d62728"}
    markers = {"spectral": "o", "mfpt": "s"}
    linestyles = {4.0: ":", 6.0: "--", 8.0: "-"}

    # (a) MFPT speedup vs Omega_rho  (Pareto curve; on 1D Omega_J is
    #     structurally 1 because the tridiagonal chain forces a single path,
    #     so the meaningful mechanism axis is Omega_rho.)
    for (U_max, tag), rows in sweep.items():
        Omega_rho = [r["Omega_rho"] for r in rows]
        speedup   = [tau0 / r["tau_AB"] for r in rows]
        ax_a.plot(Omega_rho, speedup, color=colors[tag], ls=linestyles[U_max],
                  marker=markers[tag], ms=4, lw=1.3, alpha=0.85,
                  label=f"{tag}, $U_{{\\max}}={U_max:.0f}$")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"transition-tube overlap $\Omega_\rho$")
    ax_a.set_ylabel(r"MFPT speedup")
    ax_a.set_title(r"(a) Acceleration vs $\Omega_\rho$ (1D: $\Omega_J\equiv 1$)")
    ax_a.legend(fontsize=7.5, loc="lower left")
    ax_a.grid(alpha=0.3)
    ax_a.invert_xaxis()   # higher overlap = better mechanism, plot left

    # (b) Omega_J and Omega_rho vs alpha
    for (U_max, tag), rows in sweep.items():
        alpha_v   = [r["alpha"]     for r in rows]
        Omega_J   = [r["Omega_J"]   for r in rows]
        Omega_rho = [r["Omega_rho"] for r in rows]
        ax_b.plot(alpha_v, Omega_J,   color=colors[tag],
                  ls=linestyles[U_max], lw=1.5, alpha=0.9)
        ax_b.plot(alpha_v, Omega_rho, color=colors[tag],
                  ls=linestyles[U_max], lw=1.5, alpha=0.5)
    ax_b.set_xlabel(r"bias scale $\alpha$ ($b = \alpha\, b^*$)")
    ax_b.set_ylabel(r"overlap")
    ax_b.set_title(r"(b) $\Omega_J$ (solid) and $\Omega_\rho$ (faint) vs $\alpha$")
    ax_b.grid(alpha=0.3)

    # (c) D_edge vs alpha
    for (U_max, tag), rows in sweep.items():
        alpha_v = [r["alpha"]  for r in rows]
        D_edge  = [r["D_edge"] for r in rows]
        ax_c.plot(alpha_v, D_edge, color=colors[tag],
                  ls=linestyles[U_max], lw=1.5, alpha=0.9,
                  label=f"{tag}, $U_{{\\max}}={U_max:.0f}$")
    ax_c.set_xlabel(r"bias scale $\alpha$")
    ax_c.set_ylabel(r"$D_{\rm edge}$")
    ax_c.set_title(r"(c) Edge-rate distortion vs $\alpha$")
    ax_c.legend(fontsize=7.5, loc="upper left")
    ax_c.grid(alpha=0.3)

    # (d) Comparison at alpha = 1 (full-strength bias) -- mimic panel (d)
    full_strength_rows = []
    for (U_max, tag), rows in sweep.items():
        idx_one = int(np.argmin(np.abs(np.array([r["alpha"] for r in rows]) - 1.0)))
        full_strength_rows.append((U_max, tag, rows[idx_one]))
    labels = [f"{tag}\n$U_{{\\max}}={U_max:.0f}$"
              for (U_max, tag, _) in full_strength_rows]
    bar_speedup  = [tau0 / r["tau_AB"]    for (_, _, r) in full_strength_rows]
    bar_Omega_J  = [r["Omega_J"]          for (_, _, r) in full_strength_rows]
    bar_Omega_r  = [r["Omega_rho"]        for (_, _, r) in full_strength_rows]
    bar_D_edge   = [r["D_edge"]           for (_, _, r) in full_strength_rows]
    x_pos = np.arange(len(labels))
    w = 0.2
    twin = ax_d.twinx()
    ax_d.bar(x_pos - 1.5*w, bar_speedup,  width=w, color="#999999",
             label="MFPT speedup", log=True)
    twin.bar(x_pos - 0.5*w, bar_Omega_J,  width=w, color="#1f77b4",
             label=r"$\Omega_J$", alpha=0.9)
    twin.bar(x_pos + 0.5*w, bar_Omega_r,  width=w, color="#aec7e8",
             label=r"$\Omega_\rho$", alpha=0.9)
    twin.bar(x_pos + 1.5*w, bar_D_edge,   width=w, color="#d62728",
             label=r"$D_{\rm edge}$", alpha=0.9)
    ax_d.set_xticks(x_pos); ax_d.set_xticklabels(labels, fontsize=7.5)
    ax_d.set_ylabel("MFPT speedup", color="#444")
    twin.set_ylabel(r"$\Omega_J$, $\Omega_\rho$, $D_{\rm edge}$ ")
    ax_d.set_title(r"(d) Full-strength bias ($\alpha=1$)")
    h1, l1 = ax_d.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    fig.suptitle("Mechanism-conservation audit on 1D asymmetric double well\n"
                 r"$V(x) = 21(x^2-1)^2 + x$, n=200 microstates", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = HERE / "fig_mechanism_asym1d.pdf"
    out_png = HERE / "fig_mechanism_asym1d.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")

    # Compact text table for the manuscript Section 7
    print("\n--- numbers for the manuscript table (alpha=1 full-strength bias) ---")
    print(f"{'U_max':<6} {'method':<10} {'speedup':>10} {'Omega_J':>9} "
          f"{'Omega_rho':>10} {'D_edge':>9} {'R_unsup':>9}")
    for (U_max, tag, r) in full_strength_rows:
        print(f"{U_max:<6.1f} {tag:<10} {tau0/r['tau_AB']:>10.3e} "
              f"{r['Omega_J']:>9.3f} {r['Omega_rho']:>10.3f} "
              f"{r['D_edge']:>9.3f} {r['R_unsup']:>9.3f}")


if __name__ == "__main__":
    main()
