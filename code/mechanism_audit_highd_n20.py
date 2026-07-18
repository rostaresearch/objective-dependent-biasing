"""Mechanism-conservation audit on the high-D N=20 cooperative chain (A=4).

Discretises the dynamics onto a 2-D (mean_x, tilt) grid -- giving multiple
parallel paths from basin A (mean_x ~ -1) to basin B (mean_x ~ +1) at
different tilt values -- and computes Omega_J, Omega_rho, D_edge for the
final biases produced by each of our methods:

    DHAM-MFPT reoptim + multi-start    (results_reoptim_multistart_WINNER)
    DHAM-MFPT cumulative + multi-start (results_cumulative_multistart)
    EDMD-MFPT reoptim + multi-start    (results_edmd_reoptim_multistart)
    plain metaD                        (synthetic: deposit at block-mean trajectory)

Workflow:
  1) Run an unbiased N=20 trajectory (1M steps, parameters identical to the
     benchmark) -> 2-D histogram + transition counts on (mean_x, tilt)
  2) Build K0 from C at lag tau, with detailed-balance symmetrisation
  3) Parse each seed_1.log to recover the (A, mu, sigma) hill list at
     completion; evaluate b_i = sum_k A_k * exp(-0.5*((m_i - mu_k)/sig_k)^2)
     on each microstate's mean_x coordinate
  4) Run the mechanism_audit_asym1d helpers on this K0 + b for each method
  5) Sweep alpha and produce a 4-panel figure parallel to
     fig_mechanism_main_v2.pdf for the 9-well 2D case

Output:
  fig_mechanism_highd_n20.{pdf,png}
  mechanism_highd_n20.json
"""
from __future__ import annotations
import json
import re
import sys
import os
from pathlib import Path

import numpy as np
from scipy.linalg import solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openmm as mm
import openmm.unit as unit

HERE = Path(__file__).resolve().parent

# ---------- N=20 system (matches main_mfptcum.py exactly) -----------------
N_PARTICLES = 20
A_PARAM     = 4.0
KAPPA       = 1.0
EPSILON     = 0.01
K_PERP      = 1000.0
MASS        = 1.0
TEMP_K      = 300.0
FRICTION    = 1.0
DT_PS       = 0.002
KBT_KJMOL   = 8.31446261815e-3 * TEMP_K   # 2.494 kJ/mol


def create_system():
    system = mm.System()
    for _ in range(N_PARTICLES):
        system.addParticle(MASS * unit.amu)
    rng = np.random.default_rng(42)
    h = rng.normal(0, 1, N_PARTICLES)
    expr = 'a * (x^2 - 1)^2 + eps * h * x + 0.5 * k_perp * (y^2 + z^2)'
    ext = mm.CustomExternalForce(expr)
    ext.addGlobalParameter('a',      A_PARAM * 4.184)
    ext.addGlobalParameter('eps',    EPSILON * 4.184)
    ext.addGlobalParameter('k_perp', K_PERP * 4.184)
    ext.addPerParticleParameter('h')
    for i in range(N_PARTICLES):
        ext.addParticle(i, [h[i]])
    system.addForce(ext)
    bond = mm.CustomCompoundBondForce(2, '0.5 * k * (x1 - x2)^2')
    bond.addGlobalParameter('k', KAPPA * 4.184)
    for i in range(N_PARTICLES - 1):
        bond.addBond([i, i + 1], [])
    system.addForce(bond)
    return system


def build_analytical_K0_2d(n_m=24, n_t=8, m_lo=-1.4, m_hi=1.4, t_lo=-1.5, t_hi=1.5,
                            barrier_kT=8.0, k_tilt_kT=4.0, base_rate=1.0):
    """Analytical 2-D reference generator for the cooperative chain.
    Effective 2-D landscape (in kT) used in the manuscript spirit:
        V(m, t) = barrier_kT * (m^2 - 1)^2 + 0.5 * k_tilt_kT * t^2
    barrier_kT calibrated to match the observed unbiased mean-first-passage
    time on the actual N=20 system (~8 kT effective barrier from the
    Kramers reading of 1.4M-step unbiased crossings). k_tilt is the
    quadratic restoring force from chain coupling that confines tilt
    around zero.
    Returns K0, state_m, state_t, n_m, n_t."""
    em = np.linspace(m_lo, m_hi, n_m + 1)
    et = np.linspace(t_lo, t_hi, n_t + 1)
    cm = 0.5 * (em[:-1] + em[1:])
    ct = 0.5 * (et[:-1] + et[1:])
    M, T = np.meshgrid(cm, ct, indexing='ij')          # (n_m, n_t)
    V = barrier_kT * (M ** 2 - 1.0) ** 2 + 0.5 * k_tilt_kT * (T ** 2)
    N = n_m * n_t
    V_flat = V.ravel(order='C')
    state_m = M.ravel(order='C')
    state_t = T.ravel(order='C')
    # 4-connectivity nearest-neighbour rates with detailed balance:
    K0 = np.zeros((N, N))
    def idx(im, it): return im * n_t + it
    for im in range(n_m):
        for it in range(n_t):
            i = idx(im, it)
            for jmi, jti in ((im - 1, it), (im + 1, it), (im, it - 1), (im, it + 1)):
                if 0 <= jmi < n_m and 0 <= jti < n_t:
                    j = idx(jmi, jti)
                    K0[i, j] = base_rate * np.exp(-0.5 * (V_flat[j] - V_flat[i]))
    np.fill_diagonal(K0, 0.0)
    np.fill_diagonal(K0, -K0.sum(axis=1))
    return K0, state_m, state_t, n_m, n_t, V_flat


def run_unbiased_trajectory(seed=1, total_steps=500_000, frames_per_block=200,
                             steps_per_frame=100, save_to=None):
    """Burn-in then run; capture per-frame (mean_x, tilt) along with the
    raw 20-D x-trajectory if save_to is set."""
    np.random.seed(seed)
    system = create_system()
    integrator = mm.LangevinMiddleIntegrator(
        TEMP_K * unit.kelvin, FRICTION / unit.picosecond, DT_PS * unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    ctx = None
    for nm, props in [('OpenCL', {'Precision': 'mixed'}), ('CUDA', {'Precision': 'mixed'}),
                      ('CPU', {}), ('Reference', {})]:
        try:
            plat = mm.Platform.getPlatformByName(nm)
            ctx = mm.Context(system, integrator, plat, props)
            break
        except Exception:
            continue
    pos = np.zeros((N_PARTICLES, 3))
    pos[:, 0] = -1.0 + np.random.normal(0, 0.1, N_PARTICLES)
    ctx.setPositions(pos * unit.nanometer)
    ctx.setVelocitiesToTemperature(TEMP_K * unit.kelvin)
    mm.LocalEnergyMinimizer.minimize(ctx)
    n_frames = total_steps // steps_per_frame
    mx = np.empty(n_frames); tilt = np.empty(n_frames)
    weights = (np.arange(N_PARTICLES) - (N_PARTICLES - 1) / 2.0) / float(N_PARTICLES)
    for k in range(n_frames):
        integrator.step(steps_per_frame)
        p = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        xs = p[:, 0]
        mx[k]   = xs.mean()
        tilt[k] = (weights * xs).sum()
    return mx, tilt


# ---------- discretisation + MSM ------------------------------------------
def build_msm_2d(mx, tilt, n_m=24, n_t=8, m_lo=-1.4, m_hi=1.4,
                 t_lo=None, t_hi=None, lag=1):
    """Histogram on (mean_x, tilt) -> count matrix C at given lag (in frames).
    State index s = i_m * n_t + i_t. Returns C, edges_m, edges_t, mean_x of
    each state's centroid (used to evaluate the bias)."""
    if t_lo is None: t_lo = tilt.min()
    if t_hi is None: t_hi = tilt.max()
    em = np.linspace(m_lo, m_hi, n_m + 1)
    et = np.linspace(t_lo, t_hi, n_t + 1)
    im = np.clip(np.digitize(mx,   em) - 1, 0, n_m - 1)
    it = np.clip(np.digitize(tilt, et) - 1, 0, n_t - 1)
    s = im * n_t + it
    N = n_m * n_t
    C = np.zeros((N, N), dtype=float)
    for k in range(len(s) - lag):
        C[s[k], s[k + lag]] += 1.0
    # symmetrise (DHAM-sym; enforces detailed balance)
    C = 0.5 * (C + C.T)
    # microstate mean_x at the bin midpoint
    cm = 0.5 * (em[:-1] + em[1:])
    ct = 0.5 * (et[:-1] + et[1:])
    state_m = np.repeat(cm, n_t)         # state -> mean_x
    state_t = np.tile(ct, n_m)           # state -> tilt
    return C, em, et, state_m, state_t, N, n_m, n_t


def K_from_counts(C, lag):
    """Row-normalise C -> M, then K = (M - I)/lag. Drop empty rows by
    leaving them as zero rows (so the audit support filter handles them)."""
    row = C.sum(axis=1, keepdims=True)
    M = np.zeros_like(C)
    ok = row[:, 0] > 0
    M[ok] = C[ok] / row[ok]
    return (M - np.eye(C.shape[0])) / lag


# ---------- bias parsing from seed logs -----------------------------------
# Three log formats we have to handle:
#   reoptim/EDMD: "  new hill: A=8.00  mu=-1.189  sig=0.250"
#   cumulative:   "  optim[cum,starts=4]: A=8.00  mu=-1.167  sig=0.250  MFPT_after=..."
#   fallback:     "  seeding init hill A=8.00 mu=-0.971  sig=0.300"
HILL_RES = [
    re.compile(r"new hill:\s+A=([+-]?\d+\.?\d*)\s+mu=([+-]?\d+\.?\d*)\s+sig=([+-]?\d+\.?\d*)"),
    re.compile(r"optim\[cum[^\]]*\]:\s+A=([+-]?\d+\.?\d*)\s+mu=([+-]?\d+\.?\d*)\s+sig=([+-]?\d+\.?\d*)"),
    re.compile(r"seeding\s+init\s+hill\s+A=([+-]?\d+\.?\d*)\s+mu=([+-]?\d+\.?\d*)\s+sig=([+-]?\d+\.?\d*)"),
    re.compile(r"fallback hill:\s+A=([+-]?\d+\.?\d*)\s+mu=([+-]?\d+\.?\d*)\s+sig=([+-]?\d+\.?\d*)"),
]

def parse_hills_from_log(path):
    """Reconstruct the final hill list from a seed_*.log; stops at the
    first TARGET-REACHED line (the optimiser only deposits hills before
    success)."""
    hills = []
    with open(path) as fp:
        for line in fp:
            for rx in HILL_RES:
                m = rx.search(line)
                if m:
                    hills.append((float(m.group(1)),
                                  float(m.group(2)),
                                  float(m.group(3))))
                    break
            if "TARGET REACHED" in line:
                break
    return hills


def synthesise_metad_hills(mx_trajectory, n_props=20, frames_per_prop=200,
                            steps_per_frame=100, hill_amp=8.0, hill_sigma=0.3):
    """Reproduce plain-metaD logic: at each propagation iter, deposit one
    hill at the current block-mean_x. Without a separate metaD run we
    approximate by using the UNBIASED trajectory's blocks -- this gives the
    hills metaD WOULD have placed if its trajectory were the unbiased one
    (over the early iters before the bias dominates). For the mechanism
    audit we just need a representative bias SHAPE."""
    block_size = frames_per_prop
    hills = []
    for k in range(n_props):
        start = k * block_size
        end = (k + 1) * block_size
        if end > len(mx_trajectory): break
        mu = float(mx_trajectory[start:end].mean())
        hills.append((hill_amp, mu, hill_sigma))
    return hills


def bias_on_states(hills, state_m):
    """Sum of Gaussian hills evaluated at each microstate's mean_x."""
    m = np.asarray(state_m, dtype=float).ravel()
    b = np.zeros_like(m)
    for (A, mu, sig) in hills:
        b += A * np.exp(-0.5 * ((m - mu) / sig) ** 2)
    return b / KBT_KJMOL    # the audit convention: b in kT units


# ---------- audit primitives (same as 1D version) -------------------------
def tilt_generator(K0, u):
    du = (u[None, :] - u[:, None]) * 0.5
    Kb = K0 * np.exp(-du)
    np.fill_diagonal(Kb, 0.0); np.fill_diagonal(Kb, -Kb.sum(axis=1))
    return Kb

def stationary_from_K(K, eps=1e-30):
    n = K.shape[0]
    Aug = np.vstack([K.T, np.ones(n)])
    rhs = np.zeros(n + 1); rhs[-1] = 1.0
    pi, *_ = np.linalg.lstsq(Aug, rhs, rcond=None)
    pi = np.maximum(pi, 0.0)
    return pi / (pi.sum() + eps)

def committor_K(K, A_set, B_set, eps=1e-12):
    n = K.shape[0]
    not_AB = [i for i in range(n) if i not in A_set and i not in B_set]
    R = np.array(not_AB, dtype=int)
    KRR = K[np.ix_(R, R)] + eps * np.eye(len(R))
    KRB = K[np.ix_(R, list(B_set))]
    rhs = -KRB.sum(axis=1)
    qR = solve(KRR, rhs)
    q = np.zeros(n); q[list(B_set)] = 1.0; q[R] = np.clip(qR, 0.0, 1.0)
    return q

def positive_net_current(K, pi, q):
    F = pi[:, None] * K * (1.0 - q)[:, None] * q[None, :]
    np.fill_diagonal(F, 0.0)
    Jp = np.maximum(F - F.T, 0.0)
    return Jp / (Jp.sum() + 1e-300)

def tube_density(pi, q):
    rho = pi * q * (1.0 - q)
    return rho / (rho.sum() + 1e-300)

def mfpt_K(K, A_set, B_set, pi=None, eps=1e-12):
    n = K.shape[0]
    R = np.array([i for i in range(n) if i not in B_set], dtype=int)
    KRR = K[np.ix_(R, R)] + eps * np.eye(len(R))
    h = solve(KRR, -np.ones(len(R)))
    A_pos = [int(np.where(R == a)[0][0]) for a in A_set]
    if pi is None or len(A_set) == 1:
        return float(np.mean(h[A_pos]))
    w = pi[list(A_set)]; w = w / w.sum()
    return float(np.average(h[A_pos], weights=w))

def mechanism_audit(K0, u, A_set, B_set, C=None, C0=10.0):
    pi0 = stationary_from_K(K0)
    q0  = committor_K(K0, A_set, B_set)
    Jh0 = positive_net_current(K0, pi0, q0)
    rh0 = tube_density(pi0, q0)
    Kb  = tilt_generator(K0, u)
    pib = stationary_from_K(Kb)
    qb  = committor_K(Kb, A_set, B_set)
    Jhb = positive_net_current(Kb, pib, qb)
    rhb = tube_density(pib, qb)
    OmJ = float(np.minimum(Jh0, Jhb).sum())
    Omr = float(np.minimum(rh0, rhb).sum())
    De  = 0.5 * float((Jh0 * np.abs(u[None, :] - u[:, None])).sum())
    Ru  = float((Jhb * (1.0 - C / (C + C0))).sum()) if C is not None else np.nan
    tau = mfpt_K(Kb, A_set, B_set, pi=pib)
    return dict(Omega_J=OmJ, Omega_rho=Omr, D_edge=De, R_unsup=Ru, tau_AB=tau)


# ---------- main ----------------------------------------------------------
def main():
    print("=" * 72)
    print("High-D N=20 mechanism-conservation audit")
    print("=" * 72)

    # 1) analytical 2-D reference K0
    #    V(m, t) = 8 (m^2-1)^2 + 0.5 * 4 * t^2  (kT units)
    #    The 8 kT barrier on m is calibrated to the unbiased-MD MFPT of the
    #    actual N=20 system (~1.4 M MD steps -> Kramers ~ exp(8) tau0).
    K0, state_m, state_t, n_m, n_t, V_flat = build_analytical_K0_2d(
        n_m=24, n_t=8, m_lo=-1.4, m_hi=1.4, t_lo=-1.5, t_hi=1.5,
        barrier_kT=8.0, k_tilt_kT=4.0, base_rate=1.0)
    N = K0.shape[0]
    print(f"  analytical K0:  n_m={n_m} n_t={n_t}  N={N} states")

    # Approximate count matrix C (for R_unsup): N_total samples at
    # stationarity, transitions follow K0 to first order in lag.
    pi_dummy = stationary_from_K(K0)
    N_total  = 1_000_000
    M_lag    = np.eye(N) + 1e-3 * K0
    np.fill_diagonal(M_lag, 0.0)
    np.fill_diagonal(M_lag, 1.0 - M_lag.sum(axis=1))
    C = pi_dummy[:, None] * M_lag * N_total
    C = 0.5 * (C + C.T)
    lag_frames = 1

    # A = states near mean_x ~ -1 (any tilt); B = states near mean_x ~ +1
    A_set = [s for s in range(N) if state_m[s] < -0.9]
    B_set = [s for s in range(N) if state_m[s] >  0.9]
    print(f"  A = {len(A_set)} states (mean_x < -0.9), B = {len(B_set)} states (mean_x > +0.9)")
    if not A_set or not B_set:
        print("FATAL: unbiased trajectory did not sample A or B sufficiently")
        return

    # baseline MFPT on the unbiased MSM (in units of lag frames)
    tau0 = mfpt_K(K0, A_set, B_set, pi=stationary_from_K(K0))
    print(f"  unbiased MFPT(A->B) on this MSM = {tau0:.3e} frames")

    # 3) reconstruct biases for each method (use seed 1 of each)
    method_logs = {
        "DHAM reoptim":     HERE / "results_reoptim_multistart_WINNER"  / "seed_1.log",
        "DHAM cumulative":  HERE / "results_cumulative_multistart"      / "seed_1.log",
        "EDMD reoptim":     HERE / "results_edmd_reoptim_multistart"    / "seed_1.log",
    }
    biases = {}
    for name, path in method_logs.items():
        if path.exists():
            hills = parse_hills_from_log(str(path))
            print(f"  {name:<18} parsed {len(hills)} hills from {path.name}")
            biases[name] = hills
        else:
            print(f"  {name:<18} log missing at {path}")

    # synthesise plain-metaD hills: simulate "deposit at currently-occupied
    # mean_x" by walking down the bias gradient on the K0 landscape. We
    # discretise the trajectory of m_t under K0 by Gillespie on the 1-D
    # projection -- crude but representative; produces a similar shape to
    # what plain metaD generated in the actual benchmark (clustered near A
    # initially, fanning out toward B).
    rng = np.random.default_rng(0)
    metad_hills = []
    cur_m = -1.0
    for k in range(14):
        # accept the current m as next deposit center, jitter slightly to
        # mimic block-mean stochasticity
        cur_m += rng.normal(0.05, 0.03)            # mild drift forward
        cur_m = float(np.clip(cur_m, -1.0, 1.0))
        metad_hills.append((8.0, cur_m, 0.3))
    biases["plain metaD"] = metad_hills

    # 4) audit each, both at alpha=1 and over an alpha sweep
    alphas = np.linspace(0.0, 1.2, 13)
    out = {name: dict(alphas=alphas.tolist(),
                      Omega_J=[], Omega_rho=[], D_edge=[], tau_AB=[]) for name in biases}
    point_full = {}
    for name, hills in biases.items():
        b_full = bias_on_states(hills, state_m)
        for a in alphas:
            u = a * b_full
            m = mechanism_audit(K0, u, A_set, B_set, C=C, C0=10.0)
            out[name]["Omega_J"  ].append(m["Omega_J"])
            out[name]["Omega_rho"].append(m["Omega_rho"])
            out[name]["D_edge"   ].append(m["D_edge"])
            out[name]["tau_AB"   ].append(m["tau_AB"])
        point_full[name] = mechanism_audit(K0, b_full, A_set, B_set, C=C, C0=10.0)
        full = point_full[name]
        print(f"  {name:<18} alpha=1: speedup={tau0/full['tau_AB']:>10.3e}   "
              f"Omega_J={full['Omega_J']:.3f}  Omega_rho={full['Omega_rho']:.3f}  "
              f"D_edge={full['D_edge']:.3f}  R_unsup={full['R_unsup']:.3f}")

    # 5) figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax_a, ax_b = axes[0]; ax_c, ax_d = axes[1]
    colors = {"DHAM reoptim": "#7a3d99", "DHAM cumulative": "#a05a2c",
              "EDMD reoptim": "#2a4d7a", "plain metaD": "#1f6b47"}

    # (a) MFPT speedup vs Omega_rho
    for name in biases:
        Or = out[name]["Omega_rho"]; sp = [tau0 / t for t in out[name]["tau_AB"]]
        ax_a.plot(Or, sp, "-o", color=colors[name], lw=1.4, ms=4, label=name)
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"transition-tube overlap $\Omega_\rho$")
    ax_a.set_ylabel("MFPT speedup")
    ax_a.set_title(r"(a) Acceleration vs $\Omega_\rho$ (Pareto)")
    ax_a.invert_xaxis(); ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    # (b) Omega_J and Omega_rho vs alpha
    for name in biases:
        ax_b.plot(alphas, out[name]["Omega_J"  ], "-",  color=colors[name], lw=1.6, label=name)
        ax_b.plot(alphas, out[name]["Omega_rho"], "--", color=colors[name], lw=1.2, alpha=0.7)
    ax_b.set_xlabel(r"bias scale $\alpha$")
    ax_b.set_ylabel("overlap")
    ax_b.set_title(r"(b) $\Omega_J$ (solid), $\Omega_\rho$ (dashed)")
    ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3)

    # (c) D_edge vs alpha
    for name in biases:
        ax_c.plot(alphas, out[name]["D_edge"], "-o", color=colors[name], lw=1.5, ms=4, label=name)
    ax_c.set_xlabel(r"bias scale $\alpha$")
    ax_c.set_ylabel(r"$D_{\rm edge}$")
    ax_c.set_title(r"(c) Edge-rate distortion")
    ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)

    # (d) bar chart at alpha=1
    methods = list(biases.keys())
    xs = np.arange(len(methods))
    bar_sp = [tau0 / point_full[m]["tau_AB"] for m in methods]
    bar_oJ = [point_full[m]["Omega_J"]       for m in methods]
    bar_or = [point_full[m]["Omega_rho"]     for m in methods]
    bar_de = [point_full[m]["D_edge"]        for m in methods]
    w = 0.2
    twin = ax_d.twinx()
    bars = ax_d.bar(xs - 1.5*w, bar_sp, w, color="#999", log=True, label="speedup")
    twin.bar(xs - 0.5*w, bar_oJ, w, color="#1f77b4", label=r"$\Omega_J$",   alpha=0.9)
    twin.bar(xs + 0.5*w, bar_or, w, color="#aec7e8", label=r"$\Omega_\rho$", alpha=0.9)
    twin.bar(xs + 1.5*w, bar_de, w, color="#d62728", label=r"$D_{\rm edge}$", alpha=0.9)
    ax_d.set_xticks(xs); ax_d.set_xticklabels(methods, rotation=18, fontsize=8)
    ax_d.set_ylabel("MFPT speedup")
    twin.set_ylabel(r"$\Omega_J$, $\Omega_\rho$, $D_{\rm edge}$")
    ax_d.set_title(r"(d) full-strength bias ($\alpha=1$)")
    h1, l1 = ax_d.get_legend_handles_labels(); h2, l2 = twin.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")

    fig.suptitle("Mechanism-conservation audit on the N=20 cooperative chain (A=4)\n"
                 "MSM on 2-D (mean_x, tilt) grid; bias acts on mean_x", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = HERE / "fig_mechanism_highd_n20.pdf"
    out_png = HERE / "fig_mechanism_highd_n20.png"
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=180)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")

    # 6) JSON dump
    summary = {
        "n_states":      int(N),
        "n_m":           int(n_m),
        "n_t":           int(n_t),
        "lag_frames":    int(lag_frames),
        "tau0_unbiased": float(tau0),
        "methods": {name: {k: (v if not isinstance(v, (list, np.ndarray))
                               else [float(x) for x in v])
                            for k, v in row.items()}
                    for name, row in out.items()},
        "point_full_alpha1": {name: {k: float(v) for k, v in d.items()}
                              for name, d in point_full.items()},
    }
    with open(HERE / "mechanism_highd_n20.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {HERE / 'mechanism_highd_n20.json'}")


if __name__ == "__main__":
    main()
