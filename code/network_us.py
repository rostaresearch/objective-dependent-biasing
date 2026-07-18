"""
Umbrella Sampling on Discrete-State Networks via DHAM
=====================================================

Reproduces Ahmed Fadhluddin's 2022 MSci experiments and extends them:
   - Tests several reaction coordinates (CVs): Basic / EigVec2-5 / History / Entropy
   - For each: place K harmonic umbrella windows along the chosen ordering
   - Build a *biased* rate matrix per window using detailed-balance preservation
   - Sample stochastic trajectories on the biased matrix
   - Recover unbiased FE via DHAM (Rosta & Hummer, JCTC 11 276 2015)
   - Score against the reference free energy via RMSD

Author: drafted for E. Rosta's contribution to LiveCoMS MSM Roundtable,
        2026-05-21.

Dependencies: numpy, scipy.  matplotlib for plotting (optional).

Usage:
    python network_us.py            # runs the demo
    >>> from network_us import *    # use as library
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as sla


# ---------------------------------------------------------------------
#  Reference-system utilities
# ---------------------------------------------------------------------

def stationary_from_rate(K: np.ndarray) -> np.ndarray:
    """Right null vector of K (column-stochastic rate matrix, columns sum to 0)."""
    n = K.shape[0]
    w, V = np.linalg.eig(K)
    idx = int(np.argmin(np.abs(w)))
    pi = np.real(V[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def free_energy(pi: np.ndarray, kT: float = 1.0) -> np.ndarray:
    pi = np.clip(pi, 1e-300, None)
    F = -kT * np.log(pi)
    return F - F.min()


def transition_matrix(K: np.ndarray, dt: float) -> np.ndarray:
    """Row-stochastic transition matrix from rate matrix:
       K is column-stochastic in the (K p) convention, columns sum to 0.
       T = expm(K * dt), then transpose for row-stochastic."""
    T = sla.expm(K * dt).T
    # Row-normalise to clean up numerical drift
    T = T / T.sum(axis=1, keepdims=True)
    return T


# ---------------------------------------------------------------------
#  Build a biased rate matrix preserving detailed balance
# ---------------------------------------------------------------------

def bias_rate_matrix(K: np.ndarray, u: np.ndarray, kT: float = 1.0,
                     bias_cap: float = 20.0) -> np.ndarray:
    """
    Bias K so that the new equilibrium is pi_bias_i = pi_i * exp(-u_i / kT) (up to norm).
    Detailed-balance-preserving form (analogue of Metropolis at the rate level):
        K^b_{ij} = K_{ij} * exp(-(u_i - u_j) / (2 kT))      for i != j
    `bias_cap` clips the exponent magnitude to avoid numerical overflow.
    """
    n = len(u)
    Kb = K.copy().astype(float)
    diff = (u[:, None] - u[None, :]) / (2.0 * kT)           # (n,n)
    diff = np.clip(diff, -bias_cap, bias_cap)
    factor = np.exp(-diff)
    Kb = K * factor
    np.fill_diagonal(Kb, 0.0)
    # Diagonal so columns sum to zero
    for j in range(n):
        Kb[j, j] = -Kb[:, j].sum()
    return Kb


# ---------------------------------------------------------------------
#  Reaction-coordinate (CV) options
# ---------------------------------------------------------------------

def cv_basic(n: int, **_) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def cv_eigvec(K: np.ndarray, k_eig: int, dt: float = 1.0) -> np.ndarray:
    """Return the k-th right eigenvector of the transition matrix T = expm(K dt).
       k_eig = 2 → second-largest eigenvalue's eigenvector (slowest non-stationary mode).
    """
    T = transition_matrix(K, dt)
    w, V = np.linalg.eig(T.T)        # left eigenvectors via T.T
    order = np.argsort(-np.real(w))  # descending
    v = np.real(V[:, order[k_eig - 1]])
    return _scale01(v)


def cv_history(visits: np.ndarray, **_) -> np.ndarray:
    """Cumulative visit-count ordering: states visited most → high CV."""
    visits = np.asarray(visits, dtype=float) + 1.0
    cum = np.cumsum(np.sort(visits))
    # rank-based, normalised
    rank = np.argsort(np.argsort(visits)).astype(float)
    return rank / (len(visits) - 1)


def cv_entropy(visits: np.ndarray, **_) -> np.ndarray:
    """Per-state information-content ordering: rare AND common states pulled to centre.
       p_i = (visits_i + 1) / sum.  s_i = -p_i log p_i (normalised by log n).
    """
    visits = np.asarray(visits, dtype=float) + 1.0
    p = visits / visits.sum()
    s = -p * np.log(p) / np.log(len(visits))
    return _scale01(s)


def _scale01(v: np.ndarray) -> np.ndarray:
    v = v - v.min()
    rng = v.max()
    return v / rng if rng > 0 else v


# ---------------------------------------------------------------------
#  DHAM for discrete-state networks
# ---------------------------------------------------------------------

def dham_unbias(count_mats: list[np.ndarray],
                bias_per_window: list[np.ndarray],
                kT: float = 1.0,
                symmetrise: bool = True,
                bias_cap: float = 20.0,
                floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Inputs
        count_mats     : list of (n,n) transition-count matrices, one per window
        bias_per_window: list of length-n bias arrays u^(k) (same units as kT)
    Returns
        T_unbias  (n,n)   row-stochastic estimate of unbiased transition matrix
        pi_unbias (n,)    stationary distribution of T_unbias
    """
    K = len(count_mats)
    n = count_mats[0].shape[0]
    Nij = np.sum(count_mats, axis=0).astype(float)
    if symmetrise:
        Nij = 0.5 * (Nij + Nij.T)
    # Laplace prior: avoid empty rows/columns in transition counts
    Nij = Nij + 1e-3
    Ni_per_window = np.array([cm.sum(axis=1) for cm in count_mats])  # (K, n)

    # Vectorised denominator: D[i,j] = sum_k Ni_per_window[k,i] * exp(-(u^k_j - u^k_i)/(2kT))
    # Add Laplace prior to per-window visit counts too so DHAM defines the reweighting
    # for state i even in windows that didn't actually visit i.
    Ni_per_window = Ni_per_window + 1e-3
    D = np.zeros((n, n))
    for k in range(K):
        u = bias_per_window[k]
        diff = (u[None, :] - u[:, None]) / (2.0 * kT)        # diff[i,j] = u_j - u_i
        diff = np.clip(diff, -bias_cap, bias_cap)
        D += Ni_per_window[k, :, None] * np.exp(-diff)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.where(D > 0, Nij / D, 0.0)

    # row-normalise; states never seen as source get uniform self-loop to keep T stochastic
    row_sums = T.sum(axis=1)
    bad = row_sums <= 0
    T[bad] = 0.0
    T[bad, np.arange(n)[bad]] = 1.0   # absorbing self-loop for empty rows
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / np.maximum(row_sums, floor)

    # stationary distribution (largest real eigenvalue)
    w, V = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    pi = np.abs(np.real(V[:, idx]))
    if pi.sum() <= 0:
        pi = np.ones(n) / n
    else:
        pi = pi / pi.sum()
    pi = np.clip(pi, floor, None)
    pi = pi / pi.sum()
    return T, pi


# ---------------------------------------------------------------------
#  Simulator: short Markov-chain trajectory on biased T
# ---------------------------------------------------------------------

def simulate_window(K_b: np.ndarray, dt: float, n_steps: int, start: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a discrete-time chain whose 1-step matrix is T_b = expm(K_b * dt).
       Returns (trajectory, count_matrix)."""
    n = K_b.shape[0]
    T_b = sla.expm(K_b * dt).T
    T_b = np.clip(T_b, 0.0, None)
    T_b = T_b / T_b.sum(axis=1, keepdims=True)
    traj = np.empty(n_steps + 1, dtype=int)
    traj[0] = start
    cdf = np.cumsum(T_b, axis=1)
    for s in range(n_steps):
        r = rng.random()
        traj[s + 1] = int(np.searchsorted(cdf[traj[s]], r))
    counts = np.zeros((n, n), dtype=int)
    for s in range(n_steps):
        counts[traj[s], traj[s + 1]] += 1
    return traj, counts


# ---------------------------------------------------------------------
#  Top-level: run one US experiment for a given CV
# ---------------------------------------------------------------------

def run_umbrella_sampling(K_ref: np.ndarray,
                          cv: np.ndarray,
                          n_windows: int = 10,
                          kappa: float = 50.0,
                          dt: float = 1.0,
                          steps_per_window: int = 2000,
                          kT: float = 1.0,
                          seed: int = 0) -> dict:
    """
    Place n_windows umbrella centres equally along cv[min..max], harmonic with force constant kappa.
    Simulate the biased dynamics in each window.  Recover FE via DHAM.
    Returns dict with keys: 'F_dham', 'F_ref', 'pi_dham', 'rmsd', 'counts', 'bias'
    """
    n = K_ref.shape[0]
    rng = np.random.default_rng(seed)

    centres = np.linspace(cv.min(), cv.max(), n_windows)
    count_mats, biases = [], []
    for k, x0 in enumerate(centres):
        u = kappa * (cv - x0) ** 2
        Kb = bias_rate_matrix(K_ref, u, kT=kT)
        start = int(np.argmin(np.abs(cv - x0)))
        _, counts = simulate_window(Kb, dt, steps_per_window, start, rng)
        count_mats.append(counts)
        biases.append(u)

    T_d, pi_d = dham_unbias(count_mats, biases, kT=kT)
    F_d = free_energy(pi_d, kT=kT)
    F_r = free_energy(stationary_from_rate(K_ref), kT=kT)
    # Align by mean to remove constant offset
    F_d_a = F_d - F_d.mean() + F_r.mean()
    rmsd = float(np.sqrt(np.mean((F_d_a - F_r) ** 2)))
    return dict(F_dham=F_d_a, F_ref=F_r, pi_dham=pi_d,
                rmsd=rmsd, counts=count_mats, bias=biases)


# ---------------------------------------------------------------------
#  Demo: small reference network with multimodal stationary distribution
# ---------------------------------------------------------------------

def make_demo_network(n: int = 12, seed: int = 0, n_basins: int = 3) -> np.ndarray:
    """Build a small connected random graph with a multimodal equilibrium.
       Returns a column-stochastic rate matrix K (units of 1/time)."""
    rng = np.random.default_rng(seed)
    # Random Erdos-Renyi adjacency, then make symmetric
    p_edge = 0.35
    A = (rng.random((n, n)) < p_edge).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    # ensure connectivity by adding a spanning chain
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    # impose a 3-basin equilibrium: assign each state a free-energy depth
    F_target = np.zeros(n)
    centres = np.linspace(0, n - 1, n_basins)
    for c in centres:
        F_target -= 2.0 * np.exp(-(np.arange(n) - c) ** 2 / 4.0)
    F_target -= F_target.min()
    pi_target = np.exp(-F_target)
    pi_target /= pi_target.sum()
    # construct symmetric K with detailed balance: K_ij = A_ij * sqrt(pi_j / pi_i) * k0
    K = np.zeros((n, n))
    k0 = 0.5
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > 0:
                K[i, j] = k0 * np.sqrt(pi_target[i] / pi_target[j])  # rate i<-j
    for j in range(n):
        K[j, j] = -(K[:, j].sum() - K[j, j])
    return K


def unbiased_baseline(K: np.ndarray, dt: float, n_steps: int, seed: int = 0,
                      kT: float = 1.0) -> float:
    """Run an unbiased trajectory and return RMSD of the empirical pi vs reference."""
    rng = np.random.default_rng(seed)
    n = K.shape[0]
    _, counts = simulate_window(K, dt, n_steps, start=0, rng=rng)
    visits = counts.sum(axis=1) + 1
    pi_emp = visits / visits.sum()
    F_emp = free_energy(pi_emp, kT=kT)
    F_ref = free_energy(stationary_from_rate(K), kT=kT)
    F_emp_a = F_emp - F_emp.mean() + F_ref.mean()
    return float(np.sqrt(np.mean((F_emp_a - F_ref) ** 2)))


def deep_basin_network(n: int = 40, n_basins: int = 4, depth: float = 6.0,
                       p_edge: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Multi-basin random graph.  Returns (K, F_target)."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p_edge).astype(float)
    A = np.triu(A, 1); A = A + A.T
    # spanning chain so it's always connected
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    F_target = np.zeros(n)
    centres = np.linspace(2, n - 3, n_basins)
    for c in centres:
        F_target -= depth * np.exp(-(np.arange(n) - c) ** 2 / 4.0)
    F_target -= F_target.min()
    pi_t = np.exp(-F_target); pi_t /= pi_t.sum()
    K = np.zeros((n, n))
    k0 = 0.5
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > 0:
                K[i, j] = k0 * np.sqrt(pi_t[i] / pi_t[j])
    for j in range(n):
        K[j, j] = -(K[:, j].sum() - K[j, j])
    return K, F_target


def sanity_check_dham(K: np.ndarray, n_steps: int = 30000, seed: int = 0) -> float:
    """DHAM with zero bias on a single 'window' must recover the true FE."""
    rng = np.random.default_rng(seed)
    n = K.shape[0]
    _, counts = simulate_window(K, dt=1.0, n_steps=n_steps, start=0, rng=rng)
    T_d, pi_d = dham_unbias([counts], [np.zeros(n)])
    F_ref = free_energy(stationary_from_rate(K))
    F_est = free_energy(pi_d)
    F_est = F_est - F_est.mean() + F_ref.mean()
    return float(np.sqrt(np.mean((F_est - F_ref) ** 2)))


def demo():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 68)
    print("  Network umbrella-sampling test -- deep 4-basin random graph")
    print("=" * 68)
    K, _ = deep_basin_network(n=50, n_basins=3, depth=8.0, p_edge=0.10, seed=1)
    F_ref = free_energy(stationary_from_rate(K))
    n_states = len(F_ref)
    print(f"  States: {n_states},  FE range: {F_ref.max() - F_ref.min():.2f} kT")

    # 0) DHAM sanity check on unbiased trajectory: should give ~0 RMSD with enough samples
    rmsd_sanity = sanity_check_dham(K, n_steps=60000, seed=7)
    print(f"  DHAM sanity check (zero bias, 60k steps): RMSD = {rmsd_sanity:7.3f} kT")

    # 1) Unbiased baseline (same total step budget as one US run)
    n_windows = 10
    steps_per_window = 1500
    kappa = 2.0                  # CV in [0,1], so dU at max = kappa = 2 kT
    total_steps = n_windows * steps_per_window
    rmsd_un = unbiased_baseline(K, dt=1.0, n_steps=total_steps, seed=42)
    print(f"  Unbiased baseline ({total_steps} steps) : RMSD = {rmsd_un:7.3f} kT")

    cvs_static = {
        'Basic':    cv_basic(len(F_ref)),
        'EigVec2':  cv_eigvec(K, k_eig=2),
        'EigVec3':  cv_eigvec(K, k_eig=3),
        'EigVec4':  cv_eigvec(K, k_eig=4),
        'EigVec5':  cv_eigvec(K, k_eig=5),
    }

    # Replicate-averaged RMSDs
    n_rep = 5
    print("\n  --- Static CVs ---")
    visits_accum = np.zeros(len(F_ref))
    static_results = {}
    for name, cv in cvs_static.items():
        rmsds = []
        for r in range(n_rep):
            res = run_umbrella_sampling(K, cv, n_windows=n_windows, kappa=kappa,
                                        dt=1.0, steps_per_window=steps_per_window, seed=100 + r)
            rmsds.append(res['rmsd'])
            visits_accum += sum(cm.sum(axis=1) for cm in res['counts'])
        m, s = float(np.mean(rmsds)), float(np.std(rmsds))
        static_results[name] = (m, s)
        print(f"    {name:9s} : {m:6.3f} +/- {s:5.3f} kT")

    print("\n  --- Adaptive CVs (seeded from accumulated history) ---")
    adapt_results = {}
    for name, builder in [('History', cv_history), ('Entropy', cv_entropy)]:
        cv = builder(visits_accum)
        rmsds = []
        for r in range(n_rep):
            res = run_umbrella_sampling(K, cv, n_windows=n_windows, kappa=kappa,
                                        dt=1.0, steps_per_window=steps_per_window,
                                        seed=200 + r)
            rmsds.append(res['rmsd'])
        m, s = float(np.mean(rmsds)), float(np.std(rmsds))
        adapt_results[name] = (m, s)
        print(f"    {name:9s} : {m:6.3f} +/- {s:5.3f} kT")

    # Pretty summary
    print("\n  -- Summary (lower = better) --")
    print(f"    {'Unbiased':9s} : {rmsd_un:6.3f}  (no biasing)")
    for name, (m, s) in {**static_results, **adapt_results}.items():
        flag = " *" if m < rmsd_un * 0.6 else ""
        print(f"    {name:9s} : {m:6.3f} +/- {s:5.3f} kT{flag}")


if __name__ == '__main__':
    demo()
