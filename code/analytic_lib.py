"""
Analytic library for biased kinetic networks.

Adopted from E. Rosta's draft module, with minor reorganisation:
  - committor_K defined before reactive_flux_rate_K
  - added umbrella_family_metrics (referenced in original demo)
  - small numerical safety on solves

Core operations are CLOSED FORM on the continuous-time row-generator K0:
  - tilt by potential b_i = beta U_i:    K_b[i,j] = K0[i,j] * exp(-(b_j-b_i)/2)
  - biased stationary:                    pi_b[i] propto pi0[i] * exp(-b_i)
  - MFPT(start->target), spectral gap, committor, reactive capacity
  - integrated autocorrelation time via Poisson eq:  -K g = f - <f>_pi
  - per-window variance of time-average given sample_time

No DHAM, no Monte Carlo.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm, eig
from scipy.optimize import minimize

EPS = 1e-14


# ---------- utilities ---------------------------------------------------------

def normalize(v: ArrayLike) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("Cannot normalize vector with non-positive sum.")
    return v / s


def stationary_distribution_from_P(P: np.ndarray) -> np.ndarray:
    w, vl = eig(P.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(vl[:, idx])
    if pi.sum() < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    return normalize(pi)


def stationary_distribution_from_K(K: np.ndarray) -> np.ndarray:
    w, vl = eig(K.T)
    idx = int(np.argmin(np.abs(w)))
    pi = np.real(vl[:, idx])
    if pi.sum() < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    return normalize(pi)


def check_row_generator(K: np.ndarray, atol: float = 1e-9) -> None:
    off = K.copy()
    np.fill_diagonal(off, 0.0)
    if np.any(off < -atol):
        raise ValueError("K has negative off-diagonal rates.")
    if not np.allclose(K.sum(axis=1), 0.0, atol=atol):
        raise ValueError("K rows do not sum to zero.")


def check_row_stochastic(P: np.ndarray, atol: float = 1e-9) -> None:
    if np.any(P < -atol):
        raise ValueError("P has negative entries.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=atol):
        raise ValueError("P rows do not sum to one.")


# ---------- network construction ---------------------------------------------

def grid_2d_generator(nx: int = 30, ny: int = 20,
                      barrier_height: float = 8.0,
                      bottleneck: bool = True,
                      base_rate: float = 1.0
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])
    x = coords[:, 0]; y = coords[:, 1]
    F = barrier_height * (x**2 - 1.0)**2 + 2.0 * y**2
    if bottleneck:
        F += 6.0 * np.exp(-(x/0.20)**2) * (1.0 - np.exp(-(y/0.18)**2))
    n = nx * ny
    K = np.zeros((n, n))
    def idx(ix, iy): return ix * ny + iy
    for ix in range(nx):
        for iy in range(ny):
            i = idx(ix, iy)
            for jx, jy in ((ix-1, iy), (ix+1, iy), (ix, iy-1), (ix, iy+1)):
                if 0 <= jx < nx and 0 <= jy < ny:
                    j = idx(jx, jy)
                    K[i, j] = base_rate * np.exp(-0.5 * (F[j] - F[i]))
    np.fill_diagonal(K, -K.sum(axis=1))
    return K, coords, F


# ---------- biasing -----------------------------------------------------------

def tilt_generator(K0: np.ndarray, b: ArrayLike) -> np.ndarray:
    """K_b[i,j] = K0[i,j] * exp(-(b_j - b_i)/2)  for i != j."""
    K0 = np.asarray(K0, dtype=float)
    b = np.asarray(b, dtype=float)
    check_row_generator(K0)
    db = b[None, :] - b[:, None]            # db[i,j] = b_j - b_i
    Kb = K0 * np.exp(-0.5 * db)
    np.fill_diagonal(Kb, 0.0)
    np.fill_diagonal(Kb, -Kb.sum(axis=1))
    return Kb


def biased_pi_from_reference(pi0: np.ndarray, b: np.ndarray) -> np.ndarray:
    logw = np.log(pi0 + EPS) - b
    logw -= logw.max()
    return normalize(np.exp(logw))


def harmonic_bias(cv: ArrayLike, center: float, kappa: float) -> np.ndarray:
    cv = np.asarray(cv, dtype=float)
    return 0.5 * kappa * (cv - center) ** 2


def linear_combination_bias(features: np.ndarray, theta: ArrayLike) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    b = features @ theta
    return b - b.mean()


# ---------- kinetic observables ----------------------------------------------

def mfpt_K(K: np.ndarray, start: Sequence[int], target: Sequence[int]) -> float:
    check_row_generator(K)
    n = K.shape[0]
    target = np.array(sorted(set(target)), dtype=int)
    start = np.array(sorted(set(start)), dtype=int)
    mask = np.ones(n, dtype=bool); mask[target] = False
    non = np.where(mask)[0]
    KNN = K[np.ix_(non, non)]
    tau = np.linalg.solve(-KNN, np.ones(len(non)))
    full = np.zeros(n); full[non] = tau
    return float(np.mean(full[start]))


def spectral_gap_K(K: np.ndarray) -> float:
    """Continuous-time spectral gap: -Re(lambda_2)."""
    check_row_generator(K)
    vals = eig(K, left=False, right=False)
    vals = np.sort(np.real(vals))[::-1]
    return float(-vals[1])


def implied_timescale_K(K: np.ndarray) -> float:
    return float(1.0 / max(spectral_gap_K(K), EPS))


def right_eigvec2_K(K: np.ndarray) -> np.ndarray:
    vals, vecs = eig(K)
    order = np.argsort(np.real(vals))[::-1]
    v = np.real(vecs[:, order[1]])
    return (v - v.mean()) / (v.std() + EPS)


def committor_K(K: np.ndarray, A: Sequence[int], B: Sequence[int]) -> np.ndarray:
    check_row_generator(K)
    n = K.shape[0]
    A = np.array(sorted(set(A)), dtype=int)
    B = np.array(sorted(set(B)), dtype=int)
    fixed = np.zeros(n, dtype=bool); fixed[A] = True; fixed[B] = True
    free = np.where(~fixed)[0]
    q = np.zeros(n); q[B] = 1.0
    Kff = K[np.ix_(free, free)]
    KfB = K[np.ix_(free, B)]
    rhs = -KfB @ np.ones(len(B))
    q[free] = np.linalg.solve(Kff, rhs)
    return np.clip(q, 0.0, 1.0)


def reactive_flux_rate_K(K: np.ndarray, pi: np.ndarray,
                         basin_A: Sequence[int], basin_B: Sequence[int]
                         ) -> Dict[str, float | np.ndarray]:
    """TPT capacity & two-state rates."""
    check_row_generator(K)
    pi = normalize(pi)
    A = np.array(sorted(set(basin_A)), dtype=int)
    q = committor_K(K, basin_A, basin_B)
    n = K.shape[0]
    A_mask = np.zeros(n, dtype=bool); A_mask[A] = True
    cap = 0.0
    for i in A:
        for j in range(n):
            if not A_mask[j] and i != j:
                cap += pi[i] * K[i, j] * q[j]
    pA = float(np.sum(pi[A])); pB = float(np.sum(pi[basin_B]))
    kAB = cap / max(pA, EPS); kBA = cap / max(pB, EPS)
    return dict(q=q, capacity=float(cap), pA=pA, pB=pB,
                kAB=float(kAB), kBA=float(kBA),
                mfptAB_metastable_approx=float(1.0 / max(kAB, EPS)),
                mfptBA_metastable_approx=float(1.0 / max(kBA, EPS)))


def main_path_weights_from_committor(q: np.ndarray, pi0: np.ndarray,
                                     mode: str = "tube") -> np.ndarray:
    pi0 = normalize(pi0); q = np.asarray(q, dtype=float)
    if mode == "tube":
        w = pi0 * q * (1.0 - q)
    elif mode == "path":
        w = q * (1.0 - q)
    elif mode == "basins_and_tube":
        w = pi0 * (0.25 + q * (1.0 - q))
    else:
        raise ValueError(mode)
    return w / max(np.sum(w), EPS)


def basin_free_energy_barrier(pi: np.ndarray, basin_A: Sequence[int],
                              basin_B: Sequence[int], barrier_set: Sequence[int]
                              ) -> Dict[str, float]:
    pi = normalize(pi)
    pA = float(np.sum(pi[basin_A])); pB = float(np.sum(pi[basin_B]))
    pS = float(np.sum(pi[barrier_set]))
    return dict(pA=pA, pB=pB, pBarrier=pS,
                DeltaF_A_to_barrier=float(-np.log(max(pS, EPS)) + np.log(max(pA, EPS))),
                DeltaF_B_to_barrier=float(-np.log(max(pS, EPS)) + np.log(max(pB, EPS))))


# ---------- autocorrelation via Poisson equation -----------------------------

def integrated_autocorr_time_K(K: np.ndarray, pi: np.ndarray,
                               f: np.ndarray) -> float:
    """tau_int(f) = <g, f - <f>>_pi / Var_pi(f),  with -K g = f - <f>_pi."""
    f = np.asarray(f, dtype=float)
    pi = normalize(pi)
    fc = f - np.dot(pi, f)
    var = float(np.dot(pi, fc * fc))
    if var < EPS:
        return 0.0
    A = -K.copy()
    rhs = fc.copy()
    A[-1, :] = pi
    rhs[-1] = 0.0
    try:
        g = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        g = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return float(np.dot(pi, g * fc) / var)


def window_equilibrium_variance(Kb: np.ndarray, pib: np.ndarray,
                                observable: np.ndarray, sample_time: float
                                ) -> float:
    """Approx variance of time-average estimator: 2 tau_int(f) Var_pi(f) / T."""
    f = np.asarray(observable, dtype=float)
    mean = float(np.dot(pib, f))
    var = float(np.dot(pib, (f - mean) ** 2))
    tau = integrated_autocorr_time_K(Kb, pib, f)
    return float(2.0 * tau * var / max(sample_time, EPS))


def reweighting_ess(pi0: np.ndarray, pib: np.ndarray, sample_count: float) -> float:
    w = pi0 / np.maximum(pib, EPS)
    ew = float(np.dot(pib, w))
    ew2 = float(np.dot(pib, w * w))
    return float(sample_count * ew * ew / max(ew2, EPS))


def overlap_matrix(distributions: Sequence[np.ndarray]) -> np.ndarray:
    Pis = [normalize(p) for p in distributions]
    m = len(Pis)
    O = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            O[a, b] = float(np.sum(np.sqrt(Pis[a] * Pis[b])))
    return O


# ---------- umbrella family analysis -----------------------------------------

def umbrella_family_metrics(
    K0: np.ndarray,
    cv: np.ndarray,
    centers: np.ndarray,
    kappa: float,
    pi0: Optional[np.ndarray] = None,
    sample_time_per_window: float = 1.0,
    state_weights: Optional[np.ndarray] = None,
    barrier_set: Optional[Sequence[int]] = None,
    basin_A: Optional[Sequence[int]] = None,
    basin_B: Optional[Sequence[int]] = None,
    eta_floor: float = 1e-9,
) -> Dict[str, np.ndarray | float]:
    """
    Analytic Poisson/local-coverage approximation of umbrella-sampling
    estimator variance for a family of W harmonic windows.

    For each window k we compute analytically:
       pi^{(k)}_i           biased stationary
       tau_k                slowest implied timescale (1 / spectral_gap(K_b))
       N_k^{eff} = T_k / (2 tau_k)    effective independent samples in window k

    Then approximate
       Var(pi_hat_i) / pi_i^2  ≈  1 / sum_k  N_k^{eff} * pi^{(k)}_i

    Unbiased baseline (W=1, no bias) has  pi^{(1)} = pi0,
    so Var/pi^2 ≈ 1/(N_eff_total * pi_i).

    The unbiased autocorrelation time is taken from K0 itself, so its
    N_eff = T_total / (2 tau_0).

    This formula is *not* exact MBAR covariance; it is the coverage
    approximation that ignores correlations across states and the
    nontrivial weight structure of the MBAR estimator.  It is, however,
    sharp enough to rank protocols.
    """
    if pi0 is None:
        pi0 = stationary_distribution_from_K(K0)
    n = K0.shape[0]
    W = len(centers)

    pis, timescales, gaps, esses = [], [], [], []
    biases = []
    N_eff_per_window = []
    T_per_window = float(sample_time_per_window)

    for c in centers:
        b = harmonic_bias(cv, center=c, kappa=kappa)
        Kb = tilt_generator(K0, b)
        pib = biased_pi_from_reference(pi0, b)
        pis.append(pib); biases.append(b)
        gp = spectral_gap_K(Kb)
        tau_k = 1.0 / max(gp, EPS)
        gaps.append(gp); timescales.append(tau_k)
        N_k_eff = T_per_window / max(2.0 * tau_k, EPS)
        N_eff_per_window.append(N_k_eff)
        esses.append(reweighting_ess(pi0, pib, N_k_eff))

    pis = np.asarray(pis)
    N_eff_per_window = np.asarray(N_eff_per_window)
    timescales = np.asarray(timescales)

    # Coverage approximation, effective-sample version:
    #     I_i = sum_k N_k^eff * pi^{(k)}_i
    #     Var(pi_hat_i)/pi_i^2  ≈  1/I_i
    I_i = (N_eff_per_window[:, None] * pis).sum(axis=0)
    eta_eff = I_i / max(N_eff_per_window.mean(), EPS)       # rescaled "effective coverage"
    var_pi_rel_US = 1.0 / np.maximum(I_i, eta_floor)

    # Unbiased baseline at the SAME total wall-clock budget
    T_total = T_per_window * W
    tau0 = 1.0 / max(spectral_gap_K(K0), EPS)
    N_eff_unbiased = T_total / max(2.0 * tau0, EPS)
    var_pi_rel_un = 1.0 / np.maximum(N_eff_unbiased * pi0, eta_floor)

    out: Dict[str, np.ndarray | float] = dict(
        centers=np.asarray(centers),
        biases=np.asarray(biases),
        pi_windows=pis,
        gap=np.asarray(gaps),
        timescale=timescales,
        N_eff_per_window=N_eff_per_window,
        N_eff_total_US=float(N_eff_per_window.sum()),
        N_eff_unbiased=float(N_eff_unbiased),
        tau_unbiased=float(tau0),
        neighbor_overlap=np.asarray(
            [float(np.sum(np.sqrt(pis[k] * pis[k+1]))) for k in range(W - 1)]
        ),
        reweighting_ess_to_pi0=np.asarray(esses),
        total_reweighting_ess_to_pi0=float(np.sum(esses)),
        kappa=float(kappa),
        W=W,
        I_i=I_i,
        var_pi_relative=var_pi_rel_US,
        var_pi_relative_unbiased=var_pi_rel_un,
    )

    if barrier_set is not None and basin_A is not None and basin_B is not None:
        S = np.array(list(barrier_set))
        p_dagger = float(np.sum(pi0[S]))
        var_pS_US = float(np.sum(pi0[S] ** 2 * var_pi_rel_US[S]))
        var_pS_un = float(np.sum(pi0[S] ** 2 * var_pi_rel_un[S]))
        out["p_dagger"] = p_dagger
        out["E_barrier_US"] = var_pS_US / max(p_dagger ** 2, EPS)
        out["E_barrier_unbiased"] = var_pS_un / max(p_dagger ** 2, EPS)
        out["barrier_speedup"] = out["E_barrier_unbiased"] / max(out["E_barrier_US"], EPS)

        # Coverage failure diagnostic: fraction of p_dagger sitting on states
        # with effectively zero coverage.  I_i is an expected effective *count*
        # (sum_i I_i = total effective samples), so "one effective observation"
        # is I_i = 1; a state counts as effectively unsampled when I_i < 1.
        # (Previously 1/<N_eff>, which was dimensionally wrong -- it compared a
        # count to an inverse-count and scaled with the budget.)
        epsilon = 1.0
        uncovered = I_i[S] < epsilon
        out["coverage_failure_pdagger"] = float(
            (pi0[S] * uncovered).sum() / max(p_dagger, EPS)
        )

        q = committor_K(K0, basin_A, basin_B)
        rho = main_path_weights_from_committor(q, pi0, mode="tube")
        E_rate_US = float(np.sum(rho * var_pi_rel_US))
        E_rate_un = float(np.sum(rho * var_pi_rel_un))
        out["E_rate_US"] = E_rate_US
        out["E_rate_unbiased"] = E_rate_un
        out["rate_speedup"] = E_rate_un / max(E_rate_US, EPS)
        out["committor"] = q
        out["rate_relevance_weights"] = rho

        # Coverage failure for rate tube as well
        rho_uncovered = (rho * (I_i < epsilon)).sum()
        out["coverage_failure_rate"] = float(rho_uncovered)

    return out


def oracle_flatten_bias(pi0: np.ndarray) -> np.ndarray:
    """The exact analytic flattening bias:  beta * u_i = log pi_i.
       Then pi^b_i ∝ pi_i exp(-beta u_i) = pi_i / pi_i = const, i.e. uniform.

       Note the sign:  u_i = +kT log pi_i = -F_i,  i.e. the bias compensates the
       free-energy landscape state-by-state."""
    return np.log(np.clip(pi0, EPS, None))


# ---------- optimisation of bias ---------------------------------------------

@dataclass
class BiasOptResult:
    theta: np.ndarray
    objective: float
    mfpt: float
    timescale: float
    gap: float
    bias: np.ndarray
    success: bool
    message: str


def optimize_bias_on_generator(K0, features, objective,
                               start=None, target=None,
                               l2=1e-3, max_abs_bias=8.0,
                               theta0=None) -> BiasOptResult:
    check_row_generator(K0)
    X = np.asarray(features, dtype=float)
    n, m = X.shape
    if theta0 is None: theta0 = np.zeros(m)
    if objective == "mfpt" and (start is None or target is None):
        raise ValueError("mfpt requires start and target")

    def make_bias(theta):
        b = linear_combination_bias(X, theta)
        return np.clip(b, -max_abs_bias, max_abs_bias)

    def fun(theta):
        b = make_bias(theta)
        Kb = tilt_generator(K0, b)
        if objective == "mfpt":
            val = mfpt_K(Kb, start, target)
        elif objective == "timescale":
            val = implied_timescale_K(Kb)
        else:
            raise ValueError(objective)
        return float(val + l2 * float(np.dot(theta, theta)))

    res = minimize(fun, theta0, method="Nelder-Mead",
                   options=dict(maxiter=600, xatol=1e-5, fatol=1e-5))
    b = make_bias(res.x); Kb = tilt_generator(K0, b)
    gap = spectral_gap_K(Kb); ts = 1.0 / max(gap, EPS)
    mfpt = np.nan if (start is None or target is None) else mfpt_K(Kb, start, target)
    return BiasOptResult(theta=res.x, objective=float(res.fun),
                         mfpt=float(mfpt), timescale=float(ts), gap=float(gap),
                         bias=b, success=bool(res.success), message=str(res.message))
