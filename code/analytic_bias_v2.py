"""
Analytic biased-network exploration for the MSM Roundtable contribution.

All computations are closed-form linear algebra on the biased rate / transition
matrix.  No Monte Carlo, no DHAM.

We support two operator types:
    - K  : row-stochastic *rate* matrix (rows sum to 0).  Continuous time.
    - T  : row-stochastic *transition* matrix (rows sum to 1).  Discrete time
            with lag tau (e.g. an MD-derived MSM).
Both satisfy detailed balance with stationary pi.

Biased operator (sign convention: K_ij or T_ij = i -> j):
    K^b_ij = K_ij * exp(-beta/2 * (u_j - u_i))   for i != j;  K^b_ii fills row.
    T^b_ij = T_ij * exp(-beta/2 * (u_j - u_i))   for i != j;  T^b_ii = 1 - row_sum.
Both produce DB w.r.t.   pi^b_i ∝ pi_i exp(-beta u_i).

Observables computed analytically on the biased operator:
    - lambda_2 (slowest non-stationary eigenvalue) — global relaxation
    - MFPT(A -> B)
    - committor q_AB

We compare reaction-coordinate (CV) choices, bias functional forms,
and optimise the bias parameters.

Drafted for E. Rosta's MSM Roundtable contribution.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize


# =====================================================================
#  Operators: stationary, lambda_2, MFPT, committor
# =====================================================================

def stationary_T(T: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    pi = np.abs(np.real(V[:, idx]))
    return pi / pi.sum()


def lambda2_T(T: np.ndarray) -> float:
    """Second-largest real eigenvalue of T.  -1/log(lambda_2) = relax time in lagtimes."""
    w = np.real(np.linalg.eigvals(T))
    w_sorted = np.sort(w)[::-1]
    return float(w_sorted[1])


def mfpt_T(T: np.ndarray, target: int) -> np.ndarray:
    """MFPT (in lagtimes) from every state to `target`.
       (I - T_int) m = 1, m_target = 0."""
    n = T.shape[0]
    keep = np.array([i for i in range(n) if i != target])
    A = np.eye(len(keep)) - T[np.ix_(keep, keep)]
    rhs = np.ones(len(keep))
    m_sub = np.linalg.solve(A, rhs)
    m = np.zeros(n)
    m[keep] = m_sub
    return m


def committor_T(T: np.ndarray, A: int, B: int) -> np.ndarray:
    """Forward committor q_i = P(hit B before A | start in i)."""
    n = T.shape[0]
    inter = np.array([i for i in range(n) if i not in (A, B)])
    M = np.eye(len(inter)) - T[np.ix_(inter, inter)]
    rhs = T[np.ix_(inter, [B])].flatten()
    q_int = np.linalg.solve(M, rhs)
    q = np.zeros(n)
    q[A] = 0.0; q[B] = 1.0; q[inter] = q_int
    return q


# =====================================================================
#  Biasing
# =====================================================================

def bias_T(T: np.ndarray, u: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """T^b_ij = T_ij * exp(-beta/2 * (u_j - u_i)),  diagonal fixes row sum."""
    du = (u[None, :] - u[:, None])               # du[i,j] = u_j - u_i
    factor = np.exp(-0.5 * beta * du)
    Tb = T * factor
    np.fill_diagonal(Tb, 0.0)
    row_sums_off = Tb.sum(axis=1)
    diag = 1.0 - row_sums_off
    # If bias is too strong for a row, diag goes negative -> truncate (rare for moderate bias).
    diag = np.clip(diag, 0.0, 1.0)
    np.fill_diagonal(Tb, diag)
    # Final safety: renormalise rows (no-op if math worked)
    Tb = Tb / Tb.sum(axis=1, keepdims=True)
    return Tb


def biased_pi(pi: np.ndarray, u: np.ndarray, beta: float = 1.0) -> np.ndarray:
    lw = np.log(np.clip(pi, 1e-300, None)) - beta * u
    lw -= lw.max()
    w = np.exp(lw)
    return w / w.sum()


# =====================================================================
#  Reaction coordinates (CVs)  — all return arrays scaled to [0, 1]
# =====================================================================

def _scale01(v: np.ndarray) -> np.ndarray:
    v = v - v.min()
    rng = v.max()
    return v / rng if rng > 0 else v


def cv_basic(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def cv_eigvec(T: np.ndarray, k_eig: int) -> np.ndarray:
    w, V = np.linalg.eig(T.T)
    order = np.argsort(-np.real(w))
    v = np.real(V[:, order[k_eig - 1]])
    return _scale01(v)


def cv_committor(T: np.ndarray, A: int, B: int) -> np.ndarray:
    return _scale01(committor_T(T, A, B))


def cv_mfpt_to_B(T: np.ndarray, B: int) -> np.ndarray:
    """The MFPT to B itself, used as a CV.  Smaller MFPT -> closer to B."""
    m = mfpt_T(T, B)
    # Reverse so high CV = close to B
    return _scale01(m.max() - m)


def cv_logpi(pi: np.ndarray) -> np.ndarray:
    return _scale01(-np.log(np.clip(pi, 1e-300, None)))


def cv_eigvec_pair(T: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Linear combination alpha*v2 + beta*v3 (optimisable)."""
    w, V = np.linalg.eig(T.T)
    order = np.argsort(-np.real(w))
    v2 = np.real(V[:, order[1]])
    v3 = np.real(V[:, order[2]])
    return _scale01(alpha * v2 + beta * v3)


# =====================================================================
#  Bias functional forms (return per-state u_i given parameters)
# =====================================================================

def bf_harmonic(cv: np.ndarray, kappa: float, x0: float) -> np.ndarray:
    """Centered harmonic well: u_i = kappa (x_i - x0)^2"""
    return kappa * (cv - x0) ** 2


def bf_inverse_well(cv: np.ndarray, kappa: float, x0: float) -> np.ndarray:
    """Inverted harmonic: pushes states AWAY from x0.  Anchored so min(u) = 0."""
    u = -kappa * (cv - x0) ** 2
    return u - u.min()


def bf_linear(cv: np.ndarray, alpha: float) -> np.ndarray:
    """Constant force / metadynamics-style linear pull, anchored at u_min = 0."""
    u = alpha * cv
    return u - u.min()


def bf_two_well(cv: np.ndarray, kappa: float, xA: float, xB: float) -> np.ndarray:
    """Pins the two endpoints (favours staying near either A or B)."""
    u = np.minimum(kappa * (cv - xA) ** 2, kappa * (cv - xB) ** 2)
    return u


def bf_partial_flatten(pi: np.ndarray, gamma: float) -> np.ndarray:
    """u_i = gamma * (-kT log pi_i) -> at gamma=1, biased landscape is flat.
       Acts as the 'analytical ceiling' for any bias that knows pi."""
    F = -np.log(np.clip(pi, 1e-300, None))
    return gamma * (F - F.min())


# =====================================================================
#  Optimisation
# =====================================================================

def optimise_bias(T: np.ndarray, A: int, B: int,
                  cv_fn, bf_fn, x0_params: list[float],
                  objective: str = 'mfpt',
                  beta: float = 1.0,
                  u_budget: float = 5.0) -> dict:
    """
    Optimise free parameters of bf_fn(cv, *params) to minimise MFPT(A->B) or
    minimise lambda_2 (maximise |log lambda_2|).

    All bias profiles are *rescaled* so max(u) - min(u) == u_budget, ensuring
    every CV/form is allowed the same energy range and we measure efficiency,
    not raw bias magnitude.
    """
    cv = cv_fn()
    pi = stationary_T(T)

    def rescale_u(u: np.ndarray) -> np.ndarray:
        rng = u.max() - u.min()
        if rng <= 0:
            return u
        return (u - u.min()) * (u_budget / rng)

    def objective_fn(params):
        try:
            u_raw = bf_fn(cv, *params)
        except Exception:
            return 1e12
        if not np.all(np.isfinite(u_raw)):
            return 1e12
        u = rescale_u(u_raw)
        Tb = bias_T(T, u, beta=beta)
        if not np.all(np.isfinite(Tb)) or np.any(Tb < -1e-9):
            return 1e12
        if objective == 'mfpt':
            try:
                m = mfpt_T(Tb, B)[A]
                return float(m) if np.isfinite(m) and m > 0 else 1e12
            except Exception:
                return 1e12
        elif objective == 'lambda2':
            l2 = lambda2_T(Tb)
            if l2 <= 0 or l2 >= 1:
                return 0.0
            # we want to MINIMISE lambda_2 (closer to 0 = faster relaxation)
            return float(l2)
        raise ValueError(objective)

    res = minimize(objective_fn, x0_params, method='Nelder-Mead',
                   options=dict(xatol=1e-4, fatol=1e-5, maxiter=400))
    u_raw_best = bf_fn(cv, *res.x)
    u_best = rescale_u(u_raw_best)
    Tb_opt = bias_T(T, u_best, beta=beta)
    return dict(params=res.x, objective_value=res.fun,
                u=u_best, T_biased=Tb_opt,
                mfpt=float(mfpt_T(Tb_opt, B)[A]),
                lambda2=float(lambda2_T(Tb_opt)),
                u_budget=u_budget)


# =====================================================================
#  Per-state u optimisation (theoretical ceiling)
# =====================================================================

def optimise_per_state(T: np.ndarray, A: int, B: int,
                       u_budget: float = 5.0,
                       objective: str = 'mfpt',
                       beta: float = 1.0,
                       seed: int = 0,
                       n_random_starts: int = 3) -> dict:
    """Optimise each u_i independently subject to ||u||_∞ <= u_budget.
       Multi-start L-BFGS-B for robustness."""
    n = T.shape[0]
    rng = np.random.default_rng(seed)

    def obj_fn(u):
        Tb = bias_T(T, u, beta=beta)
        if objective == 'mfpt':
            try:
                m = mfpt_T(Tb, B)[A]
                return float(m) if np.isfinite(m) and m > 0 else 1e12
            except Exception:
                return 1e12
        elif objective == 'lambda2':
            l2 = lambda2_T(Tb)
            if l2 <= 0 or l2 >= 1:
                return 0.0
            return float(np.log(l2))    # closer to 0 (less negative) is worse
        raise ValueError(objective)

    best_val = np.inf
    best_u = None
    starts = [np.zeros(n)] + [rng.uniform(-u_budget/2, u_budget/2, n)
                              for _ in range(n_random_starts - 1)]
    for u0 in starts:
        bounds = [(-u_budget, u_budget)] * n
        res = minimize(obj_fn, u0, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=200, gtol=1e-6))
        if res.fun < best_val:
            best_val = res.fun
            best_u = res.x
    Tb_opt = bias_T(T, best_u, beta=beta)
    return dict(u=best_u, objective_value=best_val,
                T_biased=Tb_opt,
                mfpt=mfpt_T(Tb_opt, B)[A],
                lambda2=lambda2_T(Tb_opt))


# =====================================================================
#  Network builders
# =====================================================================

def threebasin_K(n: int = 60, n_basins: int = 3, depth: float = 6.0,
                 p_edge: float = 0.08, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic continuous-time rate matrix with multi-basin equilibrium."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p_edge).astype(float)
    A = np.triu(A, 1); A = A + A.T
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    F = np.zeros(n)
    centres = np.linspace(2, n - 3, n_basins)
    for c in centres:
        F -= depth * np.exp(-(np.arange(n) - c) ** 2 / 6.0)
    F -= F.min()
    pi = np.exp(-F); pi /= pi.sum()
    k0 = 1.0
    S = k0 * A * np.sqrt(np.outer(pi, pi))
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                K[i, j] = S[i, j] / pi[i]
        K[i, i] = -K[i, :].sum()
    return K, F


def K_to_T(K: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Discrete-time transition matrix at lag tau."""
    return sla.expm(K * tau)


if __name__ == '__main__':
    # Quick smoke test
    K, F = threebasin_K(n=40, seed=1)
    T = K_to_T(K, tau=1.0)
    pi = stationary_T(T)
    print(f"n=40, FE range = {-np.log(pi).max() - -np.log(pi).min():.2f} kT")
    print(f"lambda_2(T) = {lambda2_T(T):.4f}")
    A_state = int(np.argmin(cv_eigvec(T, 2)))
    B_state = int(np.argmax(cv_eigvec(T, 2)))
    print(f"MFPT({A_state}->{B_state}) = {mfpt_T(T, B_state)[A_state]:.2f} lagtimes")
