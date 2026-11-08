"""Shared helpers for the pentapeptide MSM analysis.

Everything here is model-level: it is computed exactly on the estimated
250-state MSM and is conditional on that estimate.

Objective and gradient code is imported, not copied:
  per_state_ceiling.gap_and_grad / optimise_per_state_analytic  (spectral gap)
  mfpt_per_state.mfpt_and_grad / optimise_mfpt_per_state        (A->B MFPT)
  analytic_lib.tilt_generator, mfpt_K, spectral_gap_K, committor_K
  mechanism_audit_asym1d.positive_net_current                    (Omega_J)
  common/exact_last_crossing.law              (C_TS)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

sys.dont_write_bytecode = True               # leave the imported folders untouched

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root
REPO = ROOT
AUDIT = ROOT / 'common'
for p in (str(ROOT), str(AUDIT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import analytic_lib as L                     # noqa: E402
import per_state_ceiling as PSC              # noqa: E402
import mfpt_per_state as MPS                 # noqa: E402
import mechanism_audit_asym1d as MA          # noqa: E402
import exact_last_crossing as ELC            # noqa: E402

T_PATH = ROOT / 'data' / 'pentapeptide_T.npy'
US_RESULTS = ROOT / 'data' / 'pentapeptide_results.json'
# Crisp 5-macrostate PCCA+ labels distributed with the same transition matrix
# (pentapeptide_us.py loads T from this folder).
PCCA_PATH = ROOT / 'data' / 'pentapeptide_pcca5_crisp.dat'
TAU = 1.0          # lag time; not recorded in the available files. Times are in units of tau.


def rel(path):
    """Repository-relative path for provenance records."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.name


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_T() -> np.ndarray:
    return np.load(T_PATH)


def generator_from_T(T: np.ndarray, tau: float = TAU) -> np.ndarray:
    """K = (T - I)/tau. Off-diagonal K_ij = T_ij/tau >= 0, rows sum to 0, and the
    jump chain K_ij/(-K_ii) = T_ij/(1 - T_ii) equals that of T."""
    n = T.shape[0]
    K = T / tau
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(K, -K.sum(axis=1))
    return K


def stationary(K: np.ndarray) -> np.ndarray:
    return L.stationary_distribution_from_K(K)


def reversible_spectrum(K: np.ndarray, pi: np.ndarray):
    """Eigenvalues (descending) and pi-orthonormal right eigenvectors of a
    reversible generator, via the symmetric matrix D^1/2 K D^-1/2."""
    d = np.sqrt(pi)
    S = d[:, None] * K / d[None, :]
    S = 0.5 * (S + S.T)
    w, U = eigh(S)
    o = np.argsort(-w)
    return w[o], U[:, o] / d[:, None]


def gap_sym(K: np.ndarray, pi: np.ndarray) -> float:
    """Spectral gap by the symmetric route (independent of L.spectral_gap_K)."""
    w, _ = reversible_spectrum(K, pi)
    return float(-w[1])


def mfpt_fundamental(K: np.ndarray, pi: np.ndarray, A, B) -> float:
    """MFPT A->B (uniform average over A) by a route independent of the linear
    solve in analytic_lib.mfpt_K.

    Single target state j: the fundamental-matrix identity
        m_ij = (Z_jj - Z_ij)/pi_j,   Z = (1 pi^T - K)^{-1} - 1 pi^T.
    Target set B: eigendecomposition of the symmetrised block of -K on the
    complement N of B (valid for a reversible K),
        m_N = D^-1/2 U diag(1/|w|) U^T D^1/2 1.
    """
    A = np.atleast_1d(np.asarray(A, int)); B = np.atleast_1d(np.asarray(B, int))
    n = K.shape[0]
    if len(B) == 1:
        j = int(B[0])
        Z = np.linalg.inv(np.outer(np.ones(n), pi) - K) - np.outer(np.ones(n), pi)
        m = (Z[j, j] - Z[:, j]) / pi[j]
        return float(np.mean(m[A]))
    # Target set: Kemeny-Snell for sets.  m_i = sum_k (Z_kk - Z_ik) ... does not
    # reduce simply, so use the Laplace-transform / eigen-route: solve on the
    # complement with eigendecomposition of the symmetrised restricted block.
    N = np.setdiff1d(np.arange(n), B)
    d = np.sqrt(pi[N])
    S = d[:, None] * K[np.ix_(N, N)] / d[None, :]
    S = 0.5 * (S + S.T)
    w, U = eigh(S)                      # all w < 0
    # m_N = (-K_NN)^{-1} 1 = D^-1/2 U diag(-1/w) U^T D^1/2 1
    m = (U @ ((U.T @ d) / (-w))) / d
    full = np.zeros(n); full[N] = m
    return float(np.mean(full[A]))


def ess_fraction(pi0: np.ndarray, pib: np.ndarray) -> float:
    """ESS/N = 1 / sum_i pib_i (pi0_i/pib_i)^2  (Kish, reweighting pib -> pi0)."""
    return float(1.0 / np.sum(pi0 ** 2 / pib))


def ess_fraction_closed(pi0: np.ndarray, b: np.ndarray) -> float:
    """Independent closed form for the tilt pib ∝ pi0 exp(-b):
       ESS/N = 1 / ( <exp(-b)>_pi0 * <exp(+b)>_pi0 )."""
    c = b - b.mean()
    return float(1.0 / (np.dot(pi0, np.exp(-c)) * np.dot(pi0, np.exp(c))))


def omega_J(K0, pi0, q0, Kb, pib, qb) -> float:
    J0 = MA.positive_net_current(K0, pi0, q0)
    Jb = MA.positive_net_current(Kb, pib, qb)
    return float(np.minimum(J0, Jb).sum())


def omega_J_bhs(K0, pi0, q0, Kb, pib, qb) -> float:
    """Second route: Berezhkovskii-Hummer-Szabo reversible form
       J+_ij = max(pi_i K_ij (q_j - q_i), 0), normalised over all edges."""
    def jhat(K, pi, q):
        J = pi[:, None] * K * (q[None, :] - q[:, None])
        np.fill_diagonal(J, 0.0)
        J = np.maximum(J, 0.0)
        return J / J.sum()
    return float(np.minimum(jhat(K0, pi0, q0), jhat(Kb, pib, qb)).sum())


def ipr(p) -> float:
    p = np.asarray(p, float)
    return float(1.0 / np.sum(p ** 2))


class System:
    """Unbiased reference quantities for one (K0, A, B) choice."""

    def __init__(self, K0: np.ndarray, A, B, label: str = ''):
        self.label = label
        self.K0 = K0
        self.n = K0.shape[0]
        self.A = np.atleast_1d(np.asarray(A, int))
        self.B = np.atleast_1d(np.asarray(B, int))
        self.pi0 = stationary(K0)
        self.gap0 = L.spectral_gap_K(K0)
        self.t0 = L.mfpt_K(K0, self.A, self.B)
        self.t0_BA = L.mfpt_K(K0, self.B, self.A)
        self.q0 = L.committor_K(K0, self.A, self.B)
        self.K0s = csr_matrix(K0)
        self.ref = ELC.law(self.K0s, self.pi0, self.A, self.B, self.q0)
        self.p0 = self.ref['probability']

    def metrics(self, b: np.ndarray) -> dict:
        b = np.asarray(b, float)
        Kb = L.tilt_generator(self.K0, b)
        pib = L.biased_pi_from_reference(self.pi0, b)
        pib_eig = stationary(Kb)
        qb = L.committor_K(Kb, self.A, self.B)
        lb = ELC.law(csr_matrix(Kb), pib, self.A, self.B, self.q0)
        assert np.array_equal(lb['i'], self.ref['i']) and np.array_equal(lb['j'], self.ref['j'])
        gap = L.spectral_gap_K(Kb)
        t = L.mfpt_K(Kb, self.A, self.B)
        tBA = L.mfpt_K(Kb, self.B, self.A)
        w, _ = reversible_spectrum(Kb, pib)
        out = dict(
            gap=gap, gap_speedup=gap / self.gap0,
            mfpt_AB=t, mfpt_AB_speedup=self.t0 / t,
            mfpt_BA=tBA, mfpt_BA_speedup=self.t0_BA / tBA,
            lambda_2_to_6=(-w[1:6]).tolist(),
            Omega_J=omega_J(self.K0, self.pi0, self.q0, Kb, pib, qb),
            C_TS=float(np.minimum(self.p0, lb['probability']).sum()),
            ESS_over_N=ess_fraction(self.pi0, pib),
            n_at_bound=int(np.sum(np.abs(np.abs(b) - np.abs(b).max()) < 1e-6)),
            max_abs_b=float(np.abs(b).max()),
            range_b=float(b.max() - b.min()),
            last_crossing_probability=lb['probability'].tolist(),
            capacity_relative_residual=float(lb['capacity_relative_residual']),
            pib_route_difference=float(np.abs(pib - pib_eig).max()),
        )
        # Independent routes (cross-checks)
        out['check'] = dict(
            gap_sym=gap_sym(Kb, pib),
            mfpt_AB_fundamental=mfpt_fundamental(Kb, pib, self.A, self.B),
            mfpt_BA_fundamental=mfpt_fundamental(Kb, pib, self.B, self.A),
            Omega_J_bhs=omega_J_bhs(self.K0, self.pi0, self.q0, Kb, pib, qb),
            Omega_J_law_current=float(self.ref['current'].minimum(lb['current']).sum()),
            ESS_closed_form=ess_fraction_closed(self.pi0, b),
            ESS_analytic_lib=float(L.reweighting_ess(self.pi0, pib, 1.0)),
        )
        return out


class SymGap:
    """Fast spectral-gap engine for a reversible K0 under the tilt
    (independent implementation, used for multi-start and cross-checks).

    With pib ∝ pi0 exp(-b), the symmetrised biased generator
    S^b = D_b^1/2 K^b D_b^-1/2 has the SAME off-diagonal as S^0; only its
    diagonal K^b_ii changes. Hence for a simple eigenpair (lambda_k, u_k) of S^b
        d lambda_k / d b_i = (1/2) [K^b^T (u_k * u_k)]_i ,
    and gamma = -lambda_2.
    """

    def __init__(self, K0: np.ndarray, pi0: np.ndarray):
        self.K0off = K0.copy(); np.fill_diagonal(self.K0off, 0.0)
        d = np.sqrt(pi0)
        S = d[:, None] * self.K0off / d[None, :]
        self.S0off = 0.5 * (S + S.T)
        self.n = K0.shape[0]

    def tilt(self, b):
        Kb = self.K0off * np.exp(-0.5 * (b[None, :] - b[:, None]))
        np.fill_diagonal(Kb, -Kb.sum(axis=1))
        return Kb

    def eigs(self, b, m=1):
        """Top m+1 eigenpairs of S^b (descending): 0, lambda_2, ..."""
        Kb = self.tilt(b)
        S = self.S0off.copy()
        S[np.diag_indices(self.n)] = Kb.diagonal()
        w, U = eigh(S, subset_by_index=[self.n - m - 1, self.n - 1])
        return Kb, w[::-1], U[:, ::-1]

    def gap_grad(self, b):
        Kb, w, U = self.eigs(b, 1)
        u2 = U[:, 1]
        return float(-w[1]), -0.5 * (Kb.T @ (u2 * u2))

    def softmin_grad(self, b, beta, m=8):
        """Smooth lower bound on the gap: -(1/beta) log sum_k exp(-beta gamma_k),
        k = 2..m+1, and its gradient."""
        Kb, w, U = self.eigs(b, m)
        g = -w[1:]
        z = -beta * g
        zmax = z.max()
        e = np.exp(z - zmax)
        val = -(zmax + np.log(e.sum())) / beta
        p = e / e.sum()
        grad = np.zeros(self.n)
        for k in range(m):
            uk = U[:, k + 1]
            grad += p[k] * (-0.5 * (Kb.T @ (uk * uk)))
        return float(val), grad


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(path, obj):
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2) + '\n')
