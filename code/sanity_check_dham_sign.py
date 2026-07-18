"""Synthetic round-trip check for the DHAM unbiasing sign.

Manuscript convention:    K^b_ij = K^0_ij * exp[-(b_j - b_i)/2]
Therefore inverse:         K^0_ij = K^b_ij * exp[+(b_j - b_i)/2]

Procedure:
  1. Build a known K^0 (1D nearest-neighbour tridiagonal generator).
  2. Tilt with a known b via tilt_generator -> K^b_true (this defines
     what 'biased dynamics' should produce).
  3. Synthesize a count matrix C_b from K^b_true (single short trajectory
     at stationarity), use it as the input to dham_unbias.
  4. Compare dham_unbias(C_b, b) -> K^0_recovered against K^0_true.

If signs are consistent, K^0_recovered should match K^0_true elementwise.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Local helpers (no scipy needed here)
def stationary_1d(F):
    """Stationary distribution for a generator with detailed balance to exp(-F)."""
    p = np.exp(-F); return p / p.sum()


def asym_1d_K(F):
    n = len(F)
    K = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                K[i, j] = np.exp(-0.5 * (F[j] - F[i]))
    np.fill_diagonal(K, -K.sum(axis=1))
    return K


def tilt_generator(K0, u):
    """Manuscript convention:  K^b_ij = K^0_ij * exp(-(u_j - u_i)/2)."""
    du = (u[None, :] - u[:, None]) * 0.5
    Kb = K0 * np.exp(-du)
    np.fill_diagonal(Kb, 0.0)
    np.fill_diagonal(Kb, -Kb.sum(axis=1))
    return Kb


def dham_unbias_MINUS_BUGGY(C, V, KbT=2.479):
    """The CURRENT (Ras script) buggy form: post-reweighting symmetrise.
    Because (C*W + (C*W).T)/2 = C * cosh[(V_j-V_i)/2kT] when C is already
    symmetric, the sign of the exponent cancels out -- both 'MINUS' and
    'PLUS' produce the same (and wrong) answer."""
    dV = V[None, :] - V[:, None]
    W  = np.exp(-0.5 * dV / KbT)
    Cu = 0.5 * (C * W + (C * W).T)
    row = Cu.sum(axis=1, keepdims=True); ok = row[:, 0] > 0
    M  = np.zeros_like(Cu); M[ok] = Cu[ok] / row[ok]
    return M


def dham_unbias_PROPER_MINUS(C, V, KbT=2.479):
    """Proper DHAM_sym with MINUS sign (manuscript convention if K^b
    were equal to K^0 * exp[+(b_j-b_i)/2])."""
    Csym = 0.5 * (C + C.T)
    dV = V[None, :] - V[:, None]
    W  = np.exp(-0.5 * dV / KbT)
    Cu = Csym * W
    row = Cu.sum(axis=1, keepdims=True); ok = row[:, 0] > 0
    M  = np.zeros_like(Cu); M[ok] = Cu[ok] / row[ok]
    return M


def dham_unbias_PROPER_PLUS(C, V, KbT=2.479):
    """Proper DHAM_sym with PLUS sign (manuscript convention: inverse of
    K^b_ij = K^0_ij * exp[-(b_j-b_i)/2] is exp[+(b_j-b_i)/2])."""
    Csym = 0.5 * (C + C.T)
    dV = V[None, :] - V[:, None]
    W  = np.exp(+0.5 * dV / KbT)
    Cu = Csym * W
    row = Cu.sum(axis=1, keepdims=True); ok = row[:, 0] > 0
    M  = np.zeros_like(Cu); M[ok] = Cu[ok] / row[ok]
    return M


def synthesize_counts(K, n_total=200_000, dt=1e-2):
    """Sample a stationary trajectory under K, build the lag-1 count matrix.
    Approximates a long unbiased / biased trajectory."""
    n = K.shape[0]
    # transition matrix at small dt
    M = np.eye(n) + dt * K
    M = np.maximum(M, 0)
    M = M / M.sum(axis=1, keepdims=True)
    # stationary
    eig_vals, eig_vecs = np.linalg.eig(M.T)
    pi = np.real(eig_vecs[:, np.argmax(np.real(eig_vals))])
    pi = np.abs(pi); pi /= pi.sum()
    rng = np.random.default_rng(0)
    s = rng.choice(n, p=pi)
    C = np.zeros((n, n))
    for t in range(n_total):
        sn = rng.choice(n, p=M[s])
        C[s, sn] += 1.0
        s = sn
    return C


def analytical_counts(K, dt=1e-2, N_total=1_000_000):
    """Build the noise-free count matrix expected at stationarity:
        C_ij  =  N * pi_i * M_ij    where M = I + dt*K, row-stochastic.
    """
    n = K.shape[0]
    M = np.eye(n) + dt * K
    M = np.maximum(M, 0); M = M / M.sum(axis=1, keepdims=True)
    eig_vals, eig_vecs = np.linalg.eig(M.T)
    pi = np.abs(np.real(eig_vecs[:, np.argmax(np.real(eig_vals))]))
    pi = pi / pi.sum()
    return N_total * (pi[:, None] * M), pi, M


def main():
    KbT = 2.479
    n   = 20
    xs  = np.linspace(-1.5, 1.5, n)
    F   = 4.0 * (xs**2 - 1)**2          # potential in kBT units
    K0_true  = asym_1d_K(F)
    b        = -2.0 * xs                # bias in kBT units, linear tilt
    V_kjmol  = b * KbT                  # in kJ/mol so dham_unbias gets V/kT right
    Kb_true  = tilt_generator(K0_true, b)

    # Analytical biased counts -- NO MC noise
    C_b, pi_b, Mb_true = analytical_counts(Kb_true, dt=1e-2, N_total=1_000_000)
    C_b = 0.5 * (C_b + C_b.T)
    # The ground truth M^0 at the same effective lag
    _, _, M0_true = analytical_counts(K0_true, dt=1e-2, N_total=1)

    M0_BUGGY        = dham_unbias_MINUS_BUGGY(C_b,  V_kjmol, KbT=KbT)
    M0_PROPER_MINUS = dham_unbias_PROPER_MINUS(C_b, V_kjmol, KbT=KbT)
    M0_PROPER_PLUS  = dham_unbias_PROPER_PLUS (C_b, V_kjmol, KbT=KbT)

    # Diagnostics: only compare tridiagonal entries (where M^0 has support)
    def tri_mask(M):
        m = np.zeros_like(M, dtype=bool)
        for i in range(M.shape[0]):
            for j in (i-1, i, i+1):
                if 0 <= j < M.shape[0]:
                    m[i, j] = True
        return m
    mask = tri_mask(M0_true)
    def rel(M):
        return float(np.max(np.abs(M[mask] - M0_true[mask]) /
                            np.maximum(M0_true[mask], 1e-12)))

    print("Round-trip sanity check (analytical counts, no MC noise)")
    print("=" * 70)
    print(f"  manuscript convention: K^b_ij = K^0_ij * exp[-(b_j-b_i)/2]")
    print(f"  => inverse: K^0_ij = K^b_ij * exp[+(b_j-b_i)/2]  (PLUS sign)")
    print(f"  (b = V/kT; V used here is in kJ/mol, kT = {KbT:.3f} kJ/mol)")
    print()
    print(f"  M^0_true range on tridiag : [{M0_true[mask].min():.4e}, "
          f"{M0_true[mask].max():.4e}]")
    print(f"  max rel err (tridiag-only):")
    print(f"     CURRENT Ras code (sym-after-reweight): "
          f"{rel(M0_BUGGY):.3e}")
    print(f"     proper DHAM_sym, MINUS exp[-(V_j-V_i)/2kT]: "
          f"{rel(M0_PROPER_MINUS):.3e}")
    print(f"     proper DHAM_sym, PLUS  exp[+(V_j-V_i)/2kT]: "
          f"{rel(M0_PROPER_PLUS ):.3e}")
    print()
    print("VERDICT")
    print("-" * 64)
    rs = {"buggy": rel(M0_BUGGY),
          "MINUS": rel(M0_PROPER_MINUS),
          "PLUS":  rel(M0_PROPER_PLUS)}
    winner = min(rs, key=rs.get)
    for k, r in rs.items():
        marker = "  <-- WINS" if k == winner else ""
        print(f"  {k:<8} max rel err = {r:.3e}{marker}")


if __name__ == "__main__":
    main()
