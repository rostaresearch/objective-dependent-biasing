"""Can the umbrella-sampling coverage score be optimized by gradient descent?

The coverage score E_AB = sum_i rho_i / I_i, with
    I_i    = sum_k N_k^eff pi_i^(k),
    N_k^eff= T_k gamma_k / 2,     gamma_k = -lambda_2(K^(k)),
    K^(k)  = tilt(K0, b^(k)),     b_i^(k) = (kappa/2)(xi_i - xi_k)^2,
    pi_i^(k) prop pi0_i exp(-b_i^(k)),
is a SMOOTH function of the continuous window parameters -- the centres xi_k and
the force constant kappa -- for a fixed number of windows W and fixed budget T_k.
Only W is discrete.  This script derives the closed-form gradient w.r.t. xi_k and
kappa and checks it against central finite differences.

Chain of derivatives (theta = xi_k or kappa):
  d b_i^(k)/d xi_k   = -kappa (xi_i - xi_k)
  d b_i^(k)/d kappa  =  (1/2)(xi_i - xi_k)^2
  d pi_i^(k)/d theta = pi_i^(k) [ -d b_i^(k)/d theta + <d b^(k)/d theta>_k ]   (response)
  d gamma_k/d theta  = sum_i (d gamma_k/d b_i^(k)) (d b_i^(k)/d theta)          (Hellmann-Feynman)
  d N_k^eff/d theta  = (T_k/2) d gamma_k/d theta
  d I_i/d theta      = sum_k [ (d N_k^eff/d theta) pi_i^(k) + N_k^eff (d pi_i^(k)/d theta) ]
  d E_AB/d theta     = -sum_i rho_i I_i^{-2} d I_i/d theta
"""
from __future__ import annotations
import os, sys
import numpy as np

PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root; override via MSM_ROOT
sys.path.insert(0, os.path.join(PATH, 'code'))
import analytic_lib as L
from per_state_ceiling import gap_and_grad

# ---- network + CV -----------------------------------------------------
K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0, bottleneck=True)
pi0 = L.stationary_distribution_from_K(K0)
xi = coords[:, 0].astype(float)                    # 1D CV = x-coordinate
A = np.where(coords[:, 0] < -0.85)[0]
B = np.where(coords[:, 0] > 0.85)[0]
q = L.committor_K(K0, A, B)
rho = pi0 * q * (1.0 - q); rho = rho / rho.sum()   # reactive-tube weight

W = 5
centres = np.linspace(xi.min(), xi.max(), W)
kappa = 4.0
T = np.ones(W)                                      # equal budget per window


def window_bias(xi_k, kap):
    return 0.5 * kap * (xi - xi_k) ** 2


def coverage_pieces(cent, kap):
    """Return I (n,), and per-window pi^(k), gamma_k, Neff_k."""
    piks, gammas, neffs = [], [], []
    I = np.zeros_like(pi0)
    for k in range(W):
        b = window_bias(cent[k], kap)
        Kk = L.tilt_generator(K0, b)
        gk = L.spectral_gap_K(Kk)
        w = pi0 * np.exp(-b); pik = w / w.sum()
        nk = T[k] * gk / 2.0
        piks.append(pik); gammas.append(gk); neffs.append(nk)
        I += nk * pik
    return I, piks, gammas, neffs


def E_AB(cent, kap):
    I, *_ = coverage_pieces(cent, kap)
    return float(np.sum(rho / I))


def grad_E_AB(cent, kap):
    """Closed-form gradient of E_AB w.r.t. each centre and w.r.t. kappa."""
    I, piks, gammas, neffs = coverage_pieces(cent, kap)
    dE_dcent = np.zeros(W)
    dI_dkappa = np.zeros_like(pi0)
    for k in range(W):
        b = window_bias(cent[k], kap)
        pik = piks[k]
        # HF spectral gradient d gamma_k / d b_i
        _, dg_db = gap_and_grad(b, K0)
        # --- w.r.t. centre xi_k ---
        db_dxi = -kap * (xi - cent[k])
        dpik_dxi = pik * (-db_dxi + np.dot(pik, db_dxi))
        dgamma_dxi = float(np.dot(dg_db, db_dxi))
        dNeff_dxi = T[k] / 2.0 * dgamma_dxi
        dI_dxi = dNeff_dxi * pik + neffs[k] * dpik_dxi         # only window k depends on xi_k
        dE_dcent[k] = -np.sum(rho * dI_dxi / I**2)
        # --- w.r.t. shared kappa (accumulate over windows) ---
        db_dk = 0.5 * (xi - cent[k]) ** 2
        dpik_dk = pik * (-db_dk + np.dot(pik, db_dk))
        dgamma_dk = float(np.dot(dg_db, db_dk))
        dNeff_dk = T[k] / 2.0 * dgamma_dk
        dI_dkappa += dNeff_dk * pik + neffs[k] * dpik_dk
    dE_dkappa = -np.sum(rho * dI_dkappa / I**2)
    return dE_dcent, float(dE_dkappa)


# ---- finite-difference verification -----------------------------------
g_cent, g_kap = grad_E_AB(centres, kappa)

print("E_AB =", E_AB(centres, kappa))
print("\n-- centres: analytic vs central finite difference --")
eps = 1e-5
for k in range(W):
    cp = centres.copy(); cp[k] += eps
    cm = centres.copy(); cm[k] -= eps
    fd = (E_AB(cp, kappa) - E_AB(cm, kappa)) / (2 * eps)
    rel = abs(g_cent[k] - fd) / (abs(fd) + 1e-30)
    print(f"  xi_{k}: analytic {g_cent[k]:+.6e}  FD {fd:+.6e}  rel err {rel:.2e}")

print("\n-- force constant kappa --")
fd_k = (E_AB(centres, kappa + eps) - E_AB(centres, kappa - eps)) / (2 * eps)
print(f"  kappa: analytic {g_kap:+.6e}  FD {fd_k:+.6e}  "
      f"rel err {abs(g_kap - fd_k)/(abs(fd_k)+1e-30):.2e}")

# ---- one gradient-descent step, to confirm it lowers E ----------------
print("\n-- one gradient step on the centres (should lower E_AB) --")
E0 = E_AB(centres, kappa)
gn = g_cent / (np.linalg.norm(g_cent) + 1e-30)
for step in (1e-3, 1e-4, 1e-5):
    cnew = np.clip(centres - step * gn, xi.min(), xi.max())
    try:
        E1 = E_AB(cnew, kappa)
        print(f"  step {step:.0e}: E_AB {E0:.6f} -> {E1:.6f}  change {E1-E0:+.3e}")
    except Exception as e:
        print(f"  step {step:.0e}: skipped ({type(e).__name__})")
