"""
1D asymmetric double-well: per-state optimal bias under pure L-infinity
box constraint |u_i| <= U_max, no zero-mean constraint.

Potential:  V(x) = A (x^2 - 1)^2 + B x      (B != 0 breaks symmetry)
Discretised on [-X_max, X_max] with n bins.
Nearest-neighbour symmetric (detailed-balance) rates:
    K[i, j] = base_rate * exp(-0.5 (F[j] - F[i]))     for j = i +/- 1

Two objectives, both with per-state freedom (no basis, no zero-mean):
  1. Maximise spectral gap        gamma = -lambda_2(K^b)
  2. Minimise MFPT(A -> B)        with A = left basin, B = right basin

Outputs:
  asymmetric_1d.json   - JSON summary
  asymmetric_1d.mat    - MATLAB-readable struct for plotting
"""
from __future__ import annotations
import json
import numpy as np
from scipy.io import savemat

import analytic_lib as L
from per_state_ceiling import optimise_per_state_analytic
from mfpt_per_state import optimise_mfpt_per_state

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def asym_1d_generator(n: int = 200, A: float = 4.0, B: float = 1.0,
                       X_max: float = 2.0, base_rate: float = 1.0
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric 1D double well: V(x) = A(x^2-1)^2 + B*x, nearest-neighbour
       symmetric rate matrix with detailed balance w.r.t. exp(-V)."""
    xs = np.linspace(-X_max, X_max, n)
    F = A * (xs**2 - 1.0)**2 + B * xs
    K = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                K[i, j] = base_rate * np.exp(-0.5 * (F[j] - F[i]))
    np.fill_diagonal(K, -K.sum(axis=1))
    return K, xs, F


def main():
    n = 200
    A_pot = 21.0         # saddle height V(0)=A, gives barrier ~22 k_B T
    B_pot = 1.0          # asymmetry; F(+1)-F(-1) = 2B = 2 k_B T
    X_max = 1.5          # tighten to keep boundary values reasonable
    U_max_list = [4.0, 6.0, 8.0]

    print(f"Building 1D asymmetric double well:")
    print(f"  V(x) = {A_pot}(x^2-1)^2 + {B_pot} x   on x in [-{X_max}, {X_max}]")
    print(f"  n = {n}")
    K0, xs, F = asym_1d_generator(n=n, A=A_pot, B=B_pot, X_max=X_max)

    pi0 = L.stationary_distribution_from_K(K0)
    gap0 = L.spectral_gap_K(K0)

    # Basins: left well minimum (deep), right well minimum (shallow)
    half = n // 2
    iA = int(np.argmin(F[:half]))
    iB = half + int(np.argmin(F[half:]))
    A_set = [iA]
    B_set = [iB]
    mfpt0 = L.mfpt_K(K0, A_set, B_set)

    print(f"\nLandscape:")
    print(f"  left well  : x={xs[iA]:+.3f}   F={F[iA]:.4f} k_B T")
    print(f"  right well : x={xs[iB]:+.3f}   F={F[iB]:.4f} k_B T")
    print(f"  barrier ~  : F_max - min(F) = {F.max() - F.min():.3f} k_B T")
    print(f"  asymmetry  : DeltaF(L->R)   = {F[iB] - F[iA]:+.3f} k_B T")

    print(f"\nBaseline (unbiased) kinetics:")
    print(f"  gamma_0 = {gap0:.5e}")
    print(f"  MFPT(L->R)_0 = {mfpt0:.5e}")

    # ------------------------------------------------------------------
    #  Sweep U_max
    # ------------------------------------------------------------------
    results = []
    for U_max in U_max_list:
        print(f"\n=== U_max = {U_max} k_B T ===")

        # Spectral-gap-optimal per-state bias
        print(f"  Optimising spectral gap...")
        rs = optimise_per_state_analytic(K0, U_max=U_max, maxiter=3000)
        Kb_s = L.tilt_generator(K0, rs['u'])
        gap_s = L.spectral_gap_K(Kb_s)
        mfpt_s = L.mfpt_K(Kb_s, A_set, B_set)
        print(f"    gamma   speedup = {gap_s/gap0:.2f}x")
        print(f"    MFPT    speedup = {mfpt0/mfpt_s:.2f}x")

        # MFPT-optimal per-state bias
        print(f"  Optimising MFPT(L->R)...")
        rm = optimise_mfpt_per_state(K0, A_set, B_set, U_max=U_max,
                                       maxiter=3000, verbose=True)
        Kb_m = L.tilt_generator(K0, rm['u'])
        gap_m = L.spectral_gap_K(Kb_m)
        mfpt_m = rm['mfpt']
        print(f"    MFPT    speedup = {mfpt0/mfpt_m:.2f}x")
        print(f"    gamma   speedup = {gap_m/gap0:.2f}x")

        results.append(dict(
            U_max=float(U_max),
            u_spectral=rs['u'].tolist(),
            u_mfpt=rm['u'].tolist(),
            gap_spectral=float(gap_s),
            mfpt_spectral=float(mfpt_s),
            gap_mfpt=float(gap_m),
            mfpt_mfpt=float(mfpt_m),
            speedup_gap_spectral=float(gap_s / gap0),
            speedup_mfpt_spectral=float(mfpt0 / mfpt_s),
            speedup_gap_mfpt=float(gap_m / gap0),
            speedup_mfpt_mfpt=float(mfpt0 / mfpt_m),
        ))

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------
    summary = dict(
        n=n, A_pot=A_pot, B_pot=B_pot, X_max=X_max,
        x=xs.tolist(),
        F=F.tolist(),
        pi0=pi0.tolist(),
        gap0=float(gap0),
        mfpt0=float(mfpt0),
        iA=int(iA), iB=int(iB),
        U_max_list=[float(u) for u in U_max_list],
        results=results,
        notes=("Per-state bias under pure L-infinity box constraint, no "
               "zero-mean. Constant-shift gauge is undetermined; absolute "
               "u-values are the optimiser's report."),
    )
    with open(f'{PATH}/asymmetric_1d.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved asymmetric_1d.json")

    # MATLAB MAT file -- arrays only, easy to load
    mat = dict(
        x=xs, F=F, pi0=pi0,
        gap0=gap0, mfpt0=mfpt0,
        iA=iA + 1, iB=iB + 1,    # 1-indexed for MATLAB
        U_max_list=np.asarray(U_max_list),
        u_spectral=np.column_stack([np.asarray(r['u_spectral']) for r in results]),
        u_mfpt=np.column_stack([np.asarray(r['u_mfpt']) for r in results]),
        gap_spectral=np.asarray([r['gap_spectral'] for r in results]),
        mfpt_spectral=np.asarray([r['mfpt_spectral'] for r in results]),
        gap_mfpt=np.asarray([r['gap_mfpt'] for r in results]),
        mfpt_mfpt=np.asarray([r['mfpt_mfpt'] for r in results]),
        speedup_gap_spectral=np.asarray([r['speedup_gap_spectral'] for r in results]),
        speedup_mfpt_spectral=np.asarray([r['speedup_mfpt_spectral'] for r in results]),
        speedup_gap_mfpt=np.asarray([r['speedup_gap_mfpt'] for r in results]),
        speedup_mfpt_mfpt=np.asarray([r['speedup_mfpt_mfpt'] for r in results]),
    )
    savemat(f'{PATH}/asymmetric_1d.mat', mat)
    print('Saved asymmetric_1d.mat')


if __name__ == '__main__':
    main()
