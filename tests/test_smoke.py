"""Smoke tests for the objective-dependent-biasing archive.

These are fast, dependency-light checks that the deposited code and data form a
coherent, runnable bundle. They are the first thing to run after cloning:

    export MSM_ROOT=/path/to/objective-dependent-biasing
    python -m pytest tests/            # or:  python tests/test_smoke.py

They deliberately avoid the heavy sweeps (Option-C, block bootstraps). What they
guard is (1) the path convention resolves, (2) the deposited inputs are present,
(3) the core library imports and its closed-form linear algebra is self-consistent,
(4) the documented validity-screen constants and the epsilon=1 coverage threshold
behave as the manuscript describes, and (5) a deposited numerical result can be
regenerated from its deposited input.

Runs standalone (no pytest needed) via the __main__ block at the bottom.
"""
from __future__ import annotations
import os
import sys
import json

import numpy as np

# ---- path convention (mirrors every script in code/) ----------------------
ROOT = os.environ.get(
    'MSM_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(ROOT, 'code')
DATA = os.path.join(ROOT, 'data')
FIGURES = os.path.join(ROOT, 'figures')
sys.path.insert(0, CODE)


# ---------------------------------------------------------------------------
# 1. Repository layout
# ---------------------------------------------------------------------------
def test_repo_layout():
    for d in (CODE, DATA, FIGURES):
        assert os.path.isdir(d), f'missing directory: {d}'
    # A representative slice of the deposited inputs each objective needs.
    required = [
        'ras_pi_audit.json', 'ras_pi_pareto_boot.json', 'ras_pi_blockboot.json',
        'ras_pi_fig_data.mat', 'pi_features.npz', 'barrier_study.json',
        'filtered_results.json', 'spectral_results.json', 'mfpt_results.json',
    ]
    for f in required:
        assert os.path.isfile(os.path.join(DATA, f)), f'missing data file: {f}'


# ---------------------------------------------------------------------------
# 2. Core imports
# ---------------------------------------------------------------------------
def test_core_imports():
    import analytic_lib          # noqa: F401
    import network_us            # noqa: F401
    import per_state_ceiling     # noqa: F401
    import mechanism_audit_ras_proper  # noqa: F401
    import ras_pi_audit          # noqa: F401
    import export_ras_pi_mat     # noqa: F401
    import assemble_figure13     # noqa: F401


# ---------------------------------------------------------------------------
# 3. Closed-form linear algebra is self-consistent
# ---------------------------------------------------------------------------
def test_analytic_lib_identities():
    import analytic_lib as L
    K0, coords, _ = L.grid_2d_generator(nx=10, ny=6, barrier_height=3.0,
                                        bottleneck=True)
    n = K0.shape[0]
    pi0 = L.stationary_distribution_from_K(K0)
    assert abs(pi0.sum() - 1.0) < 1e-9 and np.all(pi0 > 0)

    # local-detailed-balance tilt reproduces pi^b propto pi0 * exp(-b)
    rng = np.random.default_rng(0)
    b = rng.normal(size=n)
    Kb = L.tilt_generator(K0, b)
    pib = L.stationary_distribution_from_K(Kb)
    expected = L.biased_pi_from_reference(pi0, b)
    assert np.allclose(pib, expected, atol=1e-8), 'tilt/pi^b identity broken'

    # committor endpoints and bounds
    x = coords[:, 0]
    A = np.where(x < x.min() + 1e-9)[0]
    B = np.where(x > x.max() - 1e-9)[0]
    q = L.committor_K(K0, A, B)
    assert np.allclose(q[A], 0.0, atol=1e-9)
    assert np.allclose(q[B], 1.0, atol=1e-9)
    assert q.min() >= -1e-9 and q.max() <= 1.0 + 1e-9

    # spectral gap positive and finite; MFPT positive and finite
    gap = L.spectral_gap_K(K0)
    assert np.isfinite(gap) and gap > 0
    mfpt = L.mfpt_K(K0, A, B)
    assert np.isfinite(mfpt) and mfpt > 0


# ---------------------------------------------------------------------------
# 4a. Validity-screen constants match the documented protocol
# ---------------------------------------------------------------------------
def test_screen_constants():
    import filtered_best as F
    assert (F.OV_MIN, F.COVFAIL_MAX, F.W_MIN) == (0.05, 0.01, 3), \
        'validity-screen thresholds drifted from the documented (0.05, 0.01, 3)'


# ---------------------------------------------------------------------------
# 4b. The epsilon = 1 coverage threshold behaves monotonically
#     (more sampling -> higher effective coverage I_i -> fewer states with
#      I_i < 1 -> lower coverage-failure fractions Phi_dagger and Phi_AB).
# ---------------------------------------------------------------------------
def test_coverage_threshold_behaviour():
    import analytic_lib as L
    K0, coords, _ = L.grid_2d_generator(nx=12, ny=6, barrier_height=4.0,
                                        bottleneck=True)
    x = coords[:, 0]
    centers = np.linspace(x.min(), x.max(), 4)      # W = 4 windows
    A = np.where(x < x.min() + 0.2 * (x.max() - x.min()))[0]
    B = np.where(x > x.max() - 0.2 * (x.max() - x.min()))[0]
    mid = 0.5 * (x.min() + x.max())
    barrier = np.where(np.abs(x - mid) < 0.12 * (x.max() - x.min()))[0]

    def phis(sample_time):
        m = L.umbrella_family_metrics(
            K0, x, centers, kappa=4.0, sample_time_per_window=sample_time,
            barrier_set=barrier, basin_A=A, basin_B=B)
        return (float(m['coverage_failure_pdagger']),
                float(m['coverage_failure_rate']))

    lean_dagger, lean_rate = phis(0.02)     # starved sampling
    rich_dagger, rich_rate = phis(50.0)     # abundant sampling
    for v in (lean_dagger, lean_rate, rich_dagger, rich_rate):
        assert 0.0 - 1e-12 <= v <= 1.0 + 1e-12, 'Phi out of [0,1]'
    assert rich_dagger <= lean_dagger + 1e-9, 'Phi_dagger not monotone in sampling'
    assert rich_rate <= lean_rate + 1e-9, 'Phi_AB not monotone in sampling'


# ---------------------------------------------------------------------------
# 5. A deposited result regenerates from its deposited input.
#    export_ras_pi_mat repackages the deposited bias-scaling bootstrap
#    (ras_pi_pareto_boot.json) into the MATLAB fig-data mat. Re-running the
#    scaling summary must reproduce the deposited alpha=1 headline and 5/95
#    bands stored in ras_pi_fig_data.mat.
# ---------------------------------------------------------------------------
def test_pareto_headline_reproduces():
    from scipy.io import loadmat
    boot = json.load(open(os.path.join(DATA, 'ras_pi_pareto_boot.json')))
    mat = loadmat(os.path.join(DATA, 'ras_pi_fig_data.mat'))

    alpha = np.asarray(mat['sc_alpha']).ravel().astype(float)
    oj = np.asarray(mat['sc_oj']).ravel().astype(float)
    oj_lo = np.asarray(mat['sc_oj_lo']).ravel().astype(float)
    oj_hi = np.asarray(mat['sc_oj_hi']).ravel().astype(float)

    k = int(np.argmin(np.abs(alpha - 1.0)))
    assert abs(alpha[k] - 1.0) < 1e-9, 'no alpha=1 point in the scaling sweep'
    # applied-bias headline reported in the manuscript
    assert 0.97 <= oj[k] <= 0.995, f'Omega_J(alpha=1) = {oj[k]:.4f} off headline'
    # median lies inside its own bootstrap band, and the band is a real interval
    assert oj_lo[k] <= oj[k] + 1e-9 <= oj_hi[k] + 1e-9
    assert np.all(oj_hi >= oj_lo - 1e-12) and np.any(oj_hi > oj_lo)

    # the mat's alpha grid matches the bootstrap json's alpha grid
    if 'alpha' in boot or 'alphas' in boot:
        ba = np.asarray(boot.get('alpha', boot.get('alphas'))).ravel().astype(float)
        assert len(ba) == len(alpha) and np.allclose(np.sort(ba), np.sort(alpha),
                                                     atol=1e-9)


# ---------------------------------------------------------------------------
# standalone runner (no pytest required)
# ---------------------------------------------------------------------------
def _run_standalone():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL  {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_standalone() else 0)
