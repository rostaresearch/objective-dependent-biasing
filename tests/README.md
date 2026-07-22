# Smoke tests

Fast, dependency-light checks that the archive is a coherent, runnable bundle.
Run them first after cloning:

```bash
export MSM_ROOT=/path/to/objective-dependent-biasing   # or let it auto-detect
python -m pytest tests/          # needs pytest
python tests/test_smoke.py       # standalone, no pytest
```

What they guard (`test_smoke.py`):

1. **Layout** — the `MSM_ROOT` path convention resolves and the representative
   deposited inputs each objective needs are present in `data/`.
2. **Imports** — the core library and the Ras/figure scripts import cleanly.
3. **Closed-form identities** — the locally-detailed-balance tilt reproduces
   `pi^b ∝ pi0 · exp(−b)`, committors hit their `0/1` endpoints and stay in
   `[0,1]`, and the spectral gap and MFPT are positive and finite.
4. **Validity screen** — the documented thresholds `(O_min, Φ_max, W_min) =
   (0.05, 0.01, 3)`, and the `ε = 1` coverage threshold: more sampling raises the
   effective coverage `I_i`, so the coverage-failure fractions `Φ_‡` and `Φ_AB`
   fall monotonically.
5. **Reproduction** — the deposited bias-scaling headline and 5/95 bootstrap
   bands in `ras_pi_fig_data.mat` are internally consistent with their source
   `ras_pi_pareto_boot.json` (what `export_ras_pi_mat.py` produces).

These intentionally skip the heavy sweeps (Option-C, block bootstraps) and the
MATLAB figure scripts. The GitHub Actions workflow `.github/workflows/ci.yml`
runs this suite on Python 3.9 and 3.12.
