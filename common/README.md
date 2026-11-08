# common

Shared library imported by the analysis folders.

- `analytic_lib.py`: biased generator (local-detailed-balance tilt), spectral gap, MFPT, analytical gradients
- `optimal_spectral_bias.py`, `per_state_ceiling.py`, `mfpt_per_state.py`: per-state spectral and MFPT optimization
- `exact_last_crossing.py`: exact last-crossing law and C_TS (writes to `../last_crossing/`)
- `grid_sensitivity.py`: nine-well potential (parameters in SI Table S3) and grid generators
- `validate_sampling.py`: reactive-path sampler
- `mechanism_audit_asym1d.py`: reactive-current overlap and edge distortion
