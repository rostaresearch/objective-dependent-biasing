# pentapeptide

Spectral and MFPT optima on the 250-state pentapeptide MSM (`../data/pentapeptide_T.npy`).

Run order: `step1_check_T.py` (transition matrix and generator checks) → `step3_optimize.py` (optima at U_max = 1, 2, 3) → `step5_evaluate.py` (speedups, Ω_J, C_TS, macrostate-grouped crossings, reweighting cost) → `step6_sampling_check.py` (independent-path reference values) → `step7_structure.py` → `step8_tables.py` (`tables_C.md`, values behind Fig. 2).
