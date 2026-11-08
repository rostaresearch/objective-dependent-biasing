# nine_well_langevin

Overdamped Langevin simulations with fixed biases interpolated from the nine-well network optima (SI S5.2).

- `network_predictions.py`, `network_aligned.py`, `network_fd_check.py`, `network_fpt_distribution.py`: fine-grid network MFPTs and passage-time distributions
- `langevin.py` (numba), `campaign.py` with `jobs_*.json`: Langevin runs, saved in `runs/` and `runs_markov/`
- `analysis.py`, `summary_tables.py` → `results.json`, `results_tables.md` (Table 2, Table S4)
- `ks_check.py`, `run_consistency.py`, `escape_test.py`, `em_bias_1d.py`, `test_*.py`: checks
