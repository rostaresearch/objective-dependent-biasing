# ras

Ras–GAP phosphate release: 73 KOMBI runs with 20 ps bias updates.

- `features/`: per-run projected coordinates (P–Mg and phosphate–Gln61 distances), applied bias and Gaussian bias parameters per propagation
- `extract_periodic.py`: builds the features from the raw trajectories (needs the raw data and mdtraj)
- `joint_dham.py`: joint discretization of all runs, self-consistent directed DHAM reference (`dham_P`; `normalized=False` gives plain DHAM), biased model and synthetic-run reference values
  - `python joint_dham.py real`: overlaps for all discretizations (Table S6)
  - `python joint_dham.py null w0.25_g3 200 average 1` (and `2`): synthetic-run reference values (Table 3)
- `verify_normalized.py`: forward–inverse check; the self-consistent estimator recovers the generating model exactly from expected synthetic counts, plain DHAM does not
- `current_shares.py`: shares of the outgoing current by P–Mg range (main text Section 4.1)
- `cts_ras.py`: approximate last-crossing overlap and its synthetic-run reference values (not reported: undetermined with these data)
- `primary_grid.py`: reference/biased overlaps of the MD data on the primary discretization (Table 3)
- `current_analysis.py`, `analyse_directed.py`: TPT solver and the earlier fixed 14 × 8 grid, used as a validation case
- Outputs in `results/`
