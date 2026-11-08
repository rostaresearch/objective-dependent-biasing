# Objective-Dependent Biasing of Markov Networks for Relaxation and First Passage

Code and processed data for

> Q. Wang, A. Fadhluddin, E. Rosta, *Objective-Dependent Biasing of Markov Networks for Relaxation and First Passage*, J. Chem. Theory Comput. (revised, 2026).
> Corresponding author: Edina Rosta (e.rosta@ucl.ac.uk), University College London.

## Layout

| Folder | Contents | Manuscript items |
|---|---|---|
| `common/` | Shared library: biased generator, spectral-gap and MFPT gradients, optimizers, last-crossing law, nine-well potential (Table S3), path sampler | used by all folders |
| `data/` | Input networks: bottleneck grids, nine-well network, pentapeptide MSM | |
| `grid_refinement/` | Entropic-bottleneck and nine-well grids: refinement, reoptimization, reweighting cost | Fig. 1, Fig. S2, Tables S1–S2 |
| `pentapeptide/` | Spectral and MFPT optima on the 250-state pentapeptide MSM, last crossings, reweighting cost | Fig. 2, SI S4 |
| `last_crossing/` | Last-crossing (C_TS) and reactive-current (Ω_J) overlaps under bias scaling | Fig. 5 |
| `nine_well_controls/` | Reference values for the overlaps: independent paths, random biases, landscape-blind routes, rerouting controls | Fig. S6, Table S5 |
| `nine_well_langevin/` | Overdamped Langevin validation of the nine-well biases | Table 2, Table S4, Figs. S3–S4 |
| `ras/` | Ras phosphate release: joint DHAM reconstruction from 73 KOMBI runs and synthetic-run reference values | Table 3, Table S6, Figs. S7–S8 |

Each folder has a short README with the run order. Scripts are run from inside their folder and locate `common/` and `data/` relative to the repository root. The repository contains the analysis code and numerical results; figures are not included.

The code and data deposited with the original submission, including the budget sweep stored in `data/grid_2d_data.mat` (Fig. 3), are available in the repository history at commit `94a29cd`.

## Requirements

Python 3.9+ with numpy, scipy and matplotlib (`pip install -r requirements.txt`); numba for the Langevin simulations.

## Data availability

Raw Ras trajectories (224 GB) are available from the corresponding author on request. The projected per-frame coordinates, applied biases and bias parameters needed for every Ras result are in `ras/features/`.
