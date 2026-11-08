### Cross-objective table, primary A/B (A = state 189, B = state 219)

| U | Bias | Gap speedup | A→B MFPT speedup | B→A MFPT speedup | Ω_J | C_TS | IPR of biased last crossings | ESS/N |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | spectral optimum | 2.65 | 1.23 | 0.34 | 0.920 | 0.893 | 30.7 | 0.909 |
| 1 | A→B MFPT optimum | 1.54 | 2.01 | 1.47 | 0.862 | 0.785 | 23.5 | 0.883 |
| 2 | spectral optimum | 5.10 | 1.40 | 0.087 | 0.824 | 0.767 | 22.7 | 0.572 |
| 2 | A→B MFPT optimum | 1.87 | 3.22 | 0.893 | 0.791 | 0.716 | 20.7 | 0.516 |
| 3 | spectral optimum | 8.21 | 1.55 | 0.0172 | 0.775 | 0.728 | 21.9 | 0.214 |
| 3 | A→B MFPT optimum | 2.73 | 4.75 | 0.238 | 0.768 | 0.718 | 20.6 | 0.267 |

Fraction of the attainable speedup that each optimum reaches on the other objective:

| U | spectral optimum: A→B MFPT speedup / best A→B MFPT speedup | MFPT optimum: gap speedup / best gap speedup |
|---|---:|---:|
| 1 | 1.23 / 2.01 = 0.61 | 1.54 / 2.65 = 0.58 |
| 2 | 1.40 / 3.22 = 0.43 | 1.87 / 5.10 = 0.37 |
| 3 | 1.55 / 4.75 = 0.33 | 2.73 / 8.21 = 0.33 |

### Robustness at U = 2

| U | Bias | Gap speedup | A→B MFPT speedup | B→A MFPT speedup | Ω_J | C_TS | IPR of biased last crossings | ESS/N |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | primary A/B: spectral optimum | 5.10 | 1.40 | 0.087 | 0.824 | 0.767 | 22.7 | 0.572 |
| 2 | primary A/B: A→B MFPT optimum | 1.87 | 3.22 | 0.893 | 0.791 | 0.716 | 20.7 | 0.516 |
| 2 | PCCA+ A/B: spectral optimum (same bias) | 5.10 | 4.09 | 0.182 | 0.690 | 0.635 | 22.1 | 0.572 |
| 2 | PCCA+ A/B: A→B MFPT optimum | 4.81 | 4.28 | 0.14 | 0.729 | 0.734 | 26.7 | 0.396 |
| 2 | PCCA+ A/B: primary MFPT optimum applied | 1.87 | 1.72 | 1.21 | 0.708 | 0.671 | 22.0 | 0.516 |

### Extra checks outside the requested scope (U = 2)

| U | Bias | Gap speedup | A→B MFPT speedup | B→A MFPT speedup | Ω_J | C_TS | IPR of biased last crossings | ESS/N |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | primary A/B: B→A MFPT optimum | 0.57 | 0.49 | 20 | 0.860 | 0.798 | 23.0 | 0.591 |
| 2 | K2: spectral optimum | 4.57 | 1.24 | 0.0675 | 0.864 | 0.796 | 97.0 | 0.659 |
| 2 | K2: A→B MFPT optimum | 1.41 | 3.74 | 0.684 | 0.881 | 0.837 | 74.5 | 0.513 |
| 2 | K2: K spectral optimum applied | 4.15 | 1.24 | 0.0702 | 0.828 | 0.753 | 84.1 | 0.572 |
| 2 | K2: K MFPT optimum applied | 1.40 | 3.54 | 0.661 | 0.822 | 0.749 | 60.9 | 0.516 |

### Last crossings grouped by the macrostate entered (primary A/B)

| Bias | into PCCA+ 1 (12 edges) | into PCCA+ 2 (18 edges) | into PCCA+ 4 (13 edges) | overlap with unbiased after grouping | C_TS (43 edges) |
|---|---:|---:|---:|---:|---:|
| unbiased | 0.312 | 0.456 | 0.232 | 1 | 1 |
| U = 1, spectral optimum | 0.264 | 0.472 | 0.264 | 0.952 | 0.893 |
| U = 1, A→B MFPT optimum | 0.302 | 0.469 | 0.230 | 0.988 | 0.785 |
| U = 2, spectral optimum | 0.247 | 0.413 | 0.340 | 0.892 | 0.767 |
| U = 2, A→B MFPT optimum | 0.352 | 0.391 | 0.257 | 0.934 | 0.716 |
| U = 3, spectral optimum | 0.254 | 0.329 | 0.417 | 0.814 | 0.728 |
| U = 3, A→B MFPT optimum | 0.358 | 0.339 | 0.303 | 0.883 | 0.718 |

### Cross-checks: worst case over every bias reported above

| Quantity | Primary route | Independent route | Worst disagreement |
|---|---|---|---:|
| Spectral gap | `analytic_lib.spectral_gap_K` (eig of K^b) | eigh of D^½ K^b D^-½ | 2.7e-14 (relative) |
| A→B and B→A MFPT | `analytic_lib.mfpt_K` (linear solve) | fundamental matrix / eigendecomposition of the symmetrised block | 9.5e-12 (relative) |
| A→B MFPT | as above | Monte Carlo, 20000 first-passage trajectories per bias | max abs z = 1.80 |
| Biased stationary distribution | π⁰e^(−b) normalised | eigenvector of K^b | 5.9e-13 |
| Ω_J | `positive_net_current` (π_i K_ij (1−q_i) q_j form) | BHS form max(π_i K_ij (q_j−q_i), 0); `exact_last_crossing.law` current (backward committor from the time-reversed generator) | 4.9e-12 |
| Edge currents behind Ω_J | J_ij = π_i K_ij (q_j−q_i) | Monte Carlo net traversals per reactive path, top 40 edges, 20 batches | max abs z = 3.07, rms z ≤ 1.18 |
| C_TS | `exact_last_crossing.law` | Monte Carlo last crossings (C_TS of the sampled biased distribution against the exact unbiased one) | max abs difference 0.0043 over the biased networks; sampled-vs-exact TV ≤ 0.020 for all networks (sampling noise) |
| Last-crossing total | Σ L_ij | reactive capacity from the committor | 1.8e-15 (relative) |
| ESS/N | 1/Σ π^b (π⁰/π^b)² | 1/(⟨e^−b⟩⟨e^b⟩) over π⁰; `analytic_lib.reweighting_ess` | 2.5e-12 |
| Spectral-gap gradient | `per_state_ceiling.gap_and_grad` | central finite differences (ε = 1e-5, 24 components, 5 points); symmetric-route formula, all components | FD 3.2e-07; symmetric 7.6e-14 (relative) |
| MFPT gradient | `mfpt_per_state.mfpt_and_grad` | central finite differences (ε = 1e-5, all 250 components, 5 points) | 2.3e-09 (relative) |
| First-order optimality | — | projected gradient on the box | ≤ 5.7e-05 of the largest gradient component |
| Spectral optimum value | imported `optimise_per_state_analytic` (flatten and zero starts) | symmetric-route engine: 6 random starts, lower-U warm start, soft-min continuation | spread 1.5e-04 (relative) |
| MFPT optimum value | imported `optimise_mfpt_per_state` (three starts) | 6 random starts with L-BFGS-B; TNC from zero and from the best point | spread 1.0e-06 (relative) |

Two independent unbiased Monte Carlo samples of 20000 paths overlap at C_TS = 0.976 and Ω_J = 0.617. Edge-level currents over 28530 directed edges are not resolved by this many paths, so the check of Ω_J rests on the three formulas and the edge test above.
