### Table 1. Bottleneck, U = 3, coarse (20×10) optima interpolated onto each grid

| Grid | n | cells in A (= B) | MFPT₀·h_x² | γ₀/h_x² | MFPT opt.: S_MFPT | S_γ | Ω_J | C_TS | Spectral opt.: S_MFPT | S_γ | Ω_J | C_TS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20×10 | 200 | 20 | 81.9 | 0.0245 | 85.1 | 58.6 | 0.935 | 0.999 | 34.1 | 91.3 | 0.902 | 0.996 |
| 40×20 | 800 | 80 | 53.2 | 0.0379 | 67.0 | 48.7 | 0.941 | 0.998 | 24.9 | 76.1 | 0.918 | 0.999 |
| 80×40 | 3200 | 280 | 50.5 | 0.0399 | 65.3 | 48.0 | 0.941 | 0.998 | 23.8 | 75.4 | 0.916 | 0.999 |
| 160×80 | 12800 | 1040 | 49.3 | 0.0408 | 64.6 | 47.7 | 0.941 | 0.998 | 23.3 | 75.1 | 0.917 | 0.998 |

### Table 2. Nine-well, coarse (20×20) optima interpolated onto each grid

| U | Grid | cells in A / B | MFPT opt.: S_MFPT | S_γ | Ω_J | C_TS | Spectral opt.: S_MFPT | S_γ | Ω_J | C_TS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 20×20 | 1 / 1 | 347.1 | 1.00 | 0.779 | 0.997 | 27.26 | 90.0 | 0.912 | 0.911 |
| 3 | 40×40 | 4 / 4 | 357.8 | 0.99 | 0.796 | 0.998 | 50.66 | 92.6 | 0.928 | 0.930 |
| 3 | 80×80 | 16 / 16 | 358.1 | 0.99 | 0.793 | 0.998 | 51.83 | 94.3 | 0.925 | 0.922 |
| 3 | 120×120 | 42 / 36 | 358.2 | 0.99 | 0.795 | 0.998 | 52.35 | 94.7 | 0.930 | 0.929 |
| 6 | 20×20 | 1 / 1 | 24,001 | 6.73 | 0.674 | 0.972 | 1.16 | 357.3 | 0.742 | 0.721 |
| 6 | 40×40 | 4 / 4 | 30,923 | 5.79 | 0.687 | 0.974 | 4.30 | 340.8 | 0.748 | 0.711 |
| 6 | 80×80 | 16 / 16 | 30,981 | 5.88 | 0.685 | 0.976 | 4.03 | 343.2 | 0.756 | 0.720 |
| 6 | 120×120 | 42 / 36 | 31,057 | 5.90 | 0.688 | 0.976 | 4.24 | 343.1 | 0.755 | 0.718 |

### Table 3. Bottleneck, U = 3: per-state optima re-optimized on each grid

| Grid | Optimum | own objective, coarse optimum interpolated | own objective, re-optimized | other objective (cross value) | Ω_J | C_TS | λ₃/λ₂ | spread over starts (rel.) | stationarity | starts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20×10 | MFPT | 85.1 | **85.1** | 58.61 | 0.935 | 0.999 | 6.527 | 5.4e-13 | 1.5e-08 | 7 |
| 20×10 | spectral | 91.3 | **91.3** | 34.07 | 0.902 | 0.996 | 3.126 | 2.1e-09 | 3.3e-08 | 7 |
| 40×20 | MFPT | 67.0 | **67.6** | 49.24 | 0.930 | 1.000 | 5.428 | 8.4e-12 | 5.8e-09 | 8 |
| 40×20 | spectral | 76.1 | **78.5** | 23.71 | 0.895 | 0.996 | 2.605 | 1.3e-08 | 9.9e-08 | 8 |
| 80×40 | MFPT | 65.3 | **65.9** | 48.72 | 0.932 | 1.000 | 5.329 | 1.8e-11 | 4.0e-09 | 7 |
| 80×40 | spectral | 75.4 | **77.4** | 22.55 | 0.900 | 0.997 | 2.574 | 8.2e-09 | 2.7e-08 | 7 |

### Table 4. Nine-well: per-state optima re-optimized on each grid

| U, grid | Optimum | own objective, coarse optimum interpolated | own objective, re-optimized | other objective (cross value) | Ω_J | C_TS | λ₃/λ₂ | spread over starts (rel.) | stationarity | starts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3, 20×20 | MFPT | 347.1 | **347.1** | 1.00 | 0.779 | 0.997 | 23.133 | 1.1e-07 | 3.8e-08 | 7 |
| 3, 20×20 | spectral | 90.0 | **90.0** | 27.27 | 0.913 | 0.911 | 1.001 | 8.6e-05 | 2.3e-08 | 3 |
| 3, 40×40 | MFPT | 357.8 | **359.5** | 1.00 | 0.762 | 0.998 | 25.322 | 1.4e-07 | 2.9e-08 | 7 |
| 3, 40×40 | spectral | 92.6 | **96.6** | 40.99 | 0.900 | 0.910 | 1.010 | 2.7e-04 | 1.1e-06 | 6 |
| 3, 80×80 | MFPT | 358.1 | **359.8** | 1.03 | 0.740 | 0.998 | 25.444 | 2.8e-07 | 1.5e-08 | 6 |
| 3, 80×80 | spectral | 94.3 | **98.3** | 42.26 | 0.894 | 0.907 | 1.105 | 7.0e-06 | 3.2e-07 | 6 |
| 6, 20×20 | MFPT | 24,001 | **24,001** | 6.73 | 0.674 | 0.972 | 5.695 | 3.3e-08 | 2.0e-08 | 7 |
| 6, 20×20 | spectral | 357.3 | **357.4** | 1.16 | 0.742 | 0.721 | 1.001 | 4.2e-04 | 1.3e-07 | 3 |
| 6, 40×40 | MFPT | 30,923 | **32,262** | 5.75 | 0.649 | 0.976 | 6.123 | 1.5e-05 | 2.4e-08 | 7 |
| 6, 40×40 | spectral | 340.8 | **405.2** | 3.04 | 0.663 | 0.567 | 1.001 | 5.5e-03 | 1.5e-07 | 6 |
| 6, 80×80 | MFPT | 30,981 | **32,369** | 5.73 | 0.637 | 0.977 | 6.285 | 1.6e-05 | 5.0e-08 | 6 |
| 6, 80×80 | spectral | 343.2 | **422.0** | 2.77 | 0.622 | 0.494 | 1.010 | 5.0e-04 | 4.0e-07 | 6 |

### Table 5. Bottleneck: re-optimized optima interpolated onto finer grids

| From → to | MFPT opt.: S_MFPT | S_γ | Ω_J | C_TS | Spectral opt.: S_MFPT | S_γ | Ω_J | C_TS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40×20 → 80×40 | 65.8 | 48.4 | 0.933 | 1.000 | 22.9 | 77.3 | 0.899 | 0.998 |
| 40×20 → 160×80 | 65.1 | 48.0 | 0.933 | 0.999 | 22.4 | 76.8 | 0.899 | 0.999 |
| 80×40 → 160×80 | 65.2 | 48.4 | 0.933 | 1.000 | 22.2 | 76.9 | 0.901 | 0.998 |

### Table 6. Bias profiles: re-optimized optimum minus a coarser optimum interpolated onto the same grid

Weighted by where the objective is sensitive (MFPT optimum: occupation time on the way from A to B; spectral optimum: slow-mode density φ₂²), computed on the finer grid.

| System | Optimum | Comparison | weighted RMS / U | weight in cells where abs(Δb) > U/2 | unweighted RMS / U |
|---|---|---|---:|---:|---:|
| bottleneck, U = 3 | spectral | 40×20 vs saved 20×10 | 0.0617 | 0.0e+00 | 0.094 |
| bottleneck, U = 3 | spectral | 80×40 vs saved 20×10 | 0.0585 | 0.0e+00 | 0.112 |
| bottleneck, U = 3 | spectral | 80×40 vs re-optimized 40×20 | 0.0138 | 0.0e+00 | 0.029 |
| bottleneck, U = 3 | MFPT | 40×20 vs saved 20×10 | 0.0491 | 0.0e+00 | 0.191 |
| bottleneck, U = 3 | MFPT | 80×40 vs saved 20×10 | 0.0474 | 0.0e+00 | 0.091 |
| bottleneck, U = 3 | MFPT | 80×40 vs re-optimized 40×20 | 0.0178 | 0.0e+00 | 0.158 |
| nine-well, U = 3 | spectral | 20×20 re-optimized vs saved 20×20 | 0.0072 | 7.9e-05 | 0.384 |
| nine-well, U = 3 | spectral | 40×40 vs saved 20×20 | 0.0744 | 1.1e-04 | 0.319 |
| nine-well, U = 3 | spectral | 40×40 vs re-optimized 20×20 | 0.0737 | 2.6e-05 | 0.167 |
| nine-well, U = 3 | MFPT | 20×20 re-optimized vs saved 20×20 | 0.0001 | 1.4e-08 | 0.290 |
| nine-well, U = 3 | MFPT | 40×40 vs saved 20×20 | 0.0271 | 1.6e-06 | 0.116 |
| nine-well, U = 3 | MFPT | 40×40 vs re-optimized 20×20 | 0.0271 | 1.7e-06 | 0.262 |
| nine-well, U = 6 | spectral | 20×20 re-optimized vs saved 20×20 | 0.0122 | 1.4e-04 | 0.310 |
| nine-well, U = 6 | spectral | 40×40 vs saved 20×20 | 0.0676 | 2.9e-04 | 0.341 |
| nine-well, U = 6 | spectral | 40×40 vs re-optimized 20×20 | 0.0666 | 1.1e-04 | 0.166 |
| nine-well, U = 6 | MFPT | 20×20 re-optimized vs saved 20×20 | 0.0003 | 9.7e-08 | 0.196 |
| nine-well, U = 6 | MFPT | 40×40 vs saved 20×20 | 0.0401 | 9.0e-07 | 0.178 |
| nine-well, U = 6 | MFPT | 40×40 vs re-optimized 20×20 | 0.0401 | 8.7e-07 | 0.154 |
| nine-well, U = 3 | spectral | 80×80 vs re-optimized 40×40 | 0.0218 | 1.1e-08 | 0.068 |
| nine-well, U = 3 | MFPT | 80×80 vs re-optimized 40×40 | 0.0170 | 1.9e-09 | 0.083 |
| nine-well, U = 6 | spectral | 80×80 vs re-optimized 40×40 | 0.0362 | 4.2e-06 | 0.086 |
| nine-well, U = 6 | MFPT | 80×80 vs re-optimized 40×40 | 0.0216 | 1.7e-06 | 0.107 |

### Second-route checks (largest deviation over every evaluated bias)

- MFPT, occupation solve and eigendecomposition vs LU solve: 7.1e-09 (relative)
- γ, Arnoldi on the non-symmetrised generator and dense eig vs Lanczos: 4.2e-08 (relative)
- C_TS, forward-occupation route vs exact_last_crossing.law: 1.6e-14; largest per-edge probability difference 3.3e-16; total last-crossing mass minus 1: 1.0e-13
- Ω_J, reversible BHS form vs general form: 6.8e-14
- last-crossing flux vs reactive capacity (relative): 6.7e-14
