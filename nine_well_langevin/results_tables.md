### Network predictions (A -> B MFPT, physical time, D = 1)

| Case | vertex grid 120 x 120 (audit grid) | aligned grids n = 114 / 228 / 456 | h -> 0, SQRA | h -> 0, Scharfetter-Gummel | h -> 0, central differences |
|---|---:|---:|---:|---:|---:|
| unbiased | 7.220e5 | 7.217e5 / 7.238e5 / 7.243e5 | 7.245e5 | 7.245e5 | 7.268e5 |
| MFPT bias, U = 3 | 2015 | 2015 / 2020 / 2021 | 2022 | 2022 | 2025 |
| spectral bias, U = 3 | 13792 | 13494 / 13266 / 13148 | 13017 | 13019 | 13075 |
| MFPT bias, U = 6 | 23.25 | 23.21 / 23.22 / 23.22 | 23.22 | 23.22 | 23.22 |
| spectral bias, U = 6 | 1.704e5 | 1.590e5 / 1.498e5 / 1.452e5 | 1.408e5 | 1.408e5 | - |
| MFPT bias, U = 3 (bicubic) | 2038 | 2037 / 2042 / 2044 | 2044 | 2044 | 2048 |
| spectral bias, U = 3 (bicubic) | 14250 | 13945 / 13664 / 13516 | 13355 | 13357 | - |
| MFPT bias, U = 6 (bicubic) | 22.5 | 22.44 / 22.46 / 22.46 | 22.46 | 22.46 | - |
| spectral bias, U = 6 (bicubic) | 2.070e5 | 1.924e5 / 1.793e5 / 1.729e5 | 1.667e5 | 1.668e5 | - |

### Langevin against network

| Case | Langevin MFPT (95% CI) | per time step: dt: walkers or transitions (by host), MFPT | network, h -> 0 | Langevin / network (95% CI) |
|---|---:|---|---:|---:|
| unbiased | 7.040e5 [6.722e5, 7.386e5] | 0.0004: 1177+132 running (laptop 186, markov 991), 7.062e5; 0.0002: 558+120 running (laptop 37, markov 521), 6.993e5 | 7.245e5 | 0.972 [0.928, 1.02] |
| MFPT bias, U = 3 | 2001 [1956, 2047] | 0.0004: 3000 (laptop 1000, markov 2000), 2024; 0.0002: 3000 (laptop 1000, markov 2000), 1949; 0.0001: 1500 (laptop 500, markov 1000), 2071 | 2022 | 0.990 [0.967, 1.01]; widened x1.21 for run scatter: [0.963, 1.02] |
| spectral bias, U = 3 | 13413 [13070, 13759] | 0.0004: 2300 (laptop 300, markov 2000), 12898; 0.0002: 2300 (laptop 300, markov 2000), 13828; 0.0001: 1000 (markov), 13792 | 13017 | 1.030 [1, 1.06]; widened x1.79 for run scatter: [0.983, 1.08] |
| MFPT bias, U = 6 | 23.52 [23.19, 23.86] | 0.0004: 5000 (laptop), 23.64; 0.0002: 5000 (laptop), 23.5; 0.0001: 5000 (laptop), 23.42 | 23.22 | 1.013 [0.999, 1.03] |
| spectral bias, U = 6 | 1.444e5 [1.340e5, 1.551e5] | 0.0004: 500 (laptop 100, markov 400), 1.434e5; 0.0002: 200 (markov), 1.469e5 | 1.408e5 | 1.026 [0.952, 1.1] |
| MFPT bias, U = 3 (bicubic) | 1957 [1891, 2024] | 0.0004: 2000 (laptop 1000, markov 1000), 1991; 0.0002: 1500 (laptop 500, markov 1000), 1915 | 2044 | 0.957 [0.925, 0.99] |
| spectral bias, U = 3 (bicubic) | 13640 [12812, 14481] | 0.0004: 1000 (laptop 300, markov 700), 13640 | 13355 | 1.021 [0.959, 1.08] |
| MFPT bias, U = 6 (bicubic) | 22.87 [22.46, 23.28] | 0.0004: 5000 (laptop), 22.82; 0.0002: 5000 (laptop), 22.92 | 22.46 | 1.018 [1, 1.04] |
| spectral bias, U = 6 (bicubic) | 1.833e5 [1.660e5, 2.032e5] | 0.0004: 371+12 running (laptop 71, markov 300), 1.833e5 | 1.667e5 | 1.100 [0.995, 1.22] |

### Speedups (unbiased MFPT / biased MFPT)

| Bias | network | Langevin / Langevin (95% CI) | network unbiased / Langevin biased (95% CI) |
|---|---:|---:|---:|
| MFPT bias, U = 3 | 358.3 | 352 [334, 371] | 362 [354, 370.4] |
| spectral bias, U = 3 | 55.65 | 52.5 [49.8, 55.4] | 54.01 [52.65, 55.43] |
| MFPT bias, U = 6 | 31197 | 29928 [28501, 31456] | 30798 [30364, 31243] |
| spectral bias, U = 6 | 5.147 | 4.88 [4.48, 5.33] | 5.018 [4.671, 5.408] |
| MFPT bias, U = 3 (bicubic) | 354.4 | 360 [340, 382] | 370.2 [358, 383.1] |
| spectral bias, U = 3 (bicubic) | 54.25 | 51.6 [47.9, 55.9] | 53.11 [50.03, 56.54] |
| MFPT bias, U = 6 (bicubic) | 32255 | 30785 [29290, 32397] | 31680 [31125, 32251] |
| spectral bias, U = 6 (bicubic) | 4.345 | 3.84 [3.43, 4.28] | 3.951 [3.565, 4.365] |

### Ranking: T(spectral bias) / T(MFPT bias) = speedup(MFPT bias) / speedup(spectral bias)

| U, interpolation | network | Langevin (95% CI) |
|---|---:|---:|
| U3, bilinear | 6.439 | 6.702 [6.476, 6.935] |
| U3, bicubic | 6.533 | 6.97 [6.49, 7.473] |
| U6, bilinear | 6061 | 6137 [5690, 6600] |
| U6, bicubic | 7423 | 8017 [7239, 8904] |

### Time-step trend (diagnostic): weighted line through the per-dt means

| Case | dt values | relative change from dt -> 0 to the largest dt (95% CI) |
|---|---|---:|
| MFPT bias, U = 3 | 0.0001, 0.0002, 0.0004 | +0.7% [-6.8%, +8.7%] |
| MFPT bias, U = 3 (bicubic) | 0.0002, 0.0004 | +8.2% [-5.6%, +25.4%] |
| spectral bias, U = 3 | 0.0001, 0.0002, 0.0004 | -10.2% [-17.4%, -2.3%] |
| MFPT bias, U = 6 | 0.0001, 0.0002, 0.0004 | +1.2% [-3.2%, +6.0%] |
| MFPT bias, U = 6 (bicubic) | 0.0002, 0.0004 | -0.9% [-7.4%, +6.4%] |
| spectral bias, U = 6 | 0.0002, 0.0004 | -4.7% [-28.6%, +34.5%] |
