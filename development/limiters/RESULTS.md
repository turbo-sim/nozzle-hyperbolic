# Limiter comparison results

Cold starts; shared perfect-gas nozzle shock_35 case. Full settings, histories, failures and source hashes are in the JSON records.

| Limiter | Cells | Status | Accepted / rejected | Pressure L1/p0 | Max conserved L1/scale | Shock error [m] | Max global defect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minmod | 24 | converged | 50 / 3 | 0.00298301 | 0.00445449 | 0.00166667 | 1.29505e-10 |
| superbee | 24 | converged | 76 / 10 | 0.00559276 | 0.0090058 | 0.00666667 | 5.75749e-10 |
| van_albada | 24 | converged | 46 / 2 | 0.00291002 | 0.00411007 | 0.00166667 | 1.3138e-10 |
| van_leer | 24 | converged | 55 / 5 | 0.00251402 | 0.0030913 | 0.00166667 | 6.85157e-10 |
| minmod | 48 | converged | 64 / 6 | 0.000805094 | 0.00127547 | 0.00166667 | 4.76483e-10 |
| superbee | 48 | converged | 156 / 30 | 0.00148452 | 0.00240002 | 0.0025 | 5.538e-10 |
| van_albada | 48 | converged | 59 / 4 | 0.000603409 | 0.000686189 | 0.00166667 | 1.72635e-10 |
| van_leer | 48 | converged | 79 / 10 | 0.00072479 | 0.000909628 | 0.00166667 | 2.52918e-10 |

Field errors use analytical conserved cell averages and fixed physical scales; shock position uses the strongest downstream pressure gradient. Failure rows remain visible and do not establish converged discretization errors. Timings are compile-inclusive and concurrent-work contaminated, so no speed claim is made.

![Profiles and convergence](outputs/comparison/comparison.png)
