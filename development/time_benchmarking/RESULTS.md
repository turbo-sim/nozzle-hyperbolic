# Solver time benchmarking

72 configurations and 468 measured complete solves. 72 fine-grid configurations converge; 65 meet original analytical targets.

Open [REPORT.html](REPORT.html) for the Plotly case explorer. Select the model, physical case, reconstruction and methods. Timing, field and convergence views share the selection.

The report is generated from the saved campaign; rendering performs no solves or timing measurements.

Grid sequencing carries the preceding successful solve's last accepted CFL, capped at the configured maximum, and recomputes the timestep on each grid. Cold starts and retries use the configured initial CFL; residual normalization is fresh on every grid.

With Vanilla Stadtke, sequencing changes measured 400-cell correction cost by -5.3% to +7.8% across these cases. Complete-solve totals also reflect different convergence paths and coarse-grid overhead.

With Modified Stadtke, sequencing changes measured 400-cell correction cost by -8.8% to +3.3% across these cases. Complete-solve totals also reflect different convergence paths and coarse-grid overhead.

Repeated final states are numerically identical in 72/72 configurations. Original analytical targets pass in 65/72 configurations; misses remain flagged below.

For cold starts on one grid in the shocked annular cases, modified Stadtke's mean complete-solve duration is 0.0628–0.261 times vanilla across 4 case/reconstruction pairs. Both methods pass convergence, branch and original analytical checks in 4/4 pairs. These are distinct flux discretizations evaluated at their own converged states.

Modified Stadtke · single: measured correction cost is 0.891–1.2 times Stadtke · single across 12 case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.

Modified Stadtke · sequence: measured correction cost is 0.899–1.11 times Stadtke · sequence across 12 case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.

HLL · sequence: measured correction cost is 0.376–0.744 times Stadtke · sequence across 12 case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.

Rusanov · sequence: measured correction cost is 0.379–0.779 times Stadtke · sequence across 12 case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.

euler_annular/shock35 with muscl_van_leer (Vanilla Stadtke): single-grid and sequenced fine states differ by up to 0.005975 of a fixed field scale (maximum velocity difference 0.976 m/s). Their 92.6 s versus 3.44 s totals are descriptive convergence-path timings, not a verified time-to-identical-state speedup. Original checks passed for both: True.

euler_annular/shock70 with constant (Vanilla Stadtke): single-grid and sequenced fine states differ by up to 0.2826 of a fixed field scale (maximum velocity difference 41.1 m/s). Their 60.6 s versus 1.66 s totals are descriptive convergence-path timings, not a verified time-to-identical-state speedup. Original checks passed for both: True.

22/24 same-flux pairs agree within the separate comparison threshold of 1e-6 of fixed field scales; their maximum deviation is 6.454e-07. This check does not change solver or analytical acceptance.

## Cost per correction

Means from twenty synchronized samples. A correction is the fixed twenty-correction block divided by twenty. Standalone operations include dispatch and are not additive to a fused correction.

| Case | Reconstruction | Method | Residual [ms] | Jacobian [ms] | Linear [ms] | Direction [ms] | Correction [ms] |
|---|---|---|---:|---:|---:|---:|---:|
| euler_nozzle/shock_35 | constant | Stadtke · single | 0.261 | 1.354 | 0.7086 | 2.145 | 1.833 |
| euler_nozzle/shock_35 | constant | Stadtke · sequence | 0.2703 | 1.456 | 0.7127 | 2.227 | 1.948 |
| euler_nozzle/shock_35 | constant | Modified Stadtke · single | 0.2733 | 1.571 | 0.8829 | 2.198 | 1.956 |
| euler_nozzle/shock_35 | constant | Modified Stadtke · sequence | 0.2533 | 1.486 | 0.9121 | 2.295 | 1.89 |
| euler_nozzle/shock_35 | constant | HLL · sequence | 0.1707 | 0.5613 | 0.8525 | 1.297 | 1.206 |
| euler_nozzle/shock_35 | constant | Rusanov · sequence | 0.1853 | 0.6875 | 1.103 | 1.813 | 1.502 |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · single | 0.2872 | 1.921 | 0.993 | 3.6 | 2.376 |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · sequence | 0.2832 | 1.88 | 1.007 | 2.945 | 2.257 |
| euler_nozzle/shock_35 | muscl_van_leer | Modified Stadtke · single | 0.3194 | 1.732 | 1.185 | 2.818 | 2.536 |
| euler_nozzle/shock_35 | muscl_van_leer | Modified Stadtke · sequence | 0.3109 | 1.873 | 1.109 | 3.461 | 2.51 |
| euler_nozzle/shock_35 | muscl_van_leer | HLL · sequence | 0.1939 | 0.8824 | 1.198 | 1.941 | 1.465 |
| euler_nozzle/shock_35 | muscl_van_leer | Rusanov · sequence | 0.1847 | 0.867 | 1.147 | 2.228 | 1.657 |
| euler_nozzle/shock_70 | constant | Stadtke · single | 0.3709 | 2.044 | 1.244 | 3.198 | 2.676 |
| euler_nozzle/shock_70 | constant | Stadtke · sequence | 0.3209 | 2.048 | 1.287 | 3.166 | 2.64 |
| euler_nozzle/shock_70 | constant | Modified Stadtke · single | 0.3189 | 1.83 | 1.104 | 2.865 | 2.668 |
| euler_nozzle/shock_70 | constant | Modified Stadtke · sequence | 0.3002 | 1.883 | 1.103 | 2.786 | 2.756 |
| euler_nozzle/shock_70 | constant | HLL · sequence | 0.2064 | 0.7626 | 1.179 | 1.869 | 1.715 |
| euler_nozzle/shock_70 | constant | Rusanov · sequence | 0.1833 | 0.6991 | 1.247 | 2.185 | 1.67 |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · single | 0.2811 | 1.843 | 1.134 | 2.989 | 2.457 |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · sequence | 0.2471 | 1.775 | 1.133 | 3.241 | 2.522 |
| euler_nozzle/shock_70 | muscl_van_leer | Modified Stadtke · single | 0.3112 | 1.753 | 1.091 | 3.147 | 2.953 |
| euler_nozzle/shock_70 | muscl_van_leer | Modified Stadtke · sequence | 0.2872 | 1.692 | 1.046 | 3.093 | 2.693 |
| euler_nozzle/shock_70 | muscl_van_leer | HLL · sequence | 0.1711 | 0.781 | 1.28 | 2.009 | 1.722 |
| euler_nozzle/shock_70 | muscl_van_leer | Rusanov · sequence | 0.2095 | 0.9472 | 1.333 | 1.995 | 1.688 |
| euler_nozzle/adapted | constant | Stadtke · single | 0.2429 | 1.624 | 0.9245 | 2.381 | 1.846 |
| euler_nozzle/adapted | constant | Stadtke · sequence | 0.2545 | 1.514 | 0.9194 | 2.601 | 1.989 |
| euler_nozzle/adapted | constant | Modified Stadtke · single | 0.2574 | 1.483 | 0.93 | 2.419 | 1.982 |
| euler_nozzle/adapted | constant | Modified Stadtke · sequence | 0.2552 | 1.363 | 0.9197 | 2.313 | 1.876 |
| euler_nozzle/adapted | constant | HLL · sequence | 0.1756 | 0.6832 | 0.8247 | 1.39 | 1.199 |
| euler_nozzle/adapted | constant | Rusanov · sequence | 0.191 | 0.6869 | 0.8344 | 1.617 | 1.334 |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · single | 0.2505 | 1.777 | 0.9693 | 2.707 | 1.969 |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · sequence | 0.2694 | 1.62 | 0.9267 | 2.434 | 1.864 |
| euler_nozzle/adapted | muscl_van_leer | Modified Stadtke · single | 0.2641 | 1.62 | 0.8958 | 2.565 | 2.004 |
| euler_nozzle/adapted | muscl_van_leer | Modified Stadtke · sequence | 0.2717 | 1.759 | 0.9454 | 2.631 | 1.993 |
| euler_nozzle/adapted | muscl_van_leer | HLL · sequence | 0.1703 | 0.8021 | 0.8381 | 1.948 | 1.387 |
| euler_nozzle/adapted | muscl_van_leer | Rusanov · sequence | 0.1814 | 0.8463 | 0.9157 | 1.728 | 1.453 |
| euler_annular/shock35 | constant | Stadtke · single | 0.429 | 2.926 | 0.8004 | 3.382 | 3.231 |
| euler_annular/shock35 | constant | Stadtke · sequence | 0.4502 | 3.058 | 0.8524 | 3.402 | 3.454 |
| euler_annular/shock35 | constant | Modified Stadtke · single | 0.4197 | 2.504 | 0.8376 | 3.127 | 3.165 |
| euler_annular/shock35 | constant | Modified Stadtke · sequence | 0.4345 | 2.595 | 0.8382 | 3.408 | 3.105 |
| euler_annular/shock35 | constant | HLL · sequence | 0.1596 | 0.9702 | 0.8166 | 1.743 | 1.298 |
| euler_annular/shock35 | constant | Rusanov · sequence | 0.1294 | 0.9633 | 0.8016 | 1.67 | 1.31 |
| euler_annular/shock35 | muscl_van_leer | Stadtke · single | 0.592 | 4.271 | 1.458 | 5.451 | 5.083 |
| euler_annular/shock35 | muscl_van_leer | Stadtke · sequence | 0.4991 | 3.997 | 1.162 | 4.567 | 5.021 |
| euler_annular/shock35 | muscl_van_leer | Modified Stadtke · single | 0.717 | 4.094 | 1.535 | 4.919 | 5.074 |
| euler_annular/shock35 | muscl_van_leer | Modified Stadtke · sequence | 0.6303 | 3.742 | 1.272 | 4.855 | 4.809 |
| euler_annular/shock35 | muscl_van_leer | HLL · sequence | 0.1962 | 1.458 | 1.39 | 2.76 | 2.157 |
| euler_annular/shock35 | muscl_van_leer | Rusanov · sequence | 0.1781 | 1.407 | 1.285 | 3.104 | 2.355 |
| euler_annular/shock70 | constant | Stadtke · single | 0.3571 | 2.389 | 0.7034 | 2.901 | 2.6 |
| euler_annular/shock70 | constant | Stadtke · sequence | 0.3373 | 2.399 | 0.7051 | 2.942 | 2.562 |
| euler_annular/shock70 | constant | Modified Stadtke · single | 0.3935 | 2.006 | 0.6441 | 2.761 | 2.502 |
| euler_annular/shock70 | constant | Modified Stadtke · sequence | 0.3452 | 2.122 | 0.6936 | 2.602 | 2.543 |
| euler_annular/shock70 | constant | HLL · sequence | 0.1447 | 0.8287 | 0.6429 | 1.371 | 1.055 |
| euler_annular/shock70 | constant | Rusanov · sequence | 0.1097 | 0.791 | 0.724 | 1.41 | 1.121 |
| euler_annular/shock70 | muscl_van_leer | Stadtke · single | 0.5692 | 3.884 | 1.232 | 4.965 | 4.661 |
| euler_annular/shock70 | muscl_van_leer | Stadtke · sequence | 0.5214 | 3.989 | 1.431 | 4.865 | 4.932 |
| euler_annular/shock70 | muscl_van_leer | Modified Stadtke · single | 0.6177 | 3.562 | 1.577 | 4.886 | 4.884 |
| euler_annular/shock70 | muscl_van_leer | Modified Stadtke · sequence | 0.6229 | 3.312 | 1.195 | 4.963 | 4.886 |
| euler_annular/shock70 | muscl_van_leer | HLL · sequence | 0.1953 | 1.481 | 1.447 | 2.822 | 2.188 |
| euler_annular/shock70 | muscl_van_leer | Rusanov · sequence | 0.1603 | 1.392 | 1.385 | 2.939 | 2.189 |
| euler_annular/supersonic_expansion | constant | Stadtke · single | 0.6328 | 4.473 | 1.717 | 6.388 | 7.133 |
| euler_annular/supersonic_expansion | constant | Stadtke · sequence | 0.6496 | 4.661 | 1.724 | 7.037 | 7.139 |
| euler_annular/supersonic_expansion | constant | Modified Stadtke · single | 0.7994 | 4.275 | 1.92 | 6.065 | 6.353 |
| euler_annular/supersonic_expansion | constant | Modified Stadtke · sequence | 0.6374 | 4.255 | 1.772 | 6.207 | 6.493 |
| euler_annular/supersonic_expansion | constant | HLL · sequence | 0.1961 | 1.729 | 1.988 | 3.782 | 3.229 |
| euler_annular/supersonic_expansion | constant | Rusanov · sequence | 0.1744 | 1.656 | 1.69 | 3.292 | 3.191 |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · single | 0.5097 | 3.338 | 1.015 | 4.323 | 3.6 |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · sequence | 0.4455 | 3.116 | 0.9651 | 4.066 | 3.663 |
| euler_annular/supersonic_expansion | muscl_van_leer | Modified Stadtke · single | 0.5378 | 3.03 | 0.9711 | 3.621 | 4.011 |
| euler_annular/supersonic_expansion | muscl_van_leer | Modified Stadtke · sequence | 0.5373 | 4.589 | 1.09 | 4.191 | 4.004 |
| euler_annular/supersonic_expansion | muscl_van_leer | HLL · sequence | 0.2243 | 1.127 | 0.9964 | 2.486 | 1.756 |
| euler_annular/supersonic_expansion | muscl_van_leer | Rusanov · sequence | 0.1837 | 1.229 | 0.9403 | 1.942 | 1.737 |

## Complete-solve timing

Totals include every coarse solve, prolongation, rejected work and cold retry. Production multigrid is 50→100→200→400 grid sequencing.

| Case | Reconstruction | Method | n | Mean [s] | Min [s] | Max [s] | Accuracy |
|---|---|---|---:|---:|---:|---:|---|
| euler_nozzle/shock_35 | constant | Stadtke · single | 10 | 1.002 | 0.8293 | 1.129 | True |
| euler_nozzle/shock_35 | constant | Stadtke · sequence | 10 | 0.4072 | 0.3615 | 0.459 | True |
| euler_nozzle/shock_35 | constant | Modified Stadtke · single | 10 | 0.898 | 0.7752 | 0.9964 | True |
| euler_nozzle/shock_35 | constant | Modified Stadtke · sequence | 10 | 0.3061 | 0.2604 | 0.3342 | True |
| euler_nozzle/shock_35 | constant | HLL · sequence | 10 | 0.2489 | 0.2203 | 0.3024 | True |
| euler_nozzle/shock_35 | constant | Rusanov · sequence | 10 | 0.2455 | 0.2003 | 0.2814 | False |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · single | 10 | 1.783 | 1.608 | 2.156 | True |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · sequence | 10 | 0.3746 | 0.3193 | 0.4594 | True |
| euler_nozzle/shock_35 | muscl_van_leer | Modified Stadtke · single | 10 | 1.799 | 1.564 | 2.017 | True |
| euler_nozzle/shock_35 | muscl_van_leer | Modified Stadtke · sequence | 10 | 0.332 | 0.2806 | 0.4178 | True |
| euler_nozzle/shock_35 | muscl_van_leer | HLL · sequence | 10 | 0.2721 | 0.2366 | 0.3186 | True |
| euler_nozzle/shock_35 | muscl_van_leer | Rusanov · sequence | 10 | 0.1767 | 0.1587 | 0.192 | True |
| euler_nozzle/shock_70 | constant | Stadtke · single | 10 | 1.941 | 1.601 | 2.411 | True |
| euler_nozzle/shock_70 | constant | Stadtke · sequence | 10 | 0.7785 | 0.6426 | 0.9708 | True |
| euler_nozzle/shock_70 | constant | Modified Stadtke · single | 10 | 1.403 | 1.202 | 1.661 | True |
| euler_nozzle/shock_70 | constant | Modified Stadtke · sequence | 10 | 0.3837 | 0.3137 | 0.4745 | True |
| euler_nozzle/shock_70 | constant | HLL · sequence | 10 | 0.3269 | 0.2488 | 0.4618 | True |
| euler_nozzle/shock_70 | constant | Rusanov · sequence | 10 | 0.3058 | 0.2404 | 0.5304 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · single | 10 | 3.025 | 2.396 | 3.936 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · sequence | 10 | 0.5006 | 0.3894 | 0.6389 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Modified Stadtke · single | 10 | 2.247 | 1.789 | 2.996 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Modified Stadtke · sequence | 10 | 0.4756 | 0.3351 | 0.785 | True |
| euler_nozzle/shock_70 | muscl_van_leer | HLL · sequence | 10 | 0.5925 | 0.4147 | 0.9713 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Rusanov · sequence | 10 | 0.4207 | 0.3129 | 0.621 | True |
| euler_nozzle/adapted | constant | Stadtke · single | 10 | 0.5247 | 0.3616 | 0.7966 | True |
| euler_nozzle/adapted | constant | Stadtke · sequence | 10 | 0.2731 | 0.1778 | 0.4189 | True |
| euler_nozzle/adapted | constant | Modified Stadtke · single | 10 | 0.4951 | 0.3493 | 0.744 | True |
| euler_nozzle/adapted | constant | Modified Stadtke · sequence | 10 | 0.2733 | 0.1725 | 0.4482 | True |
| euler_nozzle/adapted | constant | HLL · sequence | 10 | 0.2368 | 0.1741 | 0.3657 | True |
| euler_nozzle/adapted | constant | Rusanov · sequence | 10 | 0.2416 | 0.1583 | 0.3368 | True |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · single | 10 | 0.3955 | 0.3674 | 0.4734 | True |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · sequence | 10 | 0.1761 | 0.1561 | 0.206 | True |
| euler_nozzle/adapted | muscl_van_leer | Modified Stadtke · single | 10 | 0.4186 | 0.3833 | 0.4858 | True |
| euler_nozzle/adapted | muscl_van_leer | Modified Stadtke · sequence | 10 | 0.1689 | 0.1558 | 0.1959 | True |
| euler_nozzle/adapted | muscl_van_leer | HLL · sequence | 10 | 0.1491 | 0.1324 | 0.1691 | True |
| euler_nozzle/adapted | muscl_van_leer | Rusanov · sequence | 10 | 0.1503 | 0.1296 | 0.1872 | True |
| euler_annular/shock35 | constant | Stadtke · single | 3 | 62.84 | 61.04 | 64.34 | True |
| euler_annular/shock35 | constant | Stadtke · sequence | 3 | 1.524 | 1.393 | 1.639 | True |
| euler_annular/shock35 | constant | Modified Stadtke · single | 3 | 5.278 | 4.781 | 5.677 | True |
| euler_annular/shock35 | constant | Modified Stadtke · sequence | 3 | 0.3604 | 0.3432 | 0.3903 | True |
| euler_annular/shock35 | constant | HLL · sequence | 3 | 0.2371 | 0.2314 | 0.2482 | True |
| euler_annular/shock35 | constant | Rusanov · sequence | 3 | 0.1973 | 0.1945 | 0.2026 | True |
| euler_annular/shock35 | muscl_van_leer | Stadtke · single | 3 | 92.64 | 81.36 | 111.1 | True |
| euler_annular/shock35 | muscl_van_leer | Stadtke · sequence | 3 | 3.437 | 2.64 | 5.016 | True |
| euler_annular/shock35 | muscl_van_leer | Modified Stadtke · single | 3 | 24.18 | 20.77 | 30.6 | True |
| euler_annular/shock35 | muscl_van_leer | Modified Stadtke · sequence | 3 | 1.362 | 1.039 | 1.922 | True |
| euler_annular/shock35 | muscl_van_leer | HLL · sequence | 3 | 1.076 | 0.8723 | 1.422 | True |
| euler_annular/shock35 | muscl_van_leer | Rusanov · sequence | 3 | 0.5519 | 0.4642 | 0.6456 | True |
| euler_annular/shock70 | constant | Stadtke · single | 3 | 60.61 | 53.3 | 65.86 | True |
| euler_annular/shock70 | constant | Stadtke · sequence | 3 | 1.662 | 1.143 | 2.082 | True |
| euler_annular/shock70 | constant | Modified Stadtke · single | 3 | 3.808 | 2.341 | 4.739 | True |
| euler_annular/shock70 | constant | Modified Stadtke · sequence | 3 | 0.464 | 0.2817 | 0.6264 | True |
| euler_annular/shock70 | constant | HLL · sequence | 3 | 0.2641 | 0.1739 | 0.3152 | True |
| euler_annular/shock70 | constant | Rusanov · sequence | 3 | 0.2766 | 0.1658 | 0.354 | True |
| euler_annular/shock70 | muscl_van_leer | Stadtke · single | 3 | 90.64 | 76.41 | 115.5 | True |
| euler_annular/shock70 | muscl_van_leer | Stadtke · sequence | 3 | 2.682 | 2.517 | 2.944 | True |
| euler_annular/shock70 | muscl_van_leer | Modified Stadtke · single | 3 | 23.38 | 21.36 | 27.05 | True |
| euler_annular/shock70 | muscl_van_leer | Modified Stadtke · sequence | 3 | 2.084 | 1.63 | 2.652 | True |
| euler_annular/shock70 | muscl_van_leer | HLL · sequence | 3 | 1.162 | 0.9059 | 1.469 | True |
| euler_annular/shock70 | muscl_van_leer | Rusanov · sequence | 3 | 0.6007 | 0.4329 | 0.8645 | True |
| euler_annular/supersonic_expansion | constant | Stadtke · single | 3 | 1.097 | 1.08 | 1.119 | False |
| euler_annular/supersonic_expansion | constant | Stadtke · sequence | 3 | 0.4451 | 0.4075 | 0.5023 | False |
| euler_annular/supersonic_expansion | constant | Modified Stadtke · single | 3 | 1.068 | 1.044 | 1.098 | False |
| euler_annular/supersonic_expansion | constant | Modified Stadtke · sequence | 3 | 0.431 | 0.3916 | 0.4902 | False |
| euler_annular/supersonic_expansion | constant | HLL · sequence | 3 | 0.2994 | 0.2587 | 0.3239 | False |
| euler_annular/supersonic_expansion | constant | Rusanov · sequence | 3 | 0.2995 | 0.2661 | 0.3231 | False |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · single | 3 | 10.23 | 8.789 | 10.96 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · sequence | 3 | 0.7143 | 0.5693 | 0.7954 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Modified Stadtke · single | 3 | 11.24 | 9.712 | 12.21 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Modified Stadtke · sequence | 3 | 0.719 | 0.6514 | 0.7965 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | HLL · sequence | 3 | 0.4503 | 0.3922 | 0.5185 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Rusanov · sequence | 3 | 0.4084 | 0.3382 | 0.4534 | True |

## Single grid versus sequencing

Each row compares the same Stadtke variant. A timing ratio is descriptive when original checks or the separate field-agreement check fail.

| Case | Reconstruction | Variant | Single [s] | Sequence [s] | Sequence / single | Original checks | Field agreement |
|---|---|---|---:|---:|---:|---|---|
| euler_nozzle/shock_35 | constant | Vanilla Stadtke | 1.002 | 0.4072 | 0.4063 | True | True |
| euler_nozzle/shock_35 | constant | Modified Stadtke | 0.898 | 0.3061 | 0.3409 | True | True |
| euler_nozzle/shock_35 | muscl_van_leer | Vanilla Stadtke | 1.783 | 0.3746 | 0.2101 | True | True |
| euler_nozzle/shock_35 | muscl_van_leer | Modified Stadtke | 1.799 | 0.332 | 0.1846 | True | True |
| euler_nozzle/shock_70 | constant | Vanilla Stadtke | 1.941 | 0.7785 | 0.4012 | True | True |
| euler_nozzle/shock_70 | constant | Modified Stadtke | 1.403 | 0.3837 | 0.2734 | True | True |
| euler_nozzle/shock_70 | muscl_van_leer | Vanilla Stadtke | 3.025 | 0.5006 | 0.1655 | True | True |
| euler_nozzle/shock_70 | muscl_van_leer | Modified Stadtke | 2.247 | 0.4756 | 0.2116 | True | True |
| euler_nozzle/adapted | constant | Vanilla Stadtke | 0.5247 | 0.2731 | 0.5205 | True | True |
| euler_nozzle/adapted | constant | Modified Stadtke | 0.4951 | 0.2733 | 0.552 | True | True |
| euler_nozzle/adapted | muscl_van_leer | Vanilla Stadtke | 0.3955 | 0.1761 | 0.4452 | True | True |
| euler_nozzle/adapted | muscl_van_leer | Modified Stadtke | 0.4186 | 0.1689 | 0.4035 | True | True |
| euler_annular/shock35 | constant | Vanilla Stadtke | 62.84 | 1.524 | 0.02426 | True | True |
| euler_annular/shock35 | constant | Modified Stadtke | 5.278 | 0.3604 | 0.06827 | True | True |
| euler_annular/shock35 | muscl_van_leer | Vanilla Stadtke | 92.64 | 3.437 | 0.0371 | True | False |
| euler_annular/shock35 | muscl_van_leer | Modified Stadtke | 24.18 | 1.362 | 0.05634 | True | True |
| euler_annular/shock70 | constant | Vanilla Stadtke | 60.61 | 1.662 | 0.02743 | True | False |
| euler_annular/shock70 | constant | Modified Stadtke | 3.808 | 0.464 | 0.1219 | True | True |
| euler_annular/shock70 | muscl_van_leer | Vanilla Stadtke | 90.64 | 2.682 | 0.02959 | True | True |
| euler_annular/shock70 | muscl_van_leer | Modified Stadtke | 23.38 | 2.084 | 0.08914 | True | True |
| euler_annular/supersonic_expansion | constant | Vanilla Stadtke | 1.097 | 0.4451 | 0.4056 | False | True |
| euler_annular/supersonic_expansion | constant | Modified Stadtke | 1.068 | 0.431 | 0.4036 | False | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Vanilla Stadtke | 10.23 | 0.7143 | 0.06981 | True | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Modified Stadtke | 11.24 | 0.719 | 0.06395 | True | True |

Measured source SHA-256: `0d206dc6c6d47c322e0be833ad53a34aecb80ea78529b2828e1b52f3858c172d`.
