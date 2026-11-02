# Production solver benchmark

60 configurations and 390 measured complete solves. All 60 fine-grid solves converge; 54 meet original analytical targets.

Open [REPORT.html](REPORT.html) for the Plotly case explorer. Select the model, physical case, reconstruction and methods. Timing, field and convergence views share the selection.

The report reads the saved campaign only. No timing or numerical solution was rerun.

With vanilla Stadtke, sequencing changes measured 400-cell correction cost by -4.8% to +3.8% across these cases. Complete-solve totals also reflect different convergence paths and coarse-grid overhead.

The public-input removal passed 87 focused tests. 12/12 vanilla single-grid final states are bitwise identical to the prior campaign with reuse enabled.

Repeated final states are numerically identical in 60/60 configurations. Original analytical targets pass in 54/60 configurations; misses remain flagged below.

Modified Stadtke · sequence: measured correction cost is 0.918–1.07 times sequenced vanilla Stadtke across 12 case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.

HLL · sequence: measured correction cost is 0.398–0.712 times sequenced vanilla Stadtke across 12 case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.

Rusanov · sequence: measured correction cost is 0.405–0.772 times sequenced vanilla Stadtke across 12 case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.

euler_annular/shock35 with muscl_van_leer: single-grid and sequenced fine states differ by up to 0.005975 of a fixed field scale (maximum velocity difference 0.976 m/s). Their 103 s versus 3.96 s totals are descriptive convergence-path timings, not a verified time-to-identical-state speedup. Original checks passed for both: True.

11/12 same-flux pairs agree within the separate comparison threshold of 1e-6 of fixed field scales; their maximum deviation is 5.9e-07. This check does not change solver or analytical acceptance.

## Cost per correction

Means from twenty synchronized samples. A correction is the fixed twenty-correction block divided by twenty. Standalone operations include dispatch and are not additive to a fused correction.

| Case | Reconstruction | Method | Residual [ms] | Jacobian [ms] | Linear [ms] | Direction [ms] | Correction [ms] |
|---|---|---|---:|---:|---:|---:|---:|
| euler_nozzle/shock_35 | constant | Stadtke · single | 0.2181 | 1.182 | 0.612 | 1.825 | 1.263 |
| euler_nozzle/shock_35 | constant | Stadtke · sequence | 0.2112 | 1.153 | 0.5711 | 1.683 | 1.245 |
| euler_nozzle/shock_35 | constant | Modified Stadtke · sequence | 0.2171 | 1.14 | 0.5552 | 1.705 | 1.225 |
| euler_nozzle/shock_35 | constant | HLL · sequence | 0.1679 | 0.4533 | 0.6192 | 0.905 | 0.8094 |
| euler_nozzle/shock_35 | constant | Rusanov · sequence | 0.1559 | 0.4452 | 0.5589 | 1.071 | 0.8516 |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · single | 0.1924 | 1.206 | 0.5863 | 1.989 | 1.272 |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · sequence | 0.221 | 1.25 | 0.6239 | 1.822 | 1.321 |
| euler_nozzle/shock_35 | muscl_van_leer | Modified Stadtke · sequence | 0.2221 | 1.155 | 0.6263 | 1.924 | 1.377 |
| euler_nozzle/shock_35 | muscl_van_leer | HLL · sequence | 0.1822 | 0.6195 | 0.7359 | 1.452 | 0.9409 |
| euler_nozzle/shock_35 | muscl_van_leer | Rusanov · sequence | 0.1722 | 0.582 | 0.6752 | 1.503 | 1.02 |
| euler_nozzle/shock_70 | constant | Stadtke · single | 0.1736 | 0.9875 | 0.5233 | 1.509 | 1.172 |
| euler_nozzle/shock_70 | constant | Stadtke · sequence | 0.3187 | 1.082 | 0.5433 | 1.836 | 1.182 |
| euler_nozzle/shock_70 | constant | Modified Stadtke · sequence | 0.1653 | 1.233 | 0.5475 | 1.468 | 1.169 |
| euler_nozzle/shock_70 | constant | HLL · sequence | 0.1248 | 0.4648 | 0.5501 | 0.9334 | 0.8012 |
| euler_nozzle/shock_70 | constant | Rusanov · sequence | 0.1267 | 0.4692 | 0.5074 | 1.025 | 0.8654 |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · single | 0.1564 | 1.184 | 0.5275 | 1.765 | 1.193 |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · sequence | 0.1706 | 1.091 | 0.587 | 1.665 | 1.229 |
| euler_nozzle/shock_70 | muscl_van_leer | Modified Stadtke · sequence | 0.1779 | 1.07 | 0.583 | 1.729 | 1.193 |
| euler_nozzle/shock_70 | muscl_van_leer | HLL · sequence | 0.138 | 0.5207 | 0.5873 | 1.221 | 0.838 |
| euler_nozzle/shock_70 | muscl_van_leer | Rusanov · sequence | 0.136 | 0.587 | 0.5563 | 1.227 | 0.851 |
| euler_nozzle/adapted | constant | Stadtke · single | 0.1945 | 1.068 | 0.629 | 1.797 | 1.25 |
| euler_nozzle/adapted | constant | Stadtke · sequence | 0.1889 | 1.076 | 0.7471 | 1.814 | 1.214 |
| euler_nozzle/adapted | constant | Modified Stadtke · sequence | 0.1967 | 0.9623 | 0.7766 | 1.676 | 1.304 |
| euler_nozzle/adapted | constant | HLL · sequence | 0.1349 | 0.4498 | 0.5552 | 1.037 | 0.8432 |
| euler_nozzle/adapted | constant | Rusanov · sequence | 0.1272 | 0.4021 | 0.5681 | 0.9865 | 0.9352 |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · single | 0.1677 | 1.158 | 0.5801 | 1.702 | 1.242 |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · sequence | 0.1738 | 1.158 | 0.5835 | 1.789 | 1.244 |
| euler_nozzle/adapted | muscl_van_leer | Modified Stadtke · sequence | 0.1797 | 1.015 | 0.5715 | 1.648 | 1.247 |
| euler_nozzle/adapted | muscl_van_leer | HLL · sequence | 0.1271 | 0.4572 | 0.5513 | 1.12 | 0.8535 |
| euler_nozzle/adapted | muscl_van_leer | Rusanov · sequence | 0.1195 | 0.4912 | 0.5714 | 1.001 | 0.8824 |
| euler_annular/shock35 | constant | Stadtke · single | 0.3539 | 2.775 | 0.6692 | 3.206 | 2.769 |
| euler_annular/shock35 | constant | Stadtke · sequence | 0.6108 | 2.58 | 0.6837 | 3.28 | 2.82 |
| euler_annular/shock35 | constant | Modified Stadtke · sequence | 0.3859 | 2.174 | 0.6741 | 2.677 | 2.59 |
| euler_annular/shock35 | constant | HLL · sequence | 0.1456 | 0.8097 | 0.6437 | 1.345 | 1.124 |
| euler_annular/shock35 | constant | Rusanov · sequence | 0.1181 | 0.8261 | 0.7322 | 1.759 | 1.141 |
| euler_annular/shock35 | muscl_van_leer | Stadtke · single | 0.3167 | 2.284 | 0.6449 | 2.896 | 2.392 |
| euler_annular/shock35 | muscl_van_leer | Stadtke · sequence | 0.3274 | 2.266 | 0.6505 | 2.712 | 2.334 |
| euler_annular/shock35 | muscl_van_leer | Modified Stadtke · sequence | 0.3803 | 1.97 | 0.6356 | 2.553 | 2.377 |
| euler_annular/shock35 | muscl_van_leer | HLL · sequence | 0.1491 | 0.813 | 0.6282 | 1.439 | 1.049 |
| euler_annular/shock35 | muscl_van_leer | Rusanov · sequence | 0.1251 | 0.7944 | 0.6185 | 1.354 | 0.9976 |
| euler_annular/shock70 | constant | Stadtke · single | 0.3087 | 2.252 | 0.5969 | 2.831 | 2.442 |
| euler_annular/shock70 | constant | Stadtke · sequence | 0.3454 | 2.296 | 0.6311 | 2.893 | 2.472 |
| euler_annular/shock70 | constant | Modified Stadtke · sequence | 0.3909 | 1.928 | 0.604 | 2.438 | 2.31 |
| euler_annular/shock70 | constant | HLL · sequence | 0.1346 | 0.8162 | 0.6391 | 1.421 | 0.9891 |
| euler_annular/shock70 | constant | Rusanov · sequence | 0.0952 | 0.7143 | 0.5936 | 1.285 | 1.031 |
| euler_annular/shock70 | muscl_van_leer | Stadtke · single | 0.3475 | 2.336 | 0.6344 | 2.812 | 2.305 |
| euler_annular/shock70 | muscl_van_leer | Stadtke · sequence | 0.3456 | 2.266 | 0.6302 | 2.779 | 2.375 |
| euler_annular/shock70 | muscl_van_leer | Modified Stadtke · sequence | 0.3647 | 2.107 | 0.6364 | 2.52 | 2.348 |
| euler_annular/shock70 | muscl_van_leer | HLL · sequence | 0.152 | 0.8422 | 0.6236 | 1.41 | 1.07 |
| euler_annular/shock70 | muscl_van_leer | Rusanov · sequence | 0.1256 | 0.874 | 0.6252 | 1.47 | 0.9957 |
| euler_annular/supersonic_expansion | constant | Stadtke · single | 0.3375 | 2.013 | 0.6531 | 3.016 | 2.497 |
| euler_annular/supersonic_expansion | constant | Stadtke · sequence | 0.3357 | 2.04 | 0.6365 | 3.022 | 2.376 |
| euler_annular/supersonic_expansion | constant | Modified Stadtke · sequence | 0.3682 | 2.019 | 0.682 | 2.761 | 2.428 |
| euler_annular/supersonic_expansion | constant | HLL · sequence | 0.1471 | 0.8139 | 0.6369 | 1.372 | 1.03 |
| euler_annular/supersonic_expansion | constant | Rusanov · sequence | 0.1184 | 0.7813 | 0.6431 | 1.317 | 1.075 |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · single | 0.3446 | 2.176 | 0.6714 | 2.854 | 2.449 |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · sequence | 0.345 | 2.255 | 0.6496 | 3.013 | 2.496 |
| euler_annular/supersonic_expansion | muscl_van_leer | Modified Stadtke · sequence | 0.3738 | 2.271 | 0.671 | 2.588 | 2.397 |
| euler_annular/supersonic_expansion | muscl_van_leer | HLL · sequence | 0.1491 | 0.7963 | 0.6554 | 1.438 | 1.092 |
| euler_annular/supersonic_expansion | muscl_van_leer | Rusanov · sequence | 0.1604 | 0.8626 | 0.6291 | 1.403 | 1.072 |

## Complete-solve timing

Totals include every coarse solve, prolongation, rejected work and cold retry. Production multigrid is 50→100→200→400 grid sequencing.

| Case | Reconstruction | Method | n | Mean [s] | Min [s] | Max [s] | Accuracy |
|---|---|---|---:|---:|---:|---:|---|
| euler_nozzle/shock_35 | constant | Stadtke · single | 10 | 0.7254 | 0.6677 | 0.7987 | True |
| euler_nozzle/shock_35 | constant | Stadtke · sequence | 10 | 0.8846 | 0.7711 | 1.065 | True |
| euler_nozzle/shock_35 | constant | Modified Stadtke · sequence | 10 | 0.8754 | 0.7586 | 1.065 | True |
| euler_nozzle/shock_35 | constant | HLL · sequence | 10 | 0.6516 | 0.5658 | 0.8442 | True |
| euler_nozzle/shock_35 | constant | Rusanov · sequence | 10 | 0.6806 | 0.5728 | 0.9446 | False |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · single | 10 | 2.218 | 1.971 | 3.053 | True |
| euler_nozzle/shock_35 | muscl_van_leer | Stadtke · sequence | 10 | 1.09 | 0.9177 | 2.024 | True |
| euler_nozzle/shock_35 | muscl_van_leer | Modified Stadtke · sequence | 10 | 1.044 | 0.8433 | 1.549 | True |
| euler_nozzle/shock_35 | muscl_van_leer | HLL · sequence | 10 | 0.7657 | 0.6495 | 1.086 | True |
| euler_nozzle/shock_35 | muscl_van_leer | Rusanov · sequence | 10 | 0.7792 | 0.6745 | 1.077 | True |
| euler_nozzle/shock_70 | constant | Stadtke · single | 10 | 1.542 | 0.9414 | 2.025 | True |
| euler_nozzle/shock_70 | constant | Stadtke · sequence | 10 | 1.288 | 0.8094 | 1.675 | True |
| euler_nozzle/shock_70 | constant | Modified Stadtke · sequence | 10 | 1.148 | 0.6581 | 1.561 | True |
| euler_nozzle/shock_70 | constant | HLL · sequence | 10 | 0.8216 | 0.5055 | 1.109 | True |
| euler_nozzle/shock_70 | constant | Rusanov · sequence | 10 | 0.8515 | 0.4796 | 1.154 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · single | 10 | 2.097 | 1.949 | 2.349 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Stadtke · sequence | 10 | 0.8574 | 0.8035 | 0.9148 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Modified Stadtke · sequence | 10 | 0.8531 | 0.7913 | 1.006 | True |
| euler_nozzle/shock_70 | muscl_van_leer | HLL · sequence | 10 | 0.5548 | 0.5151 | 0.6322 | True |
| euler_nozzle/shock_70 | muscl_van_leer | Rusanov · sequence | 10 | 0.6149 | 0.5613 | 0.6461 | True |
| euler_nozzle/adapted | constant | Stadtke · single | 10 | 0.3281 | 0.2829 | 0.5801 | True |
| euler_nozzle/adapted | constant | Stadtke · sequence | 10 | 0.5925 | 0.529 | 0.9172 | True |
| euler_nozzle/adapted | constant | Modified Stadtke · sequence | 10 | 0.5942 | 0.5319 | 0.8683 | True |
| euler_nozzle/adapted | constant | HLL · sequence | 10 | 0.4994 | 0.4186 | 0.7767 | True |
| euler_nozzle/adapted | constant | Rusanov · sequence | 10 | 0.4655 | 0.4075 | 0.8996 | True |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · single | 10 | 0.3593 | 0.3231 | 0.4491 | True |
| euler_nozzle/adapted | muscl_van_leer | Stadtke · sequence | 10 | 0.5422 | 0.4766 | 0.6522 | True |
| euler_nozzle/adapted | muscl_van_leer | Modified Stadtke · sequence | 10 | 0.5572 | 0.4931 | 0.7722 | True |
| euler_nozzle/adapted | muscl_van_leer | HLL · sequence | 10 | 0.3989 | 0.3795 | 0.4629 | True |
| euler_nozzle/adapted | muscl_van_leer | Rusanov · sequence | 10 | 0.4295 | 0.3814 | 0.6705 | True |
| euler_annular/shock35 | constant | Stadtke · single | 3 | 55.14 | 51.72 | 60.54 | True |
| euler_annular/shock35 | constant | Stadtke · sequence | 3 | 3.002 | 2.518 | 3.924 | True |
| euler_annular/shock35 | constant | Modified Stadtke · sequence | 3 | 1.704 | 1.352 | 2.288 | True |
| euler_annular/shock35 | constant | HLL · sequence | 3 | 0.9053 | 0.6854 | 1.321 | True |
| euler_annular/shock35 | constant | Rusanov · sequence | 3 | 0.6959 | 0.6732 | 0.7096 | True |
| euler_annular/shock35 | muscl_van_leer | Stadtke · single | 3 | 103.3 | 76.04 | 125 | True |
| euler_annular/shock35 | muscl_van_leer | Stadtke · sequence | 3 | 3.961 | 3.395 | 4.977 | True |
| euler_annular/shock35 | muscl_van_leer | Modified Stadtke · sequence | 3 | 3.26 | 2.332 | 4.158 | True |
| euler_annular/shock35 | muscl_van_leer | HLL · sequence | 3 | 1.615 | 0.9532 | 1.953 | True |
| euler_annular/shock35 | muscl_van_leer | Rusanov · sequence | 3 | 1.187 | 0.8924 | 1.748 | True |
| euler_annular/shock70 | constant | Stadtke · single | 3 | 38.67 | 36.92 | 41.02 | True |
| euler_annular/shock70 | constant | Stadtke · sequence | 3 | 2.162 | 2.134 | 2.204 | True |
| euler_annular/shock70 | constant | Modified Stadtke · sequence | 3 | 1.34 | 1.251 | 1.505 | True |
| euler_annular/shock70 | constant | HLL · sequence | 3 | 0.7533 | 0.6617 | 0.9212 | True |
| euler_annular/shock70 | constant | Rusanov · sequence | 3 | 0.6562 | 0.6384 | 0.6887 | True |
| euler_annular/shock70 | muscl_van_leer | Stadtke · single | 3 | 66.98 | 65.13 | 69.07 | True |
| euler_annular/shock70 | muscl_van_leer | Stadtke · sequence | 3 | 3.906 | 3.694 | 4.217 | True |
| euler_annular/shock70 | muscl_van_leer | Modified Stadtke · sequence | 3 | 2.128 | 2.006 | 2.25 | True |
| euler_annular/shock70 | muscl_van_leer | HLL · sequence | 3 | 1.031 | 0.9648 | 1.092 | True |
| euler_annular/shock70 | muscl_van_leer | Rusanov · sequence | 3 | 1.015 | 0.9761 | 1.089 | True |
| euler_annular/supersonic_expansion | constant | Stadtke · single | 3 | 0.5568 | 0.549 | 0.5679 | False |
| euler_annular/supersonic_expansion | constant | Stadtke · sequence | 3 | 0.9834 | 0.9592 | 1.002 | False |
| euler_annular/supersonic_expansion | constant | Modified Stadtke · sequence | 3 | 1.028 | 0.9729 | 1.076 | False |
| euler_annular/supersonic_expansion | constant | HLL · sequence | 3 | 0.5161 | 0.5004 | 0.5311 | False |
| euler_annular/supersonic_expansion | constant | Rusanov · sequence | 3 | 0.5128 | 0.4886 | 0.5555 | False |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · single | 3 | 5.387 | 5.345 | 5.45 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Stadtke · sequence | 3 | 1.001 | 0.9466 | 1.07 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Modified Stadtke · sequence | 3 | 1.003 | 0.9732 | 1.037 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | HLL · sequence | 3 | 0.5515 | 0.5292 | 0.5741 | True |
| euler_annular/supersonic_expansion | muscl_van_leer | Rusanov · sequence | 3 | 0.5786 | 0.5569 | 0.6007 | True |

Measured source SHA-256: `6dbb0f45bfe70a627f907207cace8e5c99e7272d2292dffd51c70d1bda4230e9`.
