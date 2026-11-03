# Local pseudo-time stepping results

HLL; constant and MUSCL Van Leer; three warmed samples per method/configuration, after one excluded complete warmup. Grid sequencing carries accepted CFL. Single-grid solves start cold on 400 cells.

MUSCL global/local mean-time ratios in this session are 1.17–1.25 with sequencing and 1.08–1.34 on cold 400-cell grids. Constant reconstruction ratios are 0.97–1.03. These small constant-reconstruction differences should not be treated as a robust acceleration. The 35% constant cold run has substantial scatter; all samples are retained. For the 35% constant sequence, local stepping uses more Newton corrections (581 versus 571). Grid sequencing improves complete solve time in every comparison. This development experiment changes no production defaults.

## 50 → 100 → 200 → 400

All four comparisons pass the original equation/boundary/budget/branch audits and final-state agreement checks. Global/local mean-time ratios range from 1.03 to 1.25. A ratio above one favors local stepping; below one favors global stepping.

| Case | Reconstruction | Method | Mean [s] | Min–max [s] | Converged | Accepted / rejected | Newton corrections | Global / local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shock_35 | constant | global | 0.360240 | 0.352573–0.373406 | 3/3 | 100 / 12 | 571 | 1.031x |
| shock_35 | constant | local | 0.349256 | 0.348551–0.350120 | 3/3 | 99 / 12 | 581 | 1.031x |
| shock_35 | muscl_van_leer | global | 0.285517 | 0.268821–0.302518 | 3/3 | 107 / 12 | 673 | 1.171x |
| shock_35 | muscl_van_leer | local | 0.243906 | 0.240476–0.247756 | 3/3 | 95 / 9 | 577 | 1.171x |
| shock_70 | constant | global | 0.234270 | 0.231723–0.237127 | 3/3 | 78 / 6 | 409 | 1.030x |
| shock_70 | constant | local | 0.227445 | 0.226451–0.228233 | 3/3 | 73 / 5 | 384 | 1.030x |
| shock_70 | muscl_van_leer | global | 0.490756 | 0.399464–0.538352 | 3/3 | 117 / 17 | 769 | 1.247x |
| shock_70 | muscl_van_leer | local | 0.393665 | 0.351919–0.428785 | 3/3 | 108 / 15 | 712 | 1.247x |

## Cold 400-cell grid

All four comparisons pass the original equation/boundary/budget/branch audits and final-state agreement checks. Global/local mean-time ratios range from 0.97 to 1.34. A ratio above one favors local stepping; below one favors global stepping.

| Case | Reconstruction | Method | Mean [s] | Min–max [s] | Converged | Accepted / rejected | Newton corrections | Global / local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shock_35 | constant | global | 0.893878 | 0.835812–1.000833 | 3/3 | 109 / 15 | 781 | 0.974x |
| shock_35 | constant | local | 0.917279 | 0.755541–1.188209 | 3/3 | 104 / 14 | 738 | 0.974x |
| shock_35 | muscl_van_leer | global | 2.021973 | 1.924233–2.098254 | 3/3 | 219 / 44 | 1828 | 1.339x |
| shock_35 | muscl_van_leer | local | 1.510032 | 1.454834–1.542570 | 3/3 | 168 / 31 | 1476 | 1.339x |
| shock_70 | constant | global | 0.978582 | 0.973355–0.981864 | 3/3 | 131 / 21 | 1030 | 1.010x |
| shock_70 | constant | local | 0.969330 | 0.958277–0.981700 | 3/3 | 130 / 21 | 1017 | 1.010x |
| shock_70 | muscl_van_leer | global | 2.084568 | 2.036464–2.130210 | 3/3 | 217 / 44 | 2013 | 1.083x |
| shock_70 | muscl_van_leer | local | 1.925662 | 1.874251–1.963402 | 3/3 | 205 / 41 | 1908 | 1.083x |

Timing includes complete solves, grid transfers, monitoring, rejected work and any cold retries. Model/kernel binding, final audits, output and plots are excluded. Every original per-equation residual criterion and independent boundary/global check is unchanged. All final states must agree within 1e-6 in fixed variable scales for a ratio to be eligible, and the measured shock must lie within two cells of its reference (coordinate roundoff allowed at the boundary).

The local update uses old-state cell wave times frozen during Newton. It changes storage weighting, not fluxes, geometry, source terms or algebraic trace equations. No physical-time interpretation is assigned to local iterates. The driver still adapts a common CFL; it is carried across successful grids for both methods. The original source bytes are verified before/after the campaign.

Cold-start and sequenced solutions are compared independently. Cross-strategy normalized fine-state spreads:

- euler_nozzle/shock_35, constant: 4.344e-07.
- euler_nozzle/shock_35, muscl_van_leer: 4.776e-07.
- euler_nozzle/shock_70, constant: 9.270e-08.
- euler_nozzle/shock_70, muscl_van_leer: 5.519e-08.

| Case | Reconstruction | Cold / sequence, global | Cold / sequence, local |
| --- | --- | --- | --- |
| shock_35 | constant | 2.481x | 2.626x |
| shock_35 | muscl_van_leer | 7.082x | 6.191x |
| shock_70 | constant | 4.177x | 4.262x |
| shock_70 | muscl_van_leer | 4.248x | 4.892x |

Three samples are a small session-specific comparison. Actual median-duration trajectories are shown; all failures, warmup records, settings, states and histories remain in the evidence.

![All timing samples and means](outputs/comparison/timing.png)

![Original residual histories](outputs/comparison/convergence.png)

![Final pressure profiles](outputs/comparison/profiles.png)
