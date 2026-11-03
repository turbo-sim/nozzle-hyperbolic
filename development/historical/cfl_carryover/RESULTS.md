# CFL carryover results

HLL, backward Euler, meshes 50 → 100 → 200 → 400. Three measured complete hierarchies per policy/configuration, after one excluded warmup per policy. MUSCL means Van Leer.

| Case | Reconstruction | Policy | Mean [s] | Min–max [s] | Accepted / rejected | Newton corrections | Reset / carry | Eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shock_35 | constant | reset | 0.602762 | 0.600815–0.605918 | 200 / 2 | 854 | 1.739x | True |
| shock_35 | constant | carry | 0.346703 | 0.338567–0.354968 | 100 / 12 | 571 | 1.739x | True |
| shock_35 | muscl_van_leer | reset | 0.690331 | 0.667355–0.711647 | 231 / 12 | 1156 | 2.533x | True |
| shock_35 | muscl_van_leer | carry | 0.272565 | 0.260543–0.296575 | 107 / 12 | 673 | 2.533x | True |
| shock_70 | constant | reset | 0.624425 | 0.604371–0.646259 | 197 / 2 | 822 | 2.421x | True |
| shock_70 | constant | carry | 0.257955 | 0.256892–0.259213 | 78 / 6 | 409 | 2.421x | True |
| shock_70 | muscl_van_leer | reset | 0.817739 | 0.805740–0.826566 | 233 / 15 | 1305 | 1.918x | True |
| shock_70 | muscl_van_leer | carry | 0.426273 | 0.399979–0.467739 | 117 / 17 | 769 | 1.918x | True |

All four comparisons pass the original residual/budget/branch and fine-state agreement checks. Carrying CFL makes these hierarchies 1.74–2.53 times faster by mean elapsed time. The savings come from avoiding the repeated CFL ramp on refined grids. Carryover sometimes increases rejected steps and backtracking, but lowers total Newton corrections in every configuration.

These timings were recorded before the grid driver adopted CFL carryover as its production default. The numerical source is retained in `outputs/comparison/source_at_measurement.zip`. Current production calls use `carry_cfl=False` to select reset; saved timings are not relabeled as updated-driver measurements.

Reset/carry is the ratio of mean complete-hierarchy times; values above one favor carryover. A ratio is eligible only when all original residual/budget/branch audits pass, the shock is within two cells, and all six measured fine-grid states agree within 1e-6 in fixed variable scales. Stage and hierarchy times include interpolation, monitoring and any failed warm/cold-retry work. Audits, serialization and plots are excluded.

The first grid always starts at CFL 0.2. Carryover transfers the preceding successful grid's last accepted CFL, not its timestep or next proposed CFL. Cold retries reset to 0.2. Both policies use the same strict inner Newton criteria and original outer convergence policy.

- euler_nozzle/shock_35, constant: fine-state normalized spread 1.725e-07; original audits True; shock check True.
- euler_nozzle/shock_35, muscl_van_leer: fine-state normalized spread 3.507e-07; original audits True; shock check True.
- euler_nozzle/shock_70, constant: fine-state normalized spread 1.739e-08; original audits True; shock check True.
- euler_nozzle/shock_70, muscl_van_leer: fine-state normalized spread 8.143e-08; original audits True; shock check True.

Three samples are a small session-specific comparison; timing scatter, warmup/setup records and all failures remain in the downloadable evidence. Median-duration actual histories are shown, not averaged trajectories.

![Hierarchy and stage timings](outputs/comparison/timing.png)

![Residual and CFL histories](outputs/comparison/convergence.png)

![Final pressure profiles](outputs/comparison/profiles.png)
