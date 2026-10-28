# Backpressure boundary-condition comparison

Open [REPORT.html](REPORT.html). Three linked, offline pages share light/dark themes:

1. **Uniform initialization**: the two outlet treatments start from uniform
   stagnation conditions, $p=200$ kPa, $T=300$ K and $u=0$ m/s.
2. **[Adapted initialization](ADAPTED.html)**: both start from the same numerical
   adapted expansion, obtained by implicit integration.
3. **[Theory and equations](THEORY.html)**: the two outlet laws, finite-volume
   assembly, implicit corrections and CFL controller. See also [THEORY.md](THEORY.md).

Perfect-gas air flows through the existing 0.2 m symmetric nozzle with
exit/throat area ratio two, reservoir pressure 200 kPa and temperature 300 K.
Two backpressures put analytical shocks at 35% and 70% of the divergent section.
Analytical profiles define/check cases; they do not initialize the numerical fields.
The mesh contains 201 points, including two algebraic ghosts (597 interior unknowns).

Only two outlets are compared with otherwise matched implicit Euler integration:

- The current Mach-selected pressure/extrapolation boundary.
- A nonlinear pressure-wave boundary flux with a temporary one-way wall safeguard.

Both use corrected FVS interior fluxes, ordinary MUSCL Van Leer, colored AD and
banded linear solves. The new law replaces the final numerical flux with the
physical incoming-wave trace flux when necessary. Its wall treatment is an
iteration safeguard, not a model for exterior-reservoir backflow; it must be
inactive at the verified final solution.

## Recorded outcomes

| Initialization | Case | Current BC updates / shock verified | Pressure-wave BC updates / shock verified |
|---|---|---|---|
| Uniform stagnation | shock35 | 148 / yes | 135 / yes |
| Uniform stagnation | shock70 | 252 / yes | 233 / yes |
| Adapted expansion | shock35 | 0 / no | 869 / yes |
| Adapted expansion | shock70 | 0 / no | 642 / yes |

Both treatments work from rest in these two cases. The pressure-wave treatment's
main demonstrated benefit is admitting a shock from an initially supersonic
exit, where the current outlet ignores the imposed pressure and remains at a
valid discrete supersonic root. A small residual alone does not verify branch
selection. The new outlet is not uniformly more accurate: its maximum normalized
mean error is smaller for shock35 and slightly larger for shock70, all below
0.006 for the verified endpoints. No general speed or robustness guarantee
follows from this single mesh and pair of backpressures.

Saved-state and source-hash checks passed for all eight records; all 2,279
accepted steps satisfy the strict inner criteria. The 22 local boundary checks
and three-page browser audit also passed, including both themes, exact plotted
residual/CFL arrays and equation blocks without internal scrolling.

## CFL and inner Newton settings

The study adopts the conservative recommendation from the
[annular CFL investigation](../euler_annular/cfl_investigation/REPORT.md):
initial CFL **0.2**, accepted-step CFL growth **1.2**, rejected-trial shrink
**0.5**, maximum CFL $10^6$, a maximum of **eight Newton corrections** and
**twelve retries** per update. These are fixed-growth CFL controls, with no SER
or work-threshold ramp. Growth applies to CFL, not directly to dimensional dt.

The unchanged nozzle full-step Newton engine is used for both variants;
the annular backtracking engine is not substituted. Both relative Newton
correction and fixed-storage-scaled implicit residual must be below $10^{-10}$.
The original steady residual target remains $10^{-8}$. Full-state increment
limits remain 0.2 for pressure, 0.1 for enthalpy and 0.2 for scaled velocity.
The study-local controller records rejected work and every accepted actual CFL.

The earlier SER/one-correction results are superseded in the report. The two
initializations and both outlets have been rerun with these matched settings.
Single recorded durations include checked-step first-call JIT compilation;
other first-call costs are recorded separately. This is not a speed benchmark.

## Reproduce

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.BC_backpressure.scripts.launch
poetry run python -B -m development.roberto.BC_backpressure.scripts.launch --resume
poetry run python -B -m development.roberto.BC_backpressure.scripts.launch --audit-only
poetry run python -B -m development.roberto.BC_backpressure.scripts.launch --plot-only
poetry run python -B -m development.roberto.BC_backpressure.scripts.audit_outlet_laws
```

The default output is `outputs/recommended_cfl/`; use `--resume` when it already
exists, or `--output` with a new path inside this study for a fresh run.
`--plot-only` regenerates all three HTML pages without solving.
`--audit-only` independently reevaluates saved fields, seeds, Newton acceptance
and CFL records, then rebuilds the reports. The launcher snapshots working source
on the first run and reuses it on resume. Production and other study directories
remain read-only; `-B` prevents external bytecode writes.

| Location | Purpose |
|---|---|
| `scripts/outlet_laws.py` | Current outlet and pressure-wave/wall traces |
| `scripts/problem.py` | Reused FVM physics, local boundary-flux assembly and implicit-defect monitor |
| `scripts/time_integration.py` | Fixed-growth CFL controller and verified step acceptance |
| `scripts/run_study.py` | Matched starts, eight comparisons and saved-state audit |
| `scripts/launch.py` | Source snapshot and isolated child imports |
| `scripts/audit_html.py` | Optional headless Edge audit using an existing Playwright environment |
| `scripts/audit_outlet_laws.py` | Local conservation, wave-selection and JIT/AD checks |
| `scripts/report.py`, `scripts/report_assets/` | Offline themed reports, plots and equations |
| `outputs/recommended_cfl/` | Eight comparison records, controller histories, audits and source snapshot |

Generated outputs follow the repository ignore rules. The committed HTML embeds
its displayed numerical data, audit evidence and local plotting/equation libraries.
Source assembly requires the current factory's public physical-residual,
boundary and solver-metadata operators. Source fingerprints identify that
working-tree baseline, which includes changes outside this study; this study's
commit does not include or alter those unrelated changes.
