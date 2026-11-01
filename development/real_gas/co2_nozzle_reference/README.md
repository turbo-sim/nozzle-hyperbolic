# CO2 nozzle references with bicubic real-gas properties

This study establishes four **continuous quasi-one-dimensional Euler reference
solutions**, without an FVM calculation or production-source changes:

| Reservoir | Adapted expansion | Internal normal shock |
|---|---|---|
| Supercritical CO2, $p_0=30$ MPa, $T_0=400$ K | Supercritical throughout; $A_e/A_t=1.15$ | Prescribed $p_b=20.4$ MPa |
| Lettieri-type CO2, $p_0=8.474$ MPa, $T_0=313.88$ K | Expansion into equilibrium liquid/vapor mixture; $A_e/A_t=1.69$ | Prescribed $p_b=7.0$ MPa |

All flow-state properties use **JAXProp `FluidBicubic`, 100 by 100 tables**.
JAXProp generates the equilibrium tables from its HEOS property source; the
reference never calls a direct HEOS/CoolProp fluid to evaluate a flow state.
Pressure/enthalpy bounds are enforced before every evaluation.

Start with the generated [results and plots](REPORT.md), then read the
[equations and assumptions](EQUATIONS.md). The term *semi-analytic* is deliberate:
the continuum conservation laws are reduced to an ODE and scalar roots. There
is no closed-form two-phase CO2 solution, and a finite property table cannot
provide an exact physical EOS. Numerical errors and interpolation effects are
reported separately.

The retained two-phase case has a **23% local mismatch between the separately
interpolated sound speed and the acoustic speed derived from density** near
the saturation boundary. The conservative reference uses the latter. This is
an interpolation diagnostic, not a failed shock or backpressure solve; it
matters when comparing an FVM implementation that uses the former. Smooth
table-entropy drift is much smaller, about 0.0019 J/(kg K).

## Reproduce

Run from the repository root in its existing Poetry environment:

```powershell
poetry run python -B -u -m development.real_gas.co2_nozzle_reference.scripts.run
```

If the Poetry command is unavailable but the existing project environment is
present, the equivalent command is:

```powershell
.\.venv\Scripts\python.exe -B -u -m development.real_gas.co2_nozzle_reference.scripts.run
```

The full command builds/loads both tables, computes all four profiles, performs
conservation and independent quadrature audits, repeats the reference with a
tighter ODE solve, and generates PNG/PDF plots and `REPORT.md`. It fails if a
required audit fails. `--stations 1001` changes only the profile sampling used
for CSV/plots, not an FVM mesh or the ODE error control. `--skip-refinement` is
available for diagnostic runs and is recorded by the absence of a refinement
audit; the retained reference includes refinement.

Rebuild only the figures/report from saved numerical data:

```powershell
.\.venv\Scripts\python.exe -B -m development.real_gas.co2_nozzle_reference.scripts.plot_results
```

An optional feasibility scan prints adapted conditions and the backpressures
associated with several trial shock positions:

```powershell
.\.venv\Scripts\python.exe -B -u -m development.real_gas.co2_nozzle_reference.scripts.probe
```

The final shock locations are found from the **fixed backpressures** in
[`cases.json`](cases.json), not prescribed in that file. Both examples have
outer shock-position iterations and an inner Rankine–Hugoniot iteration.

## Files and future comparisons

- [`scripts/reference.py`](scripts/reference.py): geometry, smooth Euler primitive,
  choking, shock and backpressure roots, arbitrary station sampling.
- [`scripts/thermodynamics.py`](scripts/thermodynamics.py): bounded bicubic
  property calls, reservoir inversion, automatic differentiation of density.
- [`scripts/audit.py`](scripts/audit.py): conservation, branch selection and
  independent pressure-path quadrature.
- [`scripts/run.py`](scripts/run.py): reproducible driver, tighter integration
  comparison and source fingerprints.
- [`scripts/plot_results.py`](scripts/plot_results.py): presentation from saved
  data; no flow solve or property evaluation.
- `outputs/summary.json`: settings, versions, table hashes, reservoir/throat/shock
  traces, backpressure iteration history, audit metrics and thresholds.
- `outputs/<case>_adapted.csv`, `outputs/<case>_shock.csv`: dimensional SI fields;
  `x`, `A`, `side`, `p`, `h`, `rho`, `u`, `T`, `s`, both sound speeds/Mach numbers,
  raw quality/phase flag, table compressibility and density/entropy derivatives.
  Shock CSVs contain two coincident positions labeled `shock_pre`/`shock_post`.
- `outputs/<case>_choking_curve.csv`: pressure-parameterized mass-flux curve;
  `p_over_p0` is $p/p_0$ and `G_over_Gstar` is $G/G_*$, with all dimensional
  thermodynamic fields also retained.
- `outputs/fluid_tables/`, `outputs/*_table.log`: cached tables and complete-node
  generation evidence. `outputs/source_before.json` and `source_after.json`
  record all non-bytecode files beneath `src/` and must match.

Generated `outputs/` files are ignored by the repository's existing rules; no
ignore rule or file outside this new directory was edited by this study. Plots
are also saved as PDF for export. Regenerate outputs before sharing a checkout
that omits ignored files.

For a later FVM check, use the same case-specific area function, reservoir,
backpressure and **identical table bounds/resolution**. Compare shock traces and
location separately from errors in smooth regions; interpolating across the
jump gives a sampling-dependent error. For cell averages, integrate each
reference branch over the cell and split a cell containing the shock. No such
FVM projection or FVM solution is performed here.

## Physical scope

The two-phase closure is homogeneous equilibrium: common phase velocity,
temperature and pressure, and instantaneous phase equilibrium. It is suitable
for verifying an equilibrium Euler solver. A nonequilibrium, relaxation,
metastable or two-fluid solver solves different equations.

The second reservoir matches Case 5 in Table 3 of
[Lettieri et al. (2018)](https://doi.org/10.1115/1.4038082). The nozzle here is a
documented smooth synthetic geometry with Lettieri-inspired dimensions/area
ratios, **not a reproduction of their measured nozzle contour**. Their
nonequilibrium condensation observations are not validation data for this
equilibrium reference.
