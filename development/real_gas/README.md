# Real-gas CO2 finite-volume verification

This benchmark compares **Rusanov, HLL, general-EOS Roe, corrected Stadtke and
Osher** against continuous CO2 nozzle references. Osher uses five-point
quadrature with both Gauss and sonic-partitioned integration. It covers a
supercritical reservoir and a Lettieri-type
equilibrium condensation case, each with an adapted expansion and an internal
normal shock. Each flux uses constant reconstruction and linear MUSCL with the
Van Leer limiter on 50, 100, 200, 400 and 800 cells.

Open [REPORT.html](REPORT.html) for the standalone interactive report. The top
selectors choose the fluid, operating condition, mesh, reconstruction and fluxes.
The report contains analytic comparisons, convergence histories, mesh refinement
and downloadable results. All data and Plotly are embedded for offline use.

The reusable continuous reference is implemented in the package's
[density-EOS analytical module](../../src/nozzle_hyperbolic/models/euler_nozzle/analytical/density_eos.py).
This directory contains its CO2 case inputs, table adapter and FVM benchmark
drivers. Reference algorithms are independent of the numerical evolution solver.

## Results

All **240 mesh stages** pass native convergence and independent checks of the
original spatial residual, boundary equations and global conservation: six method
configurations, four physical operating points, two reconstructions and five meshes.

| Flux | Constant | Linear MUSCL | Total verified |
|---|---:|---:|---:|
| Rusanov | 20/20 | 20/20 | 40/40 |
| HLL | 20/20 | 20/20 | 40/40 |
| Roe | 20/20 | 20/20 | 40/40 |
| Stadtke, both corrections | 20/20 | 20/20 | 40/40 |
| Osher, Gauss 5 | 20/20 | 20/20 | 40/40 |
| Osher, sonic-partitioned 5 | 20/20 | 20/20 | 40/40 |

`settings.json` explicitly enables Stadtke's `custom_jump_correction` and
`custom_endpoint_envelope`. The two Osher report IDs, `osher_gauss` and
`osher_partitioned`, both select the production `flux="osher"` with
`quadrature_points=5`; their `integration` settings are `gauss` and
`sonic_partitioned`, respectively. Quadrature points do not specify spatial
reconstruction order. The original Stadtke formulation and individual
correction ablations are outside this comparison.

Discrete convergence and accuracy against the continuous reference are assessed
separately. In the supercritical adapted case, the Rusanov numerical mass-flow
error decreases from 14.70% to 1.54% with constant reconstruction, and from
0.654% to 0.00129% with MUSCL between 50 and 800 cells.

## Setup

The reservoir conditions, geometry and shock backpressures are defined in the
[continuous-reference cases](cases.json). The FVM
[settings](settings.json) specify table domains and numerical controls. All
flow-state properties use JAXProp `FluidBicubic`, with equilibrium CO2 tables
of 100 by 100 nodes:

| Case | Pressure domain | Enthalpy domain |
|---|---:|---:|
| Supercritical | 4–60 MPa | 260–600 kJ/kg |
| Equilibrium condensation | 1–18 MPa | 250–520 kJ/kg |

Each FVM reference uses exactly the same table bytes as its numerical solutions.
The reference integrates $dh/dp=1/\rho$, locates choking and solves the
Rankine–Hugoniot conditions while iterating shock position to match backpressure.
See the [reference equations](EQUATIONS.md).

The solver uses implicit Euler with coupled endpoint equations, colored automatic
differentiation and a banded linear solver. Newton permits ten corrections with
backtracking. The initial CFL is 0.2, with growth 1.2 up to $10^6$; no filter is
applied. Per-equation, volume-weighted, fixed-scale RMS residuals must satisfy
relative $10^{-6}$ or absolute $10^{-9}$, with separate $10^{-9}$ boundary and
global-conservation checks. [NUMERICS.md](NUMERICS.md) defines the spatial
discretization, Roe regularization and verification quantities.

All 48 chains use the production `solve_grid_sequence` driver on
**50 → 100 → 200 → 400 → 800 cells**. Each finer mesh receives the preceding
converged numerical state through componentwise linear prolongation and starts
with fresh residual normalization. This is multigrid initialization acceleration
through grid sequencing. Analytic profiles and shock locations are assessment
data, rather than numerical initial guesses.
An unsuccessful transferred-state attempt is retained before a cold restart;
if no preceding mesh converged, the next mesh starts cold. The convergence
page can display the final attempt or all solve attempts with separate counters.

For adapted cases, the cold state uses linear pressure and velocity ramps,
from inlet $u=0.02a_0$ to an outlet table Mach number of 1.5, with
$h=h_0-u^2/2$. Shock cases use the production default cold state. Adapted cases
use outgoing supersonic endpoint equations; shocked cases impose backpressure.
The [table-bounds policy](TABLE_BOUNDS.md) applies to every accepted cell, trace
and reconstructed face.

Execution budgets stop unconverged Osher attempts after 600 seconds and other
methods after 180 seconds at driver callbacks, or after 64 accepted steps with
CFL at most $10^{-6}$ and less than
$10^{-7}$ relative residual improvement. A budget stop is an unsuccessful result.
Per-stage records carry the execution controls used for that stage when applicable.
`--resume --retry-budget-stops` revisits wall-budget stops while preserving
their full states, histories and elapsed work in the stage's attempt records.

## Run and reproduce

Run from the repository root using its existing environment:

```powershell
.\.venv\Scripts\python.exe -B -u -m development.real_gas.scripts.run
```

Each chain runs in an isolated process, sharing its fluid object across all five
meshes. Completed stages are saved immediately. Use `--resume` to reuse saved
stages with matching case settings and source fingerprints. Select subsets with
`--fluid-case`, `--regime`, `--flux`, `--reconstruction` and `--grids`.

Rebuild the report from saved results without numerical solves:

```powershell
.\.venv\Scripts\python.exe -B -m development.real_gas.scripts.run --assemble-only
.\.venv\Scripts\python.exe -B -m development.real_gas.scripts.audit_campaign
.\.venv\Scripts\python.exe -B -m development.real_gas.scripts.audit_report
```

The equivalent `poetry run python -B -m ...` commands work when Poetry is on PATH.
Source fingerprints identify the implementation used by each saved result;
new work requires the pinned execution source. Table fingerprints and effective
case-settings fingerprints identify the physical and numerical configuration.

## Files

- `settings.json`: numerical controls and FVM table domains.
- `cases.json`: reservoir, nozzle geometry and backpressure definitions.
- `EQUATIONS.md`: continuous-reference equations and physical assumptions.
- `scripts/geometry.py`, `scripts/thermodynamics.py`: case geometry and CO2 tables.
- `scripts/reference_audit.py`: conservation, quadrature and refinement checks
  for the continuous reference.
- `scripts/prepare_reference.py`: matching continuous references and table checks.
- `scripts/problem.py`: physical model factory, geometry and reference sampling.
- `scripts/run.py`, `scripts/worker.py`: grid-chain execution and result assembly.
- `scripts/audit.py`, `scripts/audit_campaign.py`: numerical and saved-data checks.
- `scripts/report.py`, `scripts/report_assets/`: standalone report generation.
- `outputs/runs/<chain>/N<cells>.json`: full state, fields, convergence history,
  errors, timings and independent audits for each stage.
- `outputs/reference_summary.json`, `outputs/reference_*.csv`: matched references.
- `outputs/fluid_tables/`: bicubic tables used by FVM and matched references.
- `outputs/campaign.json`, `outputs/provenance.json`: campaign, environment,
  configuration and source fingerprints.
- `outputs/source_*/`: source snapshots indexed by their
  fingerprints in provenance.
- `outputs/logs/`: solver logs; `outputs/browser_audit/`: report verification.
- `outputs/reference_validation/`: supplementary continuous-reference fields,
  tables and figures, with their own table domains recorded in `summary.json`.
  These samples are separate from the table-matched FVM comparisons.

Generated `outputs/` are ignored by the repository. The HTML embeds complete
histories as lossless gzip/base64 and uses the browser's `DecompressionStream`.
The report generator also accepts `--data-encoding plain`.

The supplementary reference figures can be rebuilt from their saved fields with
`python -B -m development.real_gas.scripts.plot_reference`. This performs no
thermodynamic evaluations or flow solves.

## Interpretation

The continuous reference uses the bicubic density EOS. The FVM model uses the
separately interpolated table sound speed and Grüneisen coefficient for wave
data. Both table and density-derived Euler Mach numbers are reported. The
equilibrium-condensation references have maximum local acoustic mismatches of
about 14.5% for adapted expansion and 17.4% for the shocked case. Mesh convergence,
thermodynamic-table accuracy and convergence of the discrete residual are
distinct checks. The two-phase model assumes homogeneous velocity and immediate
mechanical, thermal and phase equilibrium; it does not model delayed nucleation.

Field errors compare numerical cell primitives with the reference at cell centers,
weighted by exact physical cell volume. Shock position is estimated from the
largest positive adjacent-cell pressure jump in the divergent section. Numerical
face-flux conservation is distinct from the spread of cell-state $\rho uA$.

Stage elapsed time includes solve/JIT time and timestep work, excluding post-solve
audits. Histories retain accepted and rejected steps, CFL, Newton work and all
per-equation residuals. These single-run timings describe this accuracy study;
no repeated performance measurement or multigrid speedup is inferred.
Independent chains may execute concurrently, with the limit recorded in
provenance. Each chain's mesh sequence remains serial. Shared-host wall times
are not used to rank flux performance.
