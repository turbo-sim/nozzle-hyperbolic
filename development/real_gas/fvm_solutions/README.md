# Real-gas CO2 finite-volume verification

This study compares the package's **Rusanov, HLL and general-EOS Roe** fluxes
against the established [continuous CO2 references](../co2_nozzle_reference/README.md).
Each flux uses both constant reconstruction and piecewise-linear MUSCL with the
Van Leer limiter (`muscl_van_leer`). The two physical reservoirs each have an
adapted expansion and an internal normal-shock operating point.

Each of the 24 case/method/reconstruction chains uses the production
`solve_grid_sequence` driver on **50 → 100 → 200 → 400 → 800 cells**. This is
the requested multigrid initialization acceleration: a converged numerical
state is prolonged to the next mesh. Each mesh solves its own original steady
equations with fresh residual normalization. It is grid sequencing, not an FAS
defect-correction cycle. Analytic profiles and shock locations are never used
as numerical initial states.

Open [REPORT.html](REPORT.html) for the self-contained interactive report. Its
top controls select physical case, operating condition, mesh, reconstruction
and fluxes. The report includes analytic comparisons, residual histories, grid
error trends and explicit failure evidence. All JavaScript and numerical data
are embedded for offline use.

## Retained results

All 120 requested stages completed, with 92 passing native convergence and the
independent spatial/boundary/conservation audits:

| Flux | Constant verified | MUSCL verified | Total verified |
|---|---:|---:|---:|
| Rusanov | 20/20 | 20/20 | 40/40 |
| HLL | 20/20 | 20/20 | 40/40 |
| Roe | 12/20 | 0/20 | 12/40 |

Roe is not robust for these bicubic cases and this initialization/control
profile. Its 28 unsuccessful stages remain selectable. There were six retained
failed warm attempts and two successful cold rescues. The targeted read-only
`outputs/diagnostics/roe_warm_start_audit.json` proves that the supercritical
800-cell prolonged state is physically admissible but fails the unchanged
Roe pressure-secant guard on 31 faces. That explains that warm rejection;
it is not a diagnosis of every unsuccessful Roe stage.

Converged Rusanov/HLL fields improve with mesh refinement. For example, in the
supercritical adapted case, Rusanov mass-flow error decreases from 14.70% to
1.54% with constant reconstruction, and from 0.654% to 0.00129% with MUSCL
between 50 and 800 cells. Verification of the discrete root and accuracy
against the continuous reference are separate results in the report.

## Numerical and physical setup

The [settings file](settings.json) records one common numerical profile for all
methods: implicit Euler, coupled endpoint equations, colored AD/banded linear
algebra, ten Newton corrections, Newton backtracking, no filter, initial CFL
0.2 with growth 1.2 up to $10^6$. Backtracking is an explicit study setting;
the production default is unchanged. The residual policy uses volume-weighted
fixed-scale RMS for each conservation equation: relative $10^{-6}$ **or**
absolute $10^{-9}$, with separate $10^{-9}$ boundary and global-conservation
tests. A small state increment alone is not a steady-convergence criterion.

The model, geometry and physical conditions come from the existing baseline.
Every property evaluation uses JAXProp `FluidBicubic`, equilibrium tables with
100 by 100 nodes. Upper pressure bounds are extended from 32 to 60 MPa for
the supercritical fluid and from 9 to 18 MPa for the condensation fluid to
accommodate intermediate startup states. Other bounds are unchanged. Each
continuous reference is recomputed with exactly the same extended table as
its FVM calculations, and its drift from the original reference is recorded.
All table SHA-256 fingerprints are recorded. There are no perfect-gas
approximations, clipped properties or direct HEOS flow-state calls.

The adapted cold guess prescribes a linear pressure ramp and a linear velocity
ramp from $0.02a_0$ to an outlet table Mach number of 1.5, with
$h=h_0-u^2/2$. A scalar bicubic property solve sets that outlet velocity. It
uses neither the analytic profile nor a choked mass flow. This avoids the
production default guess's excessive inlet mass flow for the large contraction
ratio. Shock cases retain the production cold guess. The same rule is used
for every flux/reconstruction. Preliminary failed narrow-table/default-seed
attempts are retained under `outputs/diagnostics/` and in report metadata.

Uniform physical cells use exact integrated area/volume measures, including
cells crossing the throat. The source is $p_i[A(x_{i+1/2})-A(x_{i-1/2})]$.
This is the production midpoint-pressure discretization of the geometric
source, not an exact integration of varying pressure within a cell.

Adapted cases use outgoing supersonic endpoint equations. Their reference exit
pressure is a cold-start datum, rather than a prescribed subsonic outlet law.
Shock cases use the specified pressure outlet. See the baseline
[equations](../co2_nozzle_reference/EQUATIONS.md) for reservoir and shock data.

The baseline solves smooth Euler paths using the bicubic **density** EOS. Its
Euler acoustic speed is obtained by differentiating that interpolant. The
production model uses the separately interpolated table sound speed for wave
speeds and boundary characteristics. The retained two-phase table has a local
23% acoustic mismatch near saturation in the original table, and about 14.5%
in the adapted / 17.4% in the shocked extended-table profiles. Both Mach definitions are available in the
results. Tight discrete convergence therefore needs to be assessed
separately from accuracy against the reference and from property-table accuracy.

## Run and reproduce

From the repository root, using its existing environment:

```powershell
.\.venv\Scripts\python.exe -B -u -m development.real_gas.fvm_solutions.scripts.run
```

The driver runs each grid chain in an isolated process to release JAX compiled
memory between chains. All five meshes within a chain share the same fluid
object. Every completed mesh is saved immediately. To continue an interrupted
campaign without discarding existing evidence:

```powershell
.\.venv\Scripts\python.exe -B -u -m development.real_gas.fvm_solutions.scripts.run --resume
```

Resume requires unchanged effective case settings and production-source hashes.
Immutable global profile versions are retained in `outputs/profile_registry.json`;
changing another fluid's table does not invalidate an unchanged case. Original
global profile hashes remain on their records, and compatibility is checked
from that case's physics, table, numerical controls and grid sequence.
Completed numerical states, rather than analytic data, seed subsequent grids.
The native driver retains failed prolonged attempts before trying a cold
restart; neither attempt is hidden. After a failed mesh, the next mesh starts
cold. Missing/incomplete stages remain visible in the report.

Study-only execution budgets stop an attempt after 180 seconds (checked at
driver callbacks), or after 64 accepted steps with CFL at most $10^{-6}$ and
less than $10^{-7}$ relative residual improvement. The actual last accepted
state and full history are retained and independently audited. These stops
are labeled explicitly and are never called converged. Production equations,
Newton acceptance and residual targets are unchanged. Each result declares
its execution controls; earlier records without this field used only native
termination, including the retained 16,000-step Roe failure.

For a controlled subset, add for example
`--fluid-case supercritical --regime adapted --flux rusanov --reconstruction constant --grids 50`.
Use `--resume` when any stage data already exist. This selects work to execute;
the assembled report still shows the full requested campaign and any missing
stages.

Reassemble saved stages and regenerate the report without solving:

```powershell
.\.venv\Scripts\python.exe -B -m development.real_gas.fvm_solutions.scripts.run --assemble-only
```

The equivalent `poetry run python -B -m ...` commands work when Poetry is on PATH.
No dependencies or files under `src/` are changed by this study. The driver
records before/after source manifests and stops rather than mixing runs if
another task changes the source during the campaign.

The driver first prepares/verifies matching continuous references and table
fingerprints. See [table-bound behavior](TABLE_BOUNDS.md) for the rejection
policy, the documented domain extension and the distinction from Fluent's
solution limits and NIST lookup fallback.

## Evidence and interpretation

- `outputs/campaign.json`: all physical definitions, numerical settings,
  source/table provenance, profiles, errors, histories and statuses.
- `outputs/runs/<chain>/N<cells>.json`: actual full `(p,u,h)` state with endpoint
  traces, physical-cell fields, native history, retained failed warm attempts,
  independent audit and pointwise analytic error norms.
- `outputs/logs/<chain>.log`: complete solver/process logs.
- `outputs/fluid_tables/`: bicubic table bytes shared by numerical and analytic
  solutions; the two extended 100 by 100 tables.
- `outputs/reference_summary.json`, `outputs/reference_*.csv`: the continuous
  references prepared for the exact table settings used in this campaign.
- `scripts/problem.py`: model factory, exact geometry and reference sampling.
- `scripts/audit.py`: original spatial/boundary residual recomputation and
  independent NumPy stopping/conservation reductions.
- `scripts/worker.py`, `scripts/run.py`: isolated grid-chain execution and
  resumable aggregation.
- `scripts/report.py`: standalone report generation from saved evidence.
- `outputs/source_snapshot/`: exact production source bytes matching the run
  source fingerprints, archived for reproducibility and never imported.
- `scripts/audit_campaign.py`: a solve-free completeness and consistency audit
  of all 120 requested configurations, statuses, source/settings identities
  and successful coarse-to-fine initialization transfers.

Errors compare the numerical physical-cell primitive state with the continuous
reference at the cell center, weighted by exact physical cell volume. These
are **not** errors against primitives obtained by inverting exact conservative
cell averages. Shock thickness and the largest positive pressure-jump position
are diagnostics; no arbitrary analytic-error cutoff is used to claim nonlinear
convergence.

Convergence histories retain the original per-equation fixed-scale norms,
relative norms, boundary/global defects, accepted/rejected steps, CFL and inner
Newton work. The canonical plotted scalar `residual` is the maximum of the
three fixed-scale per-equation RMS norms. Each mesh starts a new residual
normalization; curves from different meshes must not be interpreted as one
uninterrupted iteration.

This is one accuracy/robustness run per grid chain. Stage elapsed time includes
JIT compilation and all retained warm/cold solve attempts, excluding post-solve
audits. Histories and accepted-step counts describe the final attempt. This is
not a repeated performance benchmark or
a measured multigrid speedup against cold fine-grid runs. Generated `outputs/`
are ignored by the repository's existing ignore rules.

The HTML stores large datasets as lossless gzip/base64 and decodes them in a
modern browser using `DecompressionStream`; no external library is fetched.
Full histories remain available through embedded JSON/CSV downloads. The
report generator also accepts `--data-encoding plain` for older browsers.
