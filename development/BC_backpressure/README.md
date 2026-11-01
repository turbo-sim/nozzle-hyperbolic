# Backpressure boundary-condition sensitivity

This study compares the production **acoustic** and **finite-wave** pressure
outlets for perfect-gas and real-gas Euler flows. Each shocked target is solved
from uniform stagnation and from a separately computed numerical adapted
expansion. Both starts use the native **50 → 100 → 200 cell** grid sequence.

Open [REPORT.html](REPORT.html) for the standalone, offline Plotly report.
Its selectors compare cases, boundary formulations, initial states, fluxes,
reconstructions and mesh stages. The theory page explains both general-EOS
outlet laws, their acoustic limit and their finite-amplitude differences.
[THEORY.md](THEORY.md) contains the same derivation in readable source form.

## Recorded results

All **288 mesh stages** pass native convergence and independent checks of the
original equations, boundary conditions, conservation and admissibility. All
**96 final 200-cell runs** and **16 numerical adapted seeds** are verified.

| Outlet | Initial state | Verified roots at 200 cells | Shock diagnostics at 200 cells |
|---|---|---:|---:|
| Acoustic | Stagnant | 24/24 | 22/24 |
| Acoustic | Adapted | 24/24 | 22/24 |
| Finite wave | Stagnant | 24/24 | 22/24 |
| Finite wave | Adapted | 24/24 | 22/24 |

The eight flagged final configurations are the **supercritical CO2 case with
constant reconstruction**, for both fluxes, both outlets and both starts.
Their equation roots converge, but the 200-cell solution does not resolve a
supersonic region upstream of the intended shock. MUSCL resolves this shock.
Across the full final-mesh matrix, MUSCL passes the shock diagnostics in
48/48 configurations, and constant reconstruction in 40/48.
These are resolution/accuracy findings, distinct from nonlinear solver failure.
Analytical field errors and shock-position errors remain visible even when
the branch diagnostics pass.

All **48 matched initialization pairs** reach the same computed 200-cell root:
the largest pressure difference is $1.7\times10^{-12}$ of the fixed pressure
scale, and the largest normal-Mach difference is $3.7\times10^{-12}$.
This is an observed result for this matrix, rather than a general uniqueness
proof.

The **48 matched boundary-formulation pairs** retain finite-mesh differences.
Their largest pointwise pressure gaps are 7.16% of the fixed pressure scale
with constant reconstruction and 0.102% with MUSCL. The corresponding maximum
symmetric differences in mean face mass flow are 0.0310% and 0.000585%.
Large pointwise pressure gaps localize around the shock; the report supplies
shock positions, integrated errors and profiles alongside these maxima.

The generalized outlet also passes **37 focused production tests**, including
independent Hugoniot conservation, numerical rarefaction and sonic sampling,
forward/reverse AD, exact table-bound anchors and the zero-pressure-jump limit.

## Physical cases

| Target | Canonical definition / reference |
|---|---|
| Euler nozzle, shock at 35% of divergence | `euler_nozzle/shock_35` |
| Euler nozzle, shock at 70% of divergence | `euler_nozzle/shock_70` |
| Radial annular Euler with swirl, shock at 35% | `euler_annular/shock35` |
| Radial annular Euler with swirl, shock at 70% | `euler_annular/shock70` |
| Supercritical CO2 nozzle with internal shock | Established real-gas supercritical case |
| Equilibrium two-phase CO2 nozzle with internal shock | Established Lettieri-type case |

Perfect-gas geometry, inlet data, shock locations and backpressures are loaded
from the package's [physical verification cases](../../src/nozzle_hyperbolic/verification/README.md).
Their analytical solvers live beside the nozzle and annular models in `src/`.
The CO2 cases reuse the [real-gas physical setup](../real_gas/cases.json),
its established FVM table domains and the package's
[density-EOS reference](../../src/nozzle_hyperbolic/models/euler_nozzle/analytical/density_eos.py).
Reference-selected backpressure defines each target; the target shock profile
is never used to initialize the solver.

All CO2 flow properties use the existing JAXProp **100 by 100 bicubic tables**.
The numerical outlet uses their tabulated sound speed. Density-derived sound
speed, where supplied by the analytical assessment, is a separate diagnostic.
There is no property clipping or property-backend substitution.

## Controlled comparison

The complete target matrix has 6 physical cases × 2 outlet laws × 2 initial
states × 2 numerical fluxes × 2 reconstructions: **96 independent chains** and
**288 mesh stages**. The final comparison is on 200 cells; 50 and 100 cells
provide initialization acceleration and retain their own diagnostics.

- Fluxes: HLL and Stadtke, with both Stadtke custom corrections enabled.
- Reconstruction: constant and linear MUSCL with the Van Leer limiter.
- Time stepping: 64-bit implicit Euler, coupled boundary traces, colored AD and a
  banded linear solve. Newton permits ten corrections with backtracking.
- Initial CFL: 0.2, growth 1.2, maximum $10^6$, rejection shrink 0.5.
- Steady stopping: fixed-scale, volume-weighted per-equation RMS; relative
  $10^{-6}$ or absolute $10^{-9}$, with independent $10^{-9}$ boundary and
  global-conservation requirements.
- Execution budget: 600 seconds per attempt at driver callbacks, plus an
  explicitly recorded collapsed-CFL plateau stop. A budget stop is a failure.
- At most two independent chains run concurrently. Mesh stages within a chain
  are sequential. Recorded solve/JIT times are diagnostic, not speed rankings.
  Workers share a persistent JAX compilation cache only within
  `outputs/jax_cache`, storing kernels that take at least one second to compile.
  Stagnant-start chains run before adapted-start chains so the latter can reuse
  compiled code. Flow states and convergence histories remain independent.
  Shared-cache and overlapping-worker timings are diagnostic; they do not
  provide matched performance comparisons.

The full effective configuration is in [settings.json](settings.json).
Both laws use static shocked-target boundary settings. The acoustic automatic
pressure/sonic/outgoing option is described in the theory, but is a separate
operating mode from this controlled comparison.

The **stagnant start** sets reservoir pressure and enthalpy throughout the
physical cells, with zero normal velocity and zero annular angular momentum.
Physical inlet and outlet equations still act at the boundary faces.

The **adapted start** uses a separately converged 50-cell numerical supersonic
expansion for each geometry/fluid, flux and reconstruction: 16 numerical seeds.
The two pressure-outlet laws and the two perfect-gas shock positions share the
same relevant adapted seed. Seed preparation is recorded separately and uses
no analytical flow field. Successful target states are prolonged to finer
meshes. Failed transferred-state attempts and any cold restarts remain labeled
with their actual initialization and complete solve histories.

## Verification and interpretation

The report distinguishes native solver convergence, independent verification
of the original equations, and agreement with the intended physical target.
Independent checks recompute signed spatial residuals, boundary equations,
global conservation, admissibility and table-domain validity. A small update
or a converged inner Newton solve alone is insufficient.

Physical diagnostics include a resolved supersonic-to-subsonic compression,
shock-position error in cells, analytical field errors and mass-flow behavior.
The packaged perfect-gas shock-position criterion is reported explicitly.
CO2 shock-position errors are reported without inventing a new accuracy
tolerance. Coarse constant reconstruction can converge while failing to
resolve the supercritical shock. Such a result remains visible in the report.

The nonlinear finite-wave curve is exact for a consistent EOS and classical
wave structure up to its numerical integration/root errors. Independently
interpolated bicubic entropy, density and sound speed can violate the identities
used by that construction. The [theory](THEORY.md) states the resulting limits;
convergence is not evidence that these thermodynamic inconsistencies vanished.

## Reproduce

Run from the repository root with the existing project environment:

```powershell
poetry run python -B -m development.BC_backpressure.scripts.run
```

On a clean checkout, first prepare the shared CO2 tables and continuous
references, whose generated files are ignored by Git:

```powershell
poetry run python -B -m development.real_gas.scripts.prepare_reference
```

This prerequisite generates analytical references and the 100 by 100 property
tables, without running the real-gas FVM campaign. Existing reference/table
bytes are reused and checked; the backpressure campaign refuses missing or
stale reference inputs.

The driver snapshots production source and numerical inputs before execution,
prepares numerical adapted seeds, runs missing target stages and assembles the
campaign. Repeating the command reuses saved stages with matching execution
identities. Source or numerical-setting changes require a separate campaign
epoch; they must not silently relabel earlier solutions.

Recorded execution paths and snapshots remain unchanged. The
[relocation manifest](relocation.json) validates the directory spelling
correction and the launcher/verification changes when reusing saved results.

For a selected subset:

```powershell
poetry run python -B -m development.BC_backpressure.scripts.run --case nozzle_shock35 --outlet acoustic finite_wave --flux hll --reconstruction constant --start stagnant adapted
```

To inspect saved evidence and rebuild the report without numerical solves:

```powershell
poetry run python -B -m development.BC_backpressure.scripts.run --assemble-only --no-report
poetry run python -B -m development.BC_backpressure.scripts.audit_campaign
poetry run python -B -m development.BC_backpressure.scripts.report
poetry run python -B -m development.BC_backpressure.scripts.audit_report
```

`audit_campaign --fresh` additionally reevaluates the original equations from
saved states using production operators and the same trusted study-local
compilation cache. Cached code does not reuse residual assessments or flow
states. Report generation imports no solver
or thermodynamic backend. The HTML embeds all plotting resources, equations,
data and offered downloads, so it can be copied and opened independently.

## Files

```text
BC_backpressure/
  README.md, THEORY.md     physical setup, protocol and equations
  settings.json           declared numerical matrix and controls
  relocation.json         verified mapping for recorded execution paths
  REPORT.html             standalone interactive results and theory
  scripts/                execution, independent audits and report generator
  outputs/                ignored states, histories, seeds and source snapshots
```
