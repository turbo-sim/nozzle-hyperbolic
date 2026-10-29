# EulerNozzle physical verification cases

## Current baseline reports before the shared-engine migration

Open either standalone offline report:

- **[Steady nozzle: eight backpressure regimes](STEADY_REPORT.html)** — 64 solves:
  eight operating conditions, constant and Van Leer MUSCL reconstruction, and
  four independent meshes.
- **[Sod shock tube: physical-time baseline](SOD_REPORT.html)** — one standard
  Riemann problem on 50/100/200/400 physical cells, with three snapshots from
  each transient run.

Both follow the annular report's presentation: case/time and mesh selectors,
geometry or initial-state views, six numerical/reference field plots without
markers, history/conservation panels, refinement errors, case tables, collapsible
results sections, light/dark themes, and internal Theory and Reproduce pages.
All libraries, numerical evidence and JSON downloads are embedded. Copying
either HTML file alone is sufficient for offline reading.

The steady cases are subsonic flow, the exactly sonic-throat/subsonic-exit
analytical limit, shocks at 35% and 70% of divergence, shock exactly at the exit,
an overexpanded pressure below the shock-at-exit value, exactly adapted
discharge, and an underexpanded pressure below the adapted value. The last
three share the same ideal internal profile; external jet adjustment is outside
this model. At a shock exactly at the exit, the plotted reference uses the
upstream limit on the exit plane.

The steady report preserves the established **51/101/201/401-node convention**:
two boundary slots and **49/99/199/399 physical points**, with a node at the
throat. These are not annular's 50/100/200/400-cell meshes. All solves start
independently from the ordinary simple initialization; no exact field or
coarse-grid solution is supplied as a seed.

The source FVS/Newton/AD/banded kernels and boundary maps are unchanged. The
steady host controller is reused from the retained backpressure study, not
annular's new monitored driver. Its stopping criteria are study RMS ≤ $10^{-8}$
and native source SSR < $10^{-8}$. Additional per-equation relative residuals
are recorded as diagnostics; **nine-order reduction does not control this
baseline**. The controller rejects a timestep after nine failed trials, whereas
the production application driver can force acceptance on its last retry.
The report identifies this distinction explicitly. Both use the production
implicit kernel with one unchecked Newton correction per pseudo-time update.

All 64 steady runs meet the recorded residual targets; 15/16 finest-grid results
pass the established pressure/density/Mach mean-error thresholds. The constant
sonic-limit case fails the Mach error check. The finest MUSCL sonic-limit
result passes those mean-error checks but still has throat Mach about 0.890
instead of 1; constant gives about 0.809. This pointwise discrepancy is retained
and reported separately from domain-mean accuracy.

For Sod, one standard physical case is enough for this controlled baseline:
it already contains a rarefaction, contact and shock. Refinement and three times
provide additional checks without introducing new physics. The existing
HLL/constant/forward-Euler path is retained, with one global physical timestep
clipped before each output. All snapshots pass on 100/200/400 cells; the
50-cell run misses the velocity-error target at later times. Conservation
budgets remain near roundoff. Additional Riemann and smooth-wave tests can
extend verification later.

Configuration files:
[cases/steady.yaml](cases/steady.yaml) and [cases/sod.yaml](cases/sod.yaml).
They import the physical setup and baseline numerical constants from the
existing examples below; the backpressure reference is reused from
[`backpressure_sensitivity`](../backpressure_sensitivity/README.md).
The 81 production Python source files are fingerprinted before/after running
and checked again by the audits. This records the current working tree, not
an assumed clean version of HEAD.

From the repository root:

```powershell
.venv/Scripts/python.exe -u -m development.roberto.euler_nozzle.scripts.run_baselines steady
.venv/Scripts/python.exe -u -m development.roberto.euler_nozzle.scripts.run_baselines sod
.venv/Scripts/python.exe -m development.roberto.euler_nozzle.scripts.audit_baselines steady
.venv/Scripts/python.exe -m development.roberto.euler_nozzle.scripts.audit_baselines sod
.venv/Scripts/python.exe -m development.roberto.euler_nozzle.scripts.report_baselines
.venv/Scripts/python.exe -m development.roberto.euler_nozzle.scripts.audit_reports
```

`outputs/baseline_steady/` and `outputs/baseline_sod/` contain strict JSON,
actual histories, fields, provenance and audit records. The HTML builder only
reads saved evidence; presentation changes do not require numerical reruns.
The steady runner resumes completed entries if source/config match; use
`--output` to retain a different campaign. Browser audits use isolated copies
and profiles under `outputs/html_audit/`, with no live user browser profile.

Independent audits passed for all **64 steady states and four Sod runs**.
The Sod audit replays every saved timestep using independent NumPy HLL algebra,
including fields, physical clock and integrated boundary-flux budgets. The
steady audit reevaluates the source residuals and independent analytical errors.
These audits verify the saved baseline; they do not certify unimplemented
refactor code.

When comparing the refactor, keep these artifacts and their source fingerprints
unchanged. Save new campaigns separately. Structural extraction should reproduce
the corresponding operator values and discrete roots. The planned mesh,
coupled-boundary and Newton/stopping migrations are intentional numerical
changes: compare matched physical domains/resolutions, analytical errors,
conservation, sonic/shock locations and actual convergence criteria rather than
requiring identical iteration counts. For Sod, compare the same physical times
and boundary-flux budgets. Do not reinterpret old relative-residual observations
as evidence that the future nine-order criterion was used.

## Original standalone examples and historical verification

These two examples compare numerical Euler solutions with independent
analytical or semi-analytical references. They verify the implementation
against known solutions, rather than validating it against experiments.
Each script puts the physical conditions, mesh, method, and tolerances at
the top, so refactors can be checked against the same problems.

Both use `jaxprop.FluidPerfectGas(name="Air")`. The references use the same
heat-capacity ratio and gas constant as the backend; Air is not forced to
have exactly $\gamma=1.4$.

## Running and outputs

From the repository root, after `poetry install`:

```powershell
poetry run python -m development.roberto.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development.roberto.euler_nozzle.verify_euler_nozzle_sod --no-show
```

The files can also be run directly by path. Omit `--no-show` to display figures.
Each saves PNG/PDF comparisons, numerical/reference CSV files, and a
`summary.json` in a neighboring `results_euler_nozzle_steady/` or
`results_euler_nozzle_sod/` directory. Generated outputs are ignored by Git.
Existing local results were moved here with the scripts; they are not
committed reference fixtures. Regenerate them when changing solver versions.

Plots use `plasma` between 0.2 and 0.8. A color identifies a back pressure or
a physical time. Solid lines show the reference; hollow circles show the
numerical solution. A failed check saves the comparisons and returns a
nonzero exit status.

## 1. Steady converging-diverging nozzle

The quasi-1D flow is inviscid and adiabatic, without wall friction. The Euler
equations include the pressure-area momentum source. The symmetric area is

$$
A(x)=A_{\rm in}-4(A_{\rm in}-A_t)\frac{x}{L}\left(1-\frac{x}{L}\right).
$$

| Condition | Value |
|---|---|
| Length $L$ | 0.2 m |
| Inlet and exit area | $3.0\times10^{-4}$ m$^2$ |
| Throat area $A_t$ | $1.5\times10^{-4}$ m$^2$ |
| Throat position | 0.1 m |
| Reservoir stagnation pressure $p_0$ | 200,000 Pa |
| Reservoir stagnation temperature $T_0$ | 300 K |
| Uniform grid | 201 points including two boundary/ghost points; 1 mm spacing |

The inlet uses reservoir enthalpy and entropy. The model outlet imposes static
pressure for subsonic outflow and extrapolates for supersonic outflow.

| Reference regime | Approximate $p_b/p_0$ | Physical behavior |
|---|---:|---|
| Supersonic exit, no internal shock | 0.093739 | Choked throat and supersonic expansion |
| Shock at 35% of divergence | 0.888338 | Supersonic acceleration, normal shock at $x=0.135$ m, then subsonic flow |
| Shock at 70% of divergence | 0.708722 | A stronger normal shock farther downstream, at $x=0.170$ m |
| Sonic throat, subsonic exit | 0.937112 | Limiting isentropic branch with $M_t=1$ and subsonic flow on both sides |
| Entirely subsonic | 0.956415 | Unchoked branch with reference throat Mach $M_t=0.6$ |

The references combine isentropic nozzle relations and, where needed,
normal-shock jump conditions. Subsonic cases use the subsonic area-Mach root
on both sides. The unchoked case has a virtual critical area below the
geometric throat area and a mass flow below the choked value.

Shock positions and throat Mach numbers specify the reference branches and
their back pressures. The numerical solver receives the geometry, reservoir
conditions, and back pressure. It receives neither an imposed shock position
nor the exact flow as initialization; it uses its ordinary simple startup.

The numerical method is Städtke FVS with primitive Van Leer MUSCL and implicit
backward-Euler/Newton pseudo-time marching with one Newton correction per
timestep. The starting CFL is 10, maximum SER CFL is $10^6$, SER exponent is
one, and the timestep growth bound is 1.5 per accepted update. Adaptive
increment checks can halve a trial timestep. The budget is 5000 updates and
the residual target is $10^{-8}$.
Pseudo-time is a convergence device, not a time-accurate nozzle startup.

The combined CFL 10 / growth 1.5 / maximum $10^6$ setting was run on all five
cases. Accepted-update counts were 17 (supersonic), 114 (shock 35%), 199
(shock 70%), 43 (sonic limit), and 41 (subsonic). All met the residual target;
the same four analytic checks passed, with the known sonic-limit discrepancy
remaining. This is a verified usable setting, not a claim of minimum work for
every case. `dt_growth_factor` is a production-driver parameter with fallback
1.2 for callers that do not specify it.


**Why it is useful:** smooth branches check the geometric source and
isentropic acceleration; choking checks the sonic region and operating
branch; shock cases check discontinuity capture, downstream compression,
and shock position. All branches expose mass-flow imbalance and the
difference between residual convergence and physical accuracy. These are
standard nozzle-verification objectives; [NASA's CDV example](https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html)
uses different geometry and conditions from ours.

The script compares pressure, density, and Mach number. Acceptance requires
convergence, boundary mass-flow imbalance below $10^{-4}$, and full-domain
normalized mean absolute errors below 0.03. Pressure and density use reservoir
scales; Mach uses one. Additional smooth-region errors exclude three grid
spacings around a reference shock, but acceptance includes the shock region.
Shock-position and throat-Mach errors are reported separately.

## 2. Unsteady Sod shock tube

A diaphragm at $x=0.5$ m separates two initially stationary states in a
constant-area, one-metre domain. Removing it at $t=0$ creates a Riemann
problem, with no area source, wall friction, or heat transfer.

| Condition | Left | Right |
|---|---:|---:|
| Density | 1.0 kg/m$^3$ | 0.125 kg/m$^3$ |
| Pressure | 100,000 Pa | 10,000 Pa |
| Velocity | 0 m/s | 0 m/s |

The mesh has 400 physical cells and two transmissive ghost cells, with the
diaphragm on a face. The method is the existing JAX HLL flux, piecewise-constant
reconstruction, forward Euler, and CFL 0.45.

One simulation advances through $\tau=t a_L/L=0.1,0.2,0.3$, approximately
0.267, 0.534, and 0.801 ms for this backend. All cells share a physical timestep,
clipped before the update to reach each requested time. All waves remain
away from the ends, so the infinite-domain Riemann reference is appropriate.

The exact solution contains a left-moving rarefaction, a contact discontinuity,
and a right-moving shock. Pressure and velocity are continuous across the
contact while density jumps. A scalar star-pressure solve determines states
and wave speeds; sampling uses $(x-x_d)/t$. See the
[Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html)
for the wave structure and conservation laws.

**Why it is useful:** comparing all three waves checks propagation speeds,
the physical clock, shock/contact diffusion, and the rarefaction profile.
It complements the steady case, whose residual cannot reveal an incorrect
transient clock. The first-order method intentionally diffuses discontinuities.

Normalized mean errors in density, velocity, and pressure must be below 0.04.
Independent mass, momentum, and energy budgets must agree with the numerical
boundary flux integrals to $10^{-10}$ in normalized units:

$$
\int Q(x,t)\,dx-\int Q(x,0)\,dx
=\int_0^t\left(F_{\rm left}-F_{\rm right}\right)\,dt.
$$

Momentum need not be constant: unequal end pressures exert a net force even
before a wave reaches an end. Accounting for boundary fluxes is essential.

## Verification status and scope

Both examples were rerun against an isolated copy of the committed solver,
without pending FVS/limiter/root-finding modifications. Sod passes all three
snapshots. All five steady cases converge; four pass the accuracy checks.
The sonic-limit case fails the unchanged 3% threshold, with numerical throat
Mach about 0.846 instead of 1. This failure is retained: the steady script
currently exits unsuccessfully after saving every comparison.

The scripts use the installed solver implementation. Pending working-tree
changes can affect a rerun; see the [numerical changes review](../../../docs/NUMERICAL_CHANGES_REVIEW.md).
This commit adds examples and documentation without approving those changes.

These are fixed-mesh verification baselines, not mesh/order studies or
certification of real-fluid and TwoFluid physics. Sod checks HLL physical-time
integration, not the nozzle driver's FVS pseudo-time path.
