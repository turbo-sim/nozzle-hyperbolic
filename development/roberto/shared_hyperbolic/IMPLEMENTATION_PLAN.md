# Shared numerical infrastructure for one-dimensional hyperbolic models

Assessment and implementation plan, 29 October 2026.

## Recommendation

Proceed with an incremental refactor of the **fixed-mesh, one-dimensional,
conservative balance-law infrastructure**, starting with EulerNozzle's modern
JAX implicit path and EulerAnnular. Keep the present public entry points and
array conventions through adapters during migration. The final shared Euler
paths must use coupled boundary residuals, a unified mesh/MUSCL implementation,
and the safeguarded Newton and monitored convergence approach of annular.
Document the intentional numerical changes separately from structural moves.
Do not attempt a universal matrix-based solver or a simultaneous rewrite of
every backend.

This is practical because substantial infrastructure is already shared. The
remaining task is principally to expose clean physical and numerical operators,
then share their assembly and orchestration. It is still a substantial refactor,
best delivered in several independently useful changes with numerical parity
checks. It should not be justified by an assumed speedup: maintainability,
consistent diagnostics, and cheaper addition of models are the primary benefits.

Linear advection and Burgers are useful early tests of the abstraction. Two-fluid
migration is a separate project: its current equations and nonlinear solve do
not satisfy the same conservative, square-system contract.

**This document is a plan, not an implementation of the refactor.** The review
covered the current working tree, which contains ongoing changes; the HEAD was
`d117b0d9f7bf06bd9cbd150e6e143af4a3455102`. References below point to the source
reviewed, rather than assuming every historical report describes that source.

## Required final design

The following decisions define the destination, not optional alternatives:

1. **Coupled boundary equations.** Boundary traces and physical-cell states are
   unknowns in the same nonlinear system. Each model supplies its boundary laws;
   one shared assembler appends their algebraic residuals with zero storage.
   Elimination maps are only temporary baseline/compatibility machinery.
2. **One mesh representation and MUSCL implementation.** Both Euler models use
   the same physical-cell/face indexing, volume and coordinate conventions,
   gradient/limiter operations and face evaluation, including nonuniform meshes.
   Geometry builders supply data, not separate numerical reconstruction paths.
3. **Model-owned physical representations.** Conserved-variable definitions,
   angular/transverse momentum transformations, characteristic data and physical
   boundary/source laws stay with the particular model. The shared numerical
   machinery calls these operations without knowing what angular momentum is.
4. **One Newton and convergence approach.** Both Euler models use annular's
   safeguarded Newton/backtracking, admissibility-aware step acceptance, retry
   logic and monitored steady stopping framework. The new nozzle path does not
   retain legacy full-step Newton, SSR/plateau stopping or a separate controller
   as competing production defaults.

These decisions strengthen the original plan: old behavior is retained to
establish a baseline and diagnose migration differences, not as a permanent
second numerical architecture. Public wrappers may preserve signatures and
array layouts; they must not silently select the old solver after migration.
Physical laws and reference data remain model-specific. Changes to an existing
boundary law, rather than merely how it is solved, require explicit case-level
validation and documentation.

## What the code already shares

| Component | Current implementation | Assessment |
|---|---|---|
| Finite-volume flux differences and coupled boundary rows | [solvers/finite_volume.py](../../../src/nozzle_hyperbolic/solvers/finite_volume.py) | Already accepts callbacks and a variable equation count. |
| Colored AD and block-Jacobian construction | [numerics/jacobians.py](../../../src/nozzle_hyperbolic/numerics/jacobians.py) | Already parameterized by block count, variable count and stencil radius. |
| Banded linear solution | [numerics/linear.py](../../../src/nozzle_hyperbolic/numerics/linear.py) | Shared algebra; its bandwidth and pivoting assumptions must remain explicit. |
| Newton iteration and backtracking | [time_marching/implicit.py](../../../src/nozzle_hyperbolic/solvers/time_marching/implicit.py) | Already supports both legacy full steps and safeguarded steps. |
| Regularized characteristic-flux algebra | [numerics/characteristic.py](../../../src/nozzle_hyperbolic/numerics/characteristic.py) | Both modern Euler fluxes use the same dimension-independent contraction. |
| Limiter arithmetic | [numerics/reconstruction.py](../../../src/nozzle_hyperbolic/numerics/reconstruction.py) | Some pieces are generic; other helpers still assume three variables or a uniform mesh. |
| Residual reductions, scalar monitor windows and output | [residuals.py](../../../src/nozzle_hyperbolic/solvers/residuals.py), [monitoring.py](../../../src/nozzle_hyperbolic/solvers/monitoring.py), [convergence_output.py](../../../src/nozzle_hyperbolic/solvers/convergence_output.py) | Generic foundations already exist; model adapters and solve loops remain separate. |

The `fvm_solver/` compatibility modules redirect imports to current modules.
They are not independent copies of the algorithms and should not be removed
as supposed duplication.

The largest remaining coupling is in model-specific solver factories: they
combine state conversion, reconstruction, boundary treatment, flux/source
assembly, nonlinear solution and diagnostics. In particular,
[EulerNozzle fvm.py](../../../src/nozzle_hyperbolic/models/euler_nozzle/fvm.py)
contains multiple distinct backends, whereas
[EulerAnnular fvm.py](../../../src/nozzle_hyperbolic/models/euler_annular/fvm.py)
is a smaller coupled implicit formulation.

## Why matrices alone are insufficient

For a nonlinear system, the numerical method needs the conserved density
$\mathbf q(\mathbf W;m)$, physical flux $\mathbf F(\mathbf W;m)$, sources,
admissible states and boundary laws. A quasilinear matrix by itself does not
specify the correct weak solution across a shock, a numerical entropy rule, or
the integration of a geometric source. For example, Burgers has characteristic
speed $u$, but the conservative flux is $u^2/2$; replacing its flux difference by
a generic discretization of $u\,u_x$ does not preserve its shock speed.

For fixed geometry, the shared finite-volume structure can be written as

$$
\mathbf C_i(\mathbf W_i)
=\int_{\Omega_i} A\,\mathbf q(\mathbf W_i;m)\,dm.
$$

$$
\mathbf R_i(\mathbf W)
=\boldsymbol\Phi_{i+1/2}(\mathbf W)
-\boldsymbol\Phi_{i-1/2}(\mathbf W)
-\boldsymbol\Sigma_i(\mathbf W).
$$

Here $\boldsymbol\Phi$ is an area-integrated numerical face flux and
$\boldsymbol\Sigma$ is the integrated source. Backward Euler then solves

$$
\mathbf C_i(\mathbf W_i^{n+1})-\mathbf C_i(\mathbf W_i^n)
+\Delta t\,\mathbf R_i(\mathbf W^{n+1})=\mathbf0.
$$

Coupled boundary equations have zero storage and are appended separately.
Their residuals are not multiplied by the time step. This is already the
structure implemented by `make_coupled_residual`.

For local analysis, the model may provide

$$
\mathbf C_W=\frac{\partial\mathbf q}{\partial\mathbf W},
\qquad
\mathbf K_W=\frac{\partial\mathbf F}{\partial\mathbf W},
\qquad
\mathbf A_W=\mathbf C_W^{-1}\mathbf K_W.
$$

Analytical matrices and projectors should be optional accelerators, verified
against AD. Rusanov and scalar models should not need an eigensystem. In
particular, do not rebuild characteristic projectors through an inverse flux
Jacobian: the existing regularized Euler implementations deliberately avoid
that singular construction at rest and sonic states.

## Current differences and their final treatment

### State variables and the frame used by the flux

Nozzle's modern implicit path advances primitive variables $(p,u,h)$ and forms
three conservative densities. Annular advances $(p,v_m,\ell,h)$, with
$\ell=r v_\theta$, and stores angular momentum as its third density.

However, annular's numerical flux constructs its characteristic projectors and
conservative jump in a **transverse-momentum frame** at one common face radius.
It subsequently multiplies the third flux by that radius and all fluxes by the
face area. See [annular flux_jax.py](../../../src/nozzle_hyperbolic/models/euler_annular/flux_jax.py).
Feeding an angular-momentum jump directly into these transverse projectors
would be incorrect.

An equivalent angular-frame transformation can be derived with
$\mathbf T=\operatorname{diag}(1,1,r,1)$ and
$\mathbf P_\ell=\mathbf T\mathbf P_\theta\mathbf T^{-1}$, but changing the
floating-point evaluation order is unnecessary in the first migration.
Initially, an adapter should call the current model-owned face-frame operation.
This transformation stays in the physics package in the final design as well;
it is not a special case inside shared flux, reconstruction or Newton code.
Models may reconstruct their declared primitive/state variables through the
same numerical algorithm without sharing the same physical variable meanings.

There is encouraging existing evidence: the annular physics tests check that
its zero-swirl flux reduces to the three-equation nozzle flux. This supports
sharing numerical algebra; it does not establish equivalence of the full
solvers, whose meshes and boundaries differ.

### Meshes, reconstruction and source integration

Annular uses physical cell centers, endpoint boundary traces, nonuniform widths
and integrated volumes. Its MUSCL reconstruction uses actual distances from
centers to faces. Nozzle's current modern reconstruction uses uniform half-cell
increments and a legacy layout that includes boundary slots. Equal array sizes
therefore need not mean equal physical cell counts or equal domains.

The final mesh contract uses $N$ physical cells, $N+1$ physical faces and separate
boundary traces at the domain endpoints. Cell centers, face locations, widths,
volumes and face areas have one definition used by both models. Geometry
builders prepare these arrays before JIT; annular radius, gap and radius
derivative are additional model data. Geometry parameterization and sampling
remain model-specific, while indexing, mesh validation and reconstruction are
shared and contain no model-name branches.

Use one nonuniform-coordinate MUSCL implementation: form one-sided gradients,
apply the selected limiter, and evaluate the resulting slope at each actual
face. Constant reconstruction is the zero-slope option of that same pipeline.
The model declares reconstruction variables and admissibility constraints; it
does not supply a separate MUSCL implementation. On an identical uniform mesh,
the common formula should recover the existing half-cell algebra where the
coordinates and boundary-trace conventions coincide.

First expose old layouts through adapters, then migrate nozzle's physical-cell
and endpoint conventions explicitly. Baseline comparison must use matched
domains and physical resolution; retaining two geometry conventions indefinitely
would not meet the final requirement.

Source integration is not just a pointwise source multiplied by a volume.
Annular's pressure contribution uses $p_i(A_{i+1/2}-A_{i-1/2})$ to balance its
face-pressure flux at rest. Nozzle's main driver constructs compatible area
differences, while direct factory callers can supply a different `dAdx`.
Record these existing choices explicitly. The final common mesh supplies the
same face-area differences to each model's integrated-source operation, so
pressure flux and geometric source remain compatible. The physical source and
its required balanced integration remain model-owned; they must not be replaced
by a universal point-source-times-volume formula.

### Boundary laws and boundary solution strategies

Annular solves independent boundary traces together with the cells, using
explicit subsonic/supersonic regimes and algebraic boundary rows. Nozzle's
modern path eliminates boundary states through maps inside each residual
evaluation. Its inlet map uses a finite number of entropy Newton iterations
and clipping; its outlet uses a smooth Mach-dependent pressure blend.

The final solver must solve boundary residuals coupled with the physical-cell
residuals for both models. Nozzle's eliminated maps are retained only for the
baseline and transition. Extract physical inlet/outlet equations from their
current nested implementations and express them directly as residuals, with
one canonical definition per model. Model-specific prescriptions and regime
choices need not become identical merely because their assembly is shared.

Replacing a finite-iteration/clipped boundary map with a coupled root is an
intentional numerical change, to be checked independently of structural moves.
The
[boundary-condition study](../BC_residual/README.md) demonstrates the feasibility
of a coupled nozzle formulation using public operators; it is a starting point,
not proof of exact equivalence to the production map for every state. Where
retained, the outlet blend can itself be expressed as a coupled residual. Any
decision to change that physical boundary prescription is a separate decision.

The boundary strategy also contributes to Jacobian bandwidth. Annular needs
radius two even with constant reconstruction because a boundary equation reads
a second physical neighbor. Do not infer bandwidth from reconstruction alone.

### Nonlinear solution, time stepping and diagnostics

Nozzle preserves an unchecked Newton full-step mode, while annular uses
admissibility checks and backtracking. Nozzle's driver also has legacy residual,
CFL and plateau stopping policies. The final Euler engine replaces these nozzle
policies with annular's safeguarded approach and its `solve_steady_monitored`
semantics, not the older annular `solve_steady` stopping definition. Use one
implementation and common default policies. Case files may set documented
numerical parameters; model identity must not select a different algorithm.

The common approach includes:

- Admissibility-aware Armijo backtracking, inner residual and update checks,
  explicit Newton status, and rejection/retry of failed implicit steps.
- Annular's accepted-step CFL growth and failed-step shrink/retry mechanism,
  with shared settings semantics rather than nozzle's legacy SER/plateau rules.
- Fixed-scale, volume-weighted per-equation residual norms; each active equation
  is normalized by its own initial norm and must reach $10^{-9}$ relative
  residual as well as its absolute safeguard. Initially negligible equations
  use the existing explicit absolute-only rule, without an invented denominator.
- Separate boundary and global-conservation checks, configured monitor windows,
  and a fresh final audit of the returned state. Stagnation above a target is
  failure, not convergence. Model adapters supply equation names, reference
  scales, admissibility and regime audits.

The same algorithm does not imply identical dimensional scales or identical
boundary equations. Numerical parameters and model data remain distinct.

Nozzle also has NumPy and explicit JAX paths. The NumPy diagonal approximation
called implicit is not the modern Newton solve. The existing explicit JAX
Runge--Kutta path reuses some thermodynamic/source data across stages; a new
implementation that reevaluates the complete RHS at every stage changes its
numerical behavior. Likewise, its Lax--Friedrichs flux depends on the time step,
so not every existing flux fits a time-step-independent steady operator.
Keep these paths behind compatibility wrappers initially.

There is a concrete diagnostic inconsistency to audit: nozzle's modern residual
can add a conservative flux filter, but `face_mass_flux` currently reconstructs
the unfiltered face flux. The target design should expose one canonical face
operator used by residuals and diagnostics. Correcting this discrepancy belongs
in a separate, documented change after a baseline is recorded.

Annular residual monitoring deliberately preserves an execution boundary around
the compiled spatial residual. At the requested nine-order relative reduction,
roundoff can affect a stopping classification. A fused JIT graph is not
automatically a harmless structural change even if algebraically equivalent.

## Proposed separation of responsibilities

Use small functions and immutable setup objects, composed by factories. Avoid a
large inheritance hierarchy, a symbolic equation language, or a universal
configuration object with many model-specific switches.

| Layer | Owns | Must not assume |
|---|---|---|
| Physical model | State conversion, physical flux, EOS, signal bounds, characteristic data if available, admissibility, source/closure laws and boundary laws | A particular time integrator or nonlinear iteration loop |
| Model discretization adapter | Geometry-aware integrated sources, model-owned face-frame transforms, declared reconstruction variables and admissibility | Its own mesh indexing, MUSCL algorithm or Newton implementation |
| Shared mesh/spatial assembly | Common cell/face layout, nonuniform-coordinate MUSCL, numerical-flux dispatch, integrated flux differences and coupled boundary layout | Pressure, sound speed, positive velocity, three/four equations, or an EOS |
| Shared algebra/time integration | Storage-plus-residual assembly, AD, banded/dense directions, annular-style safeguarded Newton, integration where supported | A specific physical system or boundary regime |
| Shared driver/monitoring | Annular-style acceptance, retry/CFL and convergence policies, scalar monitors, history and printing | That Cp exists or that models need different stopping algorithms |
| Case/model adapter | YAML validation, geometry/fluid setup, named quantities, reference solutions and reporting | A second implementation of the numerical solver |

Suggested contracts, expressed as responsibilities rather than final API names:

- A model provides `conserved(W, geometry)`, `physical_flux(W, geometry)`, signal
  bounds and an admissibility predicate; characteristic information is optional.
- A bound spatial operator exposes integrated cell storage, raw signed spatial
  residuals, the actual face fluxes, coupled boundary rows, and static layout
  metadata. Units and ownership of area weighting must be documented.
- A boundary provider supplies algebraic rows for independent boundary traces.
  The core handles indexing and coupled assembly; the model handles physical
  equations. An eliminated-map adapter exists only during baseline comparison.
- Layout metadata declares state/equation counts, physical and boundary slots,
  dependency radius, and supported linear-solver structure. Reject unsupported
  combinations before compiling.
- Monitoring receives scales, equation names and quantity evaluators as data or
  callbacks. Cp and mass flow belong to Euler adapters, not generic reducers.

The initial implicit contract is square. Internal arrays should be cell-major
`(npoints, nvariables)`; compatibility wrappers preserve older transposed public
arrays. Bind callbacks and validate static choices before JIT, avoiding Python
model dispatch in compiled inner loops. Generic numerical modules should not
import model packages or depend on a particular EOS library.

Physical state and flux definitions should have one canonical implementation
within each model. Their Jacobians may be obtained by AD or supplied analytically
with consistency tests. This does not require forcing all host-side legacy EOS
recovery code into JAX during the first migration.

Reuse existing `numerics/` and `solvers/` packages. Add only the small contracts,
spatial assembly, boundary assembly or driver modules that an actual migration
needs. If thermodynamic Euler helpers become common, prefer composition in a
small shared Euler helper module over making annular inherit from nozzle.

## Which numerical methods can be generic?

| Method | Shareable part | Remaining model responsibility |
|---|---|---|
| Rusanov | Dissipation and flux combination | Conserved states, physical flux and valid speed bound |
| HLL | Two-wave flux algebra | Consistent left/right signal bounds |
| Regularized characteristic flux | Existing projector contraction and Harten arithmetic | Projectors, averaging state, frame, width scale and validity |
| HLLC or Euler Roe flux | Some low-level algebra | Current star-state/Roe construction is Euler-specific; leave it model-owned |
| MUSCL | One mesh convention and nonuniform gradient/limiter/face-evaluation implementation | Declared reconstruction variables and admissibility constraints |
| Conservative filters | Difference algebra and conservative face insertion | Sensor, physical scale, admissibility and placement in the full operator |
| Newton stabilization | One safeguarded backtracking and acceptance policy, with status reporting | Model admissibility and reference scales |

An entropy fix is not a universal entropy-stability guarantee. A positivity
fallback appropriate to an EOS table is not appropriate to every scalar law.
Capabilities should select valid combinations; the core should not contain
branches such as `if model == 'euler_annular'`.

## Implementation sequence and gates

### 0. Freeze the baseline and compatibility inventory

The retained [EulerNozzle steady report](../euler_nozzle/STEADY_REPORT.html) now
covers eight backpressure regimes, both reconstructions and four meshes;
the separate [Sod report](../euler_nozzle/SOD_REPORT.html) covers one Riemann
problem at three times on four meshes. Their
[README](../euler_nozzle/README.md) identifies actual drivers, mesh conventions,
stopping rules, source hashes and independent audits. Preserve these artifacts
and use a separate output campaign when implementing the shared engine. They
record current nozzle policies, not the final annular-style convergence policy.

Record the actual working-tree source snapshot, public callers, array layouts,
return tuples, option defaults, parameter-dictionary side effects and JIT
boundaries. Build a matrix of nozzle backends and annular methods rather than
using the presence of an installed dependency to identify the active solver.

Retain numerical fields, signed residuals, boundary defects, selected Jacobians,
Newton outcomes and histories for representative cases. Include nozzle's steady
regimes and Sod, and annular's zero-swirl, swirling, radial/curved and wall-source
checks. Record filtered and unfiltered nozzle behavior separately.

Current annular campaign notes report 68/72 converged grid/method solves and
16/18 finest-grid accuracy passes. Four finest-grid MUSCL solves reach a
roundoff floor above the requested reduction; some converged cases still fail
analytical accuracy. These are existing limitations to preserve and explain,
not failures to remove by relaxing criteria. Recheck these counts against the
saved artifacts when starting implementation.

**Gate:** reproducible baseline and explicit compatibility list. No production
algorithm changes in this step.

### 1. Introduce bound-operator contracts and adapters

Wrap the modern nozzle and annular operators behind a small common interface,
initially delegating their existing implementations. Specify integrated versus
unweighted quantities, cell layout, face frame and storage convention. Retain
public factories and return shapes. Add capability validation for fluxes,
boundary layouts and Jacobian solvers.

**Gate:** old and adapted operator values, public results and AD derivatives
agree within documented floating-point tolerances; no added EOS work per face.

### 2. Unify the mesh, MUSCL reconstruction and spatial assembly

Introduce the common physical-cell/face and endpoint-trace contract, then one
coordinate-based MUSCL implementation using existing limiter kernels. Use
adapters to reproduce the old layouts first; subsequently migrate nozzle onto
the common mesh explicitly. Keep old-layout comparisons as baseline tools,
not selectable permanent nozzle/annular reconstruction policies.

Share the sequence of boundary-trace access, reconstruction, model face
operation, area ownership, flux difference and integrated-source addition.
Keep physical geometry builders and state/frame transformations in the model
packages. Distinguish mesh-convention changes from extraction of unchanged
algebra, so each observed numerical difference has a traceable cause.

Keep the annular face-frame adapter and each model's source quadrature intact.
Centralize actual filtered face fluxes and diagnostics, but make any correction
to the current diagnostic mismatch a separate reviewed change.

**Gate:** parity for the adapter-only step; for the common mesh, independent
operator/Jacobian checks, matched-resolution accuracy, uniform-state consistency,
rest balance, conservation and unchanged source signs. No model-specific MUSCL
or mesh-indexing branches remain. A full solve alone is insufficient.

### 3. Couple all Euler boundary equations and unify safeguarded Newton

Extend the existing coupled residual helper to serve both Euler models. Port
nozzle inlet/outlet laws into explicit residual callbacks, using the boundary
study as a reference and retaining the selected physical prescriptions.
Boundary traces are unknowns of the same Newton system as physical cells;
do not run a separate nested boundary solve in the final residual evaluation.
Compute or declare the union of reconstruction, source, filter and boundary
dependencies. Retain dense-Jacobian audits of colored/banded paths.

Use annular's safeguarded Newton, admissibility-aware backtracking and inner
convergence checks for both Euler models. Generalize model callbacks and scales,
not the choice between legacy and safeguarded production policies. Preserve
public return tuples through wrappers; expose shared status and diagnostics
through an opt-in result or adapter while transitioning callers.

**Gate:** boundary rows have zero storage; both models use the same coupled
assembly and safeguarded Newton path. Compare the old map and new coupled root
where laws match, checking the boundary equations directly and documenting
differences due to old clipping or incomplete inner solves. Do not require
bitwise parity for this intentional algorithm change.

### 4. Test the abstraction with scalar models

Add minimal model packages for constant-speed linear advection and Burgers,
using the same reconstruction, Rusanov, residual assembly and appropriate time
integrator. These should require no dummy pressure, EOS or sound-speed fields.

Advection must allow signed transported values and either velocity sign, with
inflow applied at the correct end. Handle zero wave speed explicitly. Burgers
uses flux $u^2/2$ and signal bound $\max(|u_L|,|u_R|)$ for Rusanov; verify shock
speed and rarefaction behavior, not only smooth fields. Assess smooth order
before shock formation with limiter effects stated.

Start with supported inflow/outflow layouts. Periodic wraparound creates corner
couplings outside an ordinary banded matrix; it needs a dense audit or an
explicit cyclic-solver capability, not a false radius-one declaration.

**Gate:** adding a scalar model changes no core physics branches. If it requires
Euler-shaped dummy data, simplify the interface before continuing.

### 5. Share solve orchestration and monitoring

Extract accepted/rejected-step bookkeeping, selected CFL policies, termination
and histories from annular's monitored driver into the common engine. Both
Euler models adopt that same retry/CFL mechanism and normalized per-equation
stopping, including nine-order relative reduction, absolute safeguards,
boundary/global checks and configured monitors. Keep legacy histories labeled
with their actual definitions for comparison; do not retain legacy SSR/plateau
stopping as an alternate default in the final shared Euler path.

Reuse the [residual-monitoring design](../../residual_monitoring/README.md),
existing reducers and compact printer. Keep model-specific YAML schemas and
named quantity evaluators thin; a runner can request Cp or mass flow without
the core understanding them. Preserve canonical residual evaluation boundaries
and distinguish accepted iteration history from rejected trials.

**Gate:** reported residuals refer to the actual solved operator; all transport
equations and boundary criteria are checked independently. Monitor and printing
choices do not change the discrete solution. Both Euler cases use the same
convergence decision code and settings semantics, with model-specific scales
and named quantities supplied as data.

### 6. Promote the shared Euler paths and measure cost

Run the representative regressions, then the full nozzle/annular verification
campaigns. Compare fields, accuracy, convergence classifications, conservation,
AD results, compile cost, steady-state step cost and EOS evaluation work.
Warm up JIT separately and use repeated matched timings before claiming a
performance improvement or regression.

Keep wrappers for public imports, tuple layouts and driver side effects. These
wrappers route migrated Euler cases to the common coupled-boundary, mesh/MUSCL,
safeguarded Newton and monitored driver implementation. Retire temporary legacy
adapters from the production selection once the migration gates pass. Remove
duplicate assembly only after both production paths use the shared version.
Migrate older explicit/NumPy paths only when useful; a corrected fresh-stage RK
method needs a separate numerical change and temporal-order verification.

**Gate:** the four required final-design decisions above are all satisfied;
existing public callers remain supported and known accuracy limits remain
visible. Stop here before considering two-fluid migration.

## Validation strategy

Use layered checks, because matching a final pressure curve can hide boundary,
Jacobian or source errors:

1. Independent state/flux identities and AD checks, including shifted energy
   references, rest, sonic states and supported non-ideal EOS behavior.
2. Local integrated operator and boundary-row comparisons; verify area and
   radius factors are applied exactly once.
3. Dense versus colored Jacobians, banded versus dense directions, and the
   linear residual of each computed direction. Include boundary and filter
   stencil extensions, not just interior constant reconstruction.
4. Signed local/global conservation audits using the same actual face operator
   and integrated sources as the solve. Distinguish cancellation in global
   balances from small cellwise residuals.
5. Matched full-case results and histories; retain independently specified
   analytical accuracy checks and known failures. Separate parity checks for
   structural moves from accuracy/conservation checks for the planned mesh,
   coupled-boundary and solver-policy migrations. Exact legacy iteration
   counts are not an acceptance criterion for those intentional changes.
   Do not change a convergence tolerance to compensate for roundoff.

Existing foundations include [shared FVM tests](../../../tests/test_shared_fvm.py),
[nozzle steady-operator tests](../../../tests/test_euler_steady_operator.py),
[annular physics tests](../../../tests/test_euler_annular_physics.py),
[annular FVM tests](../../../tests/test_euler_annular_fvm.py), and flux and
reconstruction regularity tests. Extend these around meaningful contracts;
avoid tests that simply reproduce the implementation formula.

## Why two-fluid is deferred

The current [two-fluid solver](../../../src/nozzle_hyperbolic/models/two_fluid/solver.py)
can use eight iteration variables for six transported balances, with alternative
EOS constraints, projections and rectangular/augmented solve strategies. Its
[matrices](../../../src/nozzle_hyperbolic/models/two_fluid/matrices.py) include a
nonconservative contribution, and its [flux implementation](../../../src/nozzle_hyperbolic/models/two_fluid/flux.py)
has a dedicated nonconservative correction. Ordering that correction relative
to filtering also matters.

These are not details that can be handled by replacing `nvar=3` with `nvar=8`.
Later work must specify a nonconservative discretization contract, constraint
and unknown counts, EOS manifold treatment, and compatible nonlinear/linear
solvers. Existing shared low-level algebra can remain useful now, but the new
Euler interface must not claim two-fluid support or insert a zero correction
as a placeholder for it.

## Effort, risks and stopping rules

Expect several focused pull requests, with a useful first milestone at the
operator adapters and spatial assembly. There is no need for a big-bang switch.
The greatest risks are changing boundary laws, applying area/radius twice,
breaking source balance, underdeclaring Jacobian bandwidth, and changing
roundoff-sensitive convergence through JIT fusion. These are manageable with
the staged gates above.

Reconsider the design if it accumulates model-name branches, giant option
objects, duplicated physics in adapters, extra EOS evaluations, or mandatory
eigensystems for a scalar model. A smaller shared kernel is preferable to an
abstraction that merely relocates duplication.

The recommended end point is one reusable conservative numerical engine with
explicit physical and discretization adapters. It is not one universal matrix
solver for every PDE. That narrower goal is feasible, worthwhile, and consistent
with the strongest parts of the current codebase.

## Checks performed for this assessment

The existing focused test suite passed: **52 tests**, covering
`test_shared_fvm.py`, `test_euler_steady_operator.py` and
`test_euler_annular_physics.py`. This verifies relevant current foundations;
it does not validate an unimplemented refactor. No simulation campaign was
rerun for this architectural assessment. All document links were checked.
