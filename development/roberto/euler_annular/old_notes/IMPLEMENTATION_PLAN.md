# General-EOS EulerAnnular FVM implementation plan

Date: 2026-10-28. Baseline: `roberto-branch`, commit
`dee35eaf91acfdf7cecf788dc8d2f282e40de606`.
This plan records accepted design choices and the original staged gates.
Implementation is now present in `src/`; see
[IMPLEMENTATION_REPORT.md](../REPORT.html#use-overview) for execution evidence and
remaining verification limits. A converged discrete root does not by itself
complete an analytical-accuracy gate.

## 1. Outcome and scope

Implement an unsteady conservative FVM whose steady limit solves flow in a
stationary annular channel with a general curvilinear meridional midline, varying
radius and height, and tangential velocity. Define the geometry using NURBSpy
B-spline/NURBS midline and height curves, with radial and conical channels as
particular cases of the same representation and metric builder. Target the
latest version in the local NURBSpy repository (1.6.0 at review), replacing
the planning baseline's 1.4.0. Version 1.6.0 is now installed with its NumPy
geometry backend, as documented below. The solver must support real-gas
thermodynamics through any JAXProp fluid object providing the required state
properties and recovery operations. Use a perfect-gas backend for the initial
verification cases, with zero heat transfer and zero friction. Include shock capture, subsonic and
supersonic boundaries, physical conservation diagnostics and analytic verification.
The current application focus is steady flow reached by implicit time marching.
Retain physical time-dependent storage and transient support in the architecture.

This general-EOS requirement is specified by the user. The numerical FVM,
state maps, characteristic fluxes, boundary conditions, initialization, timestep
calculation and residual scaling must not use $\gamma=c_p/c_v$ or perfect-gas
closure formulas. Gamma-based expressions belong only to analytical references
and verification-fixture construction. Use the same numerical path for the
perfect-gas tests and real-fluid backends; do not introduce a perfect-gas solver
branch or defer general-EOS support to a later implementation.

Two further requirements are fixed by the user: boundary states and their
algebraic equations belong to the overall nonlinear residual block, with no
nested nonlinear BC solves, and every verification field comparison shows
exactly three trends: analytical reference, FVM with constant reconstruction,
and FVM with MUSCL reconstruction.

The accepted model is quasi-1D and axisymmetric, with four balances. It does not
resolve circumferential detonation waves, two-dimensional shock structure,
cross-passage pressure distributions, separation or secondary flows. An RDE-like
benchmark represents its mean annular discharge. The answered
[questionaire.md](questionaire.md) records this scope and the selected numerical
choices.

The selected verification geometries use constant normal channel width $b$,
except straight conical channels whose width follows the semi-divergence and
cant angles. The general NURBS model still supports variable width. A shaped
radial C-D passage is an optional, deferred fixture, not an approved mandatory
benchmark. Constant-width purely radial inflow supports acceleration/choking
from a subsonic inlet but cannot provide a smooth internal transonic branch.
Choosing another fixture for the full adapted/internal-shock family does not
block implementation of the base general model and selected feasible cases.

Reuse has two meanings: existing numerical infrastructure can be extracted and
shared; new annular physical equations still require model-specific code. Adding
swirl to energy, angular-momentum transport and sonic compatibility is more than
appending a passive scalar to the three-equation nozzle solver.

## 2. Source audit before implementation

This table is the historical planning audit, not a description of the current
implementation. Paths are relative to `src/nozzle_hyperbolic/` unless stated
otherwise. Line numbers identify the reviewed checkout. Current annular
assembly is in `models/euler_annular/fvm.py`; shared characteristic contraction,
colored Jacobians and banded solves are in `numerics/`, residual assembly in
`solvers/finite_volume.py`, and Newton construction in
`solvers/time_marching/implicit.py`.

| Component | Observed implementation | Proposed treatment |
|---|---|---|
| `models/euler_annular/` | Reservation docstrings in `__init__.py`, `boundaries.py` and `closures/__init__.py` | Implement here through the staged work below |
| `models/euler_nozzle/fvm.py:377` | `make_jax_stadtke_step`: three primitive unknowns, nested BCs, storage, flux/source assembly, AD and Newton | Keep public wrapper/signature; extract reusable numerical internals |
| Same file, lines 793, 844, 860, 937 | Colored JVPs, block extraction, superblock banded solve, Newton step | Parameterize variable count and actual stencil; share one implementation |
| Same file, lines 1003–1058 | Optional steady residual/Jacobian/Newton operators | Preserve API concept; expose annular operators from the same residual |
| Same file, line 1067 onward | `fvm_euler_1d`: nozzle setup, integration/controller, histories, plotting/output | Extract only reusable orchestration; do not copy the whole driver |
| `numerics/reconstruction.py` | Safe Van Leer, optional smooth limiter and array MUSCL arithmetic | Reuse limiter kernels; add mesh-aware interface placement |
| `numerics/linear.py` | Equilibrated square solve/least squares | Already dimension-independent; reuse where needed |
| `solvers/time_marching/cfl.py` | NumPy scalar uniform-spacing CFL helper | Extend narrowly for mesh-aware JAX use; not already a full integration framework |
| `models/euler_nozzle/flux_jax.py:243,255` | Harten absolute and characteristic flux/viscosity contraction | Share dimension-independent algebra for the accepted flux family |
| Same file, lines 173, 283 | Three-variable projectors and regularized characteristic flux | Derive annular four-variable projectors; do not reuse unchanged |
| `models/euler_nozzle/state.py` | Three-variable NumPy state conversion | Add annular maps with swirl kinetic energy and radius-aware angular momentum |
| `geometry/duct.py` | Existing nozzle geometry and some NURBSpy-based definitions | Reuse applicable spline tools; add one general annular spline/metric builder; keep analytic area laws as references |
| `thermodynamics/` and `jaxprop` | State recovery and fluid objects | Reuse general fluid/state recovery interfaces; `FluidPerfectGas` is a verification backend only |
| `models/euler_nozzle/analytical/` | Nozzle and shock-tube references | Reuse zero-swirl area/Mach and shock relations after checking their interfaces |
| `fvm_solver/` | Legacy compatibility import shims | Leave as shims; never add a second solver here |
| `models/two_fluid/solver.py` | Separate model-specific factory with six storage balances and eight nonlinear variables | Leave migration outside this task |

The current perfect-gas steady example deliberately uses `perfect_gas=False`
with a `FluidPerfectGas` object to select the general-EOS implicit path
(`development/roberto/euler_nozzle/verify_euler_nozzle_steady.py:182–188`).
Do not infer the EOS or active integrator from that flag or from installed
Optimistix/PySolverView dependencies. The inspected implicit backend uses JIT
Newton, AD coloring and a banded solve.

Relevant boundary logic is nested in `fvm.py`, starting near lines 533 and 575;
the standalone nozzle `boundaries.py` is not the complete active implementation.

## 3. Physical and discrete contract

See [the companion derivation](../REPORT.html#theory-overview) for equations and caveats.

- Let $m$ increase from inlet to outlet, with positive physical cell lengths.
  Use signed $r'=dr/dm$: $+1$ for radial outflow, $-1$ for radial inflow.
- Geometry is defined by a NURBSpy midline $\mathbf C(\xi)=(x,r)$ and normal
  height spline $B(\sigma)$ with $\sigma=m/L$. Use the common NURBSpy NumPy
  arclength adapter and public analytical curve derivatives with the correct
  coordinate chain rule; fixed geometry is preprocessed before JAX kernels.
  Geometry supplies $r(m)>0$, $b(m)>0$, $A=2\pi rb$, signed derivatives,
  face coordinates/metrics and $V_i=\int_i A\,dm$. It validates arclength
  consistency rather than accepting unrelated $r$, $\phi$ and $A$ arrays.
  Selected verification uses constant $b$, with the straight-conical
  semi-divergence/cant-angle exception; special cases still enter as NURBS.
- Use `v_m` and `v_theta` as the velocity variable names in proposed code,
  interfaces, configuration and output fields, with $v_m$ and $v_\theta$ in
  mathematical notation. Keep these names throughout the annular model.
- Conserve angular momentum $\ell=r v_\theta$ as the third transported quantity,
  as accepted by the user:
  $q=(\rho,\rho v_m,\rho\ell,\rho E)$, with
  $E=e+(v_m^2+v_\theta^2)/2$. This removes the inviscid geometric torque source.
- Use the accepted primitive nonlinear/reconstruction variables
  $W=(p,v_m,\ell,h)$ to stay close to the existing three-variable
  pressure/meridional-velocity/enthalpy implementation;
  output $v_\theta=\ell/r$.
- Define a cell-integrated storage map $\mathcal Q(W,\mathcal G)$, integrated numerical
  face flux $\Phi(W_L,W_R,\mathcal G_f)$ and cell-integrated source
  $\mathcal S(W,\mathcal G)$. Here $\mathcal G$ denotes geometry; reserve $G$
  for the Grüneisen parameter. Radius weights belong in these model callbacks.
  The generic residual must not assume that every balance uses only one
  common area factor.
- At a face, convert both traces using the same $r_f$ to ordinary transverse
  Euler states, evaluate the four-component flux, multiply its tangential
  momentum component by $r_f$, and multiply all components by $A_f$ once.
  Reconstruct physical/model primitives, not jumps between cell-area-weighted
  states as though area discontinuities were fluid discontinuities.
- Use a documented volume-lumped storage approximation initially, with exact
  cell volumes and consistent cell radius. Record how reference cell integrals
  are projected into that representation; never compare unlike cell averages.
- Balance the constant-pressure rest state using the same face areas in
  $p_i(A_{i+1/2}-A_{i-1/2})$ and the pressure flux. This is a rest-state
  preservation requirement, not an exact moving-vortex equilibrium guarantee.

For the physical cells, the shared implicit residual is

$$
\mathcal R_{\Delta t}(W^{n+1};W^n)
=\mathcal Q(W^{n+1})-\mathcal Q(W^n)
+\Delta t\,[\Phi_{i+1/2}-\Phi_{i-1/2}-\mathcal S_i]=0.
$$

The annular solve augments these physical-cell rows with boundary constraints.
For $N$ physical cells, pack the full unknown vector in physical order as
$z=(W_L,W_1,\ldots,W_N,W_R)$, with four unknowns per block and
$4N+8$ equations. Solve

$$
\mathcal F_{\Delta t}(z^{n+1};W^n)=
\begin{bmatrix}
C_L(z^{n+1})\\
\widehat{\mathcal R}_{\Delta t,1:N}(z^{n+1};W^n)\\
C_R(z^{n+1})
\end{bmatrix}=0,
$$

where each boundary supplies four algebraic rows and the hat denotes fixed
physical scaling. Boundary rows have **zero storage**, no artificial relaxation
and no factor of $\Delta t$; the interior storage uses old physical cells only.
One Newton iteration assembles and solves the full Jacobian, including boundary
cross-derivatives. Current boundary unknowns enter the face fluxes directly.
Details and study evidence are in [BOUNDARY_BLOCK_SPEC.md](../REPORT.html#boundary-equations).

The original spatial residual is
$\mathcal R_s=\Phi_{i+1/2}-\Phi_{i-1/2}-\mathcal S_i$.
Expose it directly. Compute monitoring face fluxes from the exact assembled
flux, including any numerical filter, rather than reconstructing a similar flux
in a second diagnostic path.

### Thermodynamic contract: real-gas solver, perfect-gas verification backend

Pass the JAXProp fluid object explicitly to the annular factory and model
thermodynamic callbacks. The implementation must not depend on a concrete fluid
class, a fluid name, a constant heat-capacity ratio or ideal-gas pressure laws.
Use a small adapter only to normalize state keys/input pairs and reuse existing
recovery operations; do not implement another EOS or copy thermodynamic routines.

The hot-path contract is
`fluid.get_state(jxp.HmassP_INPUTS, h, p)` with state fields `rho`, `a`, `G`
and, for entropy/reservoir boundaries, `T`, `s` (normalize existing aliases in
one place). A native JAX backend passes through unchanged. When a JAXProp
object uses host property calls, reuse JAXProp's existing compatible wrapper,
such as `FluidJAX`, preserving the supplied EOS, configuration and phase choice.
Capability checks must distinguish an adapter requirement from an unsupported
physical model; never replace a real fluid with a perfect-gas object.

For the accepted $(p,v_m,\ell,h)$ variables, evaluate the static state from
$(h,p)$ and obtain density $\rho$, sound speed $a$, dimensionless Grüneisen
parameter $G$, temperature and entropy from the fluid object. Internal energy
is $e=h-p/\rho$. The general thermodynamic identities required for analytical
storage Jacobians/projectors are

$$
G=\frac1\rho\left(\frac{\partial p}{\partial e}\right)_\rho,
\qquad
\left(\frac{\partial\rho}{\partial p}\right)_h=\frac{1+G}{a^2},
\qquad
\left(\frac{\partial\rho}{\partial h}\right)_p=-\frac{\rho G}{a^2}.
$$

Read $a$ and $G$ from the fluid state. Do not reconstruct either from a heat
capacity ratio. Apply AD to the same JAX-compatible state/storage maps when
forming implicit Jacobians; cross-check analytical thermodynamic identities
against those derivatives on admissible states using backend-appropriate
tolerances. Finite-difference wrapper derivatives and interpolated tables need
their actual accuracy/consistency reported rather than a universal roundoff gate.

Reservoir data $(p_0,T_0)$ are converted to $(h_0,s_0)$ by the fluid object.
Subsonic boundary compatibility uses that EOS and the constraints
$s=s_0$, $h+(v_m^2+v_\theta^2)/2=h_0$, with the selected swirl law.
Evaluate these constraints directly at the current independent boundary
$(h,p)$ unknowns through the HP interface. Do not solve for boundary pressure,
enthalpy or velocity in a nested root finder, finite Newton loop or fixed-point
loop inside residual evaluation, reconstruction, monitoring or accepted updates.
The full Newton solve resolves the boundary constraints together with the flow.

Reservoir conversion and any consistent initial-data preparation happen once
outside the nonlinear iteration. Recover diagnostic stagnation pressure and
temperature through supported EOS calls or an independently audited inversion
in setup/postprocessing; do not introduce a nested BC solve to compute monitors.
The compiled engine need not require native PT, PS or HS input pairs: setup
may supply $(h_0,s_0)$ directly, and coupled boundary rows evaluate their
constraints using the common HP interface. Useful general-EOS entropy
derivatives are $s_h|_p=1/T$ and $s_p|_h=-1/(\rho T)$.

Evaluate fixed reference density and sound speed from a chosen fluid reference
state. Do not use ideal-gas expressions in generic scaling, CFL calculation,
admissibility or initialization. Validate the finite, admissible thermodynamic
domain, including $\rho>0$, $T>0$ and $a^2>0$, through the selected backend.
The same state-property/recovery contract applies to every backend; identify
any missing operation explicitly instead of silently restricting the solver
to perfect gases.

Do not require viscosity/conductivity when wall closures are off, positive
absolute enthalpy/internal energy, or positive $G$. Respect the backend's
reference levels and declared domain/table bounds; avoid synthetic perfect-gas
enthalpy limits and unreported table extrapolation. Existing recovery helpers
must be audited before reuse: a bracket midpoint returned after an inversion
exception is not a valid thermodynamic root or a satisfied BC.

### NURBSpy geometry and implemented dependency choice

[GEOMETRY_SPEC.md](../REPORT.html#use-geometry) defines the curve representation, arclength
mapping, exact radial/conical specializations, quadrature and version audit.
The reviewed local NURBSpy version is 1.6.0, commit
`35d0b814d9d8343600d39a59dab23b64ea7bc6be`. The nozzle manifest, lockfile and
installed package now select NURBSpy **1.6.0 core**, with the explicit
`nurbspy.numpy` backend for fixed geometry preprocessing. JAXProp **0.5.8**
retains the compatible JAX/JAXlib **0.6.2** numerical stack. NURBSpy 1.6's
optional JAX extra requires JAX 0.12 and is not enabled. Its coordinated
dependency migration is optional future work if geometry/control-point AD is
requested; shape AD is outside the initial scope and does not block the
implemented fixed-geometry solver. Sampled mesh arrays become fixed JAX
constants at solver construction, while state/EOS/residual AD remains in JAX.

All benchmark geometries must instantiate the spline representation. Closed-form
radial/conical area laws may verify it, but must not bypass it in numerical runs.
Special-case constructors return NURBSpy curve data to the common builder, with
no geometry-type branches in FVM assembly. Precompute the fixed geometry metrics
once, using the selected current NURBSpy API. Its NumPy backend lacks a
tolerance-controlled arclength inverse, so the common adapter adds checked
quadrature of the public derivative and bracketed inversion. Verified affine
curve parameterizations use the exact map $m=L\xi$ in that same adapter.
Reuse NURBSpy's basis/evaluation/derivative implementation and do not copy the
sibling NozzleX geometry implementation.

## 4. Shared architecture and proposed file ownership

Names below are proposed, not new public APIs already available.

```text
src/nozzle_hyperbolic/
  numerics/
    reconstruction.py          shared limiter and metric-aware stencil arithmetic
    jacobians.py               colored JVP/block extraction with static layout
    linear.py                  existing kernels plus generic banded block solve
    characteristic.py          shared Harten/characteristic contraction
  solvers/
    finite_volume.py            integrated flux/source/storage assembly
    time_marching/
      implicit.py              shared backward Euler + Newton construction
      cfl.py                   timestep calculation and explicit controller inputs
  geometry/
    annular.py                 NURBSpy midline/height, special cases, common metrics
  models/euler_nozzle/
    fvm.py                     existing API, nozzle physics adapters
  models/euler_annular/
    state.py                   W <-> storage, thermodynamics, admissibility
    equations.py               physical transverse flux and characteristic data
    flux_jax.py                annular projectors/interface adaptation
    sources.py                 pressure/centrifugal terms; zero closure prototype
    boundaries.py              coupled physical BC residual blocks and metadata
    initialization.py          cold starts and explicit continuation policy
    fvm.py                     model factory wiring shared infrastructure
    analytical/
      perfect_gas.py           gamma-based verification references only
```

Start with functions and small immutable data containers. Do not introduce a
large class hierarchy or attempt a universal two-fluid interface. Shared modules
must not import concrete EulerNozzle/EulerAnnular models, plotting or YAML parsing.

Minimal shared inputs: `storage`, `integrated_face_flux`, `integrated_source`,
`boundary_residual`, `boundary_layout`, `differential_mask`, `wave_speed`,
`admissible`, fixed variable/residual scales, `nvar`, physical/boundary block
counts and proven stencil radius. Return step status, full state,
iteration counts and the original residual/flux operators. Array layout is
cell-major `(N, nvar)` internally; legacy public layouts remain supported by
the nozzle wrapper.
For the annular full state the shape is `(N + 2, 4)`, where $N$ counts physical
cells only. The shared engine supports zero-storage algebraic rows from the
outset; the existing nozzle wrapper retains its historical boundary behavior
during the separately verified structural extraction.

### Selected flux

Retain the existing regularized characteristic flux family for the first
implementation, as accepted by the user. Reuse its Harten
regularization and contraction arithmetic; derive and verify the four-variable
conservative projectors with two convective modes. Include swirl kinetic energy
in storage, jumps, fluxes and all thermodynamic recovery. A fixed-radius,
zero-swirl reduction must reproduce the three-equation flux.

Implemented default: `delta_factor=0.1`, selected after the inherited
$10^{-6}$ width caused failed fine-mesh shock cold starts. The controlled
width sensitivity and independent dense/colored/banded diagnosis are recorded
in [IMPLEMENTATION_REPORT.md](../REPORT.html#use-overview). This is an annular
regularization choice; the historical nozzle policy is preserved. Rusanov is
an explicitly labelled diagnostic, never an automatic method fallback.

The projectors and characteristic speeds must use the general-EOS state,
$a$ and $G$ above. Any HLL/Rusanov diagnostic must also use EOS-provided pressure,
enthalpy and sound speed. Neither flux family may import the perfect-gas
analytical module or derive numerical wave speeds from $\gamma$.

A simple four-variable HLL/Rusanov path is a possible independent diagnostic
baseline, not a required second framework. If chosen, extract its dimension-free
combination arithmetic rather than maintaining another copy. Do not add Roe,
HLLC, all-speed preconditioning and multiple FVS variants simultaneously.
Baseline first; add a low-Mach correction only if measured grid/Mach studies
justify it, with its own consistency and accuracy checks.

### Reconstruction, Jacobians and precision

Begin implementation with constant reconstruction. Then use shared safe Van Leer arithmetic
with gradients and interface distances appropriate to the actual mesh. Uniform
spacing in $m$ is a useful first milestone; general curvilinear geometry must
already work, and nonuniform spacing gets a separate verification gate.
Each completed numerical verification case runs **both** constant and MUSCL
reconstruction. An early constant-only smoke run is not the completed benchmark.

Use float64. Keep inactive traced arithmetic finite, with no host conversions
or data-dependent Python branching inside JIT. Fixed positive component scales
must include angular momentum even for zero-swirl cases. If the smooth limiter
is selected, state its mesh-dependent width explicitly and verify its refinement
behavior; smooth reconstruction alone does not establish update positivity.

Coloring must use the actual augmented dependency radius. Constant interior
flux normally has radius one and MUSCL radius two, but boundary compatibility
rows reading the nearest two physical cells already require radius two in the
full block ordering. The ordinary four-variable coloring then uses up to 20
colors for either reconstruction, capped by problem size. Use a narrower
pattern only when proved for the selected boundary stencil. Check against dense
AD, including boundary rows/cross-derivatives and small/odd/even cell counts.
A MUSCL Jacobian is generally block-pentadiagonal; the existing
superblock solve is a way to exploit that bandwidth.

## 5. Boundaries, initialization and time marching

For forward meridional flow the four speeds are $v_m-a,v_m,v_m,v_m+a$.

| Boundary/regime | External physical data | Interior information |
|---|---|---|
| Subsonic inlet | $p_0,T_0$ and fixed $\alpha_{in}$ for the selected pressure sweeps | One outgoing acoustic compatibility |
| Supersonic inlet | All four quantities, e.g. $p,T,v_m,v_\theta$ | None required to close the state |
| Subsonic outlet | Static $p_b$ | Three outgoing fields |
| Supersonic outlet | No imposed interior static pressure | All four outgoing fields |

Stagnation properties use total speed; characteristic counts and shock normal
Mach use meridional speed. Document whether input Mach is total or meridional.
Keep $\alpha_{in}$ fixed as backpressure changes, as selected by the user.
The boundary angular momentum remains an unknown constrained by
$(\ell_{in}/r_{in})\cos\alpha_{in}-v_{m,in}\sin\alpha_{in}=0$.
The steady inviscid solution has constant $\ell$ along a channel, but its value
may change between pressure points as mass flow changes. Reference construction
and continuation must not accidentally hold the preceding run's $\ell_{in}$
fixed. A prescribed-$\ell_{in}$ inlet is an optional alternative data family,
not the selected sweep law.

Define physical BC equations once in the annular model, with original-constraint
diagnostics. **Use coupled algebraic boundary blocks from the first annular
implementation**, as specified by the user. There is no eliminated annular
default, nested boundary nonlinear solve or automatic fallback between layouts.
Preserve the existing nozzle wrapper separately for regression compatibility.
The attached questionnaire's generic agreement with an older eliminated-BC
recommendation does not replace the earlier explicit coupled-block instruction.
The resolved decision is one coupled Newton system with no nested BC solve.

Follow the formulation and numerical findings of the
[boundary-condition study](../../BC_residual/REPORT.html#theory) and its
[matched implicit report](../../BC_residual/REPORT.html#results): physical
block ordering, independent current boundary unknowns, zero boundary storage,
one shared Newton/colored/banded implementation, and separate spatial/BC audits.
The study's three-equation/six-extra-unknown layout becomes four equations and
eight extra unknowns here. Adapt its reservoir/compatibility equations to swirl,
general EOS and the annular boundary radius. Do not copy its private-closure
adapter into production or impose ideal-gas analytical boundary formulas.
The full [boundary block specification](../REPORT.html#boundary-equations) defines the
proposed residuals, scaling, stencil and acceptance checks. The study's observed
agreement/cost does not establish unconditional robustness or an annular speedup.

Specify sonic equality handling and monitor switches. An extrapolating
supersonic outlet cannot force an internal shock merely by changing an ignored
$p_b$. Establish the primary operating map by decreasing pressure from a
subsonic/rest start, and distinguish independent cold starts from continuation.
If shock entry from an external high-pressure environment or ascending-pressure
unstart is required, explicitly design a boundary Riemann/exterior-reservoir
model or exterior extension; do not silently overwrite a supersonic trace.

Use cold physically admissible fields for convergence claims, with no analytic
solution as an undisclosed initial field. The initial swirled case may need a
physically labeled swirl ramp; include all ramp stages/cost and audit the final
target equations. Fail explicitly on unsupported reverse flow, rather than
using `abs(v_m)` to disguise a different boundary characteristic count.

Use the existing backward-Euler/Newton/banded method through the extracted
engine, as accepted by the user, to reach steady flow in the current campaign.
Give annular integration an explicit acceptance policy: reject invalid,
nonfinite or insufficiently solved steps; reduce timestep, and report failure
when retries are exhausted. Never accept the last failed trial automatically.
Keep local pseudo-time stepping separate from physical unsteady time. A global
physical timestep uses the meridional acoustic spectral radius $|v_m|+a$.

When physical transient runs are used, they must advance to the requested final
time, clipping the last timestep and disabling steady SSR/plateau termination, SER adaptation and
local pseudo-time. Demonstrate temporal refinement separately from spatial
refinement; backward Euler is first order in time. Preserve historical nozzle
driver stopping behavior during extraction rather than silently changing it.
Retain minimal transient/time-step verification for the promised solver
capability; a time-accurate application campaign is not the present focus.

Current nozzle policy includes finite retry limits, native state-dependent SSR
and a plateau stop near three times the target. Preserve those behaviors during
structural extraction; annular success must use its separately stated original
spatial-residual/BC/conservation criteria. Do not bundle fixes to historical
nozzle behavior into a supposedly behavior-preserving move.

## 6. Work packages and completion gates

Implementation status: the shared extraction, annular geometry/state/flux/source
model, simultaneous boundary blocks, constant/MUSCL reconstructions and steady/
physical-time interfaces are implemented. Analytical fixtures and the grid
campaign are executable in `scripts/verify_annular.py`; saved-state revalidation
uses `scripts/audit_results.py`. The report records actual test results, original
residuals, conservation and separate accuracy checks. Low-Mach constant
reconstruction and sonic-limit accuracy remain explicit acceptance concerns;
the original thresholds below are retained. The optional internal-transonic
geometry is still deferred.

| Phase | Work | Gate before advancing |
|---|---|---|
| 0. Freeze benchmark metadata | Record accepted scope/numerics and remaining dimensions, angle values, pressure ranges and gas-reference data; record the deferred transonic-fixture decision | Selected constant-width/conical cases fully specified; unresolved optional transonic family does not block the base model |
| 1. Capture baseline | Record checkout/options; run focused production Euler tests and existing five steady regimes plus Sod; retain baseline fields/operators | Any pre-existing failures documented; reproducible baseline |
| 1a. Align NURBSpy dependencies | Install reviewed NURBSpy 1.6 core for NumPy fixed-geometry preprocessing; retain compatible JAXProp 0.5.8/JAX 0.6.2 and update manifest/lock together; the JAX-extra/shape-AD migration is optional future work | Imported version/path verified; dependency resolution and focused pre/post-upgrade regressions pass |
| 2. Extract shared numerics | Move dimension-free AD coloring, block solve and Newton construction; keep nozzle physical callbacks and defaults | Old/new nozzle spatial residuals, Jacobians, steps, BCs and final fields agree at stated precision; legacy imports unchanged |
| 3. Add geometry/state | General NURBSpy midline/height builder, exact radial/conical constructors, arclength metrics/volumes, general-EOS state maps and zero closures | Independent special-case metrics, inverse/unit-speed/volume checks; rest cancellation; state/AD/EOS identities |
| 4. Add constant FVM | General-EOS four-state flux, angular transport, sources and simultaneous boundary blocks; one shared integration engine | Full augmented dense/colored/banded agreement; zero boundary storage; no nested BC solves or gamma dependence |
| 5. Verify smooth flow | Perfect-gas analytic low-M radial/swirl and general-geometry references with constant width, plus the conical angle-defined width exception; constant and MUSCL; Mach/grid refinement | Three-trend field comparisons; analytic accuracy, BC residual and conservation audited for each reconstruction |
| 6. Verify radial inflow | Constant-width radial inflow: subsonic acceleration and sonic-exit/choking limit, each with both reconstructions; selected fixed-angle sweeps | Three trends per feasible branch; mass/BC/conservation audits; no fabricated internal smooth transonic solution |
| 7. Verify outward cases | Distinct smooth swirl-dominated diffusion and meridional-supersonic shocked diffusion, with an optional supersonic expansion sign check, using constant width or the selected conical exception | Three trends per selected case; correct total/normal-Mach distinction and separate method audits |
| 8. Generalize mesh/release prototype | Nonuniform metrics, clean fluid-object factory/examples, EOS-independence/API audit and reports | Nonuniform verification with perfect-gas backend; general-EOS contract and numerical dependency audit; no duplicate solver |

Phases 2 and 3 can proceed independently after their contracts are frozen.
Both use the dependency baseline established in Phase 1a; do not hide upgrade
effects inside source extraction or numerical-method changes.
Never merge behavioral numerical changes into the baseline extraction without
separate review. A useful first milestone ends at Phase 5; the selected feasible
inflow/outward families and completed base prototype include Phases 6–8.

The original five-regime adapted/internal-shock radial C-D family is deferred
under the constant-width verification choice. It requires either a revised
width-law decision for the radial fixture or a different admissible fixture.
A constant-width curvilinear channel with an internal radius minimum is one
new proposal for the latter, not an approved geometry. Construct and check any
such reference before adding it to the required campaign. This optional
fixture decision does not hold up the base general solver or its accepted
constant-width/conical verification.

Phase 2 regressions must include representative regularized/legacy flux,
constant/MUSCL/smooth reconstruction, and local-JVP on/off configurations.
Preserve existing wrapper return tuple contracts even when shared internal
objects expose richer status. Aim for bitwise agreement when operation order is
unchanged; otherwise state fixed scaled comparison tolerances before refactoring
and investigate every discrepancy rather than absorbing it into a new baseline.

## 7. Verification, reporting and acceptance

Use the selected cases and accepted initial tolerances in
[PHYSICS_AND_BENCHMARKS.md](../REPORT.html#theory-overview). The following are distinct:

1. Root convergence: original spatial residual and physical BC residuals.
2. Discretization accuracy: fields/Cp/shock position versus independent references.
3. Conservation: exact assembled boundary fluxes and sources, not only sampled
   primitive invariants.
4. Physical branch: choking, entropy increase through shocks, intended outlet regime.

The user accepted the proposed Cp normalization, initial accuracy/convergence
targets, mesh sequence and JSON/CSV plus Markdown/PNG/PDF reporting. Retain
failures and sonic-limit discrepancies; do not relax those targets silently.
Verification compares against independent analytical solutions. No NozzleX
solution comparison is part of this campaign.

Every verification **field/reference comparison** must show these three trends
together on the same axes, in this legend order:

1. `Analytical reference`.
2. `FVM (constant reconstruction)`.
3. `FVM (MUSCL reconstruction)`.

Apply this to smooth and shocked cases, including pressure, $C_p$, Mach and
other reported fields. For each mesh/backpressure, hold geometry, fluid, flux,
physical BC equations/data, initialization, precision, controller and stopping
criteria fixed across the two numerical runs; change only reconstruction and
its required stencil. Use independent analytical references and consistent
finite-volume projections for error calculations. Use distinct styles/markers
when curves overlap. Put different grids/regimes in separate panels or figures
so that each comparison retains the three requested trends. Numerical residual
histories remain a separate diagnostic; do not invent an analytical iterative
trajectory. Run and report residual/BC/conservation/accuracy checks separately
for each reconstruction, retaining failed cases and actual final fields.

Focused regression commands, from the repository root:

```powershell
poetry run pytest tests/test_euler_flux_regularity.py tests/test_euler_reconstruction_regularity.py tests/test_euler_steady_operator.py
poetry run python -m development.roberto.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development.roberto.euler_nozzle.verify_euler_nozzle_sod --no-show
```

Poetry was not on this shell's PATH; dependency resolution and installation used
the existing workspace Poetry tool and project `.venv`. Verification uses
`.venv/Scripts/python.exe` as a transparent execution equivalent to `poetry run`.
NURBSpy alignment is now recorded in the dependency section above and in the
implementation manifest; this is no longer a documentation-only checkout.

New production tests should cover mathematical contracts, not mirror the source:
analytic flux consistency, conservation/telescoping, fixed-radius transverse
Euler waves, source geometry, independent AD/Jacobian and small-mesh band solves.
Include coupled-block dependency tests: current boundary changes must affect
the adjacent physical fluxes and Newton cross-blocks, boundary rows must be
independent of old boundary state and timestep storage, and the convergence
monitor must evaluate the actual coupled state without a nested inlet solve.
Keep experimental audits and benchmark drivers in this study's
`scripts/`; do not import development code from `tests/`.

Run the main analytic benchmark suite with a JAXProp perfect-gas backend, then
audit that replacing the fluid object requires no numerical-method changes.
Initial solver verification uses the perfect-gas backend, as requested; a
real-fluid application/test campaign is not required for this stage. Exercise
the general thermodynamic interface and $a/G$ identities with that backend,
including a restricted interface that exposes only the declared state fields,
to catch numerical access to perfect-gas constants. Record which other JAXProp
backends require existing JIT/AD adapters. A later real-fluid integration smoke
check may reuse one geometry and the same factory; distinguish it from the
current analytic verification and do not claim unperformed real-fluid validation.

Add a dependency audit that numerical FVM/model modules do not access
`gamma`, form `cp/cv`, use perfect-gas pressure/enthalpy/isentropic formulas or
import analytical reference code. Gas constants and gamma may be used to build
independent perfect-gas fixtures and references, which pass physical states/data
into the unchanged numerical solver.

Existing reference reuse needs one additional boundary: the backpressure study's
`scripts/reference.py::PerfectGasReference` imports fixed development geometry
and settings. Extract its general scalar relations into the appropriate model
analytical module and have both studies call them; do not import a development
class into `src/` or duplicate its complete operating-map implementation.
The zero-swirl general-EOS nozzle reference assumes
$v_m=\sqrt{2(h_0-h)}$ and cannot be used unchanged when swirl is present.

The implemented driver is `verify_annular`, with `--cases` selecting the radial,
sonic, swirl, conical, curved, supersonic-expansion and shocked fixtures.
`audit_results` independently revalidates saved states. Invoke both as modules
under `development.roberto.euler_annular.scripts`. Each field verification runs
both reconstruction variants and writes their separate results together with
one independent reference dataset.
Record `reconstruction`, `boundary_formulation="coupled"`, physical/boundary
unknown counts, row scales and boundary defects in each run's metadata.

Save effective settings, EOS/backend identity and reference state, mesh metrics, precision, source
hashes, cold/continuation initialization, all accepted/rejected work, final fields,
original residuals and BC defects in machine-readable outputs. Reports embed
relative PNG links and retain PDF figures. Use the repository plasma convention.
For each selected backpressure comparison use one panel per pressure, with the
analytical, constant and MUSCL curves in every field panel. Save separate results/statuses
for the two reconstructions rather than overwriting one with the other. Timing studies are
optional after correctness, using the repository's one excluded full solve plus
ten sequential measured solves, with JAX synchronization and all samples retained.

## 8. Risks and deliberate exclusions

- Low-Mach acoustic dissipation may dominate pressure recovery; test several Mach
  numbers and grids before deciding on a flux correction or background filter.
- Annular geometry/source imbalance can resemble convergence failure; audit
  rest cancellation and angular transport before changing Newton settings.
- Sonic-limit field error can persist at a converged discrete root, as already
  documented for EulerNozzle. Report it separately from solver convergence.
- Swirled sonic transitions require a regular effective throat, not merely the
  minimum geometric area. Construct references before claiming a regime exists.
- Constant-width purely radial inflow has no smooth internal transonic branch.
  The shaped-width five-regime fixture remains deferred; do not make it a
  completion gate for the base solver or silently substitute a new geometry.
- The sibling NozzleX ODE contains formula issues identified in the physics note;
  it is a useful geometry/physics reference, not an independent numerical oracle.
- General-EOS thermodynamic recovery and derivative support are part of the first
  implementation. The initial perfect-gas analytic suite does not justify
  ideal-gas assumptions in boundaries, fluxes or supporting numerical utilities.
- No ad hoc sibling-checkout imports, NozzleX integration/comparison, new EOS backend,
  empirical friction/diffusion/curvature losses, wall heating, rotating-wall work,
  detonation chemistry or two-fluid migration is part of this first prototype.
  Preserve extension points and add those only in later, separately verified work.

The implementation is complete when the general-EOS contract and integration
gates, selected perfect-gas benchmark families and production regressions pass,
reports expose failures and accuracy limits, and
EulerNozzle/EulerAnnular use the same extracted numerical engine. A plan, a
low residual alone, or a successful subset of branches is not completion.
