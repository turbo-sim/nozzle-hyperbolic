# Package architecture

Status checked against `src/nozzle_hyperbolic/` on **31 October 2026**. This guide describes the supported shared finite-volume path, rather than every historical implementation in the repository.

[Open the browser map](ARCHITECTURE.html) · [Open the vector diagram](ARCHITECTURE.svg)

The SVG is the portable, editable architecture reference. The HTML companion provides a convenient browser view; this Markdown document explains the choices and their limits. “Implemented” describes an available code path, and does not imply that every combination has been verified for every model.

## Overview

The package separates **physical definitions**, **spatial discretization**, and **solution machinery**. A user constructs a model and selects a numerical `Scheme`; an `Operator` binds those choices into JAX kernels. The steady, grid-sequence, and transient drivers then advance or solve the assembled equations.

| Layer | Owns | Does not need to know |
|---|---|---|
| Physical model | Variables, conserved storage, physical fluxes, wave information, integrated sources, closures, boundary equations, admissibility, scales | Which time integrator, Jacobian backend, or Newton policy is selected |
| Spatial operator | Reconstruction, numerical flux combination, fixed mesh measures, conservative residual assembly, optional filter | Euler variable identities, fluid properties, wall formulas, inlet/outlet physics |
| Numerical solution | Explicit stages, backward Euler, algebraic boundary completion/elimination, AD, linear solves, Newton, timestep control | Whether the model is nozzle Euler, annular Euler, or scalar transport |
| Drivers and assessment | Initialization, accepted/rejected histories, residual stopping, monitor windows, conservation budgets, model audits | Model-specific formulas for measured quantities |

The supported balance law has cell-local conserved storage on a fixed one-dimensional mesh:

$$
Q_i(W_i)=V_i U(W_i),\qquad
R_i(W)=A_{i+1/2}\widehat F_{i+1/2}-A_{i-1/2}\widehat F_{i-1/2}-S_i(W).
$$

Here $W$ is the model's iteration/reconstruction representation. It need not equal the conserved variables $U$. The model supplies the already integrated source $S_i$; the operator applies face area once. Physical boundary traces satisfy separate algebraic equations $B(W,t)=0$.

## Models

| Model | State and physics | Status in the shared engine |
|---|---|---|
| Euler nozzle / duct | $W=(p,u,h)$; mass, axial momentum, and total energy; variable area and optional wall sources | Implemented in [`NozzleModel`](../../src/nozzle_hyperbolic/models/euler_nozzle/model.py) |
| Annular Euler | $W=(p,v_m,\ell,h)$, with $\ell=r v_\theta$; mass, meridional momentum, angular momentum, and total energy; fixed curvilinear geometry and swirl | Implemented in [`AnnularModel`](../../src/nozzle_hyperbolic/models/euler_annular/model.py) |
| Signed linear advection | One transported scalar, prescribed inflow and extrapolated outflow; either transport direction | Implemented in [`AdvectionModel`](../../src/nozzle_hyperbolic/models/scalar/advection.py), without an EOS |
| Conservative Burgers | One nonlinear scalar; shock/rarefaction Riemann problems and flow-dependent endpoint treatment | Implemented in [`BurgersModel`](../../src/nozzle_hyperbolic/models/scalar/burgers.py), using HLL/Rusanov without an eigensystem |
| User-programmed balance law | Arbitrary variable count, subject to the contract below | Supported extension point; no model base class is required |
| Two-fluid model | Additional phase variables, interactions, closures, and potentially nonconservative terms | **Deferred**: [`models/two_fluid/`](../../src/nozzle_hyperbolic/models/two_fluid/) retains legacy code, not a model integrated with the shared FVM engine |

Euler thermodynamic properties come from **JAXProp `Fluid.get_state()`**, including the density, sound speed, entropy, and derivative data needed by the model. Choosing a perfect-gas property backend does not select a separate numerical solver. In the evolution path, the deliberate exception is the optional finite-wave outlet described below, which currently contains perfect-gas relations. Separately scoped analytical verification references may also assume a perfect gas.

[`thermodynamics/adapters.py`](../../src/nozzle_hyperbolic/thermodynamics/adapters.py) provides `as_jax_fluid`: native JAX property backends pass through, while configured host-only fluids use JAXProp's callback/custom-JVP bridge. The adapter preserves the selected fluid and its configuration. A lock protects mutable host property evaluation; this route can have different performance characteristics from a native JAX backend. Physical property calls and their derivatives remain outside the generic numerical layer.

## Geometry and closures

Geometry is prepared before binding the operator and remains fixed during a solve. The shared [`Mesh`](../../src/nozzle_hyperbolic/numerics/mesh.py) contains cell centers, physical endpoint locations, face areas, widths, and volumes. Model geometry can also provide the integrated metrics needed by its source terms.

- [`geometry/nozzle.py`](../../src/nozzle_hyperbolic/geometry/nozzle.py) supplies smooth nozzle and constant-area geometry, with exact smooth-area integrals for the supported profiles.
- [`geometry/annular.py`](../../src/nozzle_hyperbolic/geometry/annular.py) and [`geometry/fvm.py`](../../src/nozzle_hyperbolic/geometry/fvm.py) provide annular midline/height geometry and fixed-mesh preprocessing. NURBSpy geometry processing is host-side; shape differentiation through this preprocessing is not provided.
- [`models/closures/wall.py`](../../src/nozzle_hyperbolic/models/closures/wall.py) supplies `PrescribedWall`, `HaalandWall`, laminar $64/Re$ and turbulent Haaland Darcy factors, and signed shear evaluation with a finite zero-velocity limit. Density and viscosity are obtained through the fluid state. The physical model converts wall stress and heat flux into its momentum, torque, and energy sources.
- [`models/closures/mixture.py`](../../src/nozzle_hyperbolic/models/closures/mixture.py) retains mixture viscosity and two-phase friction correlations for use by models. Their presence does not make the Euler models two-fluid models.

Prescribed shear/heat profiles and custom callable wall laws are supported. Some annular closure choices remain placeholders or empirical policies; the diagram does not imply a completed turbulence, heat-transfer, or two-fluid closure model. Stationary adiabatic wall drag changes momentum, while signed wall heating supplies total-energy exchange.

## Model contract

The structural protocols in [`solvers/fvm/contracts.py`](../../src/nozzle_hyperbolic/solvers/fvm/contracts.py) define a **square conservative system**, with $N$ physical cells, $V$ variables per cell, and two endpoint trace states. Full state arrays have shape `(N+2, V)`; cell storage has shape `(N, V)`; reconstructed face arrays have shape `(N+1, V)`.

| Model declaration/callback | Purpose |
|---|---|
| `mesh`, `nvar`, `equation_names` | Geometry, variable count, and equation labels |
| `variable_scales`, `conservative_scales`, `global_flux_scales`, `reference_time` | Fixed normalization and dimensional interpretation |
| `cell_conserved(W)` | Nonlinear cell-local conserved storage |
| `face_conserved(W)`, `face_physical_flux(W)`, `face_speed_bounds(W)` | Physical face data for generic conservative fluxes |
| `face_data(W)` | Alternative fused bundle of conserved state, physical flux, speed bounds, and opaque auxiliary data; avoids repeated expensive property evaluation |
| `face_characteristics(...)` or `face_characteristics_from_data(...)` | Additional projectors, eigenvalues, and regularization widths required by the characteristic flux |
| `face_roe_data(...)` or `face_roe_data_from_data(...)` | Optional Roe capability: conservative spectral projectors, eigenvalues, nonnegative entropy widths, and a boolean validity flag for every face |
| `flux_to_conserved_frame(flux)` | Convert model face-frame flux into the cell storage frame |
| `integrated_source(full)` | Physical/geometric source already integrated over each control volume |
| `boundary_residual(full, time)` | One canonical definition of the physical endpoint equations, shape `(2, V)` |
| `admissible(full, left_faces, right_faces)` | Validate cells, traces, and reconstructed states |
| `wave_speed(cells)`, `initial_state()` | Timestep information and physical starting field |
| `source_dependency_radius`, `boundary_dependency_radius`, `boundary_coupling` | Declare physical stencil reach and local/global coupling for Jacobian assembly |

Optional hooks include endpoint physical-flux overrides, boundary wave speeds, explicit source stability bounds, conservative filter data, specialized cell/trace admissibility, engineering quantities, and final solution audits. The operator checks required callbacks, shapes, scales, coupling, and the selected scheme before compilation.

The model declares its physical stencil. The numerical layer combines this with the reconstruction/filter reach to select the actual Jacobian bandwidth. A source or boundary law that secretly reads farther than declared can invalidate colored differentiation; the declaration is part of the interface's correctness requirements.

## Spatial discretization

Reconstruction and flux choice are independent `Scheme` settings. They apply to each supported model when that model supplies the selected flux's required data.

| Choice | Exact setting | Status and role |
|---|---|---|
| Piecewise constant | `reconstruction="constant"` | Implemented; default |
| MUSCL with Van Leer limiter | `reconstruction="muscl_van_leer"` | Implemented; coordinate-aware reconstruction on the fixed mesh |
| MUSCL with smooth Van Leer limiter | `reconstruction="muscl_van_leer_smooth"` | Implemented; optional $C^2$ limiter regularization, controlled by `smooth_limiter_coefficient` |
| Characteristic flux / current FVS path | `flux="characteristic"` | Implemented; default; uses model-supplied characteristic projectors/eigenvalues and entropy regularization controlled by `delta_factor` |
| Rusanov / local Lax–Friedrichs | `flux="rusanov"` | **Already implemented**; only conserved states, physical fluxes, and signal-speed bounds are needed |
| HLL two-wave approximate Riemann flux | `flux="hll"` | **Already implemented**; uses model-supplied lower/upper signal-speed bounds |
| Roe approximate Riemann / FDS flux | `flux="roe"` | Implemented for nozzle/annular Euler; requires model-owned finite-jump Roe data, rather than the present characteristic averaging |

HLL, Rusanov and Roe are available alternatives to the default characteristic FVS path. HLL and Rusanov use signal bounds; Roe additionally needs a finite-jump linearization from the physical model. The generic Roe contraction supports any wave count, including a repeated speed represented by one higher-rank projector. A user-written model can provide this optional capability without introducing EOS or equation-specific branches into the solver.

The Euler Roe construction is shared by the physical models in [`models/euler_roe.py`](../../src/nozzle_hyperbolic/models/euler_roe.py). Both endpoints obtain density, sound speed and Gruneisen coefficient through `get_state(HmassP_INPUTS,h,p)`. Square-root-density velocity/total-enthalpy averages and a centered, dimensionless pressure-secant projection define the matrix. Annular states use transverse velocity at the same face radius, followed by the existing angular-momentum flux transformation. The generic [`numerics/roe.py`](../../src/nozzle_hyperbolic/numerics/roe.py) only contracts model-supplied eigenspaces with the conserved jump.

Roe uses acoustic Harten regularization; the grouped contact/shear field retains the exact absolute value, with its mathematical kink at zero speed. At dimensionless thermodynamic jumps below $\epsilon^{1/3}$, symmetric provisional slopes avoid amplifying property/subtraction noise. A $C^2$ blend reaches the full secant projection above $2\epsilon^{1/3}$. The mixed normalized pressure-secant audit uses $4096\epsilon$ to accommodate the existing host PH inversion's numerical resolution. This is a face-data validity check, separate from physical-solver convergence tolerances. Invalid endpoints, nonpositive Roe sound-speed squared, nonfinite data or excessive secant defect are rejected without a replacement flux. Models expose the exact policy and reference scales in `roe_metadata`.

The optional conservative fourth-difference filter is selected by positive `filter_strength`; its default is zero. It requires additional model filter callbacks and widens the dependency stencil. Explicit timesteps account for wave, source, and filter restrictions together. For Roe they also bound the selected interface dissipation speeds, which can exceed cell physical speeds after a large transverse-velocity jump. A limiter/filter is a numerical option; it does not replace physical admissibility checks.

## Boundary conditions

There are two separate decisions: **what physical law applies at an endpoint**, and **how its algebraic equations are solved**. The model owns the first; `Scheme.boundary_treatment` controls the second. Both implicit treatments call the same physical boundary residual.

### Physical endpoint laws

| Law | Available behavior |
|---|---|
| Nozzle reservoir inlet | Subsonic inlet: prescribed reservoir total enthalpy and entropy, plus outgoing acoustic compatibility |
| Annular inlet | Subsonic reservoir data plus either inlet flow angle or angular momentum; alternatively a complete prescribed supersonic inlet state |
| Acoustic pressure outlet | `outlet_formulation="acoustic"` (default): prescribed backpressure, acoustic compatibility, entropy extrapolation, and extrapolated advected components; general EOS |
| Finite-wave pressure outlet | `outlet_formulation="finite_wave"`: finite compression/expansion-wave treatment with outgoing and sonic-fan branches; **currently restricted to `jaxprop.FluidPerfectGas`** |
| Static supersonic outflow | `outlet_regime="supersonic"`: extrapolated fully outgoing endpoint state |
| Automatic outflow regime | `outlet_regime="auto"`: acoustic pressure/choking/outgoing active-set selection; the finite-wave formulation already selects its wave regime |
| Transmissive Sod endpoints | Nozzle `transmissive=True`: zero-gradient traces at both endpoints; use a domain/time interval where boundary interaction does not contaminate the intended Riemann comparison |
| Scalar endpoints | Model-owned prescribed inflow and extrapolated outflow, appropriate to the transport direction |

The shared Euler outlet implementation is [`models/euler_outlet.py`](../../src/nozzle_hyperbolic/models/euler_outlet.py). Its finite-wave law is the explicitly allowed location for perfect-gas relations, with a TODO to replace them by general-EOS Hugoniot/isentropic wave curves. The default acoustic law evaluates fluid behavior through `get_state()`.

The automatic acoustic law uses a physical pressure/sonic active set rather than a Mach-dependent blend. It remains a linearized acoustic compatibility law, rather than an exact finite-amplitude rarefaction integral. Negative trial velocities are not clipped, but a backpressure value alone does not prescribe all exterior data needed for sustained physical reverse inflow.

### Coupled and segregated solution

| Integrator | `coupled` | `segregated` | `auto` resolves to |
|---|---|---|---|
| Backward Euler | Yes: cells and endpoint traces in one Newton residual | Yes: converged trace roots are eliminated from the cell system | `coupled` |
| Explicit Euler | Rejected as an invalid setting | Required: nonlinear endpoint completion at the relevant stage | `segregated` |
| SSPRK3 | Rejected as an invalid setting | Required: nonlinear endpoint completion at each required stage | `segregated` |

**Zero storage** means that endpoint traces have no control-volume contents and no artificial time derivative. Their values can change over time, but must satisfy $B(W,t)=0$ at the stage/step time. Physical cells carry the transient storage.

Implicit segregation is a nonlinear elimination, not a lagged boundary update. Every reduced residual evaluation solves the trace subproblem at the current cells. A custom JVP differentiates the converged root through the implicit-function relation

$$
\delta W_b=-B_b^{-1}\left(B_c\,\delta W_c+B_t\,\delta t\right).
$$

It does not differentiate the history of boundary Newton iterations. This requires a locally nonsingular trace Jacobian and a consistent root branch. The reduced solve is finally audited against the original full backward-Euler and boundary equations. Coupled and segregated approaches target the same equations, although their nonlinear work and robustness can differ.

## Time integration

| Integrator | Setting | Numerical behavior |
|---|---|---|
| Forward / explicit Euler | `time_integrator="euler-explicit"` | One explicit conservative update; separate nonlinear storage recovery and boundary completion |
| Third-order SSP Runge–Kutta | `time_integrator="runge-kutta-3"` | Three-stage SSPRK3; stage combinations act on conserved contents, with storage recovery and endpoint completion |
| Backward / implicit Euler | `time_integrator="euler-implicit"` | Default; nonlinear backward-Euler equations solved with Newton, using coupled or eliminated endpoint traces |

“Explicit” describes the cell evolution formula. With nonlinear primitive-to-conservative storage and algebraic boundary laws, small auxiliary nonlinear solves can still be necessary. The explicit implementation advances $Q=VU(W)$ and recovers $W$; it does not apply conservative RK formulas directly to primitive variables.

The same integrators can be used for steady pseudo-time marching and physical-time transients, through different drivers. [`solve_single_grid`](../../src/nozzle_hyperbolic/solvers/fvm/drivers.py) marches toward the original spatial steady root. `solve_transient` advances to requested output times and checks actual time-dependent conservation budgets. A small steady residual is not a transient stopping condition. `Scheme()` defaults to characteristic FVS; `solve_transient` uses `Scheme(flux="hll")` when no scheme is supplied. Select the integrator explicitly for the intended application.

The timestep uses model wave speeds and any boundary wave-speed bound. Explicit integration additionally combines model source and optional filter bounds as additive inverse times, rather than assuming the minimum individual bound is sufficient. Invalid stages or failed nonlinear completion cause rejection/retry. The steady driver currently caps explicit CFL at `0.45`; the transient driver accepts a user CFL and defaults to `0.45`.

## Jacobian and Newton

The shared numerical path uses **JAX JIT and automatic differentiation**, with 64-bit arithmetic required when binding the operator. Physical callbacks must therefore remain traceable and differentiable on the relevant admissible branches.

| Component | Default | Alternatives / details |
|---|---|---|
| Jacobian assembly | `jacobian_mode="colored"` | Compressed forward JVPs via `jax.linearize` and `jax.vmap`; `"dense"` is available for reference/debugging |
| Linear correction | `linear_solver="banded"` | Variable-size block-banded storage and grouped block Thomas elimination; `"dense"` is available |
| Newton correction | `newton_backtracking=False` | Full undamped corrections; `True` enables admissibility-aware Armijo backtracking |
| Newton iteration cap | `newton_iterations=10` | Maximum corrections, not a forced count; user configurable |
| Newton convergence | Update tolerance `1e-10`, residual tolerance `1e-12` | Both enabled criteria must pass; either can be disabled by setting it to zero, but not both |

Colored differentiation exploits the declared local dependency radius $r$: independent block columns share seed directions, with up to $V(2r+1)$ directions. The banded solve avoids constructing a dense global Jacobian. Constant versus MUSCL/filter reconstruction and the physical source/boundary declarations determine $r$.

Supported backend combinations are colored/banded, colored/dense, and dense/dense. Dense differentiation with the banded backend is rejected. A model declaring global endpoint coupling must use dense/dense so that nonlocal blocks are not silently discarded. This safeguard is not a periodic-problem implementation.

Full Newton corrections still require finite, admissible trials and a converged final inner solve. The steady driver rejects a failed/capped solve and reduces CFL; it does **not** accept an unconverged update merely because the cap was reached. The optional Armijo policy additionally demands residual decrease for accepted Newton trials. Auxiliary boundary completion and conservative-storage recovery retain their own safeguarded backtracking even when implicit flow Newton is undamped.

The banded algorithm pivots within grouped blocks, rather than across the entire matrix. Consequently a nonsingular full matrix is not a guarantee that this particular factorization succeeds. Dense AD comparisons and linear/nonlinear diagnostics are useful extension checks; the architecture does not claim a general sparse direct or Krylov backend.

## Initialization and convergence

Three public drivers cover the main use cases:

| Driver | Purpose |
|---|---|
| `solve_single_grid(model, scheme, settings, ...)` | Solve the original steady equations on one mesh |
| `solve_grid_sequence(model_factory, cells, scheme, settings, ...)` | Solve successive coarse-to-fine meshes, prolonging successful numerical states |
| `solve_transient(model, output_times, scheme, cfl=..., ...)` | Physical-time integration with output snapshots and conservation histories |

The initialization accelerator often called “multigrid” in development notes is **grid sequencing / nested iteration**. It solves each mesh's original equations and uses componentwise linear interpolation to initialize the next mesh. There is no fine-grid defect restriction, FAS cycle, or multigrid correction operator. Failed warm starts remain recorded and may be followed by an independent cold retry. Each mesh starts its own residual normalization.

For steady marching, the source defaults are `cfl=0.2`, `cfl_growth=1.4`, `cfl_shrink=0.5`, `cfl_max=1e7`, `max_steps=4000`, and `max_retries=14`. These are package defaults, not the settings of every development benchmark. Growth follows successful steps; failed trials reduce the requested timestep.

`ResidualPolicy` checks each equation with fixed physical reference scales. The defaults are a volume-weighted L2 norm and **relative reduction of `1e-6` OR absolute norm of `1e-9`**. `criterion` can instead be `"relative"`, `"absolute"`, or `"both"`; `"either"` is the default. Equations initially below the `1e-14` floor use the absolute target. These dimensionless absolute norms are not raw residuals in SI units.

Boundary and global conservation defects must independently meet their default `1e-9` targets, alongside admissibility and finiteness. Optional model audits and engineering monitor windows can impose additional requirements. Model callbacks evaluate engineering quantities; the generic [`MonitorTracker`](../../src/nozzle_hyperbolic/solvers/monitoring.py) checks their accepted-state stationarity windows.

Transient histories retain actual accepted/rejected attempts and integrator-consistent integrated boundary/source transfer. Their conservation budgets compare that transfer with changes in total conserved contents. They are distinct from steady spatial/global convergence criteria.

The maintained [steady benchmark suite](../../development/roberto/benchmarking/README.md)
loads physical YAML resources beside each model through
[`nozzle_hyperbolic.verification`](../../src/nozzle_hyperbolic/verification/README.md).
It records case/source identity, analytical field and conservative errors,
independent grid refinement, and repeated old/current solve timings.

## Configuration example

The following shows the separation of responsibilities. Supply a suitable JAXProp `fluid` object from the application; these snippets illustrate API configuration and are not a verified physical case definition.

```python
import jax

jax.config.update("jax_enable_x64", True)

from nozzle_hyperbolic.geometry.nozzle import SymmetricNozzleGeometry
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.solvers.fvm import (
    ResidualPolicy, Scheme, SteadySettings,
    solve_grid_sequence, solve_single_grid, solve_transient,
)

geometry = SymmetricNozzleGeometry()

def model_factory(cells):
    return NozzleModel(
        geometry.build_mesh(cells),
        fluid,
        p0=2.0e5,
        T0=300.0,
        back_pressure=1.9e5,
        outlet_formulation="acoustic",
        outlet_regime="subsonic",
    )

scheme = Scheme(
    reconstruction="muscl_van_leer",
    flux="characteristic",
    time_integrator="euler-implicit",
    boundary_treatment="coupled",  # or "segregated"
    jacobian_mode="colored",
    linear_solver="banded",
    newton_iterations=10,
    newton_backtracking=False,
)
settings = SteadySettings(
    residual=ResidualPolicy(criterion="either"),
)

steady = solve_single_grid(model_factory(200), scheme, settings)
finest, stages = solve_grid_sequence(
    model_factory, cells=(50, 100, 200), scheme=scheme, settings=settings,
)

transient_scheme = Scheme(
    flux="hll",
    reconstruction="constant",
    time_integrator="runge-kutta-3",
    boundary_treatment="segregated",
)
snapshots = solve_transient(
    model_factory(200), [0.0, 1e-4, 2e-4], transient_scheme, cfl=0.45,
)
```

Inspect `Solution.success`, `reason`, `metrics`, and `history`; reaching an iteration limit or returning an array is not by itself evidence of a converged solution. Scheme selections are bound before JIT compilation. Switch model, mesh, or structural numerical choices by constructing the corresponding operator/solve configuration.

## Verification and limits

Production verification is exercised by [`tests/`](../../tests/) and the [transient boundary-treatment study](../../development/roberto/transient_problems/README.md), which use the source implementations. Earlier studies are indexed in [`development/roberto/historical/`](../../development/roberto/historical/README.md). Canonical analytical references live beside their physical models in [`euler_nozzle/analytical/`](../../src/nozzle_hyperbolic/models/euler_nozzle/analytical/), [`euler_annular/analytical/`](../../src/nozzle_hyperbolic/models/euler_annular/analytical/), and [`scalar/analytical.py`](../../src/nozzle_hyperbolic/models/scalar/analytical.py), separately from evolution kernels. The [source verification guide](../../src/nozzle_hyperbolic/verification/README.md) explains the packaged physical cases and reference assumptions. Older studies and low-level regression utilities should not be mistaken for the supported driver path.

The architecture's important boundaries are:

- **Current contract:** square conservative systems, cell-local storage, fixed one-dimensional geometry, two nonperiodic endpoint traces. Nonconservative products, rectangular systems, moving geometry, and periodic problems are not supported by this contract.
- **Two-fluid extension:** intended direction, not an already validated plug-in. Its equations and boundary/source structure may require an explicit contract extension.
- **Outflow physics:** finite-wave outlet generalization to a general EOS remains open; the acoustic alternative is general EOS. Sustained reverse inflow needs additional physical data.
- **Numerical robustness:** matching final steady roots does not imply identical Newton trajectories across boundary treatments, reconstruction choices, or controller settings. Undamped Newton and local block pivoting have finite robustness limits.
- **Roe/EOS scope:** the Euler construction assumes a smooth, thermodynamically consistent PH map on the selected branch. Stable endpoints do not prove a positive interface sound speed. The acoustic fix does not establish global positivity or entropy stability for arbitrary real-gas/nonconvex states; phase switches, independently interpolated inconsistent property tables and future two-fluid equations need separate assessment. See the [Roe study](../../development/roe_method/README.md).
- **Coverage:** the existence of a selector is not proof of every model × flux × reconstruction × integrator × boundary combination. Check the relevant verification study before claiming accuracy, robustness, or speed.
- **Physical accuracy:** convergence of the discrete residual, conservation, agreement with an analytical reference, and selection of the intended flow branch are separate checks. The previously observed sonic-limit accuracy discrepancy is not eliminated simply by reducing the nonlinear residual.

This documentation was produced by source inspection; no new flow simulations are implied by the diagram.

## Extending the package

1. Implement the structural model callbacks and declare equation count, fixed scales, source/boundary stencil reach, and coupling explicitly. Keep physical and EOS decisions out of the shared engine.
2. Provide conserved states, physical fluxes, and valid signal-speed bounds for HLL/Rusanov. Add characteristic projectors/eigenvalues if the characteristic flux is required. To select Roe, supply its optional finite-jump matrix/eigenspaces and face validity; an arbitrary mean-state Jacobian is insufficient. Reuse EOS data through `face_data` when helpful.
3. Define boundary equations once. Check the number of incoming conditions and the local trace Jacobian before using nonlinear elimination. Test coupled and segregated backward Euler against the original full residual.
4. Supply admissibility checks for cells and reconstructed states; provide source stability information when an explicit method needs it. Validate zero-velocity and other limiting states without hidden clipping.
5. Verify storage recovery, conservation, stencil declarations/colored versus dense AD, and meaningful limiting/reference cases. Then exercise the intended integrator, reconstruction, and physical boundary combinations.

Roe demonstrates the split between generic numerical-flux contraction and model-owned finite-jump averaging. A future physical model belongs behind the model contract. Full FAS multigrid or a new sparse/Krylov backend would be separate numerical extensions, rather than physical-model code.

## Source map

Paths below are relative to `src/nozzle_hyperbolic/` unless linked elsewhere.

| Location | Main responsibility |
|---|---|
| [`solvers/fvm/config.py`](../../src/nozzle_hyperbolic/solvers/fvm/config.py) | `Scheme`, `SteadySettings`, `Solution`, timestep diagnostics |
| [`solvers/fvm/contracts.py`](../../src/nozzle_hyperbolic/solvers/fvm/contracts.py) | Structural model protocols |
| [`solvers/fvm/operator.py`](../../src/nozzle_hyperbolic/solvers/fvm/operator.py) | Model validation, conservative spatial/time residuals, compiled numerical kernels |
| [`solvers/fvm/drivers.py`](../../src/nozzle_hyperbolic/solvers/fvm/drivers.py) | Single-grid, grid-sequence, and transient execution |
| [`solvers/fvm/elimination.py`](../../src/nozzle_hyperbolic/solvers/fvm/elimination.py), [`solvers/boundaries.py`](../../src/nozzle_hyperbolic/solvers/boundaries.py) | Converged trace completion and differentiable nonlinear elimination |
| [`solvers/time_marching/`](../../src/nozzle_hyperbolic/solvers/time_marching/) | Shared Newton, conservative forward Euler, SSPRK3 |
| [`solvers/residuals.py`](../../src/nozzle_hyperbolic/solvers/residuals.py), [`solvers/monitoring.py`](../../src/nozzle_hyperbolic/solvers/monitoring.py) | Fixed-scale convergence and optional engineering monitors |
| [`numerics/reconstruction.py`](../../src/nozzle_hyperbolic/numerics/reconstruction.py), [`numerics/characteristic.py`](../../src/nozzle_hyperbolic/numerics/characteristic.py), [`numerics/fluxes.py`](../../src/nozzle_hyperbolic/numerics/fluxes.py), [`numerics/roe.py`](../../src/nozzle_hyperbolic/numerics/roe.py), [`numerics/filters.py`](../../src/nozzle_hyperbolic/numerics/filters.py) | Reusable spatial arithmetic |
| [`numerics/jacobians.py`](../../src/nozzle_hyperbolic/numerics/jacobians.py), [`numerics/linear.py`](../../src/nozzle_hyperbolic/numerics/linear.py) | Colored/dense AD and block-banded/dense correction solves |
| [`models/euler_nozzle/`](../../src/nozzle_hyperbolic/models/euler_nozzle/), [`models/euler_annular/`](../../src/nozzle_hyperbolic/models/euler_annular/), [`models/scalar/`](../../src/nozzle_hyperbolic/models/scalar/) | Physical models and their model-specific helpers |
| [`models/euler_outlet.py`](../../src/nozzle_hyperbolic/models/euler_outlet.py) | Shared Euler acoustic and finite-wave endpoint laws |
| [`models/euler_roe.py`](../../src/nozzle_hyperbolic/models/euler_roe.py) | Shared physical Euler Roe averages, secant coefficients, grouped eigenspaces and validity policy |
| [`models/closures/`](../../src/nozzle_hyperbolic/models/closures/) | Canonical wall and mixture closure correlations |
| [`geometry/`](../../src/nozzle_hyperbolic/geometry/), [`thermodynamics/`](../../src/nozzle_hyperbolic/thermodynamics/) | Fixed geometric preprocessing and general-EOS adapters/helpers |

## Maintaining these documents

Use the SVG for presentations, printing, and vector editing; use the HTML for exploring the map and reading the guide in a browser. The HTML embeds the diagram, guide, scripts, styles, equation renderer, and downloadable SVG/Markdown. It works offline without a local server. Source-file links point into the repository and require the checkout to remain alongside it.

The editable diagram definition is [`_scripts/build_architecture.py`](_scripts/build_architecture.py); the browser renderer is [`_scripts/architecture_viewer.py`](_scripts/architecture_viewer.py). Edit this Markdown guide and the diagram definition, then regenerate from the repository root:

```powershell
poetry run python -B docs/architecture/_scripts/build_architecture.py
```

Regeneration reads documentation only; it does not import the numerical solver or run simulations. Keep implemented and planned labels synchronized with source changes, and inspect the SVG and browser view after editing.
