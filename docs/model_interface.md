# Physical model interface

The production entry point is `nozzle_hyperbolic.solvers.fvm`. A physical model
is a Python object satisfying the structural protocols in
[`contracts.py`](../src/nozzle_hyperbolic/solvers/fvm/contracts.py); inheritance
is unnecessary. The solver has no Euler-variable, equation-count or EOS branch.
The supported problem has $T$ transported balances and $C$ cell-local
algebraic constraints in $V=T+C$ state variables, fixed one-dimensional
geometry and two endpoint traces. Storage can be rectangular; the assembled
Newton system is square. Conservative models declare `nonconservative_mode="none"`.
An explicit flux-difference compatibility contract supports nonconservative
models without inferring a new path-conservative method.
The package supports two physical endpoints; periodic problems are outside its scope.
Interior flux/source callbacks are autonomous functions of state and fixed model
data; boundary equations additionally receive physical time. Explicitly
time-forced interior laws would require extending the callback contract.

The physical model owns its state variables, storage, physical flux,
eigenstructure, constitutive laws, admissible domain, boundary equations and
physical dependency stencils. The numerical engine owns reconstruction,
numerical flux contraction, conservative assembly, differentiation, Newton and
linear solves, time integration, timestep control and convergence assessment.
Geometry preprocessing constructs fixed mesh measures. Physical case definitions and analytical references belong beside their models;
benchmark profiles select numerical settings and import those definitions. See
[analytical references](analytical_references.md) and the
[installed verification API](../src/nozzle_hyperbolic/verification/README.md).

## Array and conservation conventions

With $N$ physical cells and $V$ variables, the full state has shape $(N+2,V)$.
Rows zero and $N+1$ are independent algebraic endpoint traces, located at the
physical end faces. Interior rows are cell values. Face arrays have shape
$(N+1,V)$ for states and $(N+1,T)$ for transported fluxes. By default, iteration
variables are also reconstruction variables. A model can instead supply
`reconstruction_state(full)`, `from_reconstruction_state(face_states)` and
positive fixed `reconstruction_scales`. These local maps let the limiter act
on physical primitives while retaining the full constrained iteration state.

`Mesh` stores increasing faces, interior centers, positive cell volumes and
positive face areas. Optional geometric data is opaque to the numerical engine.
There are no physical volumes assigned to endpoint traces.

For physical-cell storage density $Q(W)$, unweighted transported face flux $F$
and already integrated cell source $S_i$, the common residual is

$$
R_i=A_{i+1/2}F_{i+1/2}-A_{i-1/2}F_{i-1/2}+N_i-S_i,
\qquad V_i\frac{dQ(W_i)}{dt}+R_i=0.
$$

The engine applies face area exactly once. Do not pre-multiply face callbacks by
area. Include all geometry measures in `integrated_source`. Boundary residuals
are dimensionless algebraic equations with zero storage, independent of $dt$.

## Required data and callbacks

All callbacks used within assembly must support JAX JIT and differentiation.
Enable `jax_enable_x64` before constructing an `Operator`. Use fixed model
configuration rather than changing attributes after binding compiled callbacks.

| Member | Meaning and shape |
|---|---|
| `mesh`, `nvar`, `ntransport`, `nconstraints`, `nboundary` | Geometry and counts $V,T,C,V$, with $T+C=V$ |
| `equation_names`, `constraint_names` | $T$ transported and $C$ algebraic labels, separately monitored |
| `nonconservative_mode` | `none` (zero NC contribution) or `flux_difference` (explicit compatibility callback) |
| `variable_scales` | Positive fixed $(V,)$ scales for reconstruction and state corrections |
| `conservative_scales` | Positive fixed $(T,)$ scales for storage densities and spatial norms |
| `global_flux_scales` | Positive fixed $(T,)$ scales for integrated balance defects |
| `constraint_scales` | Positive fixed $(C,)$ physical scales; required when $C>0$ |
| `reference_time` | Positive fixed physical time scale |
| `source_dependency_radius` | Maximum full-state block distance used by a source row |
| `boundary_dependency_radius` | Maximum full-state block distance used by a boundary row |
| `boundary_coupling` | `local`, or `global` for nonlocal endpoint coupling |
| `initial_state()` | Full $(N+2,V)$ initial state |
| `cell_conserved(physical)` | Cell-local $(N,V)\to(N,T)$ transported storage densities |
| `cell_constraints(physical)` | Cell-local $(N,V)\to(N,C)$ dimensional algebraic defects when $C>0$ |
| `face_conserved(states)` | $(N+1,T)$ conserved densities in the model's face frame |
| `face_physical_flux(states)` | $(N+1,T)$ physical fluxes in the same face frame |
| `face_speed_bounds(states)` | Two $(N+1,)$ vectors of signed lower/upper characteristic speeds; needed for HLL/Rusanov |
| `flux_to_conserved_frame(flux)` | Convert $(N+1,T)$ numerical face-frame flux into the transported frame |
| `integrated_source(full)` | $(N,T)$ integrated physical sources |
| `boundary_residual(full,time)` | $(2,V)$ dimensionless endpoint equations |
| `admissible(full,left,right)` | Scalar boolean validating full states and reconstructed face states |
| `wave_speed(physical)` | $(N,)$ nonnegative maximum absolute signal speeds for timestep selection |

The augmented map `(cell_conserved, cell_constraints)` must be locally
invertible in the admissible domain.
Explicit integration advances $V_iQ_i$, then recovers the iteration variables by
a generic nonlinear solve. This is why `cell_conserved` must be cell-local.
Face property/flux callbacks and frame conversion act independently at each
face; characteristic data may combine that face's left and right states. They
must not introduce undeclared cross-face coupling. Source and endpoint callbacks
may use neighbors within their declared reach. Endpoint flux overrides preserve
internal faces and include their state dependencies in the boundary declaration.

The engine always uses exact local
Jacobians of the complete face-flux map and cell conserved storage, sharing
them across differentiation seeds. It does not interpret EOS properties or
geometry, and it recomputes derivatives at each linearization point. Sources,
boundary equations, reconstruction and endpoint flux overrides keep their
original differentiation paths. This is an internal numerical optimization,
not a `Scheme` input. Direct AD is retained only as an independent reference
in tests and development studies. Callback locality is a required model contract.

For expensive properties, replace the three independent face callbacks with
`face_data(states)`. Return a dictionary with `conserved`, `physical_flux`,
`speed_bounds=(lower,upper)` and `auxiliary`. Auxiliary data is an opaque JAX
pytree; the numerical engine never interprets its physical contents.

### Flux capabilities and current options

`Scheme.flux` accepts `stadtke` (default), `rusanov`, `hll`, `roe`, `osher` and
the explicitly model-owned `model_compatibility` formulation.
The previous `characteristic` selector has been removed; mathematical callback
names such as `face_characteristics` retain their meaning.

Stadtke and Osher require `face_characteristics(left,right,delta_factor)`,
returning projectors $(N+1,K,V,V)$, eigenvalues $(N+1,K)$ and widths $(N+1,K)$.
The wave-group count $K$ is independent of $V$. Widths must be positive for
Stadtke; Osher also permits zero. With fused face data,
`face_characteristics_from_data(left,right,delta_factor,aux_left,aux_right)`
reuses opaque property payloads. HLL and Rusanov require no eigenvectors.

Stadtke has two independent numerical settings, both enabled by default:
`custom_jump_correction` replaces finite physical-flux-jump dissipation with
mean-characteristic conservative-jump viscosity; `custom_endpoint_envelope`
adds an endpoint-family speed envelope to the mean dissipation. Disable both
to select the original Stadtke contraction. Other flux families ignore these
flags. They do not provide a general monotonicity theorem.

Osher integrates characteristic viscosity along a state path. By default the
path is affine in the model's reconstruction variables. A model may implement
`face_state_path(left,right,parameter) -> (state, derivative)`; `parameter` has
shape $(N+1,1)$ and both returned arrays have shape $(N+1,V)$. The derivative is
with respect to that parameter. The same unweighted conservative face frame
must be used for storage, flux and projectors throughout the path.

Osher evaluates characteristic data at coincident path states and needs a
supported JAX directional derivative of the conservative storage along the
path. An EOS wrapper can provide traceable values while lacking that derivative;
verify this capability separately. The physical model owns path admissibility
and hyperbolicity. Invalid sampled states reject the candidate; no clipping or
replacement HLL flux is applied. Finite node sampling does not certify the
entire path or all possible sonic crossings.

`quadrature_points` selects 3, 5 (default) or 9 Gauss nodes. With
`integration="gauss"` (default), one unsplit rule is used. With
`integration="sonic_partitioned"`, the selected rule is applied to panels split
at captured sonic/regularization crossings. These options describe path
integration, not spatial order or physical time integration; other fluxes
ignore them. See the [flux studies](../development/flux_methods/README.md) for
the selected variants, evidence and limitations.

### Optional Roe capability

Selecting `Scheme(flux="roe")` requires either of these model callbacks:

```python
face_roe_data(left, right, delta_factor)
face_roe_data_from_data(left, right, delta_factor, aux_left, aux_right)
```

The second alternative requires `face_data`; its two auxiliary arguments are
the opaque payloads returned for the reconstructed left and right states.
It avoids duplicate EOS/property evaluations. The structural protocols
`RoeModel` and `FusedRoeModel` describe these alternatives; inheritance is not
required. A direct Roe model does **not** need `face_speed_bounds`, because its
selected flux uses the supplied eigenspaces rather than HLL/Rusanov bounds.
If the model supplies the fused `face_data` bundle, that bundle still includes
the normal `speed_bounds` entry defined above.

Return a dictionary with these required entries:

| Entry | Shape and responsibility |
|---|---|
| `projectors` | $(N+1,K,V,V)$ conservative spectral projectors |
| `eigenvalues` | $(N+1,K)$ real interface wave speeds, in matching order |
| `width` | $(N+1,K)$ finite nonnegative entropy-regularization widths |
| `valid` | Boolean $(N+1,)$ vector identifying admissible Roe data at each face |

The wave-group count $K$ is independent of the equation count $V$. A repeated
speed may use one higher-rank projector: annular Euler uses three groups for
four equations, with contact and shear combined in its convective group.
Additional diagnostic dictionary entries are permitted. For the Roe selector,
`Operator.roe_data(full)` exposes the reconstructed model data for inspection.

The model owns the finite-jump averaging/linearization rule, its real spectrum,
entropy policy and supported physical domain. Its matrix must satisfy the Roe
consistency and finite-jump identities in its stated domain to the documented
floating-point/EOS resolution; evaluating an arbitrary mean-state Jacobian is
insufficient. Verify projector completeness,
idempotence and wave matching independently. The generic numerical layer only
contracts the supplied data:

$$
\widehat F=\frac{F_L+F_R}{2}
-\frac12\sum_{k=1}^{K}\psi(\lambda_k,\delta_k)P_k(Q_R-Q_L).
$$

Conserved states, physical fluxes and projectors must use the **same unweighted
conservative face frame**. Evaluate both states at the same fixed face geometry
(for example, the same annular face radius). The engine then calls
`flux_to_conserved_frame` once and multiplies by face area once. Do not include
either transformation or area weighting inside the Roe projectors/contraction.

Positive widths select the Harten regularized absolute value. A zero width
selects the exact absolute value, retaining its mathematical kink at zero;
the implementation avoids dividing by zero in inactive AD branches. Euler
regularizes its acoustic groups and leaves the contact/shear group unregularized.
Other physical models supply their own explicit entropy-width policy.

Set `valid=False` for unsupported/invalid interface averages, even if an
inactive arithmetic placeholder permits finite tracing. The operator combines
this flag with finite projector/speed checks and nonnegative width checks in
its admissibility predicate. Invalid data reject the nonlinear candidate or
explicit stage; they do not select a different flux or clip physical data.
These checks do not establish positivity or entropy stability for arbitrary
models/EOS. Physical admissibility and consistent thermodynamic derivatives
remain model responsibilities.

Explicit Roe timestep selection also bounds the actual interface dissipation
$\max_k\psi(\lambda_k,\delta_k)$ on both faces adjacent to each cell, together
with the existing model wave-speed bound. Interface Roe speeds can exceed
cell physical speeds after large velocity/shear jumps. This restriction uses
the callback data without interpreting physical variables; the implicit
pseudo-time wave bound retains its existing definition.

Dependency declarations are mathematical promises, not performance hints.
The engine combines them with its reconstruction/filter reach when coloring
the Jacobian. Underdeclared dependencies can discard derivatives. Declare
`boundary_coupling="global"` and select `jacobian_mode="dense",
linear_solver="dense"` for nonlocal boundary laws. The engine rejects global
coupling with the banded backend. Verify a new model's colored Jacobian against
dense AD and a directional finite difference before relying on its declarations.

## Optional capabilities

### Constrained cells and nonconservative compatibility

For $C>0$, backward Euler packs each cell's scaled transported rows followed by
`cell_constraints(new)/constraint_scales`. These algebraic rows have **zero
storage**, no volume multiplier and no $dt$ multiplier. Explicit Euler, SSPRK3
and classical RK4 recover $V$ unknowns from $T$ target contents and $C$
constraints at every stage, then complete the physical boundary equations at
that stage time. Coupled and segregated backward Euler use the same cell and
boundary rows. Coloring still operates on square $V$-row cell blocks; local
storage differentiation supports rectangular outputs.

For `nonconservative_mode="flux_difference"`, provide
`integrated_nonconservative(full, unfiltered_flux)` and
`nonconservative_dependency_radius`, and explicitly declare `supported_fluxes`.
The input flux is unweighted and already in the transported frame, **before**
additional stabilization and boundary-flux overrides. The callback returns
integrated $(N,T)$ terms added to the spatial residual. Geometry pairing and
singular limits belong to the model. This is an operational compatibility
contract; it does not implement a general path-conservative jump condition.
The declared radius covers the full-state reach with constant reconstruction;
the engine adds one state block for MUSCL to cover raw-flux dependencies.

`flux="model_compatibility"` requires the face-local callback
`numerical_face_flux(left,right,scheme)`. This enables a documented historical
flux whose physical decomposition does not fit the generic flux families.
Unsupported combinations fail at operator binding. An optional
`validate_scheme(scheme)` can enforce further model-specific restrictions.

`stabilization_flux(full,raw_flux)` returns an **additional** unweighted
transported face flux and requires `stabilization_dependency_radius`. Its
parameters belong to the model, independently of the generic fourth-difference
filter. Model filters must also supply appropriate explicit timestep bounds.
`Operator.residual_contributions` exposes conservative divergence,
stabilization, nonconservative terms and sources separately. Its `balance_rate`
includes the actual discrete NC work in physical-time budgets.
Optional fixed `invariant_projections` of shape `(invariants,T)`, descriptive
`invariant_names` and `invariant_flux_scales` let the model identify true
physical invariants. Monitoring evaluates them with the unmodified NC/source
budget and requires their steady defects to pass independently.

An explicitly equation-changing model policy can implement
`transform_cell_equations(full, integrated_rows, dt=None)`. It acts on each
cell's own transported rows: steady spatial rows when `dt=None`, or raw
backward-Euler content change plus $dtR$ otherwise. Declare
`equation_dependency_radius` for any extra state dependencies; cross-cell
mixing of the input equation rows is not supported. The generic engine then
applies its usual fixed scales and appends the cell constraints. Such policies
must declare their integration restrictions in `validate_scheme`; `steady_only`
rejects physical-time integration. With active `has_equation_pins`, monitoring
also records `unmodified_equation_norm`, `unmodified_equation_linf` and
`unmodified_global_defect`. Passing the selected equations is not evidence that
the unmodified balances vanish.

All packaged conservative models explicitly declare $C=0$ and zero NC terms;
external square models without the new declarations retain backward-compatible
defaults. The [two-fluid migration guide](two_fluid_migration.md) records the
frozen Alberto formulation, selected defaults and intentional deviations.

| Callback | Purpose |
|---|---|
| `face_roe_data(left,right,delta_factor)` or fused `face_roe_data_from_data(...)` | Optional model-owned finite-jump Roe eigenspaces, entropy widths and per-face validity; required only for `flux="roe"` |
| `cell_admissible(physical)` | Scalar boolean for safe local storage recovery |
| `boundary_admissible(full)` | Scalar boolean for safe endpoint completion |
| `boundary_wave_speed(full)` | Nonnegative $(2,)$ endpoint speeds included in the wave timestep |
| `boundary_face_flux(full,flux)` | Return $(N+1,V)$ unweighted transported flux, replacing physically prescribed endpoint fluxes where needed |
| `explicit_timestep(full)` | Scalar **absolute time limit**, positive or $+\infty$, from stiff physical sources |
| `filter_conserved(full)` | $(N+2,V)$ transported densities for the optional conservative filter |
| `filter_signal_speed(full)` | Nonnegative $(N+1,)$ physical speed/sensor weights for that filter |
| `solution_audit(full)` | Scalar boolean final model-specific acceptance result |
| `transient_solution_audit(full)` | Scalar boolean physical-time snapshot check; falls back to `solution_audit` when absent |
| `fields(full)` | Convenient physical postprocessing; outside numerical assembly |

Explicit wave, source and filter restrictions are combined by adding their
inverse time limits:

$$
dt^{-1}=(\mathrm{CFL}\,dt_{wave})^{-1}+dt_{source}^{-1}+dt_{filter}^{-1}.
$$

An inactive restriction contributes zero inverse rate. This conservative
composition accounts for simultaneous terms; taking the minimum of separately
stable limits can overstep the stability limit of their sum. The absolute source
bound is not multiplied by CFL. An implicit step does not use the explicit
source/filter restrictions. A model must
supply a suitable bound for a stiff source; a generic wave CFL cannot infer it.
The Haaland policy supplies a local drag estimate, whose limits are documented
in [physical closures](physical_closures.md).

The optional filter adds an internal conservative face correction in transported
densities. Its strength is numerical; velocity gates or other physical sensors
belong to the model. Do not put primitive-variable smoothing in the integrator.

## Integration and stopping

`Scheme.time_integrator` selects `euler-implicit`, `euler-explicit`,
`runge-kutta-3` or `runge-kutta-4`. `Scheme.boundary_treatment` selects how the same algebraic
boundary equations are solved:

| Setting | Backward Euler | Explicit Euler / SSPRK3 / RK4 |
|---|---|---|
| `auto` (default) | Coupled | Segregated |
| `coupled` | Cell storage and zero-storage boundary rows solved together | Rejected at configuration |
| `segregated` | Nonlinear trace elimination inside each cell-residual evaluation | Nonlinear endpoint solve at each required stage |

```python
implicit = Scheme(time_integrator="euler-implicit", boundary_treatment="segregated")
explicit = Scheme(time_integrator="runge-kutta-3", boundary_treatment="segregated")
```

The implicit segregated solve converges the boundary equations at each candidate
physical state. For $B(z,w,t)=0$, where $z$ contains the endpoint traces and $w$
the cells, its derivative is $dz=-B_z^{-1}(B_w\,dw+B_t\,dt)$. This implicit-function
derivative enters the reduced Newton matrix. Both implicit treatments therefore
solve the same backward-Euler equations. A boundary completion failure rejects
the candidate. Segregation requires a locally invertible trace block $B_z$;
a boundary law that cannot locally determine its own traces requires the
coupled formulation or a revised physical contract.

`Operator.effective_boundary_treatment` reports the resolved choice.
`Scheme.newton_iterations` defaults to 10. This is a maximum number of
corrections per nonlinear solve; convergence tests can stop it earlier.
Implicit flow Newton corrections are undamped by default:
`Scheme.newton_backtracking=False`. Set it to `True` for Armijo backtracking.
The option applies to both coupled and segregated implicit flow solves. Full
corrections must still produce finite admissible candidates, and an unsuccessful
nonlinear solve rejects the timestep and reduces the CFL. Auxiliary boundary
completion and conservative-storage recovery retain their safeguarded solves.
Full `residual`/`jacobian` diagnostics retain the coupled-system definition. An
implicit segregated operator also exposes `reduced_residual`,
`reduced_jacobian`, `reduced_dense_jacobian` and `reduced_direction`, accepting
`(cells_flat, old_full_flat, dt, time)`. Segregated `NewtonInfo.iterations`
counts outer physical Newton updates; nested boundary work is additional.
It is not a complete work counter for a performance comparison.

Explicit stages advance conservative storage and recover cell variables
generically, including the initial constrained state before the first rate is
evaluated. Boundary physics has one definition for all integrators. Negative
intermediate Euler velocities are permitted when the thermodynamic state is
admissible. The Euler steady final audit checks the requested forward branch;
the opt-in automatic outlet has a separate transient audit, allowing signed
intermediate evolution without applying a final steady direction constraint
at every output time.

`ResidualPolicy()` defaults to relative reduction $10^{-6}$ **or** absolute
norm $10^{-9}$, applied per equation, plus independent boundary and global
defect limits $10^{-9}$. Set `criterion` to `relative`, `absolute`, `either` or
`both`. Equations whose initial norm is at most `initial_floor` (default
$10^{-14}$) use an absolute test instead of dividing by a negligible initial
norm. Scales remain fixed during
each grid solve. Grid sequencing is numerical initialization; each grid receives
a new initial residual normalization. Final success also requires finite,
admissible states and any configured physical monitors/final audit.

`solve_grid_sequence(..., carry_cfl=True)` also transfers the last accepted
CFL from a successful preceding grid. It recomputes the timestep on the new
mesh and caps the inherited CFL at `settings.cfl_max`. Cold starts and cold
retries use `settings.cfl`; `carry_cfl=False` restores a reset on every grid.
An already-steady zero-update stage forwards its initial CFL. Missing or
invalid controller history falls back to `settings.cfl`. Stage records include
`cfl_policy`, `initial_cfl` and `last_accepted_cfl`, with failed warm-attempt
starting CFLs retained separately. The original convergence checks and fresh
relative residual baselines still apply independently on every grid.

`solve_transient` integrates to requested physical times rather than seeking a
steady residual. Its conservation audit uses the actual boundary flux and source
quadrature of the selected integrator, including SSPRK3/RK4 stage weights and
actual discrete NC work for nonconservative models.
Accepted history rows retain `integrated_transfer` and `conserved_totals`, in
each equation's physical units, for independent conservation bookkeeping.

## Using and extending the API

```python
import jax
jax.config.update("jax_enable_x64", True)

from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import Scheme, solve_transient

model = AdvectionModel(100, speed=-1.0)
snapshots = solve_transient(
    model, (0.1, 0.2),
    Scheme(flux="hll", reconstruction="muscl_van_leer",
           time_integrator="runge-kutta-3"),
    cfl=0.4,
)
assert all(snapshot.success for snapshot in snapshots)
```

For a new physical model, implement the callbacks on a new object, choose its
fixed scales and stencil declarations, and pass it to the same drivers. No
registration or numerical-core edit is required. The scalar models demonstrate
an EOS-free implementation. `tests/solvers/test_model_contract.py` also exercises an
independently programmed two-equation nonlinear-storage model and a wider source
stencil. The production Euler models show general-EOS storage/eigenstructure,
closure injection, physical geometry and characteristic endpoint laws.

Euler thermodynamics come from `jaxprop.Fluid.get_state()`. Both selectable
pressure outlets in
[`models/euler_outlet.py`](../src/nozzle_hyperbolic/models/euler_outlet.py) support
a general EOS. The finite-wave law computes PH Hugoniot shocks and numerical
expansion curves in
[`models/euler_wave.py`](../src/nozzle_hyperbolic/models/euler_wave.py), retaining
perfect-gas closed forms for that backend. Its classical-wave and bicubic
consistency limits are described in the
[backpressure theory](../development/BC_backpressure/THEORY.md). Sod
uses transmissive traces at both ends, with no pressure/reservoir condition.
Exact waves should remain away from the endpoints during reference comparisons;
finite-grid diffusive tails can still affect numerical boundary transfer. Use
the actual integrator-consistent numerical flux budget for conservation checks.

`outlet_regime="auto"` opts a nozzle or annular model into pressure-controlled,
sonic and fully outgoing traces during a transient. Acoustic auto treatment
uses EOS-based entropy/compatibility and a pressure/sonic complementarity
constraint. A 100:1 initial pressure release is a large finite wave; the
new [transient study](../development/roberto/transient_problems/README.md)
uses the permitted perfect-gas finite-wave outlet for that case. The physical
outlet law is held fixed while comparing numerical boundary treatments.

The integrated [two-fluid model](two_fluid_migration.md) uses this constrained
contract and its explicitly identified compatibility discretization. Interface
compliance alone does not establish physical well-posedness or hyperbolicity.
