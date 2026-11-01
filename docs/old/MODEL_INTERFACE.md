# Physical model interface

The production entry point is `nozzle_hyperbolic.solvers.fvm`. A physical model
is a Python object satisfying the structural protocols in
[`contracts.py`](../src/nozzle_hyperbolic/solvers/fvm/contracts.py); inheritance
is unnecessary. The solver has no Euler-variable, equation-count or EOS branch.
The supported problem is a **square conservative balance system**, with
cell-local storage, fixed one-dimensional geometry and two endpoint traces.
Periodic boundaries and nonconservative products need a future contract extension.
Interior flux/source callbacks are autonomous functions of state and fixed model
data; boundary equations additionally receive physical time. Explicitly
time-forced interior laws would require extending the callback contract.

The physical model owns its state variables, storage, physical flux,
eigenstructure, constitutive laws, admissible domain, boundary equations and
physical dependency stencils. The numerical engine owns reconstruction,
numerical flux contraction, conservative assembly, differentiation, Newton and
linear solves, time integration, timestep control and convergence assessment.
Geometry preprocessing constructs fixed mesh measures. Case files and reference
solutions belong outside these layers.

## Array and conservation conventions

With $N$ physical cells and $V$ variables, the full state has shape $(N+2,V)$.
Rows zero and $N+1$ are independent algebraic endpoint traces, located at the
physical end faces. Interior rows are cell values. Face arrays have shape
$(N+1,V)$. The model's iteration variables are also its reconstruction variables;
they need not be conservative variables.

`Mesh` stores increasing faces, interior centers, positive cell volumes and
positive face areas. Optional geometric data is opaque to the numerical engine.
There are no physical volumes assigned to endpoint traces.

For physical-cell storage density $Q(W)$, unweighted transported face flux $F$
and already integrated cell source $S_i$, the common residual is

$$
R_i=A_{i+1/2}F_{i+1/2}-A_{i-1/2}F_{i-1/2}-S_i,
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
| `mesh`, `nvar`, `equation_names` | `Mesh`, positive integer $V$, $V$ descriptive equation names |
| `variable_scales` | Positive fixed $(V,)$ scales for reconstruction and state corrections |
| `conservative_scales` | Positive fixed $(V,)$ scales for storage densities and spatial norms |
| `global_flux_scales` | Positive fixed $(V,)$ scales for integrated balance defects |
| `reference_time` | Positive fixed physical time scale |
| `source_dependency_radius` | Maximum full-state block distance used by a source row |
| `boundary_dependency_radius` | Maximum full-state block distance used by a boundary row |
| `boundary_coupling` | `local`, or `global` for nonlocal endpoint coupling |
| `initial_state()` | Full $(N+2,V)$ initial state |
| `cell_conserved(physical)` | Cell-local $(N,V)\to(N,V)$ transported storage densities |
| `face_conserved(states)` | $(N+1,V)$ conserved densities in the model's face frame |
| `face_physical_flux(states)` | $(N+1,V)$ physical fluxes in the same face frame |
| `face_speed_bounds(states)` | Two $(N+1,)$ vectors of signed lower/upper characteristic speeds; needed for HLL/Rusanov |
| `flux_to_conserved_frame(flux)` | Convert $(N+1,V)$ numerical face-frame flux into the transported frame |
| `integrated_source(full)` | $(N,V)$ integrated physical sources |
| `boundary_residual(full,time)` | $(2,V)$ dimensionless endpoint equations |
| `admissible(full,left,right)` | Scalar boolean validating full states and reconstructed face states |
| `wave_speed(physical)` | $(N,)$ nonnegative maximum absolute signal speeds for timestep selection |

The nonlinear storage map must be locally invertible in the admissible domain.
Explicit integration advances $V_iQ_i$, then recovers the iteration variables by
a generic nonlinear solve. This is why `cell_conserved` must be cell-local.
Face property/flux callbacks and frame conversion act independently at each
face; characteristic data may combine that face's left and right states. They
must not introduce undeclared cross-face coupling. Source and endpoint callbacks
may use neighbors within their declared reach. Endpoint flux overrides preserve
internal faces and include their state dependencies in the boundary declaration.

For expensive properties, replace the three independent face callbacks with
`face_data(states)`. Return a dictionary with `conserved`, `physical_flux`,
`speed_bounds=(lower,upper)` and `auxiliary`. Auxiliary data is an opaque JAX
pytree; the numerical engine never interprets its physical contents.

Selecting `flux="characteristic"` additionally requires
`face_characteristics(left,right,delta_factor)`, returning projectors of shape
$(N+1,K,V,V)$, eigenvalues $(N+1,K)$ and positive regularization widths
$(N+1,K)$. Here $K$ is the number of characteristic families. With fused face
data, `face_characteristics_from_data(left,right,delta_factor,aux_left,aux_right)`
can reuse property evaluations. HLL and Rusanov require no eigenvectors.

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
in [PHYSICAL_CLOSURES.md](PHYSICAL_CLOSURES.md).

The optional filter adds an internal conservative face correction in transported
densities. Its strength is numerical; velocity gates or other physical sensors
belong to the model. Do not put primitive-variable smoothing in the integrator.

## Integration and stopping

`Scheme.time_integrator` selects `euler-implicit`, `euler-explicit` or
`runge-kutta-3`. `Scheme.boundary_treatment` selects how the same algebraic
boundary equations are solved:

| Setting | Backward Euler | Explicit Euler / SSPRK3 |
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
generically. Boundary physics has one definition for all integrators. Negative
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

`solve_transient` integrates to requested physical times rather than seeking a
steady residual. Its conservation audit uses the actual boundary flux and source
quadrature of the selected integrator, including SSPRK3 stage weights.
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
an EOS-free implementation. `tests/test_generic_fvm.py` also exercises an
independently programmed two-equation nonlinear-storage model and a wider source
stencil. The production Euler models show general-EOS storage/eigenstructure,
closure injection, physical geometry and characteristic endpoint laws.

Euler thermodynamics come from `jaxprop.Fluid.get_state()`. The sole permitted
perfect-gas exception is the selectable finite-wave pressure outlet in
[`models/euler_outlet.py`](../src/nozzle_hyperbolic/models/euler_outlet.py), which
is explicitly restricted to a perfect-gas backend and carries a TODO for general
EOS extension. The default acoustic outlet uses general-EOS properties. Sod
uses transmissive traces at both ends, with no pressure/reservoir condition;
benchmark comparisons stop before waves reach those ends.

`outlet_regime="auto"` opts a nozzle or annular model into pressure-controlled,
sonic and fully outgoing traces during a transient. Acoustic auto treatment
uses EOS-based entropy/compatibility and a pressure/sonic complementarity
constraint. A 100:1 initial pressure release is a large finite wave; the
new [transient study](../development/roberto/transient_problems/boundary_treatment/README.md)
uses the permitted perfect-gas finite-wave outlet for that case. The physical
outlet law is held fixed while comparing numerical boundary treatments.

This interface leaves room for a future two-fluid model but does not establish
that model's well-posedness, hyperbolicity or nonconservative discretization.
