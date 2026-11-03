# Physical-parameter continuation and portable restart

`solvers.fvm.continuation.solve_parameter_sequence` accepts a model factory and
an ordered list of explicit physical parameter mappings. The model factory owns
their names, units and meaning. One numerical `Scheme` is held fixed throughout;
this API does not provide a flux/reconstruction method sequence.

```python
from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings
from nozzle_hyperbolic.solvers.fvm.continuation import solve_parameter_sequence

final, stages = solve_parameter_sequence(
    lambda parameters: AdvectionModel(40, speed=parameters["speed"]),
    [{"speed": 0.5}, {"speed": 0.75}, {"speed": 1.0}],
    Scheme(flux="hll"),
    SteadySettings(),
)
```

Only a successful immediately preceding solution may initialize the next stage.
The helper interpolates the old model's transported **densities** onto the new
centers, multiplies by the new cell volumes, and solves generic storage recovery
including every cell algebraic constraint. It then solves the model's physical
trace equations. Interpolated primitive variables provide only a Newton seed.
This supports changed physical parameters and nonuniform meshes without
requiring phase identities in the numerical engine.

The state/content meanings and domain endpoints must agree across stages.
Dimensions, transported equation names and endpoints are checked; the caller
must ensure that identical names retain identical physical meanings and units.
Bounded center interpolation is a steady initialization policy. It is not a
conservative remap for physical transients: old and target domain inventories
are recorded, including any difference produced by a new mesh or geometry.

`prepare_continuation_state(model, previous, scheme)` returns a prepared state
and a preparation record. On a numerical preparation failure the state is
`None`, and the failed candidate and reason remain in the record. It freshly
checks storage recovery, normalized cell constraints and boundary defects with
the configured fixed Newton residual tolerance. A small-update status alone
does not pass preparation. Exceptions during recovery or trace completion are
recorded before an independent cold attempt. Configuration validation, model
construction and operator setup are outside this recovery block: their errors
propagate, including incompatible model dimensions or equation names.

Every numerically attempted stage after successful setup records its physical
parameter mapping, model setup
cost, preparation cost and its recovery/boundary breakdown, all solve attempts,
states, metrics, accepted/rejected histories and total cost. Every solve has
fresh initial residual norms. By default, a failed warm solve is retained and
followed by an independent cold attempt; set `retry_cold=False` to disable that
retry. Failed stages remain visible and subsequent stages start cold. The
returned solution is the final requested stage when the sequence completes,
even when an earlier stage converged and the requested endpoint did not.

Portable restart helpers store arrays and JSON without pickled models, fluids
or executable objects:

```python
from nozzle_hyperbolic.solvers.fvm.continuation import (
    save_restart, load_restart, prepare_continuation_state,
)

save_restart("run_restart.npz", final, metadata={"case": "documented case ID"})
restart = load_restart("run_restart.npz")
state, preparation = prepare_continuation_state(target_model, restart, scheme)
```

The target model and selected property backend are constructed explicitly by
the caller. A restart contains full state, transported densities, coordinates,
volumes, equation names, success status, physical time and supplied provenance.
Files are created exclusively to avoid silently overwriting a previous result.
Loading validates the schema and finite dimensions; successful loading does
not establish compatibility with a new EOS. Preparation rechecks the target
model's domain, cell constraints and boundary laws. Unsuccessful saved solutions
are preserved as evidence but are not used as automatic continuation seeds.

For an explicitly requested diagnostic continuation of an unfinished finite
state, `prepare_continuation_state(..., allow_unsuccessful=True)` permits
recovery while recording `source_success=False` and the opt-in flag. It does
not relabel the source converged or weaken preparation audits. Automatic
parameter/grid sequences retain the successful-source-only policy.

These APIs do not replace the physical model's initialization choices or the
generic solver's convergence policy. Final physical budgets, requested branch
and reference agreement still need their own checks.

## Grid-sequence CFL policy

`solve_grid_sequence(..., scheme_factory=lambda count: selected_scheme)` can
choose reconstruction explicitly by grid, such as constant on the coarsest mesh
and MUSCL on the finer meshes. Its defaults retain the previous fallback behavior.
Use `retry_cold=False, stop_on_failure=True` for strict continuation: failed
preparation or a failed solve stops the hierarchy without an independent retry.
The returned stages describe attempted grids; clients must label unattempted
requested grids as blocked rather than assigning them the last successful state.

`SteadyWorkspace(model, scheme, residual_policy)` binds an operator and residual
sampler for repeated complete solves. Pass it as `workspace=` to
`solve_single_grid`; each call resets the field, CFL controller, histories and
initial residual norms. The workspace validates model identity, scheme and
residual policy. `workspace_factory(model, scheme, settings)` supplies the same
reuse to `solve_grid_sequence`. Constraint-consistent preparation can reuse
`workspace.operator` through `prepare_continuation_state(..., operator=...)`.
Reuse retains compiled kernels, not numerical state; callers still synchronize
device work and check compilation events when claiming compilation-free timings.

For reuse across separate Python processes, the [standalone demos](../../demos/README.md#reusing-jax-executables)
enable JAX's persistent executable cache before constructing arrays. The opt-in
`solvers.compilation.CompilationReport` observes compile/load events,
disk-cache hits and elapsed durations. The demos print one final status line
by default. The observer does not infer reuse from cache files.
Disk caching complements workspace reuse and does not eliminate tracing or all
compilation for changed configurations or ineligible kernels.

`solve_grid_sequence` defaults to `carry_cfl=True`: after a successful state
transfer it inherits the preceding solve's last accepted CFL, capped at
`settings.cfl_max`, and computes a new timestep on the target mesh. It does not
reuse the old absolute timestep or the next proposed CFL. A zero-update
successful stage forwards its initial CFL; absent/invalid controller history
falls back to `settings.cfl`. Cold initialization and retries always reset.
Pass `carry_cfl=False` for the former per-grid reset behavior. This also applies
when extending a hierarchy through `initial_solution`.

Stage metadata retains the policy, starting CFL and last accepted CFL; failed
warm attempts retain their own starting CFL. Physical-parameter continuation
retains its separate existing CFL policy. The small
[HLL shock study](../../development/historical/cfl_carryover/RESULTS.md) motivated this
grid-sequence default; it does not establish a universal speedup.
