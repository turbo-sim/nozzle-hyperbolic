# Grid sequencing, continuation and reuse

## Coarse-to-fine grid sequencing

`solve_grid_sequence` solves the same physical problem on progressively finer
meshes. A cheap coarse solution can supply the overall flow pattern before the
fine grid resolves its details. Each grid still solves its own original
equations to its own convergence targets.

```python
from nozzle_hyperbolic import NozzleModel, Scheme, solve_grid_sequence

def model_factory(cells):
    return NozzleModel(
        geometry.build_mesh(cells=cells), fluid,
        p0=2e5, T0=300.0, back_pressure=1.9e5,
    )

final, stages = solve_grid_sequence(
    model_factory,
    cells=(24, 48, 96),
    scheme=Scheme(flux="hll", reconstruction="muscl_van_leer"),
)
```

This fragment reuses the `geometry` and `fluid` construction from
[getting started](../getting_started/index.md). `final` is the last attempted
solution and `stages` retains stage records.

Only a **successful** preceding numerical state is transferred. For ordinary
unconstrained models the driver uses bounded componentwise linear interpolation
of the declared iteration variables. For constrained models it interpolates
transported densities, recovers a state satisfying cell constraints and completes
the traces. It records preparation, source/target inventories and failure.
The transfer is a steady initialization policy, not an exact conservative
finite-volume remap for physical transients.

By default `carry_cfl=True` carries the previous successful last accepted CFL
to a warm fine-grid start, capped by `settings.cfl_max`. The wave timestep is
recomputed using the new grid and state. A zero-update success forwards its
initial CFL; absent usable history, the initial configured CFL is used.
`carry_cfl=False` resets each stage. Cold starts and cold retries always use
`settings.cfl`.

The default policy retains a failed warm attempt, can retry cold and permits
a later grid to start cold after a failed stage. For a strict hierarchy use
`retry_cold=False, stop_on_failure=True`. Failed coarse states are never silently
prolonged as successful solutions. `scheme_factory(count)` can explicitly select
a scheme per mesh, such as constant reconstruction on the coarsest grid.

This is nested-grid initialization. It has no residual restriction, coarse-grid
defect equation or V-cycle, so it is not FAS multigrid and does not by itself
establish grid independence.

## Physical parameter continuation

`solve_parameter_sequence(model_factory, parameter_stages, ...)` follows an
explicit list of physical parameter mappings. The factory determines their
meaning, for example backpressure or interphase-source strength. One fixed
`Scheme` applies throughout. Every attempted stage establishes fresh residual
baselines and records preparation, warm failures and optional cold retries.

An earlier converged state never replaces an unsuccessful requested endpoint.
A source-strength ladder also does not validate the final full-source problem
unless that final stage actually converges. This API is physical parameter
continuation, not an automatically selected sequence of flux methods.

## Portable restarts

`save_restart` stores arrays and JSON metadata without pickled EOS/model objects.
`load_restart` validates that portable snapshot; use
`prepare_continuation_state` to interpolate, recover constraints and complete
traces for the new model before solving. Variable meanings, equation names/counts
and domain endpoints must match.

An explicit diagnostic `allow_unsuccessful=True` can prepare an unfinished finite
source state, retaining its unsuccessful provenance. Automatic sequences do not
enable that option. Loading a file does not make its source converged or its
inventories conservative on a new mesh.

## Compilation and workspace reuse

`Operator` retains compiled kernels for its fixed model and scheme. A
`SteadyWorkspace` can reuse the operator and residual sampler for repeated
solves with the exact same model object, scheme and residual policy.
`workspace_factory` can provide this reuse during repeated hierarchy replays.
Mutating model configuration behind a bound operator is unsupported.

`configure_compilation_cache` configures a persistent JAX executable cache
before arrays/operators are constructed. This cache is separate from JAXProp
property-table caches. First-run compilation, cache loading and warm executable
time are different costs; `CompilationReport` helps record their provenance.
Property-table preparation should likewise be reported separately when measuring
numerical solve performance.

Local face/storage derivatives are shared within each linearization, as
described in [Jacobians](jacobians.md). The production solver does not lag the
Jacobian across Newton steps, use normal equations as its generic correction,
or provide a chunked compiled steady-loop backend.

Relevant APIs: {mod}`nozzle_hyperbolic.solvers.fvm.continuation`,
{mod}`nozzle_hyperbolic.solvers.compilation` and
{func}`~nozzle_hyperbolic.solvers.fvm.drivers.solve_grid_sequence`.
