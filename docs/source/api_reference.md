# API reference

These pages are regenerated from the package's Python docstrings by the shared
builder. Install the repository editably before building, so imports describe
the checked-out source.

| Task | Interface |
|---|---|
| Euler nozzle/duct | {class}`~nozzle_hyperbolic.models.euler_nozzle.model.NozzleModel` |
| Annular Euler with swirl | {class}`~nozzle_hyperbolic.models.euler_annular.model.AnnularModel` |
| Constrained two-fluid flow | {class}`~nozzle_hyperbolic.models.two_fluid.model.TwoFluidModel` |
| Fixed mesh | {class}`~nozzle_hyperbolic.numerics.mesh.Mesh` |
| Numerical method | {class}`~nozzle_hyperbolic.solvers.fvm.config.Scheme` |
| Steady control | {class}`~nozzle_hyperbolic.solvers.fvm.config.SteadySettings` |
| Convergence policy | {class}`~nozzle_hyperbolic.solvers.residuals.ResidualPolicy` |
| Bind spatial/time kernels | {class}`~nozzle_hyperbolic.solvers.fvm.operator.Operator` |
| Single-grid steady solve | {func}`~nozzle_hyperbolic.solvers.fvm.drivers.solve_single_grid` |
| Coarse-to-fine solve | {func}`~nozzle_hyperbolic.solvers.fvm.drivers.solve_grid_sequence` |
| Physical-time solve | {func}`~nozzle_hyperbolic.solvers.fvm.drivers.solve_transient` |
| Physical continuation | {func}`~nozzle_hyperbolic.solvers.fvm.continuation.solve_parameter_sequence` |
| Portable restart | {func}`~nozzle_hyperbolic.solvers.fvm.continuation.save_restart`, {func}`~nozzle_hyperbolic.solvers.fvm.continuation.load_restart` |

The root package lazily exports the model classes and common mesh/settings/solve
interfaces, so `from nozzle_hyperbolic import NozzleModel, Scheme` is supported.
More specialized geometry, closure, verification and compilation helpers are
imported from their owning modules. The [solver chapters](numerics/index.md)
explain ordering, dimensions and physical/numerical responsibilities.

The generated tree covers the active package modules, including the separate
two-fluid thermodynamic, boundary, source, stabilization and closure modules.

```{toctree}
:maxdepth: 2

api/nozzle_hyperbolic
```
