# Steady-solver convergence study: consolidated notes

The experiment notes were consolidated during cleanup of `development_new/newton_solve/`.
The superseded drivers, auxiliary initializations, and old generated outputs were removed.

- [Approaches tested, historical outcomes, PTC results, and current comparison commands](../development_new/newton_solve/README.md).
- [Mathematical and physical explanation of convergence and stagnation](../development_new/newton_solve/CONVERGENCE_MECHANISM.md).

The retained comparison uses only the original cold initialization and the custom
PySolverView Newton backend, physical-storage PTC, and implicit time integration.
