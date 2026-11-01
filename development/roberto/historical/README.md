# Historical Roberto studies

These directories retain earlier verification examples and numerical investigations.
The supported numerical engine lives in [`src/nozzle_hyperbolic/solvers/fvm/`](../../../src/nozzle_hyperbolic/solvers/fvm/).
Canonical analytical references live beside their physical models in
[`src/nozzle_hyperbolic/models/`](../../../src/nozzle_hyperbolic/models/).

| Directory | Retained material |
|---|---|
| [`BC_residual/`](BC_residual/README.md) | Coupled versus internal boundary equations and matched timing studies |
| [`implicit_integration/`](implicit_integration/README.md) | Newton caps, CFL policies, and implicit integration sensitivity |
| [`multigrid/`](multigrid/README.md) | Single-grid versus coarse-to-fine initialization benchmarks and FAS design notes |
| [`backpressure_sensitivity/`](backpressure_sensitivity/README.md) | Nozzle operating-pressure sweep, choking, shocks, and outflow comparisons |
| [`euler_annular/`](euler_annular/README.md) | Annular Euler geometry, steady/transient verification, source checks, and accuracy studies |
| [`euler_nozzle/`](euler_nozzle/README.md) | Steady nozzle and Sod reference comparisons and baseline reports |
| [`newton_solve/`](newton_solve/README.md) | Newton, pseudo-transient continuation, initialization, and performance investigations |

The current acoustic and general-EOS finite-wave pressure-outlet comparison
lives in [BC_backpressure](../../BC_backpressure/README.md), covering shocked
nozzle, annular and CO2 flows from stagnant and numerical adapted starts.

Module entry points now begin with `development.roberto.historical.<study>`.
The relocation updates imports, repository roots, and shared report-asset paths;
it does not change numerical algorithms or migrate retired solver APIs. Some
archived numerical drivers require their historical source implementation or a
separate API migration before they can run against current production code.

Saved numerical outputs, recorded commands, source fingerprints, and already
generated HTML reports retain their measurement-time contents. Local ignored
outputs move with each study and remain ignored; this relocation does not add
them to Git. Regenerated reports use the updated source entry points.
