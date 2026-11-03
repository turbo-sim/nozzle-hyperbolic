# Roberto: verification and numerical studies

Production physical definitions and independent references are packaged in
[`nozzle_hyperbolic.verification`](../../src/nozzle_hyperbolic/verification/README.md).
The current [production benchmark](../benchmark_production/README.md) and
[flux studies](../flux_methods/README.md) select their numerical profiles.

| Study | Purpose |
|---|---|
| [sensitivity_backpressure/](sensitivity_backpressure/README.md) | Current shared-FVM pressure continuation, independent starts, field and balance audits |
| [sensitivity_CFL_settings/](sensitivity_CFL_settings/README.md) | Current shared-FVM Newton caps and accepted-step CFL controller comparisons |
| [cfl_carryover/](cfl_carryover/README.md) | Reset versus carried CFL across nested shock-case meshes |
| [local_timestep/](local_timestep/README.md) | Experimental local stepping; limited and inconsistent measured gains |
| [transient_problems/](transient_problems/README.md) | Sod and nozzle pressure release, stage boundaries and conservative budgets |
| [../benchmark_production/](../benchmark_production/README.md) | Production numerical methods and grid sequencing with separate compilation/timing evidence |
| [../newton_solve/](../newton_solve/README.md) | Current Newton/pseudo-time investigations |

The obsolete `euler_nozzle/` and `euler_annular/` studies have been removed,
including their saved outputs and reports. Use the installed nozzle/annular
case catalogue, current tests and
[annular NozzleX comparison](../../cases/verification_annular/README.md).

Run modules from the repository root with the existing Poetry environment.
Keep numerical settings and measured findings with their owning study.
The separate frozen source under [legacy/](../../legacy/README.md) is outside
this cleanup and remains subject to its manifest checks.
