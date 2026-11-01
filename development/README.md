# Development workspace

For experiment layout, verification, timing and report packaging, follow the
[numerical study conventions](../docs/numerical_studies.md).

All development work is grouped under this single directory:

- [alberto/](alberto/README.md): legacy research drivers, settings, sensitivity
  studies and shared helpers. Its [execution audit](alberto/RUN_REPORT.md)
  records the previous run results and their limitations.
- [flux_methods/](flux_methods/README.md): production Stadtke corrections,
  Osher path-quadrature variants and the general-EOS Roe study, with theory,
  controlled verification and offline Plotly reports.
- [BC_backpressure/](BC_backpressure/README.md): acoustic and general-EOS
  finite-wave pressure outlets, stagnant/adapted initialization and shocked
  nozzle, annular and CO2 target cases.
- [roberto/](roberto/README.md): current Euler verification examples, numerical
  fixes, Newton/PTC, implicit-integration and boundary-condition studies.

The implemented shared solver's residual definitions, convergence policy and
monitoring API are documented in
[residual monitoring](../docs/residual_monitoring.md).

Earlier Roberto studies are grouped under
[`roberto/historical/`](roberto/historical/README.md); their archived solver
APIs may differ from the current production interface.

Each collection retains its existing thematic organization. New controlled
studies belong under `roberto/<study>/`, with human-facing notes at the study
root, code in `scripts/` and generated files in `outputs/`.

Run modules from the repository root using the project environment:

```powershell
poetry run python -m development.roberto.historical.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development.roberto.historical.BC_residual.scripts.compare_implicit --no-show
poetry run python -m development.alberto.euler_nozzle.shock_tube_fvs
```

These are updated commands, not programs rerun during the folder consolidation.
Saved JSON/CSV, source hashes, recorded commands and settings preserve their
measurement-time provenance; old paths inside those records are historical.
Current commands and paths are in the corresponding READMEs. Relocated figures
and raw run data are retained. Production `src/` and numerical settings were
not changed by this directory reorganization.
