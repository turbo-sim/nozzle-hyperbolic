# Development workspace

For experiment layout, verification, timing and report packaging, follow the
[numerical study conventions](../docs/notes/numerical_studies.md).

All development work is grouped under this single directory:

- [flux_limiters/](flux_limiters/README.md): reproducible perfect-gas shock comparison of
  Van Leer, Minmod, Superbee and Van Albada, with analytical errors and retained convergence histories.
- [alberto/](alberto/README.md): legacy research drivers, settings, sensitivity
  studies and shared helpers. Its [execution audit](alberto/RUN_REPORT.md)
  records the previous run results and their limitations.
- [flux_methods/](flux_methods/README.md): production Stadtke corrections,
  Osher path-quadrature variants and the general-EOS Roe study, with theory,
  controlled verification and offline Plotly reports.
- [real_gas/](real_gas/README.md): bicubic CO2 nozzle analytical references and
  finite-volume verification for supercritical and two-phase flows.
- [BC_backpressure/](BC_backpressure/README.md): acoustic and general-EOS
  finite-wave pressure outlets, stagnant/adapted initialization and shocked
  nozzle, annular and CO2 target cases.
- [roberto/](roberto/README.md): current Euler verification examples, numerical
  fixes, Newton/PTC, implicit-integration and boundary-condition studies.
- [benchmark_production/](benchmark_production/README.md): production flux and
  grid-sequencing comparisons on 400 cells, with 50/100/200-cell coarse grids
  and separate fixed-work component timings.


The implemented shared solver's residual definitions, convergence policy and
monitoring API are documented in
[residual monitoring](../docs/notes/residual_monitoring.md).

Roberto's [study index](roberto/README.md) lists the retained sensitivity and
transient studies. The retired nozzle and annular example folders and their
saved evidence have been removed; their physical cases live in the installed
verification catalogue.

New controlled studies belong under `roberto/<study>/`, with human-facing
notes at the study root, code in `scripts/` and generated data in `outputs/`.
Run from the repository root in the existing project environment, for example:

```powershell
poetry run python -B -m development.roberto.sensitivity_backpressure.scripts.run_sweep --help
poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.compare_settings --help
```

Study READMEs describe current commands and distinguish recorded campaigns
from new solver executions. Production source is unchanged by this cleanup.
