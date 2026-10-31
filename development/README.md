# Development workspace

All development work is grouped under this single directory:

- [alberto/](alberto/README.md): legacy research drivers, settings, sensitivity
  studies and shared helpers. Its [execution audit](alberto/RUN_REPORT.md)
  records the previous run results and their limitations.
- [roberto/](roberto/README.md): current Euler verification examples, numerical
  fixes, Newton/PTC, implicit-integration and boundary-condition studies.
- [residual_monitoring/](residual_monitoring/README.md): shared nozzle/annular
  residual definitions, convergence policy, compatibility discussion, and proposed
  implementation plan; documentation only.

Earlier Roberto studies are grouped under
[`roberto/historical/`](roberto/historical/README.md); their archived solver
APIs may differ from the current production interface.

Each collection retains its existing thematic organization. New controlled
studies belong under `roberto/<study>/`, with human-facing notes at the study
root, code in `scripts/` and generated files in `outputs/`.

Run modules from the repository root using the project environment:

```powershell
poetry run python -m development.roberto.historical.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development.roberto.BC_residual.scripts.compare_implicit --no-show
poetry run python -m development.alberto.euler_nozzle.shock_tube_fvs
```

These are updated commands, not programs rerun during the folder consolidation.
Saved JSON/CSV, source hashes, recorded commands and settings preserve their
measurement-time provenance; old paths inside those records are historical.
Current commands and paths are in the corresponding READMEs. Relocated figures
and raw run data are retained. Production `src/` and numerical settings were
not changed by this directory reorganization.
