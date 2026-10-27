# Development verification examples

The readable EulerNozzle verification examples live in
[`euler_nozzle/`](euler_nozzle/README.md). Its note explains the physical
conditions, references, numerical settings, and acceptance checks for the
steady nozzle and unsteady Sod shock tube.

Run from the repository root after `poetry install`:

```powershell
poetry run python -m development_new.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development_new.euler_nozzle.verify_euler_nozzle_sod --no-show
```

The original research drivers remain in `development/`. The separate
[direct steady-solver experiments](newton_solve/README.md) live in
`newton_solve/`; their implementations and the pending numerical changes are
not part of the verification-example commit.

The [numerical-fix comparison](numerical_fixes/RUNNING.md) runs the shared steady
example with legacy/corrected fluxes and optional smooth reconstruction.
Its [explanatory note](numerical_fixes/README.md) separates AD/regularity repairs
from changes to the finite-volume equations.
