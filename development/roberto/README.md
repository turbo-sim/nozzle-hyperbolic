# Roberto: development verification and numerical studies

The readable EulerNozzle verification examples live in
[`euler_nozzle/`](euler_nozzle/README.md). Its note explains the physical
conditions, references, numerical settings, and acceptance checks for the
steady nozzle and unsteady Sod shock tube.

Run from the repository root after `poetry install`:

```powershell
poetry run python -m development.roberto.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development.roberto.euler_nozzle.verify_euler_nozzle_sod --no-show
```

The original research drivers are grouped under
[`development/alberto/`](../alberto/README.md). The focused
[Newton/PTC/implicit comparison](newton_solve/README.md) lives in
`newton_solve/`, alongside an overview of the earlier attempts and a
[physical and mathematical convergence explanation](newton_solve/CONVERGENCE_MECHANISM.md).

The [numerical-fix comparison](numerical_fixes/RUNNING.md) runs the shared steady
example with legacy/corrected fluxes and optional smooth reconstruction.
Its [explanatory note](numerical_fixes/README.md) separates AD/regularity repairs
from changes to the finite-volume equations.

The [implicit-integration sensitivity studies](implicit_integration/README.md)
compare inner Newton caps and CFL controllers across the same five nozzle cases.
The standalone [HTML report](implicit_integration/REPORT.html) contains internal
pages for all three studies plus theory, with convergence, pressure, actual CFL,
inner-work and recorded-time plots rebuilt from the 105 retained solves.

The [boundary-condition study](BC_residual/README.md) compares internal
ghost treatment with coupled algebraic boundary equations using matched implicit
Newton and banded solvers. Its reports retain repeated timing samples and the
same five-case residual and physical audits.

The [backpressure analysis](backpressure_sensitivity/README.md) follows a descending
operating-pressure sweep from rest through choking, internal shocks and
supersonic discharge, including overexpanded and underexpanded operation.
