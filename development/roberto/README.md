# Roberto: development verification and numerical studies

The final [Roe formulation and verification](../roe_method/README.md) lives
directly under `development/roe_method/`; its benchmark clients reuse the
shared tooling in `benchmarking/`.

Earlier studies and compatibility reference imports are now grouped in
[`historical/`](historical/README.md). The archive retains their saved
evidence and original numerical methods; some entry points use retired APIs.

The [shared hyperbolic infrastructure assessment and implementation plan](shared_hyperbolic/IMPLEMENTATION_PLAN.md)
reviews an incremental common numerical engine for EulerNozzle, EulerAnnular,
and future scalar models. It preserves model-specific physics and boundary
laws, with coupled boundary residuals, a common mesh/MUSCL implementation and
annular's safeguarded Newton and convergence policies as the final Euler
design. Two-fluid migration is deferred.

The readable EulerNozzle verification examples live in
[`historical/euler_nozzle/`](historical/euler_nozzle/README.md). Its note explains the physical
conditions, references, numerical settings, and acceptance checks for the
steady nozzle and unsteady Sod shock tube.
Its new standalone [steady baseline](historical/euler_nozzle/STEADY_REPORT.html) covers
eight backpressure regimes with constant/MUSCL reconstruction and four meshes;
the separate [Sod baseline](historical/euler_nozzle/SOD_REPORT.html) covers three physical
times and four meshes. Both retain the current source kernels, saved numerical
evidence and independent audits before the shared-engine migration.

Run from the repository root after `poetry install`:

```powershell
poetry run python -m development.roberto.historical.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development.roberto.historical.euler_nozzle.verify_euler_nozzle_sod --no-show
```

The original research drivers are grouped under
[`development/alberto/`](../alberto/README.md). The focused
[Newton/PTC/implicit comparison](historical/newton_solve/README.md) lives in
`historical/newton_solve/`. Its standalone [HTML report](historical/newton_solve/REPORT.html) combines
the cold-start comparison, PTC performance evidence, retained custom adapted
starts, mathematical explanations and useful earlier notes. The discarded
optimizer experiments and their results have been removed.

The [numerical-fix comparison](numerical_fixes/RUNNING.md) runs the shared steady
example with legacy/corrected fluxes and optional smooth reconstruction.
Its [explanatory note](numerical_fixes/README.md) separates AD/regularity repairs
from changes to the finite-volume equations.

The [implicit-integration sensitivity studies](historical/implicit_integration/README.md)
compare inner Newton caps and CFL controllers across the same five nozzle cases.
The standalone [HTML report](historical/implicit_integration/REPORT.html) contains internal
pages for all three studies plus theory, with convergence, pressure, actual CFL,
inner-work and recorded-time plots rebuilt from the 105 retained solves.

The [boundary-condition study](historical/BC_residual/README.md) compares internal
ghost treatment with coupled algebraic boundary equations using matched implicit
Newton and banded solvers. Its reports retain repeated timing samples and the
same five-case residual and physical audits.

The [backpressure analysis](historical/backpressure_sensitivity/README.md) follows a descending
operating-pressure sweep from rest through choking, internal shocks and
supersonic discharge, including overexpanded and underexpanded operation.

The [EulerAnnular prototype](historical/euler_annular/README.md) implements the general-EOS
curvilinear annular FVM with swirl, NURBSpy spline geometry and simultaneous
algebraic boundary blocks. The study contains the accepted implementation plan
and questionnaire, analytical/constant/MUSCL verification drivers, and reports
that distinguish converged discrete roots from analytical accuracy limits.

The annular solver retains angular-momentum storage and now includes named
wall-shear and heat-flux placeholders. Its updated report uses a 50/100/200/400
grid sequence with bounded primitive interpolation to initialize each finer
mesh, and separates original-equation convergence from analytical accuracy.

The [multigrid study](historical/multigrid/README.md) benchmarks EulerNozzle implicit
single-grid solves against coarse-to-fine initialization, with repeated timings
and one shareable standalone [HTML report](historical/multigrid/REPORT.html) at the study
root. Numerical artifacts remain under `outputs/`. Its separate annular design note
assesses future nonlinear FAS cycles; FAS is not implemented.
