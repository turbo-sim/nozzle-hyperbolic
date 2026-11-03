# Test suite

Run commands from the repository root using the existing Poetry environment.
Tests exercise the installable `src/nozzle_hyperbolic` package. They do not import
demos, experiments or development benchmarks.

```sh
# Maintained suite, including slow physical regressions. Optional tests may skip.
poetry run python -B -m pytest

# Fast feedback while refactoring.
poetry run python -B -m pytest -m "not slow and not optional"

# Physical accuracy and numerical compatibility.
poetry run python -B -m pytest tests/regression

# Complete maintained suite with Python branch coverage.
poetry run python -B -m pytest -m "not optional" --cov --cov-report=term-missing --cov-report=html

# Check the optional adapters, failing if NozzleX is missing.
# Set NOZZLE_REQUIRE_NOZZLEX=1 in the shell first.
poetry run python -B -m pytest -m optional

poetry run black --check tests
```

The default runs all maintained tests. Use `-m` to select cost or dependency
categories. Unknown markers, configuration errors and unexpected passes of
expected-failure tests fail the run.

## Organization

| Location | Responsibility |
|---|---|
| `geometry/` | Mesh measures, geometry validation and independent quadrature |
| `numerics/` | Limiters, characteristic flux algebra and derivative utilities |
| `thermodynamics/` | EOS adapter contracts and derivative consistency |
| `models/` | Physical laws, storage, sources, boundaries, closures and model interfaces |
| `solvers/` | Assembly, Jacobians, constraints, stage completion, drivers and reporting |
| `verification/` | Analytical-reference assumptions, identities, projections and packaged cases |
| `regression/` | Complete solves against analytical solutions and pinned numerical states |
| `_support/` | Small shared problem definitions and acceptance helpers; no numerical solver copies |

`conftest.py` enables JAX float64 before test-module collection. Per-file JAX
configuration is unnecessary. Tests that must disable precision do so in a fresh
subprocess, leaving the suite configuration intact. `slow` identifies expensive
compilation/integration checks; `optional` identifies NozzleX dependencies.
The steady regression fixture releases compilation caches between independent
physical cases, keeping both accuracy and snapshot checks on the same solved
hierarchy. Compilation-reuse tests run separately without this cleanup fixture.

Keep each test focused on one behavior with explicit setup, action and acceptance
checks. Prefer named physical values and descriptive parametrization IDs to dense
loops or unexplained tolerances. Tests for a reported numerical bug should fail
on the faulty behavior and explain the physical or numerical invariant involved.

## Physical regressions

| Model/problem | Independent acceptance checks |
|---|---|
| Euler nozzle: subsonic, internal shock, adapted expansion | Cold 32/64/128-cell hierarchy; original residuals; branch; projected conservative fields; shock position; smooth-grid improvement |
| Euler annular: conical subsonic, shock, supersonic expansion | Same complete-solve audits, including angular-momentum transport |
| Advection in both directions | Conserved cell averages, spatial refinement and integrated transfer budget |
| Burgers shock and rarefaction | Entropy solution, shock location or monotonic fan, global budget |
| Sod shock tube | Exact conserved cell averages before any wave reaches a boundary |
| Annular translating contact | Exact transport, constant pressure and conserved inventory |
| Annular homogeneous shear/heating | Exact momentum/energy trajectory, forced time-dependent exterior, complete production source/solver path |
| Two-fluid | Exact uniform rest/co-moving limits; nonuniform coupled advance; EOS constraints; integrated budgets; pinned full profile |
| Explicit RK3 and RK4 | Observed temporal order with nonlinear storage, recovered algebraic constraints and time-dependent boundary forcing |

Analytical profiles are integrated into physical cells, splitting quadrature at
known discontinuities. Comparisons use conserved quantities and physical-volume
weights; boundary traces are audited separately. Steady smooth errors must be
below $10^{-3}$ after normalization by fixed model scales, and improve when the
grid is refined. Captured shocks allow $10^{-2}$ integrated error and at most two
cell widths of position error. These are small fixed regression problems, not
the larger verification campaigns documented under `development/`.

The complete steady solves use HLL/MUSCL with backward Euler. Transient analytical
checks use HLL/MUSCL and RK3; the constrained two-fluid checks use its compatibility
flux. All grid attempts are retained; a failed stage is not replaced by a hidden
analytical warm start or cold retry. Solver success alone is insufficient: fresh
original residuals, finite/admissible states, boundary equations, EOS constraints,
physical branch and conservation are checked independently where applicable.

The two-fluid perfect-gas phase fixtures verify implementation behavior. They do
not validate real-fluid phase-change accuracy. Tests of the deprecated closure
implementation, including its three expected metastability failures, have been
removed with the historical test group.
The production Euler projector test differentiates the active model flux and
storage; historical tests containing local algebra are not treated as proof of
the current implementation.

## Saved numerical versions

`regression/data/physical_v1/` stores seven complete solutions: six steady Euler
profiles and a nonuniform two-fluid advance. The manifest records case hashes,
grids, methods, tolerances, initialization, dependency versions, precision,
backend, Git revision and dirty-source status. `source.zip` and per-file hashes
identify the exact numerical implementation even when the working tree was dirty.
Physical-case comparisons additionally hash canonical parsed data so Git line-ending
conversion does not change their identity. Geometry arrays allow only roundoff-level
differences; computed solutions retain the explicit tolerances below.
Iteration counts and timings are intentionally excluded from compatibility
comparisons. Every stored array and its checksum is checked before use.

Normalized profile comparisons use absolute tolerance $5\times10^{-8}$ and
relative tolerance $5\times10^{-7}$, allowing small floating-point variation while
detecting material changes. These guard numerical compatibility alongside the
independent analytical tests. A historical result can contain a bug, so matching
it alone does not establish correctness.

Pytest never creates or updates expected data. To propose a new numerical version:

```sh
poetry run python -m tests.regression.generate_baselines --output review_data/physical_v2
```

The destination must not exist. The generator stops if source files change during
the run. Review accuracy, conservation, changed profiles and provenance before
intentionally replacing a committed baseline. Keep the previous version available
in Git. Do not regenerate data just to make a refactor pass. Sensitivity tests
also check rejection of material field changes, NaNs and incomplete profiles.

## Coverage and CI

`.github/workflows/tests.yml` runs on pull requests and manual launches, not on
ordinary pushes. It runs fast checks on Python 3.12/3.13, the full core suite
with coverage, formatting and a wheel smoke check. A manually selected optional
job installs a pinned NozzleX revision using
the package's locked numerical stack and refuses silent dependency skips.

Python coverage measures executed Python lines and branches, including tracing;
it does not demonstrate that every XLA runtime branch or numerical regime was
exercised. Parameterized sonic/rest/reverse-flow, JIT/AD and physical tests provide
that additional evidence. All active two-fluid modules and closure laws are
included in coverage after removing the deprecated parallel implementation and
its exclusions. Coverage reports expose remaining gaps rather than making a
100% claim.

The wheel smoke check installs a built wheel into a disposable environment with
no numerical dependencies. It verifies the lazy top-level import and byte identity
of all shipped Python modules and 20 physical-case resources, from outside the
checkout. Numerical model behavior is checked separately in the locked environment.

The frozen archive and original-suite backup were removed at the user's request.
The numerical baselines in `regression/data/` remain independent of that archive.
The [two-fluid guide](../docs/notes/two_fluid_migration.md) describes the current
implementation, retained numerical references and historical research callers.
