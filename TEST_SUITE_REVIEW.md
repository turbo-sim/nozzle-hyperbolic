# Test-suite implementation and verification

Prepared on 3 November 2026 for the upcoming package refactor. This work reorganizes
and strengthens tests; it does not intentionally change production numerical
formulas or solver defaults. Other source/demo work already present in the shared
working tree was preserved.

The recorded runs include pending source-feature work in that working tree.
Tests specific to compilation reports, mixture acoustic diagnostics and the new
reporting/transient callback API remain with their corresponding uncommitted
source changes, separate from the test-reorganization and archive-removal commits.

**Subsequent removal:** the frozen implementation archive, original-suite backup
and historical test group were deleted at the user's request. The measurements
below describe the review before that removal. Current checks and remaining
references are recorded in [LEGACY_REMOVAL.md](LEGACY_REMOVAL.md).

## Preserved starting point

The exact original suite was initially saved in
`legacy/test_suites/2026-11-03_before_reorganization/tests.zip`,
with per-file SHA256 checksums and the original dependency files. All 48 archived
entries were verified against the manifest. A separately extracted copy ran with
**732 passed and three expected failures**, 735 parametrized cases total.

An AST comparison found **364 original test functions** in the reorganized suite.
**363 retain exactly the same function bodies and decorators**, independent of
formatting and module relocation. The remaining temporal-order test was extended
to check RK3 alongside RK4, preserving its nonlinear-storage, algebraic-recovery,
stage-boundary and inventory checks. Independent reference arithmetic was retained.

Historical tests were grouped under `tests/legacy/`. The three strict expected
failures document the deprecated two-fluid mass-transfer closure's metastability
discrepancies. They have not been converted into passing assertions or suppressed
as evidence about the active two-fluid implementation. That historical group
and its expected failures have subsequently been removed; the parallel deprecated
solver in `src/` is a separate remaining implementation.

## Implemented changes

- Grouped tests by geometry, numerics, thermodynamics, models, solvers, analytical
  references, physical/numerical regressions and historical ownership. Split broad
  multi-purpose modules and extracted small shared test fixtures.
- Centralized JAX float64 setup before collection. Added a fresh-process check
  that disabling precision is rejected by the implicit operator.
- Added registered execution categories, strict marker/configuration handling,
  strict expected failures and consistent Black formatting.
- Added compilation-cache cleanup at independent steady regression case boundaries.
  This releases accumulated test resources without changing solver kernels or
  clearing caches inside compilation-reuse assertions.
- Added complete nozzle and annular subsonic/shocked/expanding solves on three
  grids, with original-residual, physical-branch and analytical-accuracy checks.
- Added exact advection, Burgers, Sod, annular contact and homogeneous wall-source
  transient checks, plus complete two-fluid equilibrium and perturbed-state
  advances with EOS and accepted-transfer audits.
- Added a production Euler projector check against an independently differentiated
  conservative Jacobian. Historical local flux algebra is not used as a proxy for
  the active model.
- Saved seven complete numerical reference profiles with explicit initialization,
  settings, dependency versions, checksums and an exact source archive. Pytest reads
  fixed data; regeneration is a separate, explicit command into a new directory.
- Added fast/full/optional CI, branch coverage reports and an isolated wheel import
  and resource smoke check. Only `coverage` and `pytest-cov` were added to the lock;
  existing numerical dependency versions were retained.
- Added the [test guide](tests/README.md) and updated maintained documentation links.

## Measured verification before archive removal

| Check | Result |
|---|---|
| Archived original suite | 732 passed, 3 strict expected failures |
| Original-test preservation | All 364 original functions retained; 363 unchanged, one deliberately extended |
| New transient accuracy checks | 7 passed |
| Additional exact limits, temporal order, projectors and import checks | 21 passed |
| Complete reorganized sweep | 770 passed, 3 strict expected failures in 1361.23 seconds |
| Collection at the end of the original review | 775 cases: 772 passed, 3 strict expected failures; no unchecked case IDs |
| Final physical regression rerun | 30 passed in 339.16 seconds after portability and fixture cleanup changes |
| Configured fast selection | 437 passed, 336 deselected in 277.59 seconds; two subsequently added compilation tests passed separately |
| Concurrent compilation-report tests | 2 passed; the relocated acoustic JIT/AD diagnostic also passed separately |
| Combined Python coverage | 86.3% statements, 74.0% branches; coverage.py combined score 83.9% |
| Black | All 89 Python files pass |
| Installed wheel | Passed in a disposable environment, outside the checkout, with no numerical dependencies |
| Runtime fault injection | A 20% HLL-flux defect caused the advection analytical test to fail; production files were not edited |
| Snapshot comparison guards | Reject mass/momentum/energy perturbations, NaNs and missing cells |
| Baseline portability and integrity | LF/CRLF identities agree; altered case data differs; numerical/source checksums verified; overwrite attempt refused |

The full sweep collected 773 cases. During the work, compilation-report tests
were added concurrently and an acoustic-diagnostics test moved into the two-fluid
test folder; these were checked separately. All 30 physical regressions were
rerun after the final portability and cache-cleanup adjustments. A final audit
matched every one of the 775 then-collected case IDs to passing results or
the three explicit historical expected failures. No original test was dropped
at that point; later removal was an explicit user-directed change.

Machine-readable results are saved locally under `outputs/test_suite_audit/`.
The HTML coverage report is `htmlcov/index.html`, with XML in `coverage.xml`.
These generated reports are ignored by Git; CI uploads corresponding artifacts.
The reported durations include local compilation and concurrent machine load,
and are not performance benchmarks.

The mutation produced normalized advection error about **0.074**, exceeding the
fixed **0.004** target. This demonstrates a failing physical acceptance check even
though the same faulty flux can remain internally conservative. The clean solver
passes the same test. Snapshot sensitivity checks provide a second kind of guard
against numerically changed or incomplete results.

Local execution uses Python 3.13, JAX/jaxlib 0.6.2, JAXProp 0.5.8, NumPy 2.4.6,
SciPy 1.17.1 and pytest 8.4.2 on CPU with float64. Poetry 2.1.2 was located outside
PATH and used with the existing `.venv`. Python 3.12 and Linux CI are configured;
their remote jobs have not been run in this session.

The Windows sandbox denies access to pytest's private temporary-directory ACLs.
For the original and full/fast audit runs, a run-only plugin supplied normal
workspace temporary directories and disabled pytest's tmpdir/cache plugins. All
numerical test bodies and assertions were retained. This workaround is external
to the committed suite; ordinary CI uses pytest's standard temporary directories.
Wheel construction/checking used normal temporary-directory permissions separately.

## Scope and remaining limits

Analytical agreement and numerical compatibility are complementary. The saved
profiles pin this verified working-tree version, rather than claiming agreement
with every published historical release. Larger intentional numerical changes
require reviewed replacement baselines and independent physical acceptance.

This suite supplies a practical refactor gate for current CPU behavior. It does
not establish GPU performance, universal absence of JAX recompilation, exhaustive
real-fluid phase-change accuracy or every model/method/EOS combination. Existing
JIT/AD regularity, compilation-reuse, general-EOS and failure-path checks remain;
the new complete-solve two-fluid fixtures use perfect-gas phase backends with
closures disabled. Real-fluid closure validation and large resolution campaigns
remain separate numerical studies.

Coverage reports include active production code and omit explicitly deprecated
two-fluid files. Python tracing coverage cannot establish XLA runtime-branch
coverage, so physical regimes and numerical identities are tested explicitly.
No arbitrary 100% target or weakened physical tolerance was introduced.

The largest remaining line gaps are the older duct/workbook geometry helpers
(`geometry/duct.py`: 265 uncovered statements) and shape-fitting utility
(`fit_nozzle_shape.py`: 55). The active two-fluid model has 76 uncovered statements,
including configuration/failure branches. These gaps remain visible in coverage;
the wheel's resource check does not claim numerical coverage of Excel preprocessing.
Representative real-fluid closure validation still needs dedicated physical data
and acceptance criteria before treating it as an accuracy regression.

The CI action inputs were checked against the official
[checkout](https://github.com/actions/checkout),
[setup-python](https://github.com/actions/setup-python) and
[artifact upload](https://github.com/actions/upload-artifact) documentation.
