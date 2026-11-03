# Architecture and code-quality review

Reviewed on **3 November 2026**, on `roberto-branch`, against the working-tree snapshot at the time, including its pre-existing modified and untracked files. HEAD at the original verification was `a37140461b770ab2ee163637f043578868af1564`; the unrelated case commit made during that review did not change `src/` or `tests/`. Existing work was preserved. The source fixes and closure relocation were subsequently committed at the user's request; later implementation notes are recorded below.

## Assessment

**The shared finite-volume architecture is sound and worth keeping.** Physical models supply storage, fluxes, closures, admissibility and boundary equations; the numerical engine supplies discretization, differentiation and solution algorithms. The scalar models and constrained two-fluid port provide useful evidence that this separation extends beyond Euler equations.

The main improvements are to make that architecture easier to see in the code: retire the parallel historical implementation, divide the large operator factory, finish documenting the public interfaces, and establish a consistent formatting and testing workflow. A broad solver rewrite would add risk without addressing these maintenance problems efficiently.

The review covered all Python files through structural/static inspection, with closer reading of the shared engine, model bindings, numerical kernels, EOS adapter, geometry, verification utilities and their tests. Initially there were **92 source files / 20,873 lines** and **46 test files / 9,033 lines**. This is an architecture and implementation review, not an independent validation of every physical correlation or a GPU performance benchmark.

## Structure worth preserving

| Area | Assessment |
|---|---|
| [`models/`](src/nozzle_hyperbolic/models/) | Good ownership of physical equations and general-EOS behavior. Shared Euler outlet and Roe helpers avoid duplicating important physics. The duplicate two-fluid implementation has now been removed. |
| [`numerics/`](src/nozzle_hyperbolic/numerics/) | Mostly small, model-independent kernels. Colored differentiation, local JVP reuse and block-banded solving have meaningful dense/AD comparisons. Historical helper variants need clearer boundaries. |
| [`solvers/fvm/`](src/nozzle_hyperbolic/solvers/fvm/) | Appropriate assembly and driver layer. The operator constructor has accumulated too many responsibilities. |
| [`solvers/time_marching/`](src/nozzle_hyperbolic/solvers/time_marching/) | Useful separation of integration and Newton policies from physical models. Explicit stages advance conserved contents and complete algebraic states. |
| [`thermodynamics/`](src/nozzle_hyperbolic/thermodynamics/) | The adapter preserves the configured EOS and delegates callback derivatives to JAXProp. Serializing mutable host-fluid evaluation is appropriate. |
| [`geometry/`](src/nozzle_hyperbolic/geometry/) | Fixed physical geometry and host preprocessing are explicit. The newer mesh builders and older sampled `Geometry` helpers coexist with different conventions. |
| [`verification/`](src/nozzle_hyperbolic/verification/) | Independent references, strict case definitions and finite-volume projections are strong design choices. Keep analytical assessment separate from numerical initialization. |
| [`tests/`](tests/) | Substantial coverage of conservation, boundary treatment, invalid states, AD regularity, general EOS behavior and generic model contracts. The flat layout and unclassified compilation-heavy tests make navigation and routine checking harder. |

## Suggestions, in priority order

“High” below means a high maintenance/correctness priority. It does not mean that every item is an observed failing numerical result. Larger changes listed here remain proposals for discussion.

### 1. Remove the deprecated historical two-fluid implementation — completed

The duplicate standalone solver, old state implementation and old closure module
have been deleted. The active pinned physics now occupies the familiar
`matrices.py`, `flux.py` and `sources.py` paths; closure laws are grouped by
mechanism. Model-owned boundary, admissibility and stabilization policies have
dedicated modules. Public model/configuration imports remain unchanged.

The [current two-fluid guide](docs/notes/two_fluid_migration.md) maps Alberto's
function names, explains the retained equations and records the historical
research scripts that still need a separate driver migration. Maintained tests
import the canonical modules. Original regression data and provenance archives
remain unchanged. The earlier source/test counts in this review are historical
measurements, not inventories of the cleaned tree.

### 2. Divide `Operator` binding into a few cohesive factories — high

**Evidence:** [`Operator.__init__`](src/nozzle_hyperbolic/solvers/fvm/operator.py) originally contained roughly **930 implementation lines**, including capability discovery, shape validation, reconstruction, several flux families, source/constraint assembly, Newton construction, recovery, boundary elimination and timestep selection.

**Suggested approach:** extract model validation/capability binding, spatial/flux construction, storage/recovery construction, and timestep construction into private modules or factories. Retain `Operator` as the small public facade. Pass a clearly defined bound context rather than an expanding collection of loosely related callbacks. Keep concrete model/EOS imports out of the numerical layer.

Make this a structural change with unchanged arithmetic ordering, defaults and physical formulas. Gate it with existing dense-versus-colored, coupled-versus-segregated, residual/conservation and representative model tests, plus compilation-reuse checks. Avoid introducing a mandatory model inheritance hierarchy simply to shorten this constructor.

### 3. Complete public NumPy-style docstrings and make them buildable — high

**Evidence:** an AST inventory found **447 public declarations**: top-level public classes/functions and their direct public methods, including protocols and retained historical code. Before editing, **153 had no docstring**, and only **32** had any `Parameters`, `Returns` or `Attributes` section. This is a structural indicator, not a Sphinx quality score.

This review documents the central solver settings/results, drivers, workspace, operator, mesh, every existing protocol callback, current two-fluid state conversions, and fitting helpers. The same inventory now finds **125 declarations without docstrings** and **71 with a NumPy section**. Remaining work is concentrated in model constructors/callbacks, geometry helpers and smaller public utilities. Removing deprecated APIs will also reduce that backlog.

**Suggested approach:** document supported public APIs first. Include array shape, state ordering, SI units, sign conventions, integration/area weighting, valid domain, returned failure information and JIT/AD limitations. Describe configuration dataclasses with `Parameters`/`Attributes`. Use reStructuredText roles and math inside Python docstrings; reserve dollar delimiters for Markdown documents. Do not repeat solver theory in every one-line adapter: link to the owning interface guide.

There is no Sphinx `conf.py` or `.rst` API build under the current `docs/` tree. Add a small API documentation build using a deliberate NumPy-docstring parser configuration, and validate the maintained public API in CI. The presence of Sphinx/numpydoc dependencies alone does not provide this check.

### 4. Establish one formatting/import/lint policy — high

**Evidence:** the initial `python -m black --check src tests` reported **75 files needing formatting and 63 unchanged**. The package mixes compact one-line functions, semicolon-like packing through long expressions, several import styles, inconsistent whitespace and capitalization conventions. Some actual text-encoding damage remains in `geometry/duct.py` docstrings/comments.

**Suggested approach:** choose the project's existing Black formatter settings, configure import ordering and a small lint rule set, and run them in CI. Start linting with undefined names, unused imports and accidental shadowing; defer broad annotation/style migrations. Format the supported source and tests in a dedicated mechanical change, separate from numerical changes and historical-source removal. Normalize damaged prose using the original intended text.

A repository-wide formatting change was deliberately left for that dedicated review; it would otherwise obscure the existing substantial working-tree changes.

### 5. Make the lifetime of bound model configuration explicit and enforceable — high

**Observed correctness risk:** model objects are mutable, while operator callbacks close over Python configuration. A probe compiled an advection face-flux kernel, changed `model.speed` from 1 to 2, and observed an unchanged compiled flux while eager `wave_speed` returned 2. `SteadyWorkspace.validate` checks model identity and settings equality, so it does not detect this in-place mutation.

This is expected behavior for captured Python state in JAX; it can produce inconsistent physics if callers assume the model is editable after binding. [JAX's discussion of pure functions and captured state](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) explains the underlying constraint.

**Already done:** added explicit fixed-configuration lifetime documentation to `Operator` and `SteadyWorkspace`.

**Suggested approach:** choose an enforceable contract: immutable configuration snapshots at binding, or a model revision/fingerprint checked before reuse. Frozen dataclasses alone do not freeze embedded dictionaries/arrays. Keep evolving state as explicit array arguments; decide separately whether parameter differentiation merits dynamic parameter pytrees. Do not silently try to refresh selected callbacks after mutation.

### 6. Extend deliberate kernel reuse to retries and repeated transient/continuation solves — medium

[`solve_single_grid`](src/nozzle_hyperbolic/solvers/fvm/drivers.py) can reuse `SteadyWorkspace`, which is a useful existing feature. However, normal calls construct new operators/samplers, grid cold retries rebind without a supplied workspace, parameter continuation constructs an operator for preparation and another for solving, and `solve_transient` has no workspace/operator reuse argument.

Fresh closures across repeated solves can incur repeated compilation. New meshes genuinely require shape specialization; repeatedly binding the same mesh/model/scheme is a different source of cost. [JAX documents why newly created temporary functions interfere with compilation caching.](https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching)

**Suggested approach:** bind once per fixed model/scheme and reuse within preparation, warm solve and cold retry. Consider a small shared binding/workspace interface for transient replays. Preserve fresh solution histories, monitor windows and residual baselines. Decide whether `SteadyWorkspace` should join the package-root convenience exports: it is public in `solvers.fvm` but absent from the root export map.

### 7. Profile host synchronization and repeated EOS work before changing kernels — medium

**Specific review targets:**

- `sampler` separately evaluates spatial rows, boundary rows, constraints, reductions and admissibility, with several host conversions. Steady stepping also compares full candidate/state arrays on the host for stagnation detection.
- `NozzleModel.boundary_residual` requests cell thermodynamics for the entire interior and uses only the inlet impedance. The annular boundary factory has a similar whole-interior request for inlet information.
- `HaalandWall.__call__` obtains wall-state properties and calls `drag_coefficient`, which obtains them again. The Euler source assembly may then request `drag_coefficient` separately as well.

These are concrete opportunities to investigate, **not measured speedups**. Native JAX compilation can remove redundant work; a host callback backend can have very different behavior. The sampler explicitly preserves a separate spatial graph to avoid changing roundoff near strict targets, so indiscriminate fusion is inappropriate.

**Suggested approach:** count actual property callbacks and profile a native perfect-gas/bicubic backend and the host-fluid adapter separately. Restrict inlet property requests to their physical stencil where equivalence is demonstrated, and consider a fused wall-data result. Benchmark cold compilation and warmed execution separately, synchronizing results. [JAX's asynchronous dispatch guide explains why dispatch-only timings and host transfers must be distinguished.](https://docs.jax.dev/en/latest/async_dispatch.html)

### 8. Give solver diagnostics an explicit schema — medium

**Confirmed and fixed:** the solver records `global_defect`, while `ConvergenceTable` read `conservation_max`. The table therefore omitted G for ordinary solver rows. It now displays the maximum absolute global defect, preserves NaN/infinity diagnostics, and retains the older scalar key as a fallback.

**Suggested approach:** introduce a small `TypedDict` or documented result/history schema for shared fields. Reuse that schema in samplers, console output, JSON export and cases. Clarify the distinction between requested/actual CFL, accepted/rejected steps, inner corrections and boundary work. Keep model-specific engineering quantities in a clearly named extension mapping. Add producer-to-consumer integration checks where a schema mismatch could hide diagnostics.

### 9. Organize tests by responsibility and execution cost — medium

The flat test directory contains unit identities, model contracts, analytical references, full shared-solver steps, optional integrations and historical regression tests. Precision setup is repeated across many modules, and importing the old solver also changes global JAX precision. A passing combined collection does not by itself prove isolated-test setup is reliable.

**Suggested approach:** group tests under `numerics/`, `solvers/`, `models/`, `geometry/`, `verification/` and an explicitly historical group pending deletion. Set required precision in one early test configuration. Keep import-side-effect and disabled-precision tests in fresh subprocesses. Share reusable model fixtures sparingly; preserve independent reference formulas.

Register markers for fast checks, compilation-heavy integration and optional dependencies. The initial suite took about **13 minutes** here; the slowest case was a coupled/segregated two-fluid step at about **67 seconds**. Those measurements include this machine's compilation/environment costs and are not solver benchmarks. Maintain a fast required subset and a full numerical job. Add a separate optional NozzleX job so `importorskip` does not silently stand in for integration coverage.

### 10. Consolidate older geometry and residual helpers without losing useful references — medium

[`geometry/duct.py`](src/nozzle_hyperbolic/geometry/duct.py) mixes NumPy sampling, JAX/NURBS geometry and case-specific fitting. Its sampled `Geometry` interface differs from the newer fixed-mesh geometry objects. The perfect-gas nozzle reference still imports its symmetric-nozzle helper, so the module cannot simply be removed.

**Confirmed and fixed:** `get_lettieri_nozzle` used the first coordinate interval as a uniform `np.gradient` spacing. Nonuniform coordinate arrays produced incorrect area derivatives. It now uses the supplied coordinate vector; independent planar/axisymmetric straight-wall cases cover the correction. Endpoint differences remain first order, as before.

**Suggested approach:** choose a supported geometry-provider contract with explicit point values, physical derivatives and volume integration. Move workbook fitting/plotting into a clearly named preprocessing utility; keep case-specific demonstrations outside the installable core. Retain historical adapters only with named callers and migration plans.

Also evaluate [`solvers/finite_volume.py`](src/nozzle_hyperbolic/solvers/finite_volume.py): `make_coupled_residual` is exercised by tests but is not used by the current production operator. It duplicates a simpler square-storage assembly contract that does not express current constrained cells. Either make it an intentional supported low-level API with clear limits or retire/consolidate it. Historical nozzle flux/reconstruction variants likewise need explicit ownership before removal.

### 11. Harmonize continuation, restart and validation policies — medium

**Evidence:** `solve_grid_sequence` supports strict failure handling and optional workspace reuse. `solve_parameter_sequence` has different controls: `retry_cold=False` disables a second solve after a failed warm attempt, but a failed preparation still produces a cold first attempt. That may be a useful policy, but the flag does not express the same strict-transfer choice as the grid driver.

Restart compatibility checks compare counts, equation labels and domain endpoints. They cannot establish that two models share iteration-variable meanings, EOS configuration, units or constitutive conventions. Generic JSON values are also coerced when loading fields such as `success` rather than fully schema-validated.

**Suggested approach:** agree on explicit preparation-failure, retry and stop policies, then share the implementation. Extend restart metadata with variable/transport identifiers, model/EOS provenance and strict primitive-type validation. Document intentional parameter changes as permitted continuation changes rather than requiring all provenance to be identical. Add incompatible-state-meaning and corrupted-metadata tests. Keep restart interpolation an initialization policy; conservative remapping for physical transients is a separate feature.

Similarly, harmonize configuration validation for booleans, integer-like values, finite reals and exception types. `Scheme`, `SteadySettings`, monitor configuration and two-fluid configuration currently use different validation idioms. Avoid a large generic validation framework; a few shared host-side helpers would suffice.

### 12. Add installation and capability checks before expanding features — medium

`fit_nozzle_shape.py` directly imports pandas, but pandas is not declared directly in `pyproject.toml`; Excel reading also needs an appropriate engine. Optional NozzleX integration is tested only when that package is present. The source tree's successful editable installation does not test a built wheel's resources or dependency completeness.

**Suggested approach:** decide whether Excel preprocessing is a supported optional extra or a development-only utility, then declare its dependencies accordingly. Add a wheel smoke test in a clean environment that imports public APIs and loads packaged YAML cases. Keep dependency/lockfile changes together.

Publish a compact model/flux/integrator/backend capability matrix and test the supported combinations. The active two-fluid port explicitly accepts only `model_compatibility`; current identity and small-step tests should not be interpreted as validation of every closure, real-EOS phase limit or physical transient. Prioritize representative general-EOS/two-fluid convergence and failure-path checks before new numerical features. Shape sensitivities through host geometry and conservative transient restart remapping are separate design decisions, not features to add incidentally during cleanup.

## Small changes implemented in this review

| Change | Files / scope |
|---|---|
| Correct G-column diagnostics and retain scalar compatibility | `solvers/convergence_output.py`; regression cases in `test_convergence_output.py` |
| Correct nonuniform area-derivative spacing | `geometry/duct.py`; two new independent cases in `test_nozzle_geometry.py` |
| Mark historical two-fluid code deprecated | The eight files listed in suggestion 1; documentation notices only |
| Remove redundant process-global import-path mutation | Historical `models/two_fluid/sources.py`; its relative import already resolves the closure module |
| Add NumPy-style public documentation | `solvers/fvm/{config,contracts,drivers,operator}.py`, `numerics/mesh.py`, current `two_fluid/constrained_state.py`, fitting helpers |
| Correct stale descriptions | Current model package summary, generic linear module summary, transported projector dimensions, mapped reconstruction/path wording, and the retained NumPy Van Albada helper's misleading JAX description |
| Remove an obsolete commented script copy | Top of `fit_nozzle_shape.py`; document all three actual fitting return values |
| Add a JAX tracing-reuse regression | `test_steady_workspace.py`: changing state, timestep and time values with fixed shapes/dtypes does not retrace the bound flux callback |

No flux formulas, closure laws, Newton policies, tolerances, default schemes or physical boundary equations were changed. The geometry-spacing correction is the only numerical calculation changed; the other executable production edits concern reporting and removal of a redundant `sys.path` insertion. Larger proposals above remain unimplemented.

## Verification and limits

The existing project `.venv` was used directly because Poetry was unavailable on PATH and the available Poetry support directory was not usable. The editable import resolved to this repository's `src/` tree. Relevant versions were Python 3.13, JAX/jaxlib 0.6.2, JAXProp 0.5.8, NumPy 2.4.6, SciPy 1.17.1 and pytest 8.4.2. Execution used a CPU device.

| Check | Result |
|---|---|
| Initial collection | 724 tests collected successfully |
| Final collection | 730 tests collected successfully, including six new regression cases |
| Full initial suite: `python -B -m pytest -q --tb=short --durations=15` | **720 passed, 3 xfailed, 1 setup error** in 797 seconds. The error occurred before the restart test body: Windows denied access to pytest's temporary directory. There were also cache-directory permission warnings. |
| Focused final solver/output/workspace/legacy/continuation checks | **63 passed, 3 xfailed**. The restart test passed with a test-run-only `tmp_path` fixture using a normal workspace directory with inherited permissions; built-in tmpdir/cache plugins were disabled for that run. No repository fixture or numerical assertion was weakened. |
| New nonuniform-geometry checks | **2 passed** |
| Initial Black check | **75 would reformat, 63 unchanged**; recorded as an existing consistency issue |
| NumPy-docstring parsing/signatures | **36 changed documented signatures checked**, with no missing/extra parameter names |
| Source/test syntax and executable-change audit | All Python files parsed. Comparison to saved pre-edit files after removing docstrings limited executable changes to reporting, geometry spacing, redundant path-mutation removal and the added regression tests. |
| Import isolation | Importing the current `TwoFluidModel` loaded none of the eight deprecated modules and did not alter `sys.path`. Explicit historical source import no longer alters `sys.path`. |
| JAX tracing probe | Python flux-callback trace counts after binding/first/second/third steps were **1 / 9 / 9 / 9**; timestep values changed between calls. This checks reuse for one fixed scalar binding, not all model/backend combinations. |

The full run began before the review edits and did not include the six new parametrized test cases. Changed executable paths and the restart fixture were checked afterward as listed above. No full post-edit CFD campaign, GPU timing study, clean-wheel installation or Sphinx HTML build was performed. The three strict expected failures remain attached to the deprecated closure implementation and are listed for migration/removal, rather than reclassified as passing physics.

## Follow-up: model-owned closures

The requested closure organization has been implemented: the former
`models/closures` directory is removed. Its unchanged wall and equilibrium-mixture
implementations now live in `models/euler_nozzle/closures`. Annular Euler imports
the same wall laws and exposes them from its own closure package, alongside its
annular-specific NozzleX adapter. Two-fluid interphase laws retain their existing
model-owned locations.

This organization is suitable because the constitutive laws accept physical
properties and hydraulic diameter, while models retain wall measures, velocity
direction, torque and energy-source assembly. The relocation preserves function
bodies, parameters and numerical behavior. Maintained callers, tests and guides
use the new paths; the closure guide and migration guide document the imports.

Follow-up verification: **44 affected tests passed**, including constitutive
limits/AD, nozzle and annular wall sources, and optional NozzleX comparisons.
All **731 tests collect**. Both annular-first and nozzle-first imports passed;
the relocated wall, mixture and export modules have unchanged executable ASTs.

## Follow-up: standalone demos and consistent reporting

Five single-file clients now live in [demos/](demos/README.md), each in its own
folder: nozzle Euler, annular Euler, the real nitrogen–water NICFD two-fluid
case, Sod and nozzle pressure release. The three steady clients each solve
subsonic, shocked and adapted cases. The two transient clients generate MP4s.
All configuration is constructed in Python, with descriptive comments and
method alternatives. The two-fluid default explicitly discloses its published
CSV initial guess; it is not presented as a cold-start convergence result.

The reporting work addresses parts of suggestions 6 and 8: transient solves
can reuse the exact bound operator after initial trace completion, both drivers
support consistent attempt callbacks, and `solution_summary` distinguishes
accepted/rejected work and the separate physical defects. The console now
exposes cell constraints for steady cases and physical time/content budgets
for transients. No numerical formulas or production solver defaults changed.
The new public two-fluid acoustic diagnostic reuses the model's eigenstructure
and is checked under JIT and automatic differentiation.

All nine default steady regimes converged, and both transients reached their
requested final times. The longer Sod run has a final relative pressure span
of $1.86\times10^{-10}$. Both videos decode successfully. Detailed settings,
initialization limits and verification results belong in
[demos/VALIDATION.md](demos/VALIDATION.md). A formal history schema, broader
continuation/retry reuse and immutable binding remain design proposals.

## Suggested next changes to discuss

1. Migrate/archive the remaining historical tests and callers, then delete the deprecated two-fluid files.
2. Make a separate formatting/lint setup change for supported code.
3. Agree on immutable model binding and extract the operator factories without changing numerical arithmetic.
4. Finish supported API documentation and add a documentation build plus capability/installation checks.
5. Profile the identified EOS and host-synchronization paths before selecting performance changes.
