# Working in nozzle-hyperbolic

Keep this file limited to durable project instructions and documentation routing.
Put implementation details in guides and measured findings in study reports.
The user's current scope and corrections take precedence over these conventions.

## Start and scope

- Check the active branch and `git status --short`; preserve unrelated staged
  and working-tree changes. Do not reset or switch branches to obtain a clean tree.
- Identify the actual source entry point and backend relevant to the task.
  Historical reports and installed dependencies do not establish the active path.
- Distinguish structural refactoring, numerical changes, experiments and
  environment changes. A structural refactor preserves numerical behavior and
  function bodies; fluxes, closures, tolerances, initialization and defaults are
  separate numerical changes requiring explicit scope and review.
- Keep experiments under `development/`, normally `development/roberto/` for a
  new controlled study. Preserve established thematic folders. Do not silently
  add experimental hooks or change defaults in `src/`.
- Complete authorized implementation and appropriate verification; ask only
  for genuinely missing decisions. NozzleX integration, additional model ports,
  publishing and releases are separate work unless requested.

## Read what the task needs

| Task | Start here |
|---|---|
| Setup and repository layout | [README.md](README.md) |
| Documentation and supported capabilities | [docs/README.md](docs/README.md) |
| Architecture or physical-model extension | [Architecture](docs/architecture/ARCHITECTURE.md), [model interface](docs/model_interface.md) |
| Residuals, stopping or physical monitors | [Residual monitoring](docs/residual_monitoring.md) |
| Physical closures | [Closure guide](docs/physical_closures.md) |
| Physical cases and analytical comparisons | [Verification API](src/nozzle_hyperbolic/verification/README.md), [reference assumptions](docs/analytical_references.md) |
| Test organization, regression data and CI | [Test suite](tests/README.md) |
| Numerical experiment, timing or report | [Study conventions](docs/numerical_studies.md), then the relevant [study README](development/README.md) |
| Native/current comparison | [Benchmark](development/roberto/benchmarking/README.md), [frozen baseline](legacy/README.md) |

Read relevant guides when their topic is involved; small edits do not require
reviewing the full documentation set. Check historical instructions against the
current source before using them.

## Architecture and numerical integrity

- Physical models own storage, physical fluxes/eigenstructure, EOS, geometry,
  boundary equations, closures, admissibility and physical dependency stencils.
  The shared numerical engine must remain independent of concrete model types,
  equation identities, equation counts and EOS libraries.
- Evaluate Euler fluid behavior through the configured JAXProp state interface.
  Do not introduce perfect-gas identities into general-EOS numerical paths.
  Any explicitly scoped analytical/backend specialization must retain its limits.
- Define physical boundary equations once per model. Their coupled/segregated
  solution strategy belongs to numerical assembly; algebraic traces have zero
  storage unless an explicitly defined physical dynamic model requires otherwise.
- Keep JAX kernels compatible with the intended JIT/AD target. Avoid unsafe
  inactive arithmetic, host conversions and undeclared dependencies. Verify
  precision and the actual backend rather than assuming them from option names.
- Check convergence with the original spatial residual, independently of native
  optimizer success or small updates. Report convergence, conservation, physical
  branch and analytical accuracy separately.
- Do not hide failures through relaxed targets, clipping, dropped attempts or
  reporting an intermediate continuation stage as the requested solution.
  Distinguish accepted steps, inner corrections and rejected work.
- Reuse physical cases and independent references. Preserve failures, settings,
  states and source provenance; disclose analytical or numerical warm starts.
- Treat `legacy/pre_generic_fvm` source and tests as frozen. Verify its manifest
  before comparisons; keep adapters and new numerical code outside the snapshot.

## Environment and verification

- Use Poetry and the existing project `.venv/`, from the repository root.
  Typical checks use `poetry run python -B -m pytest <affected tests>`.
  Do not add `sys.path` workarounds or rebuild the environment for routine work.
- Production pytest tests belong in `tests/` and import `src/`, not development
  scripts. Development studies use their own executable audits and reports.
- Run checks appropriate to the change. Keep pre-existing failures distinct;
  documentation-only changes need document/link checks, not fresh CFD campaigns.
- Update `pyproject.toml` and `poetry.lock` together for dependency changes.
  Preserve local dependency comments and the documented release workflow.

## Documentation and commits

- Use `$...$` for inline Markdown math and `$$...$$` for display math.
- Update the owning guide or study README rather than duplicating solver
  defaults, campaign results or historical commands here.
- For numerical studies, follow [the study conventions](docs/numerical_studies.md)
  for fair timings, provenance, figures and standalone report packaging.
- Give concise progress updates and separate measurements, deductions and
  hypotheses. State what changed, what was checked and any material limitation.
- When asked to commit, select explicit paths or hunks and use a descriptive
  message. Preserve unrelated work and separate production and study changes
  when useful. Do not push, tag, publish or release without a request.
