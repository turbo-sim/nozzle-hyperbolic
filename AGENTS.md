# Working in nozzle-hyperbolic

This file is the starting guide for new sessions. Follow the user's current
scope and corrections; use this guide for established project conventions.
Keep it concise and current when folders or workflows change. Detailed findings
belong in study notes, not in a growing session diary here.

## Start a session

1. Read this file, inspect `git status --short` and the active branch, and read
   [README.md](README.md) plus the relevant study README. Do not reset, switch
   branches, stage, or overwrite unrelated work to make the checkout clean.
2. Identify the actual source entry point and backend before proposing a change.
   Installed dependencies and historical comments do not identify the solver
   used by a particular path. Older reports can describe retired experiments.
3. Establish the task's boundary: structural refactor, numerical experiment,
   production numerical change, or environment change. Keep these distinct.
4. Reuse the existing physical setup/reference when applicable. Begin with one
   controlled configuration, then expand only as needed for the requested study.

The work is currently focused on nozzle-hyperbolic. NozzleX integration and
porting additional branch implementations are separate tasks, not implied by
a local cleanup or solver experiment. Check the branch rather than assuming
`roberto-branch` is still active.

## Collaboration and scope

- Lead with the finding or intended action. Give concise progress updates and
  distinguish measured facts, mathematical deductions, and hypotheses.
- Carry authorized work through implementation, execution, diagnosis and report.
  Ask only about genuinely missing choices that materially affect the outcome;
  do not repeatedly request permission for already authorized routine work.
- For a numerical experiment, keep tentative implementations in
  `development/roberto/` by default. Do not silently add hooks or change defaults in
  `src/`. If source changes are explicitly requested, isolate and explain them.
  Record source hashes for studies that promise to leave source unchanged.
- A structural refactor preserves function bodies, comments/docstrings and
  numerical behavior. Moving files/imports is different from changing fluxes,
  limiters, tolerances, initialization, boundary equations or defaults. Separate
  numerical changes for review and commits; do not call them mere refactoring.
- Do not hide failures by relaxing tolerances, clipping invalid results, dropping
  unsuccessful cases, or reporting an intermediate continuation stage as a
  solution of the original target equations.
- When asked to commit, use explicit paths and descriptive, coherent commits.
  Preserve unrelated edits. Commit production changes and experimental studies
  separately when useful; do not bundle abandoned investigations or publish,
  push, tag or release unless requested.

## Repository map

| Location | Ownership / where to start |
|---|---|
| `src/nozzle_hyperbolic/` | Installable production package; import as `nozzle_hyperbolic` |
| `models/euler_nozzle/` beneath the package | EulerNozzle physics, NumPy/JAX fluxes, state maps, boundaries, initialization, sources, closures, analytical references and FVM assembly |
| `models/two_fluid/` | Two-fluid solver, state/matrix/flux/source code, boundaries, analytical references and model-specific closures |
| `models/euler_annular/` | General-EOS quasi-1D annular Euler prototype: swirl, characteristic flux, coupled boundary blocks, steady and physical-time drivers; see its verification report for limitations |
| `numerics/` | Shared reconstruction, characteristic contraction, colored block Jacobians and banded linear solves |
| `geometry/duct.py` | Duct/nozzle geometry |
| `geometry/annular.py` | NURBSpy 1.6 NumPy midline/height geometry, arclength metrics and physical annular meshes; fixed host preprocessing, no shape AD |
| `thermodynamics/` | Property helpers and recovery routines; fluid backends also come from JAXProp |
| `solvers/finite_volume.py`, `solvers/time_marching/` | Shared conservative residual assembly and implicit Newton construction; model factories retain physical callbacks and driver policies |
| `fvm_solver/` | Legacy compatibility imports forwarding to canonical modules. Edit the canonical implementation, not a shim or a second copy |
| `development/` | Single development workspace; the `alberto/` and `roberto/` collections are indexed in its README |
| `development/roberto/` | Current controlled verification examples and numerical investigations |
| `development/alberto/` | Legacy research code, grouped by topic; see its README and execution audit |
| `cases/` | Physical applications and reference datasets; start with its README and inspect each case's imports/settings |
| `cases/validation/` | Literature validation data and associated processing/plotting scripts |
| `tests/` | Production pytest tests only |
| `docs/` | Organization/audit documents, physical derivations and historical findings; verify dated claims against the current checkout |

Important Euler navigation:

- `models/euler_nozzle/fvm.py`: `fvm_euler_1d`, `make_jax_step`, and
  `make_jax_stadtke_step`. Some active boundary logic is **nested inside the
  factory**, so reading `boundaries.py` alone is insufficient.
- `models/euler_nozzle/flux_jax.py`: JAX characteristic flux implementations,
  including regularized and legacy variants.
- `numerics/reconstruction.py`: shared reconstruction/limiter arithmetic;
  `models/euler_nozzle/reconstruction.py` contains model-specific routines.
- `models/euler_nozzle/analytical/`: steady nozzle and shock-tube references.
- `models/euler_annular/fvm.py`: `make_annular_solver`, full boundary/interior
  residual operators, cold initialization, steady and physical-time drivers.
  Unknowns are `(p, v_m, ell, h)`; `v_theta=ell/r`. Numerical paths use sound
  speed and Gruneisen, with perfect-gas formulas confined to analytical references.
- `models/euler_annular/closures/wall.py`: named zero-default/prescribed
  `shear_stress_wall` and `heat_flux_wall` placeholders. Shear contributes angular
  wall torque; only heat adds total energy. Empirical correlations are deferred.
  The current report campaign uses `verify_grid_sequence` with 50/100/200/400
  physical cells and bounded primitive initialization between converged meshes.
- `solvers/residuals.py`, `solvers/monitoring.py`, and
  `solvers/convergence_output.py`: reusable fixed-scale equation norms,
  scalar-monitor windows, plain-text convergence tables and strict JSON.
  Annular's opt-in `solve_steady_monitored` connects these; nozzle and legacy
  annular drivers keep their previous stopping definitions. Annular YAML inputs
  live in `development/roberto/historical/euler_annular/cases/`; its `run_case.py` runs one
  named case with live convergence output. Keep historical histories labelled
  with their actual residual definition.
  The main report now reads `outputs/grid_sequence_monitored_full/`: all 72
  nine-order reruns, with four N400 MUSCL stagnation failures retained. Results
  uses eight native collapsible boxes and a 2-by-2 overall/CFL/local/global view.
- `models/two_fluid/solver.py`: substantial model-specific implicit assembly
  and boundary construction; do not assume it shares Euler's implementation.

Architecture direction: share genuinely reusable numerical assembly and solver
infrastructure. Keep physical boundary laws and closures beside the models
that use them. General-EOS Euler versus a perfect-gas property backend is not
necessarily a separate solver: inspect both the actual fluid object and any
implementation flags. Do not infer the EOS from a flag named `perfect_gas`.

Earlier nozzle, annular, backpressure and Newton studies, plus compatibility
reference imports, are indexed under
[`development/roberto/historical/`](development/roberto/historical/README.md).
Their saved evidence retains measurement-time paths and source hashes.

## Current development studies

Start with [development/README.md](development/README.md) for the Alberto/Roberto split,
then [development/roberto/README.md](development/roberto/README.md) for current studies.
Keep the single `development/` parent; do not recreate `development_new/` or
`development_roberto/` as additional top-level trees.

| Study | Purpose |
|---|---|
| [euler_nozzle](development/roberto/historical/euler_nozzle/README.md) | Readable steady nozzle and unsteady Sod verification against analytic solutions |
| [numerical_fixes](development/roberto/numerical_fixes/README.md) | Flux/projector/entropy-regularization and limiter repairs; legacy/corrected comparison |
| [newton_solve](development/roberto/historical/newton_solve/README.md) | Retained Newton/PTC/implicit comparison and consolidated explanations of earlier failures |
| [implicit_integration](development/roberto/historical/implicit_integration/README.md) | Inner Newton work and CFL/controller sensitivities |
| [BC_residual](development/roberto/historical/BC_residual/README.md) | Internal versus coupled BC equations with matched implicit solver machinery and repeated timings |
| [backpressure_sensitivity](development/roberto/historical/backpressure_sensitivity/README.md) | Perfect-gas implicit operating map, choking/shock-exit behavior and outlet branch selection |
| [euler_annular](development/roberto/historical/euler_annular/README.md) | General-EOS annular solver, NURBSpy geometry, answered design decisions, analytical/constant/MUSCL verification and independent saved-state audits |
| [multigrid](development/roberto/historical/multigrid/README.md) | EulerNozzle implicit single-grid versus coarse-to-fine initialization benchmark; annular multigrid design note; no FAS implementation |

The steady baseline has five regimes: supersonic exit, shocks at 35% and 70%
of the divergent section, sonic-throat/subsonic-exit limiting flow, and fully
subsonic flow. The Sod example checks time-dependent wave propagation.
References verify the result and select physical data; they are not numerical
initial guesses unless an explicitly labeled local-convergence test needs one.

Known interpretation limits: a discrete steady root can retain the sonic-limit
analytic accuracy discrepancy. PTC improved cold convergence in some shocked
cases but did not establish unconditional robustness or a speed advantage.
Read the relevant current report rather than turning these observations into
universal claims or copying historical timings into a new study.

## Organize a numerical study

Use this structure for new studies; preserve older exceptions unless their
reorganization is requested:

```text
development/roberto/<study>/
  README.md                  question, physical setup, controlled settings, commands
  <topic>_NOTES.md            equations, interpretation, design discussion as needed
  <topic>_REPORT.md           findings with embedded relative figure links
  REPORT.html                one standalone HTML with internal results/theory pages
  scripts/                   readable drivers, supporting experimental code, audits
  outputs/<run_or_variant>/  JSON/CSV, logs, figures and generated run data
```

- Human-facing files belong at the study root; implementation belongs in
  `scripts/`, generated data in `outputs/`. Python package markers are fine.
- Keep human-facing HTML reports directly at the study root, outside `outputs/`.
  Their Python generators and reusable assets belong in `scripts/`; generated
  numerical data, logs and figures belong in `outputs/`. Keep reports reproducible
  from the Python scripts. Do not commit generated HTML unless requested.
- Generate one shareable, standalone HTML file per report. Put results, theory
  and other chapters on internally navigable pages within that file, rather
  than separate HTML documents. Embed styles, JavaScript libraries, plot data,
  images and offered downloads; do not depend on CDNs, neighboring files or a
  local server. A colleague must be able to copy just the HTML file and open it
  in a browser offline. Check page navigation, plots, equations, theme controls
  and embedded downloads after consolidation.
- Regenerate presentation changes from saved numerical results without rerunning
  simulations. Preserve numerical data and timing samples, and update report
  generators, links and structural audits together.
- Put ignore rules in the repository `.gitignore`, not a study-local
  `.gitignore`. `outputs/` is generated and ignored by default. Some older
  result snapshots were explicitly committed; check tracking before cleanup.
- Keep scripts readable. Import shared geometry, physical cases, references and
  settings rather than duplicating their definitions. Numerical overrides must
  not mutate the imported baseline's global settings.
- Preserve mathematical/physical explanations separately from measured results.
  Embed PNG figures with relative links; also save PDF versions for export.
  Label axes, units, methods, residual definitions and stopping thresholds.
- Use plasma colors sampled between 0.2 and 0.8 for comparable lines unless the
  user specifies otherwise. Use distinct styles/markers when curves overlap.
  For the standard five-case comparison, use five subplots, one backpressure
  per panel, with one line per method; provide the analogous final-field figure.
- Save machine-readable settings, case/mesh/EOS details, backend identities,
  package versions and source fingerprints when relevant. Retain failed runs,
  actual final fields, accepted histories and rejected-trial information.
- When cleaning an investigation, consolidate useful negative results and why
  they matter before deleting superseded scripts/results. Update imports,
  module commands, defaults, relative links and report generators together.

## Numerical correctness and solver choices

- Hyperbolic JAX kernels should remain compatible with JIT and AD. Avoid
  unsafe inactive arithmetic, host conversions or data-dependent Python control
  flow inside traced code. Record dtype and follow the solver precision policy;
  current Euler verification uses 64-bit arithmetic.
- For new direct Newton/LM/HYBR/SLSQP investigations, use custom PySolverView
  backends with initial-step/radius controls. Record `library="pysolver"`, the
  actual method and effective options; never silently substitute vanilla SciPy.
- **Exception: a user-requested comparison that must match source uses the
  verified source algorithm.** The Euler Städtke implicit path inspected in the
  boundary study uses JIT Newton, colored blocks and a banded solve, not
  Optimistix or PySolverView. Recheck current code before asserting a backend.
- A fair formulation comparison holds physical equations, initialization,
  reconstruction, mesh, precision, timestep controller, stopping rules and
  linear/nonlinear solver machinery fixed as far as possible. Explain unavoidable
  differences; do not attribute a dense-versus-banded timing gap to BC physics.
- Validate final convergence with the **original spatial residual** independently
  of native success, small increments or a least-squares progress flag. Define
  normalization precisely: native state-dependent SSR and a fixed-scale spatial
  RMS are different metrics and can stop at different update counts.
- Report root convergence, physical/analytic accuracy, conservation and intended
  branch selection separately. Local convergence does not prove global uniqueness
  or physical well-posedness. Smaller residual need not mean smaller state error.
- Distinguish accepted outer updates, inner corrections, rejected trials,
  residual evaluations and continuation stages. An inner iteration setting is
  a maximum cap, not a forced correction count or proof of an exact timestep.
- Physical BC equations should have one definition per model. If both eliminated
  and coupled strategies are added later, select them in an assembly factory
  before JIT compilation rather than duplicating physics or silently switching
  state dimensions halfway through a solve. Algebraic ghost constraints have
  zero storage unless a separate physical dynamic model explicitly requires it.

## Performance measurements

To rebuild the retained Newton/PTC/custom-adapted report without numerical
solves, use `poetry run python -B -m development.roberto.historical.newton_solve.scripts.report`.
The standalone root HTML includes the useful mechanism/performance notes stored
under `scripts/report_assets/notes/`. Discarded optimizer experiments are not
part of this study; preserve the common problem and implicit-helper imports
used by neighboring studies when reorganizing it.

To rebuild the implicit-integration report from the retained 105-run dataset,
without importing numerical solvers or rerunning cases, use
`poetry run python -B -m development.roberto.historical.implicit_integration.scripts.report`.
It writes root `REPORT.html` with three sensitivity pages and one theory page;
saved timing samples and numerical JSON/CSV files remain unchanged. See that
study's README for the optional isolated offline browser audit.

To regenerate the EulerNozzle grid-sequencing benchmark and its HTML
reports, run from the repository root:

```powershell
poetry run python -u -m development.roberto.historical.multigrid.scripts.run_demo
poetry run python -m development.roberto.historical.multigrid.scripts.audit_results
poetry run python -m development.roberto.historical.multigrid.scripts.report_results
```

See the study README for effective settings and smaller smoke commands. The
default benchmark compares single-grid and coarse-to-fine initialization on
three cases and two reconstructions, with ten measured samples plus an excluded
complete solve per method. It changes no source and implements no FAS. Place
HTML reports directly in `development/roberto/historical/multigrid/`; keep numerical data,
logs and figures in its `outputs/` subfolder. The generator writes one
`REPORT.html` containing internal results and theory pages, with all rendering
resources and offered downloads embedded. To update only the report from saved
results, run the `report_results` command above; do not rerun `run_demo`.

For a repeated solver benchmark, default to one excluded **complete solve**
per case/method followed by ten measured solves, unless the user specifies
another protocol.

- Build kernels once and reuse them across repeats. Reset the identical initial
  field and all controller/history state on every solve; do not reuse a solution
  as a hidden warm start. Alternate method order and run measurements sequentially.
- Synchronize JAX device work before ending a timer. Excluding import/JIT warmup
  for only some operations or rebuilding closures between repeats is not fair.
- State the timing boundary: include equivalent monitoring and rejected work;
  exclude or separately report setup, reference/seed generation, final audits,
  output and plotting. Hybrid warm-start cost must be included in hybrid totals.
- Retain every sample and the excluded-run status. Report minimum, maximum and
  mean, with explicit sample count. Do not cherry-pick successful or fastest runs.
- Plot one labeled actual representative trajectory (for example the median-
  duration run), not an averaged residual curve or a best-so-far envelope.
- Only discuss time-to-solution speedups after checking success and agreement
  with the same intended discrete root across repetitions. Different backends,
  sparsity, precision or stopping rules must be disclosed.

## Verification, environment and documentation

- The pytest suite covers production code under `src/` only. Do not add pytest
  tests for `development/` experiments or import development scripts from
  `tests/`. Keep experiment verification in executable audits and reports.
- For production changes, run focused meaningful pytest checks first. For a
  study, execute its requested cases and independent residual/field audits,
  inspect generated figures, check report links, and record limitations. Keep
  pre-existing failures distinct; do not report an all-green suite without evidence.
- Use Poetry and the existing project `.venv/`; run commands from the repository
  root, typically `poetry run python -m development.roberto.<study>.scripts.<driver>`.
  Install with `poetry install --with dev` when needed. Do not add `sys.path`
  hacks, restore Conda, or rebuild environments for an ordinary numerical study.
- Treat `pyproject.toml` and `poetry.lock` as a pair for dependency changes. Preserve
  commented local JAXProp/PySolverView/NURBSpy dependencies and release instructions.
  Verify current requirements in the manifest; do not carry old package-version
  assumptions into a new session. Releases use bump-my-version, as described in
  the root README and at the end of `pyproject.toml`; releases are separate work.
- Markdown math uses `$...$` inline and `$$...$$` for display equations, never
  `\(...\)` or `\[...\]`. Keep notes self-contained and explicit about physical
  setup, equations, numerical changes, controls, evidence and unresolved issues.
