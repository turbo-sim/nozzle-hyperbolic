# Matched implicit integration: internal versus coupled boundary equations

This study compares exactly two treatments of the same five nozzle backpressures:

1. **Source/internal BCs:** call the current source timestep kernel, which
   reconstructs ghost values inside each inner residual evaluation.
2. **Experimental/coupled BC equations:** keep ghosts as unknowns and solve
   their equations together with the implicit interior balances.

Both now use the **same source-native JAX Newton algorithm, colored block
Jacobian assembly and block-banded Thomas solver code**, with the same inner
iteration cap and relative-update stopping rule. The source path does not use
Optimistix. The previous PySolverView/dense inner implementation has been
removed so it cannot distort this comparison. Running the study does not edit
production source. The current adapter consumes explicit source operators and
the shared numerical engine introduced by the later annular refactor.

- [IMPLICIT_REPORT.md](IMPLICIT_REPORT.md): measured outcomes, min/max/mean timing
  table, timing distribution, five-panel convergence and five-panel pressure figures.
- [BOUNDARY_DESIGN_RECOMMENDATIONS.md](BOUNDARY_DESIGN_RECOMMENDATIONS.md): robustness,
  coding simplicity, two-fluid considerations and a clean two-strategy design.
- [BC_IMPLEMENTATION.md](BC_IMPLEMENTATION.md): physical boundary equations,
  reduced/coupled formulations, shared numerical machinery and limitations.
- `scripts/`: comparison, tentative coupled timestep, boundary equations,
  public-operator source adapter, and plots/report generation.
- `outputs/steady/`: all timings, excluded and measured histories, fields,
  trial records, CSV files, figures and source hashes. The repository-level
  `.gitignore` excludes `outputs/`; there is no local ignore file.

## Run the benchmark

From the repository root:

```powershell
poetry run python -m development.roberto.BC_residual.scripts.compare_implicit --no-show
```

Defaults are 201 points and **one excluded full solve followed by ten measured
solves per case and method**. There are 110 full solves: 10 excluded and 100
measured. Every solve restarts from the same cold field and fresh controller
state. Compiled kernels are built once per case and reused. Method order
alternates between measured repeats. The first solve of each case/method is
excluded even if part of its work was already compiled during setup.

The timer includes synchronized solver work, the identical monitor, timestep
controller and rejected attempts. Problem setup, final audits, writing and
plotting are outside it. All solves run sequentially with one BLAS thread.
The report gives minimum, maximum and mean of all ten measured times, and
shows every sample. `timings.csv` retains 100 rows; `summary.json` also retains
the excluded runs and every accepted/trial history. Failed runs stay visible.

The benchmark explicitly pins its measured CFL 20, maximum CFL 500, SER
exponent 1, timestep growth 1.2, and inner cap 8, independently of later changes
to the readable steady-example defaults.

The same five physical cases are used: supersonic exit, shocks at 35% and 70%
of the divergent section, sonic-throat/subsonic-exit limiting flow, and fully
subsonic flow. Both methods receive the same perfect-gas backend, corrected
flux, ordinary safe Van Leer, geometry, boundary data and initial full field.
Analytic fields are used for verification, never as initial guesses.

## Figures and reports

- `convergence_histories.png/.pdf`: five subplots with one actual measured
  trajectory per method/backpressure. Each is the median-duration measured
  run for that method/case; curves are not averages or best-run selections.
- `pressure_distributions.png/.pdf`: final pressures from those same runs,
  plus analytic reference curves.
- `timings.png/.pdf`: all samples, mean markers and min?max whiskers.

Regenerate the figures and root report without solving:

```powershell
poetry run python -m development.roberto.BC_residual.scripts.compare_implicit --plot-only --no-show
```

Use `--cases shock35`, `--points 51`, or `--runs 2` for a smaller check, choosing
an alternate `--output development/roberto/BC_residual/outputs/small_mesh`
to preserve the main dataset. Custom outputs generate separately named root
reports. The default report is `IMPLICIT_REPORT.md`.

## Scope and checks

The coupled system adds six algebraic variables/rows and no ghost storage.
The source factory now exposes a physical residual that accepts independent
full new/old states, its unchanged inlet/outlet maps, and numerical construction
metadata. The adapter uses those public operators and the shared
`make_block_operators` engine for the enlarged cell count. It no longer inspects
or rebinds Python closures. Full physical cell ordering retains the width-two
stencil. The physical equations, Newton stopping formula and numerical
algorithms are preserved, with the stopping formula applied to ghosts as well
when coupled.

The retained 100-sample timing results describe the original measured revision,
which used the documented read-only closure adapter. They were not rerun or
replaced after this compatibility migration. A separate 21-point subsonic
[compatibility smoke log](../euler_annular/outputs/implementation_baseline/boundary_adapter_smoke.log)
and [saved audits](../euler_annular/outputs/implementation_baseline/boundary_adapter_smoke.json)
record source/coupled convergence from identical cold fields, plus augmented
linear and boundary-dependency checks. That small smoke is not a new timing
comparison or a replacement for the five-case benchmark.

The surrounding comparison loop uses a common fixed-scale steady RMS target
rather than the standalone source driver's state-dependent SSR/plateau rule.
Counts may consequently differ from that standalone example (56 versus 58 in
the subsonic case). This does not change the source timestep update. Original
steady residual, mass conservation and analytic accuracy are audited after
every solve, outside timing. The known sonic-limit analytic error remains.

The pytest suite is reserved for production `src/` code. Development-only
boundary formulation tests have been removed. The benchmark retains its
original-residual, algebraic-boundary, mass-conservation, analytic and source-hash
audits; recorded reports describe the checks performed at measurement time.
