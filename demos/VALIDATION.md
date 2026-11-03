# Demo validation

Measured on **3 November 2026** using the repository's existing `.venv`, CPU
backend and float64. Poetry was unavailable on PATH, so the executable was
`.venv/Scripts/python.exe -B`. The environment was Python 3.13.2, JAX/jaxlib
0.6.2, JAXProp 0.5.8, NumPy 2.4.6 and Matplotlib 3.11.2.

These are executable API examples with checked default outcomes, rather than a
grid-convergence study or a timing comparison. Every demo has its own folder,
one `run.py`, inline model configuration and an ignored `outputs/` directory.
No script loads YAML or imports a development/case runner.

## Steady outcomes

All nine requested defaults converged with the original shared-driver stopping
checks and passed independent checks on their saved physical regime. Euler
examples use MUSCL Van Leer/HLL and backward Euler on 80 physical cells.
Nitrogen–water uses MUSCL Van Albada, the model-owned Stadtke compatibility
discretization and backward Euler on 100 physical cells.

| Model | Regime | Accepted updates | Rejected attempts | Maximum Mach | Exit Mach | Maximum scaled global-balance defect |
|---|---|---:|---:|---:|---:|---:|
| Nozzle Euler | Subsonic | 56 | 0 | 0.361 | 0.172 | $6.47\times10^{-10}$ |
| Nozzle Euler | Shocked | 78 | 9 | 1.569 | 0.352 | $1.46\times10^{-10}$ |
| Nozzle Euler | Adapted | 43 | 0 | 2.184 | 2.184 | $3.04\times10^{-10}$ |
| Annular Euler | Subsonic | 61 | 1 | 0.298 | 0.144 | $8.18\times10^{-10}$ |
| Annular Euler | Shocked | 195 | 39 | 2.275 | 0.411 | $8.22\times10^{-10}$ |
| Annular Euler | Adapted | 44 | 1 | 2.499 | 2.499 | $1.45\times10^{-10}$ |
| Nitrogen–water | Subsonic | 36 | 0 | 0.836 | 0.136 | $3.73\times10^{-10}$ |
| Nitrogen–water | Shocked | 35 | 1 | 1.332 | 0.184 | $4.95\times10^{-10}$ |
| Nitrogen–water | Adapted | 37 | 0 | 1.645 | 1.645 | $2.17\times10^{-10}$ |

Mach denotes meridional Mach for annular Euler and acoustic-mixture Mach for
the two-fluid model. The regime checks require subsonic flow everywhere,
supersonic flow followed by a subsonic exit for the shocked case, or a
supersonic exit for adapted flow. The plotted profiles also resolve the shock.
Saved fields are finite, pressures and temperatures are positive, and both
two-fluid phase velocities/densities are positive with $0<\alpha_g<1$.
Final scaled boundary defects are below $2.4\times10^{-15}$; two-fluid EOS
constraint defects are below $1.9\times10^{-14}$.

The two Euler demos use their models' simple cold guesses. Analytical solutions
select boundary data and provide comparisons; their profiles do not seed the
interior solve. The annular supersonic cases prescribe a supersonic inlet in a
diverging radial channel; the subsonic case uses a reservoir inlet.

The nitrogen–water demo follows the published NICFD two-component setup:
20 bar reservoir, both stagnation temperatures 295.15 K, gas mass-flow quality
0.5, inlet slip one, and backpressures 16, 12 and 0.986 bar. Nitrogen and water
use separate real-fluid HEOS bicubic PH tables. Drag/heat exchange are active,
mass transfer is disabled, and the original number-density area law is selected.

Its default initial guess is **explicitly interpolated from the published
profile CSVs**. Those reference data are frozen at upstream revision
`2dc04ad9d474afb727857139df89537fcadb5769`, with attribution and a manifest in
[cases/NICFD2026](../cases/NICFD2026/).
The model, geometry, mesh quadrature, EOS configuration, closures, boundaries
and solver settings are constructed in the demo itself. All discrete residuals
are solved afresh. This verifies the example's branch with that initialization;
it does **not** establish cold-start robustness. The optional
`INITIALIZATION = "cold"` path is not part of the successful default validation.
See the [NICFD guide](../cases/NICFD2026/README.md) for comparison limits.

JAXProp 0.5.8 emits a static-array metadata warning when constructing each
`FluidBicubic`; the warning is retained in the run log. The demonstrated solves
and diagnostics pass on this CPU environment. Dependency-level static metadata
and broader compilation/performance behavior remain separate review work.

## Transient outcomes and MP4s

Both defaults use 100 cells, MUSCL Van Leer/HLL, SSPRK3, requested CFL 0.4 and
121 output frames. Initial algebraic traces are completed without changing
physical cell contents; integration reuses the same operator binding.

| Demo | Reached time [s] | Accepted steps | Rejections | Final maximum normalized content-budget error |
|---|---:|---:|---:|---:|
| Shock tube | 0.06684924584 | 10,320 | 0 | $5.87\times10^{-12}$ |
| Nozzle pressure release | 0.004605284745 | 4,387 | 0 | $2.04\times10^{-13}$ |

For the initial shock-tube run, the final pressure span divided by the mean is
$1.86\times10^{-10}$; the velocity span divided by the reference sound speed
is $1.12\times10^{-10}$. The resolved waves have left to numerical accuracy.
The duration is three estimated rarefaction-tail exit times. Initial velocities
remain zero; no velocity boost or late-time field replacement is used.
The zero-gradient endpoints approximate open boundaries, so the
infinite-domain analytical comparison applies before boundary interaction.
Quadratic output spacing resolves early waves and the long washout period;
movie titles show actual physical times.

Both videos decode fully without errors using FFmpeg. Each is H.264, yuv420p,
1000 by 700 pixels, 24 fps and approximately 5.04 seconds of playback:

- [Shock-tube MP4](shock_tube/outputs/transient.mp4)
- [Nozzle pressure-release MP4](nozzle_transient/outputs/transient.mp4)

FFmpeg was supplied as a portable encoder in ignored output folders; no project
dependency was added. A clean checkout needs FFmpeg on PATH or `FFMPEG_BINARY`,
as explained in the [demo guide](README.md).

## Reporting and verification

The source changes made while preparing these examples are limited to:

- `solution_summary`: common outcome, accepted/rejected work and separate final
  spatial, boundary, constraint, global-balance and transient-budget diagnostics.
- `ConvergenceTable`: a visible cell-constraint column for steady cases and a
  transient mode showing physical time, actual CFL, inner work and content budget.
- Shared drivers: callback validation, transient attempt callbacks, separate
  requested/actual CFL, removal of an unconditional extra progress print, and
  optional reuse of the exact model/scheme operator for transient integration.
- `TwoFluidModel.acoustic_diagnostics`: a public JIT/AD-compatible read-only
  diagnostic using the existing acoustic eigenvalues, so the demo does not
  access private eigenstructure helpers or assume perfect-gas sound speeds.

No flux formula, closure law, boundary equation, timestep policy, convergence
tolerance or production numerical default was changed for the demos.
Rejected attempts remain in saved histories and work totals. Failure paths
save the reached state/outcome and exit with an error.

The full maintained-suite run passed **726 tests**, with **29 legacy tests
deselected**, in 866.66 seconds. A run-only `tmp_path` fixture used a normal
workspace directory with inherited Windows permissions; pytest's built-in
temporary-directory and cache plugins were disabled to avoid the environment's
temporary-directory permission error. No numerical assertion was weakened.

The test tree was changing concurrently: a later collection found 734 active
tests, so the full-run count describes its collection snapshot. A subsequent
focused run passed **24 tests**, covering reporting, transient callbacks/operator
reuse, the acoustic diagnostic in its final model-owned test location, and the
new saved-profile guard checks. The focused command was:

```powershell
poetry run python -B -m pytest -q -p no:cacheprovider tests/solvers/test_convergence_output.py tests/solvers/test_transient_audits.py tests/models/two_fluid/test_acoustic_diagnostics.py tests/regression/test_guard_sensitivity.py
```

All five scripts parse and have no YAML/development/case-runner imports.
Black accepts the five demos, convergence-output module and new acoustic test.
NumPy-docstring parsing matches the signatures of `solution_summary`,
`acoustic_diagnostics` and the extended `solve_transient`, including return sections.
The changed source/documentation paths pass `git diff --check`.
Default flow/convergence figures and transient frames were inspected visually;
the annular plot title was corrected to avoid clipping. Presentation-only
updates were regenerated from saved states/histories without advancing CFD.
Transient endpoint defects were reevaluated on the frozen saved states using
the same JIT-compiled physical boundary function; their final values match the
original solver summaries exactly. Sampled endpoint diagnostics are retained
alongside the attempt history for subsequent plotting. Exact zeros are kept
as zero in saved data and omitted/annotated on logarithmic defect plots.

Generated plots, numerical records and MP4s remain under each demo's ignored
`outputs/`. Single-run elapsed values are retained in JSON summaries, but are
not end-to-end timings or performance claims: compilation, preparation and
rendering costs have different timer boundaries.

## Follow-up: JAX executable caching

The first cache-enabled revision configured persistent executable caching before
constructing JAX arrays. That revision printed compile/load START/END events, elapsed
durations and observed disk-cache hits, with totals in `compilation.json`.
Steady clients retain a `SteadyWorkspace` for the solve; transient clients
retain their existing `Operator`. Neither configuration performs an extra
warmup solve or changes numerical initialization.

Each client was executed in **two fresh Python processes** against the same
per-demo cache. The steady checks used the default subsonic case and physical
cell count. Transient checks used their default 100 cells but five frames over
$10^{-5}$ s, to exercise RK3, stage boundaries and MP4 creation. The earlier
full-duration videos and three-regime results above were preserved.

| Demo | First-run non-cached requests | Second-run non-cached requests | Second-run disk hits | First compile/load total [s] | Second compile/load total [s] |
|---|---:|---:|---:|---:|---:|
| Nozzle Euler | 69 | 0 | 70 | 4.7115 | 1.8333 |
| Annular Euler | 74 | 0 | 76 | 4.6833 | 2.1552 |
| Nitrogen–water | 304 | 2 | 320 | 124.0084 | 19.5303 |
| Shock tube | 51 | 0 | 51 | 6.3976 | 1.8282 |
| Nozzle pressure release | 75 | 0 | 78 | 9.2091 | 2.6145 |

The cached and first-run numerical NPZ outputs were **bitwise identical for
every saved field** in all five pairs. Logs and records are retained under each
demo's `outputs/cache_validation/{cold,warm}/`. A first run can itself have
cache hits from earlier imports or duplicate lowered computations; these are
separate from the non-cached counts in the table.

The two nitrogen–water misses were `get_state` requests. Its large
`jit(implicit_step)` executable was an observed disk hit, loaded in 8.199846 s.
The two property-state misses remain visible; these checks do not establish
that all EOS specializations or physical regimes will always avoid rebuilding.

These are single cache-functionality checks, not a repeated performance
benchmark. Totals are sums of JAX backend compile/cache-handling events,
including lookup, compilation, serialization or loading as applicable. They
exclude Python tracing/lowering before those events, and may include monitoring
and filesystem overhead. They should not be interpreted as pure compiler CPU
times or used to claim a time-to-solution speedup.

**Two new cache-observer tests passed**, including fresh-process persistent hits,
same-callable in-memory reuse with changed input values, and failure cleanup.
**Fifteen existing workspace, transient-audit and import-isolation checks
passed**. Black, NumPy-docstring signature parsing and source syntax checks
also passed for this addition. The earlier 726-test full run predates this
follow-up; it is not claimed as a full post-cache-change run.


## Current presentation and readability revision

The demo folder is named `shock_tube`. All five scripts now expose editable
variables near the top and have a plain `main()` with no CLI parsing. The
physical setup and solver API usage appear before the supporting plotting
helpers. Method and unit comments remain in the scripts.

Figures use the package's `set_plot_options`, portable STIX/serif typography,
consistent colours/markers, restrained grids and separate legends. Static
figures are exported as 300-dpi PNGs and vector PDFs with embedded fonts.
All nine stored steady flow solutions were redrawn without advancing CFD.
The three revised steady clients were also smoke-tested on their default
subsonic meshes: each passed, and every numerical field was bitwise identical
to the corresponding original saved result. The case data were reorganized
concurrently; the nitrogen-water script now reads the unchanged published
`cases/NICFD2026/two_component/*/reference.csv` files.

Both transient clients were rerun to their full physical end times on 100
cells, with 601 genuine recorded states. Changing the output grid changes the
places where steps land on requested times; neither the scheme, CFL limit,
physical equations nor solver tolerances were changed.

| Demo | Reached time [s] | Accepted updates | Rejections | Final normalized content-budget error |
|---|---:|---:|---:|---:|
| Shock tube | 0.06684924584 | 10,693 | 0 | 5.97e-12 |
| Nozzle pressure release | 0.004605284745 | 4,708 | 0 | 1.92e-13 |

The shock tube's final pressure span/mean is 1.67e-10, consistent with the
resolved waves having left. An independent analytical Riemann profile is now
included in the movie and a three-time static comparison. The first boundary
interaction estimate was corrected: the right shock reaches the right end at
0.0009020938644 s, before the rarefaction head reaches the left end. The earlier
`first_boundary_interaction_s` metadata described the head arrival and was too
late; this diagnostic correction does not change the physical initial data or
long simulation duration. After interaction, the unbounded reference is grey
and explicitly labelled as a comparison rather than an exact finite-domain
solution. The static analytical comparisons use only earlier times.

Each MP4 contains 701 encoded frames at 20 fps, including two-second initial
and three-second final holds: **35.05 seconds**, 1600 by 1000 pixels,
H.264/yuv420p. Both movies decoded without errors. Saved frames are used
throughout; no synthetic interpolated flow states were introduced. Video
layout refinements are regenerated from saved states, without new integration.
The previous transient data/figures were retained in each demo's
`outputs/before_readability_refresh/`.

Quiet JAX reporting prints exactly **one status line per script**, identifying
compiled, loaded or mixed requests and the observed compile/load duration.
Counts and timing fields remain available in `compilation.json`. All five
execution logs contained exactly one `[JAX]` line. **Three reporter tests
passed**, including quiet single-line output, actual disk reuse, same-process
reuse and failure cleanup. The earlier full-suite results above are historical;
a new full package suite was not needed for these presentation/client changes.
