# NICFD 2026 nitrogen–water reproduction

This case reproduces the two-component column of Figure 1 in Bergamin et al.,
*A non-equilibrium time-marching model for two-phase supersonic nozzle flows*.
It uses the new eight-state `TwoFluidModel` and shared finite-volume solver.
The two components are gaseous nitrogen and liquid water. Interphase drag and
heat exchange are active; interphase mass transfer is disabled.

The primary numerical reference is the authors' supplementary repository,
[alber-berg/twophase-nozzle-nicfd2026](https://github.com/alber-berg/twophase-nozzle-nicfd2026),
fetched on 2 November 2026 and frozen at
`2dc04ad9d474afb727857139df89537fcadb5769`. Unchanged settings, converged CSVs,
figure scripts and the CC BY 4.0 attribution are retained in `upstream/`.
`upstream/manifest.json` records Git blobs and SHA-256 hashes. The supplied
manuscript is available through [DTU Orbit](https://orbit.dtu.dk/en/publications/a-non-equilibrium-time-marching-model-for-two-phase-supersonic-no/).
Its local PDF is inspected but is not redistributed here.

## Published cases

All three cases use the existing Elliot geometry over $0\le x\le0.31$ m,
reservoir pressure 20 bar, both stagnation temperatures 295.15 K, inlet gas
mass-flow fraction 0.5 and inlet velocity ratio $u_g/u_l=1$.

| Regime | Backpressure [bar] | Reference row count | Physical interior rows |
|---|---:|---:|---:|
| `subsonic` | 16 | 500 | 498 |
| `supersonic_shock` | 12 | 500 | 498 |
| `supersonic_adapted` | 0.986 | 500 | 498 |

The reference's `Nx=500` includes the first/last endpoint ghosts. Their stored
coordinates are exactly 0 and 0.31 m. A new-solver run uses an explicit number
of **physical** cells plus two endpoint traces. The active study uses **50, 100,
200 and 400 physical cells**, as specified in [settings.json](settings.json).
Its interior centers and volumes follow the shared FVM convention,
which differs from the original uniformly spaced point coordinates. Comparisons
interpolate interior profiles and report this difference. Endpoint ghosts are
excluded from profile statistics. The adapted outlet pressure is selected by
the supersonic branch and need not equal the imposed backpressure.

The raw geometry is the existing `geometry.duct.get_nozzle_elliot` function.
Face areas and cell-center areas are sampled directly; volumes are integrated
with 24-point Gauss quadrature split at the contour joins. A separate audit
checks the geometry against the reference CSV and partition-independent total
volume on uniform/nonuniform meshes. No flow-based geometry fit is used.

## Physical and numerical mapping

The complete published YAML for each regime remains unchanged in `upstream/`.
Every run records both that specification and the effective new-solver inputs.

| Setting | Reproduction |
|---|---|
| Nitrogen properties | JAXProp HEOS bicubic, 145 by 200, $h=50$–400 kJ/kg, $p=0.1$–60 bar, forward gradients |
| Water properties | JAXProp HEOS bicubic, 145 by 200, $h=1$–140 kJ/kg, $p=0.1$–60 bar, forward gradients |
| Virtual mass | $k=0.5$ |
| Mass transfer | Disabled; two-component heat/drag assembly |
| Bubble/droplet counts | Both $10^{11}$ m$^{-3}$, preserved from the case |
| Unspecified closure selectors | NICFD-era number-density interfacial area (`area_law="original"`), Schiller–Naumann drag, Ranz–Marshall bubbly liquid HTC, internal gas Nu=12, `regime_htc=False`, linear regime blend |
| Flux / nonconservative form | Explicit `model_compatibility`, `nc_form="wb"`, full acoustic dissipation internally, boundary dissipation floor zero |
| Entropy fix / slip blend | $10^{-6}$, 5–20 m/s |
| Reconstruction | Generic MUSCL Van Albada in six enthalpy primitives; eight unknowns retained per cell |
| Void filter | Strength 0.05, frozen sound-speed scale, active everywhere |
| Thin-phase policies | Pins, phase clipping and additional ghost filters off |
| Physical BC | Stagnation reservoir with gas entropy at quality 0.5; characteristic outlet, pressure relaxation 1 |
| Integration | Backward Euler, colored AD, square block-banded Newton, coupled algebraic traces |
| Initial CFL / Newton cap | Published 2/5 for subsonic/adapted and 5/5 for shock; explicit diagnostic overrides are recorded |

The solver adaptation intentionally retains the shared square Newton system,
fixed scales, strict update/residual termination, backtracking, CFL/retry control
and independent original-spatial/EOS/trace/budget checks. It does not import
the original normal-equation solve, adaptive Newton column scaling, increment
limits or boundary fixed-point/isentrope-lookup machinery. Generic Van Albada
uses local sign-preserving slopes rather than the upstream global regularized
formula. Exact integrated geometry differs from the upstream point sampling.
These differences can change finite-mesh profiles and robustness; agreement
must be measured rather than assumed.

The supplementary YAML omits the interfacial-area selector. The historical
Alberto revision `e185f571f8ea0512a7a16ce53a4b2458f72c33c2`, dated 28 September
2026 and titled “Version of NICFD26 with minor changes”, calls the original
number-density law directly, with bubble diameter floor $5\times10^{-6}$ m
and droplet floor $10^{-5}$ m. This supplies the missing selector explicitly.
The polynomial law was added in October. Initial runs that inherited the
later polynomial default are retained and labelled as a different physical
closure, rather than combined with the corrected reproduction. The current
model's later pressure-work/source corrections remain active and documented.

The reference reports JAXProp 0.5.6. The installed version and full package
versions are recorded without changing the environment. All calculations use
float64. Every accepted PH stencil must have populated finite coefficients and
physical corner data. No EOS clipping, missing-node filling or extrapolation is
introduced. The prior identical water table can be reused after JAXProp metadata
validation; its source hash/copy cost is retained in `outputs/`.

## Run and compare

From the repository root, using the existing Poetry environment:

```powershell
poetry run python -B -m cases.nicfd_validation.scripts.audit
poetry run python -B -m cases.nicfd_validation.scripts.run --prepare-only
poetry run python -B -m cases.nicfd_validation.scripts.run --regime all --cells 50 100 200 400 --independent-meshes --initialization reference --newton-iterations 20 --steps 500
poetry run python -B -m cases.nicfd_validation.scripts.verify_results
poetry run python -B -m cases.nicfd_validation.scripts.report
poetry run python -B -m cases.nicfd_validation.scripts.audit_report --browser
```

The report generator uses saved results only. It builds a standalone offline
Plotly report styled after `development/real_gas/`, with all data and Plotly
embedded. Profile visibility selects **published only**, **new solver only** or
**both**. Regime, mesh and flow/thermodynamic field groups have their own
controls. Other pages show signed differences, mesh errors, per-equation
histories, rejected work and exact evidence. Figures export as SVG through the
Plotly toolbar; selected profiles/histories export as CSV and the complete
dataset as JSON. No Python Plotly dependency or CDN is needed.

The active input and verification paths are recorded in `settings.json`. The
commands above rebuild the report from that saved campaign. A new solve creates
a timestamped output directory; to report a replacement campaign, update the
`campaign` path in `settings.json` before running verification and report generation.
The twelve saved runs include eight verified cases: all subsonic meshes and the
100/400-cell shock and adapted meshes. The 50/200-cell shock and adapted runs
remain unfinished and are labelled in the report.

The builder asserts that Scheme, SteadySettings, boundaries, closures, compatibility
and stabilization match [baseline_numerics.json](baseline_numerics.json), the
saved settings of the prior 498-cell comparison. Each new mesh starts
independently from the published profile: **no grid sequencing** is used here.
Failed coarse-grid states are never used to initialize finer grids in this
campaign. The nonlinear discrete problem changes with the mesh, including shock
location relative to cell faces and the MUSCL limiter pattern. Convergence is
therefore not necessarily monotone with mesh size; the saved histories establish
stagnation/step-limit failures but do not isolate their precise cause.
Earlier 498-cell runs and diagnostic attempts remain in `outputs/`; their
summary is available on the evidence page. The former static report and its
generator are preserved under `outputs/archive_498_report/`.

Published settings say `ic: restart` without identifying the original restart.
The default independent cold seed is explicitly 10 m/s at the inlet with
area-scaled velocity, reservoir pressure and total-enthalpy conversion.
`--initialization reference` instead interpolates the published primitive
profile as an openly declared numerical seed. This is useful for checking the
new discrete branch but does not establish independent cold-start convergence.
`--restart path.npz` uses shared constraint-consistent portable preparation.
`--independent-meshes` resets the chosen initialization for each mesh.
Automatic mesh continuation reuses only a successful previous stage. A failed
stage is retained, and later requested grids use the declared base initialization. No flux-method sequence
or hidden warm start is used.

`--newton-iterations`, `--cfl`, `--steps` and `--stretch` are explicit controls.
The active mesh comparison uses a Newton cap of 20 with unchanged update,
residual, EOS, trace and budget tolerances. The earlier five-iteration coarse
shock/adapted attempts stagnated and remain recorded. The larger cap is an
adaptation to the generic nonlinear solver, not a change to the physical model.
Output directories are created exclusively, so a prior attempt is not replaced.
Each `result.json` retains all accepted/rejected attempts, initial/final scales,
original and fresh residuals, EOS/BC/budget gates, field-error measures, source
hashes and costs. Final CSV, NPZ fields and portable restart preserve the actual
state even on failure. An unsuccessful requested stage gives exit code 1.
Preparation/execution exceptions remain recorded separately.

Numerical root convergence and the intended flow regime are separate checks.
The case audit requires positive phase velocities and maximum mixture Mach below
one for subsonic flow, a supersonic region with subsonic exit for the shock case,
and a supersonic exit for adapted flow. A converged numerical root on the wrong
regime is retained as a mismatch, not a reproduced case. New runs mark its
portable snapshot unsuccessful for automatic reuse and return nonzero status.
The independent cold-to-fine attempt found such a root with a supersonic pocket;
the three close fine-grid comparisons remain reference-seeded.

Compare [RESULTS.md](RESULTS.md) and the generated `REPORT.html`. Pressure,
both phase velocities, mixture Mach, void fraction and **flow** quality reproduce
the quantities in the paper. Mixture Mach uses the model's acoustic eigenvalues,
not Wood's sound speed. Static quality is retained as a separate diagnostic.
The flashing-water reference data/figure scripts are preserved for provenance;
this client executes only the requested nitrogen–water family.

## Recorded elapsed times

These are single-run, compile-inclusive solver durations from the saved
50–400-cell campaign, including monitoring and rejected work. They are not
repeated warmed benchmarks. Preparation, final audits, file output and plotting
are outside the solver timer. The per-grid setup took another 8.5–19.2 seconds;
the shared property-table preparation loaded cached tables in 1.52 seconds.

| Physical cells | Subsonic [s] | Shock [s] | Adapted [s] |
|---:|---:|---:|---:|
| 50 | 53.6 | 115.3 (stagnation) | 211.6 (step limit) |
| 100 | 62.5 | 84.3 | 66.8 |
| 200 | 94.2 | 124.6 (stagnation) | 105.5 (stagnation) |
| 400 | 110.9 | 126.8 | 66.6 |
| Sum over four meshes | 321.2 | 451.0 | 450.5 |

Unmarked entries converged and passed the independent case audit. Failed entries
show time spent before termination, not time to a converged solution.
