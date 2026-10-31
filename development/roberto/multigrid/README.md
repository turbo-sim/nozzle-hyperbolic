# Coarse-to-fine implicit EulerNozzle demo

This study compares a cold single-grid solve with coarse-to-fine initialization
for **EulerNozzle**, using the unchanged production implicit timestep. It does
not implement FAS. The earlier [annular design assessment](MULTIGRID_NOTES.md)
remains a proposal for a different model; its coupled boundary formulation is
not silently substituted into this EulerNozzle benchmark.

## Generate the original study and the combined report

Run from the repository root in the existing Poetry environment:

```powershell
poetry run python -u -m development.roberto.multigrid.scripts.run_demo
poetry run python -m development.roberto.multigrid.scripts.audit_results
poetry run python -m development.roberto.multigrid.scripts.report_results
```

The generated [REPORT.html](REPORT.html) lives directly at this study root.
It is a **single shareable file** with internal results and method/theory pages,
offline Plotly figures, equations, light/dark themes, marker controls and
embedded downloads. Colleagues can copy just this file and open it in a browser;
no server, installation, internet connection or accompanying folder is needed.

JSON histories, timing samples, audit data and PNG/PDF exports remain in ignored
`outputs/`. The scripts and this documentation are the durable artifacts;
do not commit generated HTML unless requested. Inspect `outputs/audit.json`
before interpreting timing ratios.

To regenerate only the report from the saved results, without rerunning any
simulations:

```powershell
poetry run python -m development.roberto.multigrid.scripts.report_results
```

The generator verifies that saved summaries, numerical run files and audit data
remain unchanged, audits the report's embedded resources and links, and records
the build in `outputs/report_manifest.json`. The superseded two-file reports
under `outputs/` are removed only after the consolidated document passes its
structural audit.

The auditor distinguishes equation/branch checks from agreement between the
two final numerical profiles. It returns a nonzero status if a declared
comparison gate fails, while retaining the results for reporting. In the
initial demo, both constant-reconstruction shock comparisons exceed the
fixed-scale state-agreement threshold near the sonic throat, despite satisfying
the original equations and agreeing on shock location. Their timing ratios are
shown without a validated same-solution acceleration claim. MUSCL shock
comparisons pass that guard; both smooth subsonic controls retain a slowdown.
These are measured outcomes, not reasons to relax a tolerance or omit a run.

A smaller smoke experiment is available without overwriting the main data:

```powershell
poetry run python -u -m development.roberto.multigrid.scripts.run_demo --cases shock35 --reconstructions muscl_van_leer --repeats 1 --output development/roberto/multigrid/outputs/pilot
poetry run python -m development.roberto.multigrid.scripts.audit_results --output development/roberto/multigrid/outputs/pilot
poetry run python -m development.roberto.multigrid.scripts.report_results --output development/roberto/multigrid/outputs/pilot --report development/roberto/multigrid/REPORT_pilot.html
```

One measured sample is a smoke check, not the default timing benchmark.

## Latest four-grid rerun: 1e-8 at every stage

The default HTML opens **Four grids (tight)**. The other tabs retain the tight
two-grid campaign, the original four-grid campaign with coarse tolerance 1e-5,
and the shared theory. Existing numerical archives are preserved.

This rerun uses **50 / 100 / 200 / 400 source intervals**, corresponding to
**51 / 101 / 201 / 401 sampled points** including two ghost slots. Every stage
stops at the same original spatial RMS **1e-8**. The physical unknown counts
are 49 / 99 / 199 / 399. Physics, initial-guess rule, interpolation, CFL/SER
control, one-correction implicit kernel and fine-grid verification are unchanged.

```powershell
poetry run python -B -u -m development.roberto.multigrid.scripts.run_four_grid
poetry run python -B -m development.roberto.multigrid.scripts.audit_results --output development/roberto/multigrid/outputs/four_grid_tight
poetry run python -B -m development.roberto.multigrid.scripts.report_results
python -B development/roberto/multigrid/scripts/audit_html.py
```

Results are isolated in `outputs/four_grid_tight/`. As in the existing campaigns,
one complete solve per configuration is excluded, followed by ten measured cold
starts, alternating method order. Each hierarchy stage, transfer, monitor and
rejected attempt contributes to total time. The runner refuses to overwrite an
existing completed summary; choose another study-local `--output` for a repeat.

All 132 new attempts reach their final steady roots. Independent checks verify
all **330 stage endpoints** against 1e-8, plus unchanged source/shared helpers,
counts, transfers, physical boundary equations, conservation and branch selection.
The unchanged profile-agreement gate passes for both MUSCL shock cases and both
subsonic controls. Constant-reconstruction shock35/shock70 remain beyond the
1e-4 profile threshold, so their timing ratios are unqualified. The overall audit
returns nonzero for those two comparison failures despite passing root/integrity
checks; the report preserves this distinction.

Relative to the fresh single-grid controls, qualified MUSCL shock35 and shock70
sequences are about **3.75x** and **5.23x** faster. Subsonic constant and MUSCL
sequences remain about **7.50x** and **1.41x** slower. The report shows all ten
samples, min/max/mean, per-stage residuals and costs, and a comparison with the
tight two-grid results. The campaigns are separate timing batches; paired
controls within each campaign qualify its ratios. The coarsest mesh also differs
between the two- and four-grid paths, so no universal optimum is inferred.

`scripts/run_four_grid.py` and `run_two_grid.py` share one tight-hierarchy runner;
`scripts/tight_grid_report.py` reuses one analysis/plot path for both experiments.
The renderer discovers both datasets automatically; `--four-grid-output` and
`--two-grid-output` select explicit paths when needed. All three campaigns offer
separate embedded audit, summary, timing CSV and timing/work PDF downloads.

## Additional two-grid, tight-tolerance experiment

The standalone report now has a separate **Two grids (tight)** page, while
retaining the original four-grid results and the shared theory page.

The requested 40 and 400 elements are interpreted as **40 and 400 intervals
between sampled source points**: 41 and 401 points, including the two algebraic
ghost slots, hence 39 and 399 interior unknown cells. This keeps the exact same
401-point fine-grid equations as the original comparison. It is the existing
source mesh convention, not 40/400 agglomerated finite-volume unknowns.

Both stages solve to the original fixed-scale spatial RMS **1e-8**. The implicit
kernel still performs one Newton correction per attempted pseudo-step; the
spatial stopping tolerance does not make each backward-Euler timestep exact.
Three cases and both constant/MUSCL reconstructions use the same physics,
boundary maps, interpolation, CFL/SER controller and final verification targets.

```powershell
poetry run python -B -u -m development.roberto.multigrid.scripts.run_two_grid
poetry run python -B -m development.roberto.multigrid.scripts.audit_results --output development/roberto/multigrid/outputs/two_grid_tight
poetry run python -B -m development.roberto.multigrid.scripts.report_results
```

The runner preserves the original dataset and writes `outputs/two_grid_tight/`.
It refuses to overwrite a completed summary; select a fresh `--output` path
for another campaign. Both methods receive one excluded complete solve and ten
measured cold starts per case/reconstruction. The fresh single-grid control
and the complete two-grid sequence are timed with the existing matched policy.
All stage solves, transfers, monitors and rejected attempts are included.

The report generator automatically appends this dataset when present, or accepts
`--two-grid-output` for an explicit study-local dataset. It embeds separate
summary, audit, timing CSV and timing/work PDF downloads for each campaign.
`scripts/tight_grid_report.py` reuses the original plot/analysis code with
independent controls, rather than duplicating numerical solvers.

### Recorded findings

All 132 attempts (120 measured and 12 excluded) reach their final steady roots.
The independent audit checks all 198 stage endpoints against their declared
1e-8 spatial target; all meet it. Source and shared imported helpers remain
unchanged. These are steady-equation checks, separate from analytical accuracy.

The unchanged 1e-4 profile-agreement gate passes for MUSCL shock70 and both
subsonic controls. MUSCL shock70 is about **5.26 times faster** than its fresh
single-grid control. Subsonic constant and MUSCL take about **4.57** and **1.68**
times the single-grid time, respectively. Shock35 with constant/MUSCL and
shock70 with constant fail profile agreement, so their retained time ratios
are descriptive rather than verified same-root acceleration claims. The overall
audit intentionally returns nonzero for those three failed comparison gates;
there are no root or integrity failures and no relaxed thresholds.

All four shocked case/reconstruction combinations require more fine-grid
updates than the retained four-grid sequence. The new experiment changes both
the hierarchy and coarse tolerance. It does not establish which change causes
the different trajectory. Original four-grid wall times come from an earlier
campaign, so the fresh paired single-grid controls are the basis for the new
performance statements. Coarse/fine endpoint costs, residuals and profile
agreement are shown explicitly in the report.

Optional offline browser verification uses an existing Playwright/Edge environment:

```powershell
python -B development/roberto/multigrid/scripts/audit_html.py
```

This checks all three result pages and theory, light/dark themes, case/reconstruction
and marker controls, exact plotted histories and all twelve embedded downloads.
It copies only the HTML to an isolated directory; browser state and screenshots
remain under `outputs/`.

## Matched physical and numerical setup

Reuse the [steady nozzle verification](../historical/euler_nozzle/README.md) and
`SteadyNozzleProblem`, with its perfect-gas Air backend, symmetric
converging-diverging geometry, 200 kPa/300 K reservoir and back pressures for
shock35, shock70 and the fully subsonic control. Analytical solutions determine
physical boundary data and assess final errors; no analytical profile or imposed
shock station initializes a solve.

The hierarchy is **51, 101, 201, 401 points including two ghost slots**. The
physical unknown counts are 49, 99, 199 and 399; this is the source's nodal
mesh convention, not a hierarchy of 50/100/200/400 agglomerated cells. Each
level samples the same geometry. The single-grid method uses only 401 points.
Constant and primitive Van Leer MUSCL reconstruction are compared separately,
with the same reconstruction on every level within a sequenced solve.

Transfer is bounded linear interpolation of $(p,v_m,h)$ along $x$, followed
by the target factory's existing boundary completion. It is a nonconservative
initialization transfer, not a physical-time remap. Each new mesh resets its
SER/timestep history. The finest stage solves the original target equations;
there is no defect forcing or repeated coarse correction.

Both methods retain the source Städtke FVS with regularized flux, colored AD
Jacobian and banded direct Newton solve, CFL 10, SER exponent 1, maximum CFL
$10^6$, timestep growth bound 1.5 and adaptive increment checks that halve a
rejected timestep. The source baseline performs **one Newton correction per
trial**, including rejected trials. This is implicit pseudo-time iteration;
it does not claim a fully converged backward-Euler timestep or time-accurate
shock startup. The source's eliminated ghost maps are retained identically
for both methods. No new boundary treatment or source modification is introduced.

Coarse stages stop at fixed-scale spatial RMS $10^{-5}$, a declared
initialization tolerance. The finest stage stops at $10^{-8}$. An independent
saved-full-state audit checks the original spatial residual, physical boundary
constraints, EOS admissibility, numerical mass flux and summed integrated
balances. It compares final fields with the same single-grid discrete root
using fixed-scale primitive tolerance $10^{-4}$ and shock-station tolerance
two fine spacings. Analytical accuracy is reported separately, including failures.

## Timing and work accounting

For each case/reconstruction/method, perform one excluded **complete** solve,
then ten measured solves, alternating method order and executing sequentially.
Every single-grid solve resets to its original simple initial field. Every
sequenced solve resets to that same physical initialization rule on the
coarsest mesh; it never reuses a previous repetition's solution.

The outer solve timer includes all levels, interpolation, boundary completion,
rejected trials and common convergence monitoring. JAX work is synchronized.
Problem construction and kernel compilation are recorded separately, while
final diagnostic audits and serialization are outside the solve timer for both
methods. The excluded complete solves remain in the data. Timing statistics
include every measured sample, with failures shown and speedup claims qualified
by convergence and agreement with the same root.

Report all-stage accepted updates, fine-stage updates, rejected trials and
Newton corrections (one per trial). A trial-weighted work proxy is

$$
W_{\mathrm{fine\ equivalent}}=
\sum_{\mathrm{levels}}N_{\mathrm{trials},l}
\frac{N_{\mathrm{points},l}-2}{N_{\mathrm{points},\mathrm{fine}}-2}.
$$

This is a cell-count work proxy, not a model of elapsed time. Coarse steps
are never omitted from total iteration or timing comparisons. Residuals on
different meshes use different source scales; stage boundaries must remain
visible in cross-level histories.

The runner saves before/after hashes for every Python file under `src/`.
Independent audits recompute the hashes, transfer states, residuals and counts.
Experimental verification belongs in these executable study audits, not in
the production pytest suite.
