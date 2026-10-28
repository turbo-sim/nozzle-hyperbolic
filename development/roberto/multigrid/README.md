# Coarse-to-fine implicit EulerNozzle demo

This study compares a cold single-grid solve with coarse-to-fine initialization
for **EulerNozzle**, using the unchanged production implicit timestep. It does
not implement FAS. The earlier [annular design assessment](MULTIGRID_NOTES.md)
remains a proposal for a different model; its coupled boundary formulation is
not silently substituted into this EulerNozzle benchmark.

## Generate results and reports

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

## Matched physical and numerical setup

Reuse the [steady nozzle verification](../euler_nozzle/README.md) and
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
