# Local pseudo-time stepping for steady HLL nozzle solves

> **Assessment: limited and inconsistent benefit; not recommended as a production default.**
> The results were not compelling enough to justify adopting this method.
> Constant reconstruction showed little consistent acceleration. MUSCL gave
> modest gains, but three repetitions and noticeable timing scatter do not
> establish a broadly reliable improvement. This study is retained as a
> reproducible experimental result; global stepping with CFL carryover remains
> the recommended production approach for these cases.

Compare the production global pseudo-time step against a development-only local
step on the packaged Euler nozzle `shock_35` and `shock_70` cases. Both constant
and MUSCL Van Leer reconstruction use HLL and safeguarded backward Euler.
Two solve strategies are compared independently:

- Grid sequencing: 50 → 100 → 200 → 400 cells, carrying accepted CFL between
  successful grids with the production default.
- Single grid: a cold model-owned initial state on 400 cells.

Each strategy/method/configuration has one excluded complete warmup followed
by three measured solves. Method order alternates; timings run sequentially.
Kernels, models and samplers are bound once and reused. Every complete solve
resets the numerical state, controller and history. No analytical state or
previous measured solution initializes a solve.

## Experimental update

For each physical cell, freeze its wave time at the old outer iterate:

$$
b_i=\frac{\Delta x_i}{\max(s_i,10^{-300})},\qquad
\Delta\tau_i=\mathrm{CFL}\,b_i.
$$

Here $s_i$ is the physical model's wave-speed callback. Endpoint cells also
respect the same model-owned boundary-speed bound as the global driver. The
global method uses $\Delta\tau=\mathrm{CFL}\min_i b_i$. The local adapter
passes the vector of cell steps into the unchanged production implicit kernel,
which broadcasts its storage/spatial row weights cellwise:

$$
\frac{V_i[Q(W_i^{n+1})-Q(W_i^n)]+\Delta\tau_iR_i(W^{n+1})}
{\max(1,\Delta\tau_i/t_{\mathrm{ref}})V_iQ_{\mathrm{ref}}}=0.
$$

The local steps are fixed throughout each inner Newton solve; they are not
differentiated with the trial state. Boundary equations retain zero storage
and their original definition. This experiment supports only the coupled,
square conservative, unfiltered implicit configuration being measured. It is
steady pseudo-time continuation, not a physical-time integration algorithm.
Final success is checked against the original unweighted spatial equations,
physical trace equations, conservation budgets and branch audit.

All CFL, Newton, tolerance, EOS, geometry and boundary settings match between
methods. The local method includes its extra wave-time evaluation in the timed
work. The timestep in history is the minimum/global reference step; the local
cell steps follow the formula above. Final cell-step ratios are retained.

## Reproduce

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.local_timestep.scripts.checks
poetry run python -B -m development.roberto.local_timestep.scripts.run
poetry run python -B -m development.roberto.local_timestep.scripts.report
poetry run python -B -m development.roberto.local_timestep.scripts.audit_report
poetry run python -B -m development.roberto.local_timestep.scripts.audit
```

A new campaign needs a fresh `--output` directory. Numerical records, source
hashes, environment/settings, warmup costs, complete accepted/rejected histories,
failed attempts and actual final states are retained under `outputs/`. Setup,
final independent audits, serialization and plotting are outside solve timers;
interpolation, monitoring, rejected work and cold retries are included. Device
work is synchronized before timers stop. No `src/` files are changed; source
hashes are verified before and after the campaign. Presentation regenerates
from saved records without advancing any simulations.

See [RESULTS.md](RESULTS.md) and the standalone [HTML report](REPORT.html), which
uses the shared flux-method report theme and offline Plotly explorer. Select
case, strategy, reconstruction, mesh stage and actual measured run; inspect
fields, equation/boundary/global residuals, CFL, nonlinear work and cell-step
multipliers. Automatic/light/dark themes, shock zoom and empty method selections
are supported. Downloads include plotted JSON and a ZIP with every original
input, source snapshot, renderer and license. Three samples give a small comparison
for this machine/session, not a universal performance claim.

The saved source snapshot and the runner's end-of-campaign hash check establish
the measurement-time numerical version. The audit reports any later working-tree
source changes separately; later edits do not relabel these saved timings as
measurements of a different version.

The recorded campaign completed all 48 measured solves and 16 excluded warmups,
with all 160 mesh-stage audits passing. All methods and strategies agree within
4.8e-7 in fixed variable scales at 400 cells. Equation/Jacobian checks passed for
both reconstructions. The HTML restyling used saved evidence only; 72 original
input files are protected by SHA-256. Presentation checks exercise 80 exact
case/reconstruction/strategy/mesh/run combinations and all residual views.
Navigation, selectors, themes, Plotly rendering/zoom and embedded downloads
were also checked in headless Chrome; appending `?audit=1` to the local HTML
URL exercises browser checks. Presentation QA records live under
`outputs/report_presentation/`; the numerical audit remains under
`outputs/comparison/audit.json`.
