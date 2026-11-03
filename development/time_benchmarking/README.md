# Solver time benchmarking

This study compares production solver costs on six steady Euler cases, with
constant and MUSCL Van Leer reconstruction and a 400-cell finest grid:

1. Vanilla Stadtke, single grid.
2. Vanilla Stadtke, 50→100→200→400 sequencing.
3. Modified Stadtke, single grid.
4. Modified Stadtke, 50→100→200→400 sequencing.
5. HLL, 50→100→200→400 sequencing.
6. Rusanov, 50→100→200→400 sequencing.

Modified Stadtke enables both custom jump and endpoint-envelope corrections;
vanilla disables both. Numerical settings are in [settings.yaml](settings.yaml).
Production sequencing performs nested-grid initialization. Successful numerical
states are prolonged by componentwise linear interpolation.

## Completed campaign and validation

The `cfl_carryover_20261103` campaign completed 72 configurations, 468 measured
complete solves and 72 excluded complete solves. All configurations converge
on the intended branch, and all repeated final states are identical. Original
analytical targets pass in 65/72 configurations. The seven misses are constant
Rusanov nozzle shock35 and all six constant annular-expansion configurations.

All twelve modified Stadtke single-grid/sequence pairs agree within the
separate comparison threshold of $10^{-6}$ of fixed field scales. Vanilla
Stadtke disagrees locally for annular shock35 MUSCL and shock70 constant;
these timing ratios remain explicitly descriptive. See [RESULTS.md](RESULTS.md)
and [REPORT.html](REPORT.html) for measured durations, fields and checks.

The raw-data audit verified all 468 measurement hashes, exact source and case
archives, original residual checks, 72 component timing blocks and 1,620 CFL
stage records across measured and excluded solves. The historical audit also
verified that all 489 original measurement artifacts survived the folder rename.
All 14 production grid-sequence tests passed. Report structural and application
checks are recorded under `presentation/`. Actual browser preview was unavailable
because the native/browser access hosts could not be reached.

## CFL carryover

The campaign explicitly passes `carry_cfl: true` to the production grid-sequence
driver. A successful solve forwards its last **accepted** CFL to the next grid,
capped at `cfl_max`. The timestep is recomputed for the new grid. A zero-update
success forwards its initial CFL. Cold starts, failed preceding stages and cold
retries use the configured initial CFL of 0.2. Each grid starts fresh residual
normalization. Complete hierarchies reset numerical/controller state on every
repetition; no CFL or solution is reused between measured complete solves.

Raw and compact stage records retain the CFL policy, initial CFL, last accepted
CFL, fresh residual-normalization policy and failed prolonged attempts. The audit
checks these against actual histories for every measured and excluded solve.

## Physical cases and timing protocol

The packaged cases are nozzle `shock_35`, `shock_70`, `adapted`, and annular
`shock35`, `shock70`, `supersonic_expansion`. They use model-owned cold initial
guesses, independent analytical references and their original accuracy targets.
Convergence, physical branch, analytical accuracy and same-flux final-field
agreement are separate checks. No analytical initialization or tolerance
relaxation is used.

Each configuration receives one excluded complete solve, then ten measured
nozzle solves or three measured annular solves in rotating method order.
This gives 72 configurations, 468 measured complete solves and 72 excluded
complete solves. Production operators and samplers are prebound and reused.
Import, setup, JIT, references, field extraction and serialization are outside
timers. Every grid solve, prolongation, rejected trial, cold retry and equivalent
original equation/boundary/global monitoring is inside. Optional scalar monitors
are disabled equally. The backend is JAX CPU with float64 and one BLAS thread.

At each method's last measured fine-grid state, twenty synchronized samples
measure residual evaluation, Jacobian blocks, a preassembled banded linear solve,
a Newton direction and a fixed block of twenty checked Newton corrections.
Three warm calls are excluded. The correction block uses zero early stopping
tolerances and $\Delta t=0.2\Delta t_{\mathrm{wave}}$, retaining admissibility
checks. Different methods can have different final states; kernel cost is not
an accuracy comparison.

Read-only process probes check for known competing Python research/test jobs
before and after each block. Source, harness and machine fingerprints must stay
fixed. Exact measured source and cases are archived, including any pre-existing
uncommitted source edits; `git_status` records the workspace state. Unrelated
workspace edits are not part of this study's commit.

## Run and reproduce

From the repository root, use the existing Poetry project environment:

```powershell
.venv/Scripts/python.exe -B -m development.time_benchmarking.scripts.run
.venv/Scripts/python.exe -B -m development.time_benchmarking.scripts.run --resume
.venv/Scripts/python.exe -B -m development.time_benchmarking.scripts.report
.venv/Scripts/python.exe -B -m development.time_benchmarking.scripts.audit
```

The default campaign is `outputs/cfl_carryover_20261103/`. A fresh `--output`
starts another campaign. `--cases`, `--reconstructions`, `--methods`, `--repeats`
and `--kernel-repeats` support separate smoke runs. Resume requires matching
source, harness, numerical protocol and machine. Interrupted blocks rerun in
fresh attempt folders; partial measurements remain preserved and are not mixed.
A complete solve has a 1200 s timeout. Hard failures are retained.

[REPORT.html](REPORT.html) and [RESULTS.md](RESULTS.md) rebuild from saved
measurements without importing a flow solver or advancing simulations. Both
Stadtke variants have explicit single-grid/sequence timing and field comparisons.
The standalone HTML embeds Plotly, themes, timing samples, representative fields
and histories, CSV and exact source/results downloads. Copying the HTML alone
is sufficient for offline viewing. Full per-repeat histories remain in `raw/`.
On a fresh checkout, extract `summary.json` from the HTML's results download
and its separate `source_and_cases.zip` into the default campaign directory to
rebuild the presentation. Full numerical audits also need the retained `raw/`
directory or a fresh campaign; raw per-repeat files are not embedded in HTML.

Run the application audit through `scripts/audit_interactions.mjs` in a Node
runtime; it exports `auditInteractions(html)`. This checks every method, physical
case, reconstruction, operation, sample mapping, profile, selector, navigation
and theme. Actual browser layout is checked separately. Generated numerical
outputs remain ignored by Git; the existing standalone HTML and generated
Markdown analysis are maintained alongside their generators.

## Historical campaign

The folder was renamed from `development/benchmark_production/` to
`development/time_benchmarking/`. The original `outputs/initial_20261102/`
campaign and its exact source archive are preserved. Its five methods and
reset-CFL policy are historical evidence; they are not merged into the new
six-method carryover campaign. The original report is retained as
`outputs/initial_20261102/presentation/pre_cfl_carryover_REPORT.html`.
The old measurement-time paths inside archived downloads remain unchanged.

The validated smoke campaign is `outputs/smoke_cfl_stable_20261103/`: adapted nozzle,
constant reconstruction, all six methods, one measured repetition and two
component samples. Smoke timings are excluded from the main campaign.
The earlier `outputs/smoke_cfl_20261103/` attempt is retained as ineligible:
production source files changed during its measurement block.
