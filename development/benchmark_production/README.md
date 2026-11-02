# Production solver benchmark

The report now uses the same visual system as the flux-method studies, with
embedded Plotly, light/dark themes, six explicit physical-case cards and shared
selectors for model, operating case, reconstruction, methods and operation.
Overview shows the full campaign. Timing explorer separates fixed-work cost,
complete-solve duration, individual samples and hierarchy costs. Fields &
convergence shows the selected actual profiles, original residual histories,
CFL and analytical errors. Interpretation follows the current selection.

This presentation was rebuilt from saved results only. The 489 measurement
artifacts (summary, source archive, validation and raw runs/logs) are checked
against their pre-edit SHA-256 inventory. No timing or numerical solution was
rerun. The application audit covers all twelve case/reconstruction selections,
all five operations, exact plotted samples, method filters, keyboard navigation
and themes. Browser preview was unavailable because the browser host could not
be reached; this is separate from the structural and application checks.

The first campaign is complete: 60 configurations, 390 measured complete solves
and 60 excluded complete hierarchies. All configurations converge on the
intended branch, with 54 passing the original analytical targets. All repeats
within a configuration have identical final states, and there are no cold
retries. See [REPORT.html](REPORT.html) or [RESULTS.md](RESULTS.md).

The six original accuracy misses are constant Rusanov nozzle shock35 (six-cell
shock error, target two) and all five constant annular expansion variants
(conserved normalized L1 of 0.001615–0.003220, target 0.001). The annular
shock35 MUSCL single-grid and sequenced solutions both pass their original
checks but differ locally near the shock; their total timing ratio is flagged
as descriptive rather than a verified time-to-identical-state speedup.

This campaign compares five production `src/` configurations on 400 cells,
with constant and MUSCL Van Leer reconstruction:

1. Vanilla Stadtke, single grid.
2. Vanilla Stadtke, 50→100→200→400 grid sequencing.
3. Modified Stadtke (both custom corrections), the same hierarchy.
4. HLL, the same hierarchy.
5. Rusanov, the same hierarchy.

Production's multigrid driver performs nested-grid initialization, not FAS or
defect-correction cycles. A hierarchy starts cold on 50 cells and interpolates
successful numerical solutions. Each grid uses fresh residual normalization
and CFL control. The production driver's failed prolonged attempts and cold
retries are retained and charged to the hierarchy total.

The six physical cases match `benchmark_legacy`: nozzle `shock_35`, `shock_70`,
`adapted`; annular `shock35`, `shock70`, `supersonic_expansion`. Independent
analytical references and original accuracy targets come from the packaged
verification catalog. Numerical convergence, branch and accuracy are reported
separately. No analytical initialization or tolerance relaxation is used.

The optimized local derivatives are now unconditional production behavior.
`Scheme` no longer accepts a `local_derivative_reuse` input. Structural coloring
remains outside production.

## Timing protocol

Each case/reconstruction has reusable production operator and sampler bindings.
Each method receives one excluded complete hierarchy, followed by ten measured
nozzle solves or the previously agreed three annular solves, with rotating
method order. Initial/controller state resets every repetition. Import,
binding, compilation, reference construction, field extraction and serialization
are outside timers. Equation/boundary/global monitoring and rejected work are
inside. Optional scalar quantity monitors are disabled equally for every method.
Total hierarchy time, coarse and fine time, actual accepted steps, Newton work,
failures and cold retries are retained individually.

At each method's last measured 400-cell state, twenty synchronized samples
measure residual, Jacobian blocks, preassembled banded linear solve, Newton
direction, and twenty checked Newton corrections. Three warm calls are excluded.
The fixed-work block uses zero early stopping tolerances, retains admissibility
checks and uses $\Delta t=0.2\Delta t_{\mathrm{wave}}$. Physical states and fluxes
can differ; these diagnostics separate implementation cost from iteration count.

Process-inventory probes occur before and after each block. They detect known
competing Python research/test workloads, not every source of operating-system
noise. Source and numerical-runtime fingerprints must stay unchanged. Every
raw solve, including excluded attempts, is saved immediately. Interrupted blocks
are rerun into a new folder without mixing prior partial measurements.

## Run and reproduce

From the repository root, use the existing project environment:

```powershell
.venv/Scripts/python.exe -B -m development.benchmark_production.scripts.run --output development/benchmark_production/outputs/revision_2
.venv/Scripts/python.exe -B -m development.benchmark_production.scripts.run --output development/benchmark_production/outputs/revision_2 --resume
.venv/Scripts/python.exe -B -m development.benchmark_production.scripts.report
.venv/Scripts/python.exe -B -m development.benchmark_production.scripts.audit
```

The numerical settings and exact representative cases are in [settings.yaml](settings.yaml).
Use `--output` for a fresh campaign; `--cases`, `--reconstructions`, `--methods`,
`--repeats`, `--kernel-repeats` support separate smoke campaigns. Resume requires
matching source, runtime, machine and protocol. A complete solve has a 1200 s
timeout; hard failures are preserved and other methods continue.

The older shared benchmark directories have since been retired in a concurrent
workspace cleanup. Their serialization, physical assessment, worker protocol
and environment helper bodies are retained locally, unchanged, in
`scripts/runtime_support.py` and `scripts/environment.py`; the solver worker and
settings are unchanged. This compatibility relocation was not timed. The saved
source archive retains the original measurement-time runtime and paths. A
future campaign uses the new runtime identity and a fresh `--output`; it must
not resume or merge into the historical campaign.

The first smoke campaign in `outputs/smoke_20261102/` covers the adapted nozzle,
constant reconstruction, all five methods, one measured repetition and two
kernel samples. It is excluded from the main campaign statistics.

The public-input removal passed 87 focused production tests (local derivative
identities, generic Jacobians, model bindings and boundary treatments). The
independent dense/direct-AD reference is constructed only inside tests.
Saved-data validation also confirms that all twelve vanilla single-grid final
states are bitwise identical to the prior production campaign with reuse
enabled. This evidence is in `outputs/initial_20261102/api_removal_validation.json`
and embedded in the HTML results download.

[REPORT.html](REPORT.html) and [RESULTS.md](RESULTS.md) rebuild from saved data.
The standalone HTML embeds timing samples, representative fields/histories,
interactive Plotly figures, CSV, results, renderer assets and exact measured
source/runtime downloads. Full per-repeat hierarchy histories/states remain
under the campaign's `raw/` directory. The prior static report and PNG/PDF
exports are preserved in `outputs/initial_20261102/presentation/previous_REPORT.html`
and `figures/`. Presentation rebuilds never import a flow solver.

The renderer is `scripts/report.py`, with saved-data calculations in
`scripts/report_data.py` and HTML/CSS/JavaScript in `scripts/report_assets/`.
It embeds the existing vendored Plotly and shared flux-report theme assets.
`scripts/audit_interactions.mjs` exports `auditInteractions(html)` for a Node
runtime; it executes the embedded application against an instrumented DOM and
checks values against the saved payload. Actual browser rendering remains a
separate check. Audit records live under the campaign's `presentation/` directory.

Generated numerical output remains ignored by Git. The committed standalone
HTML carries the complete measurement summary in its results ZIP and the exact
measured source in a separate ZIP. On a fresh checkout, download/extract those
into `outputs/initial_20261102/` to rebuild the presentation. Full per-repeat
raw-file audits additionally require the retained `raw/` directory; it is not
included in the compact HTML download.
