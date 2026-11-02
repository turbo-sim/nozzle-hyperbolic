# CFL carryover during coarse-to-fine initialization

Small timing comparison on the packaged Euler nozzle `shock_35` and `shock_70`
cases, using HLL, constant reconstruction and MUSCL Van Leer. Every complete
hierarchy uses 50, 100, 200 and 400 cells. The two policies are:

- **Reset:** start every mesh at CFL 0.2.
- **Carry:** on a successful warm transfer, start at the preceding mesh's last
  accepted CFL, capped at the common maximum. Carry CFL, not absolute timestep.
  The first mesh and any cold retry start at CFL 0.2.

The experiment wraps the production grid-sequence driver inside this study;
production files and defaults were unchanged for the recorded measurement.
All other settings match: backward
Euler, coupled traces, colored AD/banded solve, Newton backtracking, CFL growth
1.2, shrink 0.5 and cap 1e6. Original per-equation, boundary and global residual
checks remain active, with fresh relative baselines on each mesh/attempt.

Run from the repository root:

```powershell
poetry run python -B -m development.roberto.cfl_carryover.scripts.run
poetry run python -B -m development.roberto.cfl_carryover.scripts.report
poetry run python -B -m development.roberto.cfl_carryover.scripts.audit_report
poetry run python -B -m development.roberto.cfl_carryover.scripts.audit
```

There is one excluded complete warmup per policy/configuration, followed by
exactly three measured hierarchies per policy/configuration. Policy order
alternates. Kernels/models/samplers are bound once and reused; each hierarchy
resets the initial field, controller and history. Measurements run sequentially.
Timers include interpolation, monitoring and failed warm/cold-retry work, and
exclude model/kernel binding, serialization, final independent audits and plots.
JAX work is synchronized before stopping timers. Warmup/setup costs are retained.

`outputs/comparison/` preserves every attempt, final fields, complete histories,
settings, source hashes and environmental identity. A fresh output directory is
required for a new campaign. Report regeneration uses saved results only.
The standalone [HTML report](REPORT.html) embeds interactive plots and downloadable evidence;
[RESULTS.md](RESULTS.md) provides a compact summary. Three samples measure only
this machine/session and do not establish a general speedup.

The recorded campaign passed all 128 saved mesh-stage audits, including the
excluded warmups. All 24 timed hierarchies converged, and each comparison's
fine-grid solutions agree within 1e-6 in fixed variable scales. The report's
navigation, themes, plotted arrays and embedded downloads are checked separately
from the numerical audits. The report uses the shared flux-study design with
Overview, Case explorer, Interpretation and Protocol pages. Select either shock,
reconstruction, any of the four meshes, and an actual median-duration run or
numbered repetition. Explore pressure, velocity, residual components, rejected
steps, nonlinear work, and the CFL entering and leaving each grid. Complete
timing samples retain their original hierarchy boundary when a mesh is selected.

The presentation is rebuilt from saved results only. It embeds Plotly, its license,
all plotted data, and a ZIP containing original evidence and report sources; it
opens offline when copied alone. SHA-256 checks preserve all 38 numerical/source
inputs and the original results narrative. Appending `?audit=1` to the local HTML
URL exercises browser interactions and displays their result. Structural and
application checks, desktop/mobile browser checks and screenshots live in
`outputs/report_presentation/`; the saved numerical audit remains in
`outputs/comparison/audit.json`.

The production grid driver subsequently adopted `carry_cfl=True` as its default.
Use `carry_cfl=False` there for the original reset policy. This study explicitly
disables that automatic carryover because its wrapper controls both policies.
The recorded numerical source is preserved in
`outputs/comparison/source_at_measurement.zip`; audits verify it against the
measurement-time hashes and list subsequent source changes. Saved timings are
not relabeled as measurements of the updated driver.

A separate production-API check on shock 35 / MUSCL, using the same four grids,
reproduced both saved final states exactly and matched the work counts: 673
Newton corrections with the new default and 1156 with `carry_cfl=False`.
This is a correctness check, not a new timing campaign; its record is
`outputs/comparison/production_default_check.json`.
