# Steady EulerNozzle: legacy versus corrected numerics

The [source-change explanation](README.md) describes the four numerical
changes and their mathematical motivation. The [illustrated report](results_steady/REPORT.md)
presents the measured comparison and figures. The existing steady-nozzle example supplies all physical setup
and solver execution; this directory only selects variants and analyzes them.

```powershell
poetry run python -m development.roberto.numerical_fixes.compare_steady --no-show
poetry run python -m development.roberto.numerical_fixes.compare_steady --repeats 3 --no-show
poetry run python -m development.roberto.numerical_fixes.compare_steady --plot-only --no-show
```

Run from the repository root after installing the project dependencies.
The numerical-fix commits provide the required variant-selection API.

The initial study snapshot under `results_steady/` is versioned explicitly
(the general results-directory ignore rule still applies to new outputs): per-variant original
example figures/CSV/JSON/logs, per-case method overlays, aggregate CSV/JSON,
field-difference measurements, and rest/sonic/entropy-switch/limiter probes.
The default run uses one mesh and one repeat; it retains the known sonic-limit
accuracy failure instead of treating it as an execution error.

The driver enables `record_history=True` to retain every accepted implicit
update. `case_*_convergence.csv` files store update number, pseudo-time,
timestep, raw native SSR, stopping SSR, and timestep-halving count. Rejected
trial steps and inner Newton iterations are not plotted. The multi-panel
`convergence_histories.png`/`.pdf` figure compares all variants at each backpressure.
`--plot-only` requires histories from a fresh run of this version.
