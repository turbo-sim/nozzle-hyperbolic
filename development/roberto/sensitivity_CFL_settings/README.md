# Newton-cap and CFL sensitivity

The numerical runner uses the current shared FVM driver and
[packaged nozzle cases](../../../src/nozzle_hyperbolic/verification/README.md).
It imports no retired verification example, monkey-patched factory or removed
Newton-study controller.

## Run current comparisons

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.compare_settings --smoke --points 10
poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.compare_settings --no-show
poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.compare_settings --plot-only
poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.audit_results
poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.report --output development/roberto/sensitivity_CFL_settings/outputs/shared_fvm
```

Use a fresh `--output` for each campaign. Defaults write to `outputs/shared_fvm/`.
`--points` includes two algebraic endpoint traces; the remaining points are
physical finite-volume cells. `--cases`, `--caps` and `--max-iterations` select
bounded comparisons. `--smoke` runs the subsonic case at Newton caps 1 and 10
with the baseline CFL policy. Failures remain saved and produce a nonzero exit.

The full campaign compares five physical cases and Newton caps 1, 3, 5, 10 and
100. Caps 1 and 100 also compare nine explicit CFL policies. All runs for one
physical case start from exactly the same cold state. Warm-ups are timed
separately; measured solves reuse immutable workspaces and reset their
controller and convergence baselines. No analytical field initializes a solve.

Current production control multiplies the CFL after accepted steps and shrinks
it after rejected trials. The baseline is 0.2 / growth 1.2 / maximum 1e6.
Variants change the initial CFL, maximum CFL or growth multiplier. The removed
SER exponent is not supported or silently emulated. The current finite-volume
mesh, corrected Stadtke flux, automatic acoustic outlet, strict Newton solve
and independent equation/boundary/global stopping checks intentionally replace
the earlier factory-based experiment. Its numerical results are not comparable
as unchanged-algorithm timings.

Each run saves the returned state, complete histories, actual Newton counts,
rejected work, final defects and independent analytical errors. The saved-state
audit independently recomputes defects and verifies caps and work counts. `summary.json`
and `results.csv` index all attempts; each campaign also gets a standalone
`REPORT.html` with case/comparison selectors, pressure, residual, CFL and work
plots. Rejected trials appear in the CFL and work plots. Report rebuilding
imports no solver. Timings are individual observations, not distributions.

## Existing saved presentation

The root [REPORT.html](REPORT.html) describes the older nodal campaign.
To rebuild only that saved presentation, use:

```powershell
poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.report
```

The builder reads the old records without evaluating the current solver.
Current-schema inputs are routed to `current_report.py` and rendered within
the selected campaign. Shared styles, Plotly/MathJax runtimes and licenses come
from the retained `sensitivity_backpressure/scripts/report_assets/` directory.
`audit_html.py` checks the existing standalone presentation in an isolated
browser profile.
