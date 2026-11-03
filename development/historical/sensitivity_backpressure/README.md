# Nozzle backpressure sensitivity

New sweeps use `NozzleModel`, `SteadyWorkspace` and `solve_single_grid` from the
current shared FVM engine. Physical geometry, reservoir data and the perfect-gas
backend come from the [packaged nozzle cases](../../../src/nozzle_hyperbolic/verification/README.md).
There is no dependency on the retired `euler_nozzle` or `euler_annular` studies.

## Current numerical driver

Run from the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.historical.sensitivity_backpressure.scripts.run_sweep --smoke
poetry run python -B -m development.historical.sensitivity_backpressure.scripts.run_sweep --ratios 1.0 0.97 --n-points 10 --output development/historical/sensitivity_backpressure/outputs/check
poetry run python -B -m development.historical.sensitivity_backpressure.scripts.audit_results
poetry run python -B -m development.historical.sensitivity_backpressure.scripts.run_sweep --plot-only
```

Default numerical output is `outputs/shared_fvm/`; select a fresh `--output`
for another execution. `--n-points` counts physical cells plus two endpoint
traces. The current mesh uses finite-volume centers and integrated smooth
geometry, rather than the former uniform nodal arrangement.

The driver selects MUSCL Van Leer, corrected Stadtke flux, coupled backward
Euler and the production automatic acoustic outlet. Each backpressure binds an
immutable model and workspace; compilation occurs in a separately recorded
warm-up before the measured solve. Successful descending solutions seed the
next pressure. Failed states are retained and do not replace the last successful
seed. Explicit `--ratios` runs only those pressures; the normal complete sweep
also attempts independent cold and supersonic-restart branch probes.

The current CFL controller starts at 0.2, grows by 1.2 after acceptance, shrinks
after rejected trials and caps at 1e6. Every transported equation, boundary
trace and global discrete balance must independently meet an absolute 1e-8
target. Inner Newton convergence remains required. These choices intentionally
replace the retired factory/controller; old timings and fields are not current
solver results.

Each new campaign writes full returned fields, accepted/rejected histories,
metrics, source fingerprints, an operating-map CSV, PNG/PDF figures and
`REPORT.md`. The saved-field audit accepts the current schema only and checks
the actual current spatial residual and numerical face mass flux without
advancing new timesteps. Analytical accuracy remains separate from convergence.

## Existing standalone presentation

[REPORT.html](REPORT.html) displays the earlier recorded sensitivity campaign.
It is self-contained. Rebuild that presentation without a solve using:

```powershell
poetry run python -B -m development.historical.sensitivity_backpressure.scripts.build_html_report
```

That builder checks its original dataset fingerprint and does not relabel new
shared-FVM data as the old campaign. Plotly/MathJax runtimes and licenses live
locally in `scripts/report_assets/vendor/`. `audit_html_report` checks the
standalone presentation using an isolated browser profile.
