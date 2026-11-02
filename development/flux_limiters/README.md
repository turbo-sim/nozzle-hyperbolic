# Generic MUSCL limiter comparison

Open [REPORT.html](REPORT.html) for the standalone Plotly explorer. The report
now follows the shared flux-study design: Overview, Case explorer,
Interpretation & model, and Protocol & data, with light/dark themes and
keyboard navigation. Select the mesh, limiters, profile range and residual
measure. The single physical case is explicit; no additional operating points
were simulated.

The modernization reads the existing eight runs only. All nine campaign JSON
inputs retain their pre-edit SHA-256 hashes. Selector, plotted-data, packaging
and theme audits passed. Actual browser preview was unavailable because the
browser access host could not be reached; the application audit uses an
instrumented DOM/Plotly interface and does not assert browser layout.

This study compares Van Leer, Minmod, Superbee and sign-preserving Van Albada
on the shared `euler_nozzle/shock_35` perfect-gas case. It uses the packaged
geometry and analytical solution, including conservative cell projections.
Every method and grid starts from the model's cold initialization. The study
records convergence, physical budgets, pressure/conserved errors and shock
location independently. It is not a timing benchmark.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -B -m development.flux_limiters.scripts.report
.venv/Scripts/python.exe -B -m development.flux_limiters.scripts.audit_report
```

These presentation commands do not import a flow model or rerun cases.
For a separately requested future simulation campaign, use
`python -B -m development.flux_limiters.scripts.compare --cells 24 48 --output <fresh-directory>`.
`compare --report-only` delegates to the same Plotly renderer.
Use a fresh `--output` directory for a new numerical attempt. Individual JSON files
preserve failures, settings, final states and accepted/rejected histories.
`outputs/comparison/RESULTS.md`, PNG and PDF plots summarize the run.
`REPORT.html` contains interactive Plotly profiles, residual/CFL trajectories,
work and error plots, original data downloads and exact input hashes. Existing
PNG/PDF exports and numerical tables remain available. Regenerating the report
does not run flow solves.
The checked two-grid results are summarized in [RESULTS.md](RESULTS.md).

The controlled settings are corrected Stadtke flux, backward Euler with
backtracking, no additional filter, default fixed residual thresholds and the
same CFL controller. A failed limiter case remains a failure; it is not warmed
up with a different flux or limiter. These small grids establish a reproducible
comparison and reveal robustness differences; they do not certify an
asymptotic convergence rate for a discontinuous solution.

The reference Andrea branch is
`bcfe6ae1ec8abd71528e57b218ac8291dc9461fa`. Its active JAX Superbee used a signed
maximum, which selects the wrong candidate for negative gradients. The generic
implementation selects maximum magnitude then restores the common sign.
Minmod adds a more diffusive option; Superbee adds a more compressive one.
Van Albada provides a less compressive rational alternative to Van Leer. The
selected Van Albada explicitly sets opposite-sign slopes to zero and avoids
dividing by zero in inactive JAX branches; it differs intentionally from the
retained ungated historical NumPy helper. None of these choices is globally
smooth at the origin, a general coupled-system positivity theorem, or a
guarantee of faster Newton convergence.

Presentation code lives in `scripts/report.py` and `scripts/report_assets/app.js`.
The offline shell, theme and data audits are shared with the flux-method
reports. SHA-256 manifests and application/structural checks live in
`outputs/report_presentation/`; the prior static report is retained there.
The numerical `run()` function in `compare.py` is unchanged.

Generated numerical output remains ignored by Git. The HTML downloads preserve
all plotted data and original JSONs. To rebuild on a fresh checkout, restore the
ZIP's `original/` contents under this study and its `input_manifest.json` under
`outputs/report_presentation/`. The renderer source is also included.
