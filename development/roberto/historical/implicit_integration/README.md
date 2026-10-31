# Implicit integration: Newton work and CFL sensitivity

Open **[REPORT.html](REPORT.html)**. It is one standalone offline file, with
embedded plots, equations, recorded outcomes and downloadable CSV/JSON data.
Colleagues can copy just this file and open it in a browser. No server,
installation, internet connection or neighboring files are needed for viewing.
Light/dark themes and configurable pressure-marker spacing are included.

The report has four internal pages:

1. **Newton caps**: maximum corrections 1, 3, 5, 10 and 100 at fixed CFL settings.
2. **CFL / cap 100**: nine controller variants with a large Newton cap.
3. **CFL / cap 1**: the same variants with one correction per timestep trial.
4. **Theory**: Euler balances, eigenvalues, spatial and implicit discretization,
   boundary handling, SER control, stopping definitions and saved-data provenance.

Each results page has five-case plots for residual versus updates, total Newton
work or recorded time; actual accepted CFL; actual inner corrections; and final
pressure with the retained analytical reference. Rejected attempts can be
overlaid in the CFL and inner-count views. Numerical curves use plasma colors,
distinct line styles and large hollow pressure markers.

## Retained evidence and interpretation

The complete saved campaign contains **105 unique numerical solves** on the
201-point EulerNozzle mesh, including two ghost slots, with a perfect-gas Air
backend. All 105 recorded original residual checks and baseline-agreement
checks pass. **21 sonic-limit analytical accuracy checks fail** and remain
explicitly labeled. Baseline runs at caps 1 and 100 are shared between the
three pages; repeated display does not make them independent measurements.

Every configuration has **one recorded timed solve**. These are descriptive
times, not repeated timing distributions or validated speedup estimates.
The report distinguishes accepted outer updates, actual inner corrections,
rejected work, root convergence and analytical accuracy. A Newton setting is a
maximum cap, not a forced count or proof of a fully converged transient step.

The report was built from existing results. **No simulations were rerun.**
The new builder checks saved arithmetic and consistency, not the current
numerical source against old states. The original versions, source hashes and
historical `development_new` paths remain intact as provenance.

## Directory organization

```text
implicit_integration/
  README.md                entry point, interpretation and reproduction commands
  REPORT.html              one standalone report with four internal pages
  __init__.py              package marker
  scripts/
    report.py              solver-free standalone HTML builder
    report_data.py         saved-record loader, consistency audit and plot payloads
    audit_html.py          isolated offline browser verification
    compare_settings.py    retained historical numerical sensitivity driver
    report_assets/         editable CSS/JS, theory and retained setup note
  outputs/                 ignored numerical records, figures and generated audits
```

The builder reuses the locally vendored Plotly/MathJax libraries and base visual
styles from `BC_residual/scripts/report_assets/`; it embeds their contents and
licenses into the HTML. The finished report has no dependency on that directory.
Original Markdown reports and PNG/PDF exports remain in `outputs/` as retained
artifacts; the HTML is the human-facing entry point.

## Rebuild only the report

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.historical.implicit_integration.scripts.report
```

This reads `outputs/summary.json` and the saved run JSON files, then replaces
root `REPORT.html`. It imports no numerical solver, rewrites no historical
summary or timing sample, and runs no cases. It creates:

- `outputs/saved_results_audit.json`: counts, histories, cap/inner flags, root
  flags, field shapes, common seeds and baseline-distance consistency.
- `outputs/report_structure_audit.json`: internal links, embedded dependencies
  and downloadable payload checks.
- `outputs/html_report_manifest.json`: report/asset hashes and verification that
  all 422 saved numerical JSON/CSV files remain unchanged for the default data.

Select another saved run directory with `--output`; use `--destination` for an
alternative HTML filename directly at this study root. Generated HTML is not
committed unless requested.

## Verify the standalone HTML in a browser

With Microsoft Edge installed:

```powershell
poetry run python -B -m development.roberto.historical.implicit_integration.scripts.audit_html
```

Use `--edge <executable>` if Edge is installed elsewhere. This audit uses a
separate study-local browser profile and a solitary report copy, disables
network resolution, and needs no Playwright installation. It checks all four
pages, both themes, exact saved plot arrays, marker spacing and invalid inputs,
equation rendering, embedded downloads and JavaScript errors. Browser evidence
and screenshots remain under `outputs/html_audit/`; no numerical solver runs.

## Numerical-driver scope

`compare_settings.py` is preserved without changing its numerical algorithm.
It uses the common nozzle problem, implicit controller and an in-memory
counter adapter from `newton_solve`. That adapter targets the historical source
layout and needs a compatibility review before launching a new campaign against
the evolved current solver. This reporting change does not migrate it or claim
fresh numerical reproducibility. Use `scripts.report` for presentation updates,
rather than the historical driver or its older Markdown/plot-only path.

The original detailed setup is preserved in
[`scripts/report_assets/historical_setup.md`](scripts/report_assets/historical_setup.md)
and embedded in the report as an expandable historical record. Editable current
theory for the report lives in
[`scripts/report_assets/theory.md`](scripts/report_assets/theory.md).
