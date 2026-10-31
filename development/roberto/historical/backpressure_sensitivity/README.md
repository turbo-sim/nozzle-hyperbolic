# Nozzle backpressure sensitivity

Open **[REPORT.html](REPORT.html)**. One standalone HTML file contains the
existing scientific report and an internal **Theory & background** tab, with
light/dark themes and the same navigation style as the boundary-condition studies.
It works offline when copied alone.

The organization and presentation have changed; the content and results have
not. All 104 recorded states, 88 analytical profiles, blue color palette,
original audit, numerical findings and provenance are preserved. No solver or
analytical-reference calculation was rerun.

## Report contents

- **Results:** the existing setup, analytical regimes, operating map, critical
  transitions, field distributions, branch comparison, residual histories,
  conservation checks, operating-point table and reproducibility information.
- **Theory & background:** the existing physical-regime and boundary-condition
  notes, with typeset equations. Original setup/execution documentation and the
  detailed Markdown results record are included as expandable background.

All existing interactions remain: pressure/startup selectors, the animation
slider and play/pause controls, ghost and alternative-state overlays, history
region selection, table filters, zoom/reset and embedded CSV/JSON downloads.
The original report wording and numerical values remain the basis of these views.

## Directory organization

```text
backpressure_sensitivity/
  README.md                  contents, organization and reproduction commands
  REPORT.html                complete offline report with Results and Theory tabs
  __init__.py                Python package marker
  scripts/                   unchanged numerical study and presentation code
    report_assets/           editable result template, themes and browser code
      notes/                 original physical notes, setup and results documents
      vendor/                local Plotly/MathJax runtimes and licenses
  outputs/                   ignored numerical records, figures and browser audits
```

Human-facing files live at the study root. The earlier Markdown files are
preserved under `scripts/report_assets/notes/`; their content is in the HTML.
The superseded HTML is retained only as an ignored presentation archive.

| Script | Purpose |
|---|---|
| `run_sweep.py` | Numerical backpressure sweep, continuation and diagnostics |
| `reference.py` | Existing perfect-gas analytical operating map |
| `audit_results.py` | Independent saved-field residual/reference audit |
| `plot_results.py` | Optional static PNG/PDF export and detailed output record |
| `build_html_report.py` | Repackage the unchanged dataset into the standalone HTML |
| `markdown_notes.py` | Render existing prose/equations into the Theory tab |
| `audit_html_report.py` | Offline desktop/laptop browser and interaction checks |

## Rebuild the HTML without solving

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.historical.backpressure_sensitivity.scripts.build_html_report
```

The builder checks the existing dataset fingerprint and uses the exact original
embedded analytical profiles and colors. It reads `outputs/baseline/report_payload.json`
or recovers that payload from an existing `REPORT.html`; it does not recompute
analytical curves. Saved `summary.json`, `audit.json` and `operating_map.csv`
are checked and left unchanged. Plotly and MathJax are vendored locally.

The original dataset guard remains: another numerical dataset requires an
explicit review of the existing narrative before publication. `--input` and
`--output` select study-local paths; `--plotly-js` can select a local runtime.

## Existing numerical and audit commands

```powershell
poetry run python -B -m development.roberto.historical.backpressure_sensitivity.scripts.run_sweep
poetry run python -B -m development.roberto.historical.backpressure_sensitivity.scripts.run_sweep --plot-only
poetry run python -B -m development.roberto.historical.backpressure_sensitivity.scripts.audit_results
```

These numerical/reference scripts are unchanged. Full setup, diagnostics,
controller settings and limits remain in the report. Separate variants should
use a separate study-local `--output` directory.

For optional HTML verification with an existing Playwright environment and Edge:

```powershell
python -B development/roberto/historical/backpressure_sensitivity/scripts/audit_html_report.py
```

This checks a solitary copy of the HTML offline, both tabs and themes, plot
controls, animation, keyboard slider, filters, exact downloads and desktop
widths of 1280, 1440 and 1920 pixels. Browser state and evidence stay in `outputs/`.
