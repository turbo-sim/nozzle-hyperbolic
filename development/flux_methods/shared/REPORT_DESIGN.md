# Flux-method report presentation

Stadtke, Roe and Osher share a visual system while retaining their own
verification populations, numerical acceptance rules and evidence models.
Presentation reads saved results and does not run a flow solver.

## Common report structure

| Page | Common purpose | Study-specific content |
|---|---|---|
| Overview | Scope, coverage, outcomes and interpretation | Correction tradeoffs, quadrature comparisons or Roe suite checks |
| Case explorer | Select a physical case and inspect saved fields, original spatial residual, CFL, work and analytical refinement | Four-field shock comparisons for Stadtke/Osher; the broader steady suite and conditional timing views for Roe |
| Theory | Read equations, assumptions, limitations and sources | Original/corrected Stadtke equations, Osher integration policies or general-EOS Roe construction |
| Protocol & evidence | Check controls, audits, source identities and downloads | Recorded/current evaluation separation; Roe algebra, Sod and actual-EOS flow checks |

The header, navigation order, typography, spacing, cards, tables, status
colors and Plotly palette are shared. Theme choices are Automatic, Light and
Dark, stored under `nh-flux-report-theme`; storage restrictions fall back
gracefully. Keyboard tab navigation, visible focus, a skip link, responsive
layouts and reduced-motion/print rules apply to all three reports.

Stadtke/Osher retain Profiles, Convergence & work and Accuracy & refinement
inside the case explorer. Their configuration IDs remain metadata, and the
explicit method parameters remain inspectable. Roe retains its broader
validation sections and evidence archive. The visible Sod profiles use the
saved numerical arrays through Plotly; the original PNG remains in the
byte-preserved evidence archive.

## Shared implementation

- `assets/report_theme.css` owns visual tokens and common components.
- `assets/report_theme.js` owns the shared theme/Plotly presentation helpers.
- `presentation.py` loads shared assets and renders Markdown prose and protected
  dollar-delimited equations with the existing offline MathJax SVG bundle.
- `report.py` and `assets/report_*` render Stadtke/Osher data.
- `../roe/scripts/presentation.py` applies the visual system to Roe without
  changing the maintained benchmark renderer's default presentation.
- `audit_report.mjs` and `audit_reports.mjs` execute report controls against
  saved data and check presentation/evidence invariants.

The full Markdown notes remain unchanged and downloadable. Roe's native
MathML and the typeset Stadtke/Osher LaTeX appear in the same equation/prose
components. All runtimes, styles, SVG math fonts, licenses and offered
downloads are embedded in each standalone HTML; no CDN or server is needed.

## Evidence rules preserved by presentation

A coverage count is not a pass count. Original-equation convergence,
analytical accuracy, physical admissibility and provenance checks remain
distinct. Failed, incomplete and unsupported outcomes stay visible. No theme
or layout change may alter stopping thresholds, numerical values or histories.

Verification-only studies do not acquire timing rankings. Genuine timing
views retain their eligibility rules and measured samples. Native execution,
metadata migration and saved-state reevaluation keep separate source identities.
The compact Stadtke/Osher download and full Roe evidence ZIP retain their
explicit scope; a common Downloads control does not imply identical contents.

## Rebuild from saved results

Use the existing project development environment (`poetry install --with dev`
when setting up a new checkout). The Markdown parser is already locked through
the documentation dependencies. From the repository root:

```powershell
.venv/Scripts/python.exe -B -m development.flux_methods.shared.report --study stadtke
.venv/Scripts/python.exe -B -m development.flux_methods.shared.report --study osher
.venv/Scripts/python.exe -B -m development.flux_methods.roe.scripts.report --presentation-only
```

These commands regenerate the HTML without numerical solves. The Roe flag
also skips independent analysis/assessment regeneration. Generated reports
stay at each study root; raw campaign data and QA records stay in ignored
`outputs/` directories. Numerical-data, physical-case, source and execution
client hashes are checked separately from the generated HTML and presentation
audit outputs.
