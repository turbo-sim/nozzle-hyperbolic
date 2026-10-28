# EulerAnnular verification

Open **[REPORT.html](REPORT.html)**. Copying that single file is enough to read
it offline. Its internal pages contain:

- **Results:** nine cases, four meshes, analytical/constant/MUSCL comparisons,
  meridional/frontal/3D channel views in one row with uniform streamline color,
  accuracy limits, convergence and four conservation histories.
- **Theory:** general-EOS equations, eigenanalysis, flux/reconstruction,
  complete implicit residual, Newton machinery and simultaneous boundary blocks.
- **Implementation:** public API, fluid and NURBS contracts, shared code and
  reproduction commands.
- **Notes:** controller evidence, full-campaign comparison, physics qualifications,
  retained failures and limitations. The original 42 controller experiments are
  available as an embedded ZIP download.

The current steady campaign converges all 72 discrete roots; 11 of 18 finest-grid
analytical accuracy targets pass. The seven misses remain documented. Its fixed
controller uses initial CFL 0.2, growth 1.2, shrink 0.5 and at most eight Newton
corrections, retaining both strict inner tolerances at $10^{-10}$. The production
solver's defaults remain separate. The numerical method accepts general JAXProp
fluids; this flow-verification campaign uses a perfect-gas backend.

## Folder structure

```text
euler_annular/
  README.md
  REPORT.html              one standalone document with four internal pages
  old_notes/               completed questionnaire, plan and planning snapshot
  scripts/                 verification, audits and one HTML report generator
    report_assets/         editable HTML chapters, CSS/JS and embedded libraries
  outputs/                 numerical data, figures, audits and provenance
```

The retired CFL investigation folder and duplicate Markdown theory/specification
notes have been removed. Their essential findings are incorporated in the
report. Current numerical data and previous campaigns are retained in `outputs/`.
The implementation/environment snapshot is in
`outputs/provenance/implementation_manifest.json`.

## Rebuild the report without rerunning simulations

Run from the repository root in the existing Poetry environment:

```powershell
poetry run python -m development.roberto.euler_annular.scripts.write_html_report
poetry run python -m development.roberto.euler_annular.scripts.audit_html_report
```

Edit chapter sources in `scripts/report_assets/` and rebuild. All styles,
Plotly/MathJax libraries, result data and offered downloads are embedded. The
static audit checks the saved samples, math, links, source hashes and embedded
evidence. JavaScript figure/navigation audits are recorded in `outputs/html_report/`;
they are distinct from browser visual inspection.

## Rerun numerical verification

```powershell
poetry run python -m development.roberto.euler_annular.scripts.verify_annular --controller cfl_recommended --cells 50 100 200 400 --max-steps 16000
poetry run python -m development.roberto.euler_annular.scripts.audit_results
poetry run python -m development.roberto.euler_annular.scripts.audit_shocks
poetry run python -m development.roberto.euler_annular.scripts.verify_nonuniform --controller cfl_recommended
poetry run python -m development.roberto.euler_annular.scripts.audit_results --output development/roberto/euler_annular/outputs/nonuniform
poetry run python -m development.roberto.euler_annular.scripts.write_manifest
poetry run python -m development.roberto.euler_annular.scripts.write_html_report
```

The verification command retains a failing exit status when analytical targets
fail even if all roots converge. `--resume` reuses matching completed files,
never their fields as warm starts. See the report for physical-time verification,
production tests, geometry limitations and deferred scope.
