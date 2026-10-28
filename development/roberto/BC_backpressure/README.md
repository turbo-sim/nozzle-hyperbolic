# Backpressure boundary-condition study

Open **[REPORT.html](REPORT.html)**. This is one standalone, offline HTML file:
copying it alone is sufficient to read the complete study, use the interactive
plots and render the equations. It embeds the numerical results, audit evidence,
styles, Plotly and MathJax. Light and dark themes are available.

The report has three internal tabs:

1. **Uniform start**: current and pressure-wave outlet treatments initialized
   at uniform stagnation conditions, compared against analytical shocked profiles.
2. **Adapted start**: the same comparison initialized from a numerical adapted
   expansion, testing whether the imposed backpressure can initiate shock entry.
3. **Theory and equations**: physical setup, Euler balances, both outlet laws,
   wall safeguard, implicit integration, CFL control, verification criteria and
   limitations. The original implementation note is included as a supplementary
   source-text appendix.

Each results tab includes the solver settings, residual and actual accepted CFL
histories, final pressure/Mach profiles, verification outcomes and downloadable
CSV. The saved evidence contains eight comparisons on the same 201-point nozzle
mesh with a perfect-gas property backend. No numerical solutions were rerun
when consolidating the presentation.

## Directory organization

```text
BC_backpressure/
  README.md             contents, organization and reproduction commands
  REPORT.html           complete standalone report, with three internal tabs
  scripts/              experimental implementation, drivers and audits
    report_assets/      editable templates, styling, theory note and vendor libraries
  outputs/              ignored numerical records, source snapshots and audit artifacts
```

The two root files are the human-facing entry points. Scripts remain separate
so the study can be reproduced and the report rebuilt. Generated working data
stay under `outputs/`; the final HTML report deliberately lives at the study root.

| Script | Purpose |
|---|---|
| `outlet_laws.py` | Current and experimental pressure-wave/wall outlet laws |
| `problem.py` | Imported source physics with study-local boundary-flux assembly |
| `time_integration.py` | Matched fixed-growth CFL control and verified implicit steps |
| `run_study.py`, `launch.py` | Numerical comparisons and source-snapshot isolation |
| `report.py` | Build the single HTML file from saved results; no numerical solves |
| `audit_outlet_laws.py` | Local conservation, wave selection, JIT and AD checks |
| `audit_html.py` | Check an isolated copy of the HTML offline in headless Edge |

## Rebuild the report

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.BC_backpressure.scripts.report
```

This reads `outputs/recommended_cfl/summary.json` and the recorded audits, then
replaces `REPORT.html`. It does not import or execute the numerical solver.
Use `--output <path-inside-this-study>` to select another saved run directory.
The HTML itself requires none of these supporting files for viewing.

## Run or audit the numerical study

```powershell
poetry run python -B -m development.roberto.BC_backpressure.scripts.launch
poetry run python -B -m development.roberto.BC_backpressure.scripts.launch --resume
poetry run python -B -m development.roberto.BC_backpressure.scripts.launch --audit-only
poetry run python -B -m development.roberto.BC_backpressure.scripts.audit_outlet_laws
```

Use `--resume` when the default results already exist, or select a new study-local
`--output` directory for a fresh run. The launcher records and reuses a working
source snapshot; production source and other studies remain unchanged. Numerical
settings, findings and their interpretation are documented in the report.

The browser audit needs an existing Python environment with Playwright and Edge:

```powershell
python -B development/roberto/BC_backpressure/scripts/audit_html.py
```

It copies only `REPORT.html` to an isolated directory, disables networking, and
checks all three tabs, both themes, plot arrays, controls, CSV downloads,
deep links and equation layout. Its browser state and evidence stay in `outputs/`.
