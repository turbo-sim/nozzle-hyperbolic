# Internal versus coupled boundary residuals

Open **[REPORT.html](REPORT.html)**. It is one standalone HTML file with three
internal tabs, interactive Plotly figures, typeset equations and light/dark
themes. Copying the HTML alone is sufficient to read it offline.

- **Results:** five nozzle cases, convergence histories, final pressure profiles,
  all timing samples with minimum/maximum/mean, physical checks and provenance.
- **Theory:** inlet/outlet laws, eliminated and coupled implicit systems, the
  Schur-complement connection, zero ghost storage, scaling and implementation
  history. It distinguishes the original closure adapter from today's public
  physical-residual/metadata interface.
- **Recommendations:** when elimination or coupling is appropriate, local
  solvability and two-fluid concerns, one canonical model-specific boundary
  residual, shared assembly, and an explicit formulation selector before JIT.

The report retains the measured benchmark: 100 timed solves, ten excluded full
solves and the actual representative trajectories. The sonic-limit analytical
check remains a failure for both methods, despite small equation residuals.
The timings were not rerun after the adapter migration or this report cleanup.
Recommendations about future two-fluid behavior are proposals, not measured
robustness claims.

## Directory organization

```text
BC_residual/
  README.md                  entry point and reproduction commands
  REPORT.html                complete offline results, theory and recommendations
  __init__.py                Python package marker
  scripts/                   numerical experiment, report builder and audits
    report_assets/           editable HTML/CSS/JS and embedded browser libraries
      notes/                 retained background and historical report sources
  outputs/                   ignored data, figures, source records and audits
```

The older Markdown documents are preserved under `scripts/report_assets/notes/`
and incorporated into the report. Detailed physical background and design
recommendations are rendered as HTML, including equations. Original benchmark
and smoke narratives remain expandable historical records; they are not new
validation runs. No companion file is required to view the HTML.

| Script | Purpose |
|---|---|
| `compare_implicit.py` | Matched source/coupled integration and repeated timing driver |
| `boundary_system.py`, `coupled_step.py` | Experimental algebraic BC equations and timestep assembly |
| `source_adapter.py` | Access to public production physical/solver operators |
| `report_implicit.py` | Build the standalone HTML from saved records and notes |
| `plot_implicit.py` | Optional static PNG/PDF export from saved results |
| `audit_html.py` | Check a solitary report copy offline in headless Edge |

## Rebuild the report without numerical solves

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.BC_residual.scripts.report_implicit
```

This reads `outputs/steady/summary.json`, checks saved timing arithmetic and
representative selection, then replaces root `REPORT.html`. It uses the
existing Markdown package, local Plotly/MathJax runtimes and no solver imports.
Custom saved runs can be selected with `--output`; their default HTML remains
inside that output directory. `--destination` can select another study-local
HTML path explicitly.

## Run the benchmark

```powershell
poetry run python -B -m development.roberto.BC_residual.scripts.compare_implicit --no-show
```

Defaults are 201 points, five cases, one excluded full solve and ten measured
solves per case/treatment. Every solve resets the same initial full field and
controller. Kernels are reused, methods alternate order, and all measured values
are retained. See the report for effective controls and timing boundaries.

For a smaller independent run, use `--cases shock35 --points 51 --runs 2`
and a separate `--output` path within this study. The driver still changes no
production source. Re-running today tests today's implementation; it does not
recreate the historical source revision merely by reusing its settings.

## Browser verification

With an existing Python environment containing Playwright and installed Edge:

```powershell
python -B development/roberto/BC_residual/scripts/audit_html.py
```

The audit copies only the HTML to an isolated folder, disables networking and
checks tabs, themes, numerical plot arrays, timing samples, downloads, deep links
and continuous equation layout. Audit files and browser state stay in `outputs/`.
