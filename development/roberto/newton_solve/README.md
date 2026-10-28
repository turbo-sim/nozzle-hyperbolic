# Newton, implicit marching and physical-storage PTC

Open **[REPORT.html](REPORT.html)**. It is one standalone offline HTML file:
copying this file alone is sufficient to use all plots, notes, equations,
light/dark themes and embedded outcome downloads. No server or supporting
folder is needed for viewing.

The report has five internal pages:

1. **Cold starts:** the five-case implicit/PTC/controlled-Newton comparison,
   actual convergence histories, final pressures and recorded solve times.
2. **PTC cost:** actual implicit inner counts, accepted CFL, correction
   equivalence, the dense/backend cost breakdown and saved controller ablations.
3. **Adapted starts:** retained custom Newton, HYBR, LM and SLSQP outcomes,
   emphasizing the difference between an equation root and the requested
   shocked outlet branch.
4. **Mechanism:** the physical and mathematical explanation of conservative
   storage, regularization, Newton directions and flow-structure development.
5. **Earlier notes:** useful earlier negative results, limits, settings and
   provenance, with clearly labeled historical measurements.

The report reuses existing results. **No simulations were rerun.** The
POUNCE-specific experiments, reports, worker records and generated results
have been removed as requested. The adapted experiment's 16 custom-method
records, common numerical seeds and implicit references are retained unchanged.

## Findings and interpretation

The cold-start comparison contains 15 calls: implicit integration and PTC meet
the original spatial target in all five cases; controlled custom Newton does
so in three, stalling on the two shocked cold starts. PTC provides a useful
physical globalization direction, but this implementation is much slower than
the retained implicit solver. Dense versus banded linear algebra and different
controller trajectories prevent an intrinsic algorithm speed ranking.

The adapted-start experiment is separate. Its eight unmodified-adapted custom
calls satisfy the equation-residual test while remaining on the wrong
supersonic-outlet branch. The eight outlet-adjusted custom calls remain
unconverged. None of the sixteen reaches the independently computed shocked
implicit reference. A small residual alone does not establish the requested
physical operating state.

Timings are single serial observations, not repeated speed benchmarks. The
sonic-limit analytical accuracy discrepancy remains a failure even when a
spatial root is reached. Historical notes include earlier exploratory timings;
they are not silently merged with the authoritative plotted five-case archive.

## Directory organization

```text
newton_solve/
  README.md                 entry point, interpretation and reproduction commands
  REPORT.html               one complete offline report with five internal pages
  __init__.py               package marker
  scripts/                  retained numerical drivers, adapters and report code
    report_assets/
      report.css            presentation-specific styling
      report.js             interactive figures and internal navigation
      notes/                editable retained mathematical and historical notes
  outputs/                  ignored numerical records, figures and audit evidence
```

The earlier top-level mechanism and performance Markdown files now live under
`scripts/report_assets/notes/` and are rendered into the HTML. Numerical and
reusable solver helpers remain in `scripts/`. No production source is modified
by this cleanup. Other studies importing the common nozzle problem and implicit
helper retain their existing paths.

## Rebuild only the report

From the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.roberto.newton_solve.scripts.report
```

This reads the saved cold comparison, performance analysis and filtered custom
adapted-start records. It does not evaluate a flow residual, run an optimizer
or advance an implicit timestep. It verifies numerical-file hashes and the
preserved custom records, then writes root `REPORT.html` plus generated audits
and a manifest under `outputs/`.
The retained dataset fingerprints guard the report's numerical narrative;
presenting a different campaign requires reviewing that narrative and its audits.

The builder reuses plotting/theme helpers and vendored Plotly/MathJax assets
from the committed neighboring reports rather than copying numerical code or
browser-library bundles. All resources and offered downloads are embedded in
the resulting HTML. Generated HTML is committed only when requested.

## Browser verification

With Edge installed:

```powershell
poetry run python -B -m development.roberto.newton_solve.scripts.audit_html
```

The audit uses an isolated profile and a solitary HTML copy, disables networking,
and checks navigation, both themes, plots against saved data, pressure-marker
controls, equations and downloads. It requires no Playwright installation.
Evidence remains in `outputs/html_audit/`; no simulations run.

## Retained implementation modules

| Module | Purpose |
|---|---|
| `problem.py` | Common scaled EulerNozzle physical problem and source operators |
| `pseudo_time.py` | Implicit marching and diagnostic/control histories; shared by other studies |
| `direct_solvers.py` | Controlled custom direct Newton comparison |
| `ptc.py`, `ptc_storage.py` | One-loop PTC controller and physical-storage Jacobian |
| `investigate_ptc.py` | Original five-case cold-start numerical driver |
| `analyze_ptc_cost.py` | Retained instrumentation, matched-correction and cost investigation |
| `adapted_solvers.py` | Retained custom Newton/HYBR/LM/SLSQP configurations |
| `experiment_support.py` | Shared signed-velocity domain and serialization helpers |
| `adapted_isolation.py` | Time-bounded workers for the four custom methods |
| `source_snapshot.py`, `run_adapted_initialization.py` | Frozen-baseline source export and launcher |
| `investigate_adapted.py` | Adapted and outlet-adjusted seed experiment, custom methods only |
| `adapted_reporting.py` | Passive handoff to the unified HTML builder |
| `report_data.py`, `report.py` | Solver-free loading, audits and report generation |

These numerical campaigns used historical source/backend versions. The cost
counter adapter targets the older implicit-factory layout and needs a
compatibility review before new live-source replays. The custom adapted driver
retains its frozen-baseline workflow; only the removed optimizer dispatch and
support dependencies were changed. No fresh numerical compatibility claim is
made by this presentation cleanup. Use the report-only command for updates.
