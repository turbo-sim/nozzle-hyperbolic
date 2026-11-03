# Numerical study conventions

Use this guide when creating, changing, running or reporting a numerical study.
It collects the detailed experiment and presentation conventions formerly in
root `AGENTS.md`. Study READMEs own physical setups, executable commands,
recorded settings and measured conclusions. Reading this guide does not restart
a paused campaign or authorize unrequested production-source changes.

For supported model responsibilities, use the [architecture](architecture/ARCHITECTURE.md)
and [model interface](model_interface.md). Locate an existing study through the
[development index](../../development/README.md). The
[maintained benchmark](../../development/time_benchmarking/README.md) owns its
runner-specific timing and resume rules.

## Organize a numerical study

New controlled studies use descriptively named `development/<study>/` folders.
Retained earlier investigations live in `development/historical/`. Preserve
existing thematic collections such as `development/flux_methods/` and their
shared infrastructure. Use the established names in an existing study:

```text
development/<study>/
  README.md                  question, physical setup, controlled settings, commands
  <topic>_NOTES.md            equations, interpretation, design discussion as needed
  <topic>_REPORT.md           findings with embedded relative figure links
  REPORT.html                one standalone HTML with internal results/theory pages
  scripts/                   readable drivers, supporting experimental code, audits
  outputs/<run_or_variant>/  JSON/CSV, logs, figures and generated run data
```

- Human-facing files belong at the study root; implementation belongs in
  `scripts/`, generated data in `outputs/`. Python package markers are fine.
- Keep human-facing HTML reports directly at the study root, outside `outputs/`.
  Their Python generators and reusable assets belong in `scripts/`; generated
  numerical data, logs and figures belong in `outputs/`. Keep reports reproducible
  from the Python scripts. Do not commit generated HTML unless requested.
- Generate one shareable, standalone HTML file per report. Put results, theory
  and other chapters on internally navigable pages within that file, rather
  than separate HTML documents. Embed styles, JavaScript libraries, plot data,
  images and offered downloads; do not depend on CDNs, neighboring files or a
  local server. A colleague must be able to copy just the HTML file and open it
  in a browser offline. Check page navigation, plots, equations, theme controls
  and embedded downloads after consolidation.
- Regenerate presentation changes from saved numerical results without rerunning
  simulations. Preserve numerical data and timing samples, and update report
  generators, links and structural audits together.
- Put ignore rules in the repository `.gitignore`, not a study-local
  `.gitignore`. `outputs/` is generated and ignored by default. Some older
  result snapshots were explicitly committed; check tracking before cleanup.
- Keep scripts readable. Import shared geometry, physical cases, references and
  settings rather than duplicating their definitions. Numerical overrides must
  not mutate the imported baseline's global settings.
- Preserve mathematical/physical explanations separately from measured results.
  Embed PNG figures with relative links; also save PDF versions for export.
  Label axes, units, methods, residual definitions and stopping thresholds.
- Use plasma colors sampled between 0.2 and 0.8 for comparable lines unless the
  user specifies otherwise. Use distinct styles/markers when curves overlap.
  For the standard five-case comparison, use five subplots, one backpressure
  per panel, with one line per method; provide the analogous final-field figure.
- Save machine-readable settings, case/mesh/EOS details, backend identities,
  package versions and source fingerprints when relevant. Retain failed runs,
  actual final fields, accepted histories and rejected-trial information.
- When cleaning an investigation, consolidate useful negative results and why
  they matter before deleting superseded scripts/results. Update imports,
  module commands, defaults, relative links and report generators together.

## Numerical correctness and solver choices

- Hyperbolic JAX kernels should remain compatible with JIT and AD. Avoid
  unsafe inactive arithmetic, host conversions or data-dependent Python control
  flow inside traced code. Record dtype and follow the solver precision policy;
  current Euler verification uses 64-bit arithmetic.
- For new direct Newton/LM/HYBR/SLSQP investigations, use custom PySolverView
  backends with initial-step/radius controls. Record `library="pysolver"`, the
  actual method and effective options; never silently substitute vanilla SciPy.
- When the user's comparison must match an existing implementation, use that
  verified source algorithm and backend. Do not replace its machinery to meet
  the general direct-solver convention above. Record the actual selected path.
- A fair formulation comparison holds physical equations, initialization,
  reconstruction, mesh, precision, timestep controller, stopping rules and
  linear/nonlinear solver machinery fixed as far as possible. Explain unavoidable
  differences; do not attribute a dense-versus-banded timing gap to BC physics.
- Validate final convergence with the **original spatial residual** independently
  of native success, small increments or a least-squares progress flag. Define
  normalization precisely: native state-dependent SSR and a fixed-scale spatial
  RMS are different metrics and can stop at different update counts.
- Report root convergence, physical/analytic accuracy, conservation and intended
  branch selection separately. Local convergence does not prove global uniqueness
  or physical well-posedness. Smaller residual need not mean smaller state error.
- Distinguish accepted outer updates, inner corrections, rejected trials,
  residual evaluations and continuation stages. An inner iteration setting is
  a maximum cap, not a forced correction count or proof of an exact timestep.
- Physical BC equations have one definition per model. Select eliminated or
  coupled numerical treatment in `Scheme` before binding/JIT compilation;
  do not duplicate physics or switch state dimensions during a solve. Algebraic ghost constraints have
  zero storage unless a separate physical dynamic model explicitly requires it.

## Scope, verification and reproducibility

Reuse physical geometry, packaged cases and independent references rather than
copying their definitions into a new client. Numerical overrides must not
mutate an imported baseline's global settings. Begin with one controlled
configuration, then expand as required by the requested study.

Keep tentative numerical implementations under `development/`; isolate any
explicitly requested production change. Record source fingerprints for a study
that promises unchanged source. Fixed settings, model/case hashes, mesh/EOS,
backend/precision, package versions and environment identity belong in the
saved evidence. Retain actual final states, accepted histories and rejected work.

The [residual guide](residual_monitoring.md) defines current spatial, boundary
and global checks. A reference is verification data, not an unreported numerical
initial guess. Distinguish root convergence, conservation, the intended branch,
analytical accuracy and timing eligibility. Never replace failed original
checks with relaxed targets or a successful intermediate continuation stage.

Production pytest tests cover `src/` only; do not import development clients
from `tests/`. Studies use their own executable audits. Run the requested cases
and independent residual/field checks, inspect figures and links, and disclose
coverage limits and pre-existing failures.

Presentation-only changes rebuild from saved data without importing a flow
solver or advancing simulations. Historical campaign commands, run counts and
source-specific conclusions stay in the owning study README/report. Obey its
pause and resume rules. A changed source, client, case or numerical policy
requires a compatible fresh campaign identity rather than merged timings.

## Performance measurements

For a repeated solver benchmark, default to one excluded **complete solve**
per case/method followed by ten measured solves, unless the user specifies
another protocol.

- Build kernels once and reuse them across repeats. Reset the identical initial
  field and all controller/history state on every solve; do not reuse a solution
  as a hidden warm start. Alternate method order and run measurements sequentially.
- Synchronize JAX device work before ending a timer. Excluding import/JIT warmup
  for only some operations or rebuilding closures between repeats is not fair.
- State the timing boundary: include equivalent monitoring and rejected work;
  exclude or separately report setup, reference/seed generation, final audits,
  output and plotting. Hybrid warm-start cost must be included in hybrid totals.
- Retain every sample and the excluded-run status. Report minimum, maximum and
  mean, with explicit sample count. Do not cherry-pick successful or fastest runs.
- Plot one labeled actual representative trajectory (for example the median-
  duration run), not an averaged residual curve or a best-so-far envelope.
- Only discuss time-to-solution speedups after checking success and agreement
  with the same intended discrete root across repetitions. Different backends,
  sparsity, precision or stopping rules must be disclosed.

## Reporting checks

Use Markdown `$...$` inline math and `$$...$$` display math. Scientific figures
need labelled units, methods, norms, thresholds and time/update coordinates.
Keep explanations separate from measured findings and retain unsuccessful cases.

After report consolidation, verify actual browser navigation, controls, plots,
equations, themes and embedded downloads in addition to structural HTML checks.
The report must remain usable offline when copied alone. Keep generator assets
and required third-party licenses reproducible; no CDN or neighboring files may
be required for viewing.

Do not infer a speed advantage from one observation, compile-inclusive cold
durations or concurrent CPU contention. Record environmental eligibility and
actual completed sample counts. Include warm-start preparation in hybrid totals,
and never compare kernel-only time against a complete-driver measurement.
