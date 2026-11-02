# Generic MUSCL limiter comparison

This study compares Van Leer, Minmod, Superbee and sign-preserving Van Albada
on the shared `euler_nozzle/shock_35` perfect-gas case. It uses the packaged
geometry and analytical solution, including conservative cell projections.
Every method and grid starts from the model's cold initialization. The study
records convergence, physical budgets, pressure/conserved errors and shock
location independently. It is not a timing benchmark.

Run from the repository root:

```powershell
poetry run python -B development/limiters/scripts/compare.py --cells 24 48
poetry run python -B development/limiters/scripts/compare.py --report-only
```

Use a fresh `--output` directory for a new attempt. Individual JSON files
preserve failures, settings, final states and accepted/rejected histories.
`outputs/comparison/RESULTS.md`, PNG and PDF plots summarize the run.
`REPORT.html` is generated as a standalone offline report with embedded
plots and a JSON download. Regenerating the report does not run flow solves.
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
