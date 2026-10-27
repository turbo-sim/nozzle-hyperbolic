# Implicit integration: inner Newton work and CFL sensitivity

This study asks two separate questions: how much inner nonlinear work is useful,
and how much the timestep controller determines progress. It reuses the same
five [EulerNozzle physical cases](../euler_nozzle/README.md), perfect-gas backend,
201-point mesh, original cold initialization, corrected FVS flux, and safe
ordinary Van Leer reconstruction as the Newton/PTC comparison.

```text
implicit_integration/
  README.md            Setup, controlled settings, and run instructions
  scripts/             Comparison driver
  outputs/REPORT.md    Combined report with figures from all three studies
  outputs/<study>/    Detailed report, figures, and results CSV
  outputs/runs/       Individual run data, accepted histories, and rejected trials
```

Open the [combined report](outputs/REPORT.md), or go directly to:

1. [Newton caps 1, 3, 5, 10, 100](outputs/newton_iterations/REPORT.md).
2. [CFL sensitivity with Newton cap 100](outputs/cfl_newton100/REPORT.md).
3. [The same CFL sensitivity with Newton cap 1](outputs/cfl_newton1/REPORT.md).

Each study has a five-panel residual-versus-update figure and a five-panel
pressure-versus-position figure. Additional figures show residual versus warm
solve time and the actual accepted CFL. Plasma colors span 0.2?0.8. The five
cases are fully subsonic, the sonic-throat/subsonic-exit limit, shocks at 35% and
70% of the divergent section, and adapted supersonic exit.

## Meaning of the Newton settings

The requested values are **maximum corrections per timestep trial**, with the
existing relative-update stopping tolerance $10^{-8}$. A cap of 100 allows the
inner solve to finish; it does not force 100 unnecessary corrections. Actual
linear-solve counts, the last relative update, and whether the inner stopping
criterion was satisfied are recorded for every accepted and rejected trial.

The inner stopping criterion is the production code's relative Newton-update
criterion, not an independently imposed norm of the transient equations. An
inner solve hitting its cap is not automatically declared an outer failure:
the existing domain/increment acceptance rules still govern timesteps.
The independent outer criterion is original fixed-scale steady residual RMS
$\leq10^{-8}$. Analytic accuracy and agreement with the cap-100 baseline root
are checked separately. The known sonic-limit accuracy discrepancy remains
visible even when residual convergence succeeds.

## CFL sensitivities

All CFL variants are tested identically at both Newton caps 100 and 1. Each
non-baseline row changes only one controller setting:

| Variant | Starting CFL | Maximum CFL | SER exponent | Maximum timestep growth per accepted update |
|---|---:|---:|---:|---:|
| Baseline | 20 | 500 | 1 | 1.2 |
| Start CFL 1 | 1 | 500 | 1 | 1.2 |
| Start CFL 100 | 100 | 500 | 1 | 1.2 |
| Max CFL 100 | 20 | 100 | 1 | 1.2 |
| Max CFL 100000 | 20 | 100000 | 1 | 1.2 |
| No SER ramp | 20 | 500 | 0 | 1.2 |
| SER exponent 2 | 20 | 500 | 2 | 1.2 |
| Timestep growth 1.05 | 20 | 500 | 1 | 1.05 |
| Timestep growth 2 | 20 | 500 | 1 | 2 |

For the existing anchored SER rule, the requested CFL cap is

$$
C_{\rm target}=C_0\operatorname{clip}
\left[(\mathrm{SSR}_1/\mathrm{SSR})^\beta,1,C_{\max}/C_0\right].
$$

A separate growth bound limits the next timestep relative to the preceding
accepted timestep. Domain/increment retries can halve it further. Consequently
actual accepted CFL can be below the base CFL, and the maximum can remain unused.
With $\beta=0$, SER acceleration is disabled while the base CFL remains active.
Its reference residual is established after the first accepted timestep.
SER uses the production state-normalized SSR; the final success check uses the
same fixed-scale spatial residual as the other development studies.

The existing pressure/enthalpy/velocity increment limits (0.2, 0.1, 0.2), maximum
eight timestep halvings, fluid, mesh, spatial discretization, and stopping rules
are held fixed. These are controlled sensitivities, not a full factorial search
or an optimal-controller claim. There are 105 unique solves: the baseline runs
at caps 1 and 100 are shared between studies, not duplicated as independent runs.

## Reproduction

From the repository root after installing project dependencies:

```powershell
poetry run python -m development_new.implicit_integration.scripts.compare_settings --no-show

# Regenerate all reports and figures from saved run data.
poetry run python -m development_new.implicit_integration.scripts.compare_settings --plot-only --no-show

# Optional separate coarse/budget-limited execution check.
poetry run python -m development_new.implicit_integration.scripts.compare_settings --points 31 --max-iterations 1000 --no-show --output development_new/implicit_integration/outputs/coarse_check
```

All variants restart independently from the same cold state, without analytic
initialization or warm-start transfer. Default outer budget is 5000 accepted
updates. Failed trials and unconverged final fields are retained and labeled.
Accepted histories distinguish actual outer updates, cumulative linear solves,
and all attempted timesteps. Final pressure CSVs omit ghost cells.

## Reuse and numerical scope

The physical problem and original spatial operators come from
`newton_solve/scripts/problem.py`; the shared implicit controller is
`newton_solve/scripts/pseudo_time.py`. The controller now exposes optional timestep
growth, retry-budget, and diagnostic-history controls while retaining its former
defaults. The production compiled backward-Euler/Newton/banded kernel is used;
this is not a new direct root solver or a substitution of a SciPy backend.

The existing in-memory counter adapter now optionally returns the final inner
relative update as well as the actual solve count. It adds return diagnostics;
it does not change the production residual or update equations. No `src` file
is changed for this study. Starting CFL is rescaled in the timestep estimate;
changing its host setting alone would otherwise leave the factory's compiled
CFL unchanged. This rescaling is applied consistently to the SER cap as well.

Each kernel is warmed once per case/Newton cap; first-call costs are saved
separately. Host CFL variants reuse that kernel. Timed solves include controller,
monitoring and rejected-trial work, but exclude compilation, plotting and final
audits. The time comparisons are single serial measurements with one BLAS thread;
use iteration/work counts for the stronger algorithmic conclusions.
