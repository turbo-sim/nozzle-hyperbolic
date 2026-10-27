# Direct steady EulerNozzle experiments

The latest experiment is the single-loop Newton/PTC solver in
`investigate_ptc.py`. It uses one custom PySolverView Newton correction per
trial with an adaptive physical-storage shift, keeping the original steady
residual and exact spatial Jacobian. Cold convergence results and the method
are documented in [NEWTON_PTC_EXPERIMENT.md](../../docs/NEWTON_PTC_EXPERIMENT.md).

```powershell
# Both shocked cases from the original cold initialization, with matched-seed references.
poetry run python -m development_new.newton_solve.investigate_ptc --phases simple --initial-cfl 1 --no-show --output development_new/newton_solve/results_ptc_simple

# Repeat from the uniform spouting field, with no implicit warm start.
poetry run python -m development_new.newton_solve.investigate_ptc --phases spouting --initial-cfl 1 --no-show --output development_new/newton_solve/results_ptc_spouting

# Redraw saved residual and shift histories without solving again.
poetry run python -m development_new.newton_solve.investigate_ptc --plot-only --no-show --output development_new/newton_solve/results_ptc_simple
```

The preceding physical-path investigation is `investigate_physical_paths.py`. Every new direct
solve uses the custom PySolverView backend, `library="pysolver"`, with explicit
initial-step control for Newton-Raphson, Levenberg-Marquardt, Powell hybrid,
and SLSQP. The implicit backward-Euler solver supplies an independently audited
reference and, in the stated warm-start experiments, a shared initial field.
See [PHYSICAL_CONVERGENCE_STUDY.md](../../docs/PHYSICAL_CONVERGENCE_STUDY.md) for the
physical-path study, results, and interpretation.

The installed `pysolver-view==0.7.4` already supplies all four custom methods.
Install the project environment with `poetry install --with dev`. The custom
SLSQP bridge requires the existing SciPy 1.17.1 pin; this compiled numerical
dependency does not change the selected custom-backend identity.

## Current runs

Run from the repository root. These commands use both shocked cases and all
four custom direct methods on the 201-point mesh:

```powershell
# Local convergence from a shared partial implicit field and a perturbed root.
poetry run python -m development_new.newton_solve.investigate_physical_paths --phases warm perturbed --methods newton lm hybr slsqp --no-show --output development_new/newton_solve/results_custom_local

# Independent cold starts: simple initialization and uniform spouting.
poetry run python -m development_new.newton_solve.investigate_physical_paths --phases simple spouting --methods newton lm hybr slsqp --no-show --output development_new/newton_solve/results_custom_cold

# Experimental residual anchoring and scalar backpressure continuation.
poetry run python -m development_new.newton_solve.investigate_physical_paths --phases continuation pressure --continuation-methods newton lm hybr slsqp --no-show --output development_new/newton_solve/results_custom_continuation

# Redraw a saved current study without rerunning its solvers.
poetry run python -m development_new.newton_solve.investigate_physical_paths --plot-only --no-show --output development_new/newton_solve/results_custom_local

# Independent verification of the spatial residual and exact Jacobian.
poetry run python -m development_new.newton_solve.verify_operator
```

The recorded pseudo-arclength settings can be tested separately. The second
command explicitly selects `shock70` and uses the saved `shock35` reference
from the local warm study above; this is a known-shock tracking experiment.

```powershell
# Cold pressure path, with the weighted arc equation.
poetry run python -m development_new.newton_solve.investigate_physical_paths --cases shock35 --phases arclength --continuation-methods newton hybr --continuation-attempts 100 --arclength-step 0.1 --arclength-max-step 0.5 --no-show --output development_new/newton_solve/results_arclength_local

# Track an existing shock35 field toward the shock70 target.
poetry run python -m development_new.newton_solve.investigate_physical_paths --cases shock70 --phases arclength_shock --continuation-methods newton hybr --continuation-attempts 180 --arclength-step 0.05 --arclength-max-step 0.2 --shock-seed-report development_new/newton_solve/results_custom_local/summary.json --no-show --output development_new/newton_solve/results_shock_arclength_local
```

Use `--cases shock35` or `--cases shock70` to select one case. Reconstruction
options are `constant`, `muscl_van_leer`, and `muscl_van_leer_smooth`. Newton and
SLSQP use `--first-step-fraction` (default 0.25) with growth 1.5. LM and Powell
hybrid use `--initial-radius` (default 0.1), an absolute scaled L2 step bound,
with radius growth 1.5 and maximum radius 100. These are different control
quantities; neither is a native MINPACK `factor`. Backend identities and
effective options are stored in each result.

The `warm` seed must remain above the final $10^{-8}$ RMS target and is shared
identically between methods. `perturbed` intentionally starts near the tight
implicit reference to test local convergence. `simple`, `spouting`, and
residual anchoring use their stated cold seeds; the reference is used only for
auditing those runs. `pressure_shock` and `arclength_shock` explicitly load a
saved shock35 implicit anchor, so they are separate from cold pressure paths.
The experimental `arclength` phase adds a path constraint to pressure
continuation; phase completion and the original physical residual are audited
separately. Successful intermediate stages do not establish a target root.

Current direct histories contain the initial state and accepted solver moves.
Continuation histories contain accepted stage endpoints, with original-target
and stage residuals distinguished; nested stage histories retain their own
equation labels. Failures are saved alongside successes. The outer direct
timer includes solver construction, and continuation timers include failed
attempts and monitoring. Implicit seed-generation costs remain separately
reported, so direct-phase time alone is not a hybrid speedup.

## Earlier studies

`compare_solvers.py` retains the five-case implicit/Newton/SLSQP comparison;
`investigate_convergence.py` retains the initial-step sweep and actual implicit
handoff study. Both use custom direct backends. Their historical results below
identify the spatial flux variant and timing scope used at measurement time.

`minpack_solvers.py` and the LM/HYBR paths in `compare_root_methods.py` are
**historical SciPy/MINPACK-only experiments**. Their archived evaluation
histories explain earlier findings, including native LM success above the root
tolerance. They are not the current investigation or recommended future test
commands. Use the four custom methods in `investigate_physical_paths.py` for
new tests.

## Shared problem and scaling

`problem.py` reuses the moved steady example's five physical cases, perfect-gas
backend, geometry, numerical settings, and simple initialization. Initialization
is explicit in each study phase: partial implicit fields and perturbed roots
are local-convergence tests, while cold phases use the simple or uniform
spouting seed. No analytic nozzle profile initializes the cold phases.
The finite-volume scheme is Städtke FVS with primitive Van Leer reconstruction;
the face-consistent pressure-area source and model boundary conditions match
the implicit solver.

Unknowns are the interior pressure, velocity, and enthalpy, scaled by the
reservoir pressure $p_0$, sound speed $a_0$, and enthalpy $h_0$. The integrated
spatial residual has mass, momentum, and energy components. Its fixed scale is

$$
R_{\mathrm{scale},i}
= \frac{V_i}{\Delta x/a_0}
\begin{bmatrix}\rho_0 & \rho_0 a_0 & \rho_0 h_0\end{bmatrix}.
$$

Every method is independently checked against
$\|R/R_{\mathrm{scale}}\|_2/\sqrt{n}\leq 10^{-8}$.
The original state-dependent pseudo-time SSR is also saved as a diagnostic;
it is a different normalization and is not mixed into the common convergence
plot. Variable/equation scaling changes conditioning, not the root equations.

The production factory has one additive, optional interface:

```python
step, compute_dt, jac_diff, face_flux, operators = make_jax_stadtke_step(
    ..., return_operators=True
)
```

The default still returns the original four objects. The extra dictionary
exposes the existing boundaries, spatial residual, AD Jacobian, optional
banded Newton direction, and SSR. The spatial residual evaluates the implicit
residual with the same current state in both storage terms, so the storage
contribution cancels. Its Jacobian must also cancel storage: it is assembled
as the implicit Jacobian at $\Delta t=1$ minus the Jacobian at $\Delta t=0$.
Using the ordinary implicit Jacobian unchanged would add a mass matrix and
would therefore not be a direct steady Newton solve.

`verify_operator.py` checks the unchanged four-return API and implicit step,
boundary and SSR equivalence, the spatial Jacobian against independent dense
AD, and the banded Newton linear residual. Enabling the operator dictionary
preserves the current four-return step. The subsequent regularity corrections
change the default flux; use `--flux-variant legacy` to reproduce historical
equations and timings.

## Historical comparison: histories, stopping, and timing

The following conventions describe `compare_solvers.py` and its archived
three-method comparison. The current four-custom-method study uses the outer
timers and phase-specific histories described above.

- All measured solves run sequentially. Dense BLAS threads are set to one
  before importing the numerical libraries.
- JAX operators are called once before timing. Their first-call costs,
  including compilation and execution, are reported separately. The warm-up
  implicit update is discarded. A tiny untimed problem loads both PySolverView
  backends before the nozzle comparisons.
- `warm_median_s` measures the complete warm solve, including model calls,
  linear algebra, interface work, and history monitoring. Reference generation,
  final diagnostics, plotting, and file output are outside the timer.
- `time_to_common_target_s` is the first accepted history record meeting the
  common RMS target. SLSQP can continue to a tighter native componentwise
  feasibility tolerance; its total solve time remains available separately.
- Repeats restart from the identical initial field. Timing CSVs use medians;
  plots use an actual middle-duration run, not a best-so-far curve.
- Histories contain the initial state and accepted updates. Derivative probes
  and rejected trials are not relabeled as iterations. Rejection counters and
  native iteration labels are retained where the backend exposes them.
- One implicit outer update contains inner Newton work. A direct Newton or
  SLSQP update means something different, so the wall-time comparison is more
  informative than comparing iteration counts alone.
- Newton and SLSQP use dense linear algebra through PySolverView. The implicit
  reference uses the existing banded solver. These timings compare the current
  implementations, not intrinsic algorithmic complexity or a promise that a
  direct method will be faster on every mesh.

The implicit experiment retains the baseline CFL/SER and adaptive increment
controls. It stops on the common fixed-scale residual rather than the old
driver's SSR/plateau rule. An update that still violates the domain or increment
limits after nine trials is reported as a failure. Root model calls reject
nonpositive pressure, enthalpy, or forward velocity through `psv.DomainError`,
allowing the direct methods to backtrack. The legacy flux is singular at rest;
the corrected flux also supports stationary states. The experiment retains
its positive forward-flow branch, and fabricates no invalid residual.

The post-solve audit distinguishes native termination, the common root target,
and agreement with the analytical solution. A converged discrete sonic-limit
solution can still have the accuracy gap documented in the baseline example.
Speedups are omitted when either method misses the common root target or when
the returned scaled solutions differ by more than $10^{-4}$ in RMS.
Every repeat is checked, including repeats other than the plotted trajectory.

## First measured comparison

The 201-point run on 2026-10-26 used three sequential warm repeats, initial
direction fraction 0.25, growth factor 1.5, and the shared simple initialization.
The table gives median seconds to the common RMS target. Operator first-call
costs are excluded and saved separately in `summary.json`.
These historical measurements use `flux_variant='legacy'`.

| Flow regime | Implicit pseudo-time | Controlled Newton | Controlled SLSQP |
|---|---:|---:|---:|
| Supersonic exit, no shock | 0.1041 | 0.2627 | 0.4280 |
| Shock at 35% of divergence | 0.6015 | Target missed | Target missed |
| Shock at 70% of divergence | 1.6073 | Target missed | Target missed |
| Sonic throat, subsonic exit | 0.4123 | 0.3099 | 0.4997 |
| Fully subsonic | 0.2941 | 0.2321 | 0.4022 |

Newton was about 1.33 times faster for the sonic/subsonic discrete problem and
1.27 times faster for the fully subsonic problem in this run. The implicit
banded implementation was fastest for the smooth supersonic case. SLSQP used
fewer accepted updates than implicit marching on the smooth cases, but its
dense/interface costs outweighed that reduction.

Both direct methods exhausted their configured budgets on the shocked cases:
Newton reached its 200-trial budget, and SLSQP reached its 200-major-iteration
limit. Their final common RMS values were approximately $5\times10^{-4}$ to
$9\times10^{-4}$, so no solution-time speedups are reported for them. Their
actual histories and returned profiles are retained and labeled unconverged.
These are experimental outcomes for this initial field, control schedule, mesh,
and budget; they do not establish a universal failure or performance ordering.

## Investigating the shocked cases

`investigate_convergence.py` tests initial direction fractions 1, 0.25, 0.1,
0.03, and 0.01 at growth factor 1.5. It also tests fraction 0.1 with growth
factors 1.0 and 1.1, to distinguish persistent damping and slower recovery from
merely starting with a small step. Newton's budget counts trial attempts,
including rejected steps; SLSQP's budget counts major iterations. Both default
to 200, and the reports retain native counters and accepted histories.

The hybrid experiment keeps the original direct schedule (fraction 0.25,
growth 1.5) and tests switches after 5, 10, 20, 40, 60, 80, 120, 160, 180, or 190 accepted
implicit updates. Each trial restarts from the same simple initial field,
actually runs that implicit prefix, and immediately calls the direct solver.
Warm counts at or beyond implicit convergence are skipped. The search stops
at the first successful **tested** count for each method and repeats that
hybrid three times. It does not find a globally optimal switch or step schedule.
Use `--skip-cold-sweep` for hybrid-only studies, and
`--warm-first-step-fraction`/`--warm-step-growth-factor` to change the direct
controls after the switch.

An enclosing timer includes the implicit prefix, direct solve, solver
construction, and handoff. Compilation, reference generation, final root and
flow audits, plotting, and output remain separate. Prefix and direct-phase
costs are reported alongside total cost. A hybrid qualifies for a speedup only
if every repeat meets the common residual target and agrees with the implicit
discrete root. The original stage histories are retained, and the combined
history omits the duplicate state at the switch. A star marks the handoff.

See [STALL_DIAGNOSIS.md](STALL_DIAGNOSIS.md) for the saved-state Jacobian,
conditioning, positivity, and directional checks. See
[CONVERGENCE_INVESTIGATION.md](CONVERGENCE_INVESTIGATION.md) for the measured
step sweep and warm-start findings. Those historical step-control studies kept
the spatial equations unchanged. The subsequent production corrections below
change the flux inside its sonic regularization band.

The subsequent [mathematical analysis](../../MATHEMATICAL_STAGNATION_ANALYSIS.md)
isolates the limiter's loss of a useful Newton neighborhood, demonstrates a
finite entropy-switch flux jump, and derives the avoidable sonic matrix
inverse and equal-state consistency defects. Reproduce those diagnostic-only
checks with:

```powershell
poetry run python -m development_new.newton_solve.verify_flux_regularity
poetry run python -m development_new.newton_solve.analyze_stagnation_mechanism
```

The implemented corrections and historical LM/HYBR/reconstruction results are
documented in [SOLVER_REGULARITY_IMPROVEMENTS.md](../../SOLVER_REGULARITY_IMPROVEMENTS.md).
The archived `compare_root_methods.py` experiments separated native SciPy
MINPACK termination from the common root target and recorded all LM/HYBR
residual requests. Some historical probes used log coordinates or an actual
short implicit prefix. They did not establish reliable direct cold-start
convergence for either shocked case. These native results are distinct from
the current custom LM/HYBR accepted-iterate studies.

The direct roots that converged in the historical comparison agree with the implicit discrete solution.
Consequently, direct solving preserves the sonic-limit spatial accuracy gap
seen in the baseline; it does not repair that discretization issue. Some
Newton/SLSQP residual-versus-update curves overlap because a square,
zero-objective equality problem can produce the same Newton direction. Distinct
line/marker styles and the elapsed-time panel expose the differences.

## Files and outputs

| File | Purpose |
|---|---|
| `investigate_physical_paths.py` | Current four-custom-method local, cold-start, and continuation study |
| `physical_initialization.py` | Uniform spouting seed from reservoir data and backpressure |
| `continuation.py` | Residual anchoring and physical backpressure paths with rollback and target audits |
| `backpressure_family.py` | Shared JAX operators with scalar backpressure as a dynamic argument |
| `arclength.py` | Experimental pressure-path continuation with an added path constraint |
| `compare_solvers.py` | Readable experiment settings, serial runs, plots, and reports |
| `problem.py` | Shared physical problem, scaling, JAX model calls, and final audit |
| `direct_solvers.py` | Custom PySolverView Newton, LM, Powell hybrid, and SLSQP adapters |
| `pseudo_time.py` | Monitored implicit reference loop using the production step |
| `verify_operator.py` | Small independent operator/AD equivalence checks |
| `diagnose_stalls.py` | Saved-state Jacobian, conditioning, and Newton-path checks |
| `investigate_convergence.py` | Cold fraction/growth sweeps and actual implicit-to-direct runs |
| `verify_flux_regularity.py` | Single-face continuity, consistency, and projector checks |
| `analyze_stagnation_mechanism.py` | Saved-state limiter, entropy/floor, and Newton-model probes |
| `minpack_solvers.py` | Historical native SciPy/MINPACK LM/HYBR adapters and evaluation histories |
| `compare_root_methods.py` | Historical native SciPy LM/HYBR flux and reconstruction comparisons |
| `diagnose_lm_stationarity.py` | Saved-state singular modes and stopping at historical native LM non-root fields |

Current physical-path output directories contain `summary.json`, `results.csv`,
per-phase accepted-history CSVs, and PNG/PDF convergence plots. JSON records
the exact seed hashes and fields, custom backend identities/options, independent
physical diagnostics, implicit reference and warm-start costs, and continuation
attempts including failures. Current reports also record numerical-library
versions and core source fingerprints. Saved shock-anchor paths record their
source report hash, source settings, and anchor-state hash. Historical warm-proof
archives can predate the added provenance fields.

`results_investigation/` contains `summary.json`, `trials.csv`, and per-case
accepted-history CSVs, cold-control sweep plots, and hybrid convergence plots
in PNG/PDF. JSON retains both complete stages, final fields, native status,
and repeated timing comparisons. Failed trials are included. The diagnostic
script writes its numerical checks to
`results_investigation_diagnostics/diagnostics.json`.
The late-switch follow-up measurements for `shock70` are in
`results_late_warm_start/`; the findings document links both measurement batches.

By default, `results_comparison/` contains:

- `timings.csv`: solve time, time to the common target, residuals, success, and
  qualified speedups for each method/case.
- `summary.json`: settings, versions, first-call costs, native status, model
  work, final diagnostics, full returned coordinates, and raw accepted histories.
- One directory per case, with residual histories against accepted updates and
  elapsed time (`residual_history.png/.pdf`), analytic/numerical flow comparisons
  (`flow_comparison.png/.pdf`), accepted-history CSVs, and profile CSVs.

Generated result directories are ignored by Git. The original nozzle and Sod
baseline scripts and their existing results now live directly in
`development_new/`; the older research code remains in `development/`.
