# Why this Newton/PTC implementation is slower than implicit integration

**Two effects multiply:** PTC takes more accepted outer corrections, and each
correction is much more expensive. The first is primarily a trajectory/controller
issue in this prototype; the second is largely dense custom-backend linear
algebra and interface work. Extra JIT compilation does not explain the measured
solve-time gap, and colored differentiation does not explain the iteration gap.

The expectation that a well-implemented PTC method could be competitive is
reasonable. The current comparison is not a comparison of equally optimized
implementations with matched controllers. We have now measured those differences
rather than inferring them from outer iteration counts.

## Evidence and reproduction

This investigation uses the [five-case comparison](outputs/REPORT.md), whose
measured operator hashes are retained with its data, plus:

- Counter-enabled replays of all five implicit runs, including rejected timestep
  trials. Every default replay exactly reproduces the archived final state and
  accepted-update count.
- A one-inner-Newton variant with the **same implicit timestep controller**.
- A matched-state, matched-timestep correction comparison using custom PySolverView.
- A timed 200-update custom-PTC prefix with nested backend/QR/condition timers.
- Three full upstream-shock PTC runs changing selected controller settings.

The instrumentation adds an inner-count return to an in-memory copy of the
implicit factory; the production file is not edited. Counter-replay timings are
not used as performance benchmarks because instrumentation adds monitoring.
All actual PTC corrections use `library="pysolver"`, `method="newton_raphson"`.
The short profiled prefix intentionally stops at its budget; it is not presented
as a new converged flow solution. The three full sensitivity runs satisfy the
original root target, agree with the implicit shocked root, and pass the analytic
check.

```powershell
# Recompute counts, correction equivalence, the short profile, and sensitivity runs.
poetry run python -m development_new.newton_solve.scripts.analyze_ptc_cost --ptc-ablations

# Redraw the diagnostic figure from saved measurements.
poetry run python -m development_new.newton_solve.scripts.analyze_ptc_cost --plot-only
```

Raw results: [analysis.json](outputs/performance_analysis/analysis.json).
The measurements use 201 points, 597 unknowns, one BLAS thread, corrected FVS,
safe ordinary Van Leer, and PySolverView 0.7.4. They are single-run diagnostics,
not repeated performance estimates. No production solver or controller was
changed by this investigation. The diagnostic script pins the measured CFL 20,
maximum CFL 500, inner cap 8, and growth 1.2 independently of later changes to
the readable steady-example defaults.

## 1. Separate number of corrections from cost per correction

For accepted outer counts $N$ and total solve times $T$,

$$
\frac{T_{\rm PTC}}{T_{\rm implicit}}
=\frac{N_{\rm PTC}}{N_{\rm implicit}}
\frac{T_{\rm PTC}/N_{\rm PTC}}{T_{\rm implicit}/N_{\rm implicit}}.
$$

These are an accounting identity, not an algorithmic complexity model. Per-update
cost includes rejected work and monitoring. One implicit outer update can contain
several inner Newton solves; one accepted PTC update normally contains one.

| Case | PTC / implicit accepted updates | Update-count ratio | Cost per accepted update ratio | Total time ratio |
|---|---:|---:|---:|---:|
| Subsonic | 138 / 56 | 2.46 | 11.76 | 29.0 |
| Sonic limit | 176 / 80 | 2.20 | 12.14 | 26.7 |
| Shock 35% | 1514 / 77 | 19.66 | 7.73 | 152.1 |
| Shock 70% | 1190 / 193 | 6.17 | 8.22 | 50.7 |
| Adapted supersonic | 82 / 19 | 4.32 | 11.93 | 51.5 |

Thus a faster linear solve alone cannot explain or eliminate the excess outer
updates. Conversely, similar update counts would not establish similar runtime.

## 2. Counting the implicit inner solves confirms the original intuition

The implicit method does perform several linear solves per timestep, capped at
eight with early termination. Counting actual executions, including discarded
trials, gives:

| Case | Implicit accepted updates | Implicit timestep trials | Actual implicit linear solves | PTC accepted updates | PTC affine linear solves |
|---|---:|---:|---:|---:|---:|
| Subsonic | 56 | 56 | 226 | 138 | 138 |
| Sonic limit | 80 | 80 | 311 | 176 | 185 |
| Shock 35% | 77 | 91 | 550 | 1514 | 1572 |
| Shock 70% | 193 | 238 | 1325 | 1190 | 1231 |
| Adapted supersonic | 19 | 19 | 77 | 82 | 82 |

For shock70, PTC actually performs **fewer linear solves** than implicit
integration, yet is about 51 times slower. The subsonic and sonic comparisons
also have fewer PTC linear solves. Therefore the concern about counting only
outer iterations is justified, but the much greater cost of each current dense
correction overwhelms that possible saving.

## 3. Measured per-correction cost: QR dominates, rather than AD

![Accepted CFL histories and the measured PTC prefix cost breakdown](outputs/performance_analysis/cost_and_cfl.png)

[Vector PDF](outputs/performance_analysis/cost_and_cfl.pdf).
The left panels compare actual accepted CFLs, not configured caps. The right
panel partitions a 200-accepted-update shock35 PTC prefix (207 trial corrections,
7.172 s). Nested timing was flattened to avoid counting the QR time twice.

| Cost in the profiled prefix | Time [s] | Fraction |
|---|---:|---:|
| Dense pivoted QR | 4.269 | 59.5% |
| QR conditioning checks | 0.288 | 4.0% |
| Other work inside the custom affine/backend call | 1.481 | 20.7% |
| Physical spatial-Jacobian evaluation | 0.417 | 5.8% |
| Physical residual, storage matrix, and CFL evaluation | 0.327 | 4.6% |
| Remaining controller/matrix work | 0.390 | 5.4% |

The whole custom correction call accounts for 84.2%; the QR and conditioning
figures are subsets of that call. Its other work includes frontend/backend
processing and operations such as scaling, matrix-vector products, triangular
solving and history handling. This profile does not separately attribute each
of those remaining operations.

The installed custom Newton backend builds a dense pivoted QR model for every
trial. At 597 unknowns, it does not exploit the fixed-width FVM stencil. The
implicit implementation instead assembles block-banded matrices and performs
small-block elimination in JIT-compiled loops. Fixed-bandwidth elimination scales
very differently from dense QR as mesh size grows. The exact code paths are
`_newton_dU`/`_banded_solve` in the production FVM and the installed
`pysolver_view/roots/solvers/linear_model.py` QR factorization.

PTC also exports dense Jacobian/storage arrays to the host and constructs a new
custom affine solve per trial. Its spatial Jacobian currently subtracts two
AD assemblies to cancel storage. Those costs are real, but the measured Jacobian
time is only about 6% of total PTC time in both full shocked runs as well.
They cannot explain a factor of 50?150 by themselves.

Both approaches already use JAX and colored differentiation. Coloring groups
local derivative probes; it is not a lower-order physical Jacobian. Operators
are warmed before the timed solves, and first-call costs are reported separately.
The PTC helper creates a new Python solver object, not a newly JIT-compiled PDE
kernel on every iteration. There is no evidence here of repeated PDE compilation
being the main warm-runtime cost.

## 4. Why the update counts differ: these controllers are not matched

The current PTC controller starts at CFL one and combines the following:

- Domain and relative-increment checks.
- A bound on error of the linear model of the **steady** residual.
- A nonmonotone residual-growth bound relative to a ten-state window.
- CFL halving after a rejected finite-shift trial.
- Multiplicative residual-ratio CFL updates, with a 5% growth probe on a
  sufficiently well-modeled, small-step plateau.

Ignoring clipping, plateau probes and rejections, its SER update is

$$
C_{k+1}=C_k(r_k/r_{k+1})^{0.7}.
$$

Residual increases can reduce CFL even after an accepted update. Rejections
introduce additional halvings. After CFL collapses, the locally small change
can satisfy the model test while advancing the flow extremely slowly. If the
5% plateau probe controls recovery, growing from $4\times10^{-6}$ back to one
would require about 255 accepted updates. This is an illustration of the
recovery rate, not an exact formula for every measured segment.

The implicit controller starts with base CFL 20 and an anchored target
$20\,\operatorname{clip}(\mathrm{SSR}_1/\mathrm{SSR},1,25)$, limits timestep
regrowth to 20% per accepted update, and retries based on componentwise physical
increments/domain validity. It does **not** impose PTC's steady-model-error or
residual-growth tests. Its target cap has a floor of 20; its actual accepted CFL
can still be lower because of increment-based retries and slow regrowth.
The two controllers also use different normalization for their SER residuals.

| Measured accepted-control statistic | Shock 35% | Shock 70% |
|---|---:|---:|
| Minimum PTC CFL | $4.37\times10^{-6}$ | $4.38\times10^{-6}$ |
| PTC updates at CFL below 0.01 | 357 | 332 |
| PTC updates at CFL below one | 700 | 608 |
| Adjacent accepted CFL ratios equal to 1.05 | 798 | 573 |
| Minimum actual implicit CFL | 10.69 | 2.20 |
| Median actual implicit CFL | 28.97 | 5.52 |
| Accepted zero-shift PTC/Newton corrections | 2 | 2 |

Low-CFL PTC blocks are updates 587?727 and 1034?1249 for shock35, and 430?594
and 793?959 for shock70. Hundreds of accepted updates are therefore spent
recovering from strong regularization, not moving rapidly with Newton-like steps.
The PTC curves in the figure make this history visible.

The extra steady-model tests can reject a move that the implicit controller
would allow. This is a supported explanation of the observed paths, not a proof
that all such rejected moves would be safe or that the safeguards should simply
be removed. The increment criteria themselves are differently scaled, so neither
controller is uniformly stricter in every variable/state.

## 5. Is one Newton correction per update fundamentally insufficient?

No. We reran the implicit controller with `newton_iter=1`, `newton_tol=0`, using
exactly one banded Newton correction per timestep trial. All five cases still
reached the original residual target:

| Case | Default implicit updates | One-correction implicit-controller updates | Current PTC updates |
|---|---:|---:|---:|
| Subsonic | 56 | 56 | 138 |
| Sonic limit | 80 | 81 | 176 |
| Shock 35% | 77 | 96 | 1514 |
| Shock 70% | 193 | 209 | 1190 |
| Adapted supersonic | 19 | 19 | 82 |

Multiple inner corrections help: for shock35 they reduce 96 updates to 77,
and for shock70, 209 to 193. They do not explain 1514 versus 77 or 1190 versus
193. A single-correction evolution method can already make much more progress
with the existing implicit controller.

At a fixed admissible anchor, first-Newton backward Euler and undamped PTC solve

$$
[J_R(z)+M(z)/\Delta t]d=-R(z).
$$

We compared them from the same cold shock35 state at CFL 20 and fraction one.
The custom-PySolverView and production one-correction states differ by only
$3.33\times10^{-16}$ in maximum scaled component. This verifies the matching
formula numerically at that state; it is not a claim of bitwise equivalence on
every possible branch.

Consequently the one-correction implicit variant is itself a linearly implicit /
PTC-style method with the production controller. Calling one path ?implicit? and
the other ?PTC? can hide how much they have in common. Their controller and
implementation choices, not the labels, need to be compared.
The distinction between one linearization and a fully solved implicit timestep
is also explicit in [PETSc's PTC documentation](https://petsc.org/release/manualpages/TS/TSPSEUDO/).

## 6. The small controlled changes did not identify one magic setting

These shock35 runs leave the physical residual and exact spatial Jacobian
unchanged and keep the custom PySolverView backend:

| PTC settings | Accepted updates | Final original RMS | Intended shocked root? |
|---|---:|---:|---|
| Original: initial CFL 1, initial fraction 0.25 | 1514 | $5.59\times10^{-14}$ | Yes |
| Change only initial CFL to 20 | 728 | $5.93\times10^{-14}$ | Yes |
| Initial CFL 20 and initial fraction one | 1420 | $5.60\times10^{-14}$ | Yes |
| Previous row, residual-growth limit relaxed to $10^6$ | 1247 | $2.64\times10^{-13}$ | Yes |

Initial CFL matters, but larger initial fractions do not monotonically help.
Relaxing the residual-growth guard still leaves nonlinear-model-error rejections
(52 in the last run) and 226 updates below CFL 0.01. This is a nonlinear path and
recovery problem, not just a mistaken starting timestep. These tests do not
exhaust possible schedules or establish a uniquely optimal controller.

The ordinary direction fraction already reaches one after the first four
accepted updates in the original PTC run. Persistent scalar under-relaxation is
therefore not the explanation for its subsequent thousand-plus updates.

## 7. What ?switching to fully accurate Newton? actually means here

PTC already uses the **exact spatial AD Jacobian at every accepted anchor**.
Regularization changes the correction equation by adding $M/\Delta t$; it does
not replace an inaccurate derivative by an accurate one at the end.

The explicit zero-shift trigger requires a thousandfold reduction in the
original RMS plus three consecutive sufficiently well-modeled steps. It is
reached very late in the shocked runs. Both finish with only two accepted
unshifted corrections: the Newton finish works, but almost all runtime was
already spent reaching that local basin.

The implicit method also uses accurate Newton steps throughout, for its
transient nonlinear equation rather than directly for $R=0$. Several inner
corrections can support a larger nonlinear timestep. That is useful, but the
one-correction replay shows it is not the dominant reason for the iteration gap
in this comparison.

## Recommended next experiment

First establish a controlled single-correction baseline using the implicit
controller and the shared residual/storage, then vary its acceptance and shift
recovery rules one at a time. The measured 96/209-update one-correction paths
provide a much stronger starting point than tuning the existing 1514/1190-update
PTC controller by initial CFL alone. Retain domain checks, original-root audits,
failed trials, and physical-branch/analytic checks.

Separately, give the custom PySolverView Newton backend a structured linear-solve
path (banded/sparse or a supplied solve callback) so it can exploit the same FVM
structure. Keep the backend custom; substituting a different root solver would
confound that comparison. Reducing repeated frontend construction and dense host
arrays is another implementation target. Optimizing AD alone has a much smaller
measured ceiling at this mesh size.

An earlier zero-shift handoff can be investigated after the trajectory is under
control, but the cold direct-Newton failures show why simply enabling full Newton
sooner is not guaranteed to help. There is no present evidence that PTC must be
slower in principle, and no evidence that the current prototype is already a
fairly optimized alternative. Its demonstrated benefit remains cold convergence
to the shocked roots; its current controller and dense implementation are the
specific performance limitations identified here.
