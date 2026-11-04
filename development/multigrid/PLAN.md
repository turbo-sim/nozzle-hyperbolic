# Plan for initialized implicit FAS versus grid sequencing

The active study compares two strategies: **FAS with coarse-to-fine
initialization** and the package's **grid sequencing**. The goal is to find
one practical FAS configuration that reduces complete solve time while
preserving the requested finest-grid solution. A gain is an experimental
question, not an assumption of this plan.

All prototype changes belong in `development/multigrid/`; package `src/`
stays unchanged. [README.md](README.md) owns the executable workflow,
[settings_fmg.json](settings_fmg.json) owns the current main configuration,
and [RESULTS.md](RESULTS.md) reports measured outcomes. The final standalone
presentation is [REPORT.html](REPORT.html). Retired generated reports and
older raw datasets are removed. The current `outputs/fmg_*` campaigns retain
the main comparison, tuning trials, controls and their source archives.

## Scope and fixed physical problem

Use the Euler **nozzle** model only: subsonic flow, internal shocks at 30%
and 70% of the diverging section, and adapted expansion. Test constant and
linear MUSCL Van Leer reconstruction, giving eight physical/reconstruction
groups. The 30% case changes the packaged 35% case's shock definition in the
study adapter. No annular or two-fluid campaign is part of this work.

The default nested hierarchy is 50/100/200/400 cells, with HLL and implicit
backward-Euler updates. Both strategies retain the same final spatial
scheme, EOS, geometry, physical boundary laws, precision and final targets.
Analytical profiles assess the numerical state; they do not initialize it.

The shared algorithm uses transported storage, integrated residuals, cell
constraints, physical trace equations, admissibility and mesh callbacks.
Case construction and analytical comparisons stay in the nozzle adapter.
Do not put Euler variable identities, perfect-gas inverses or copied boundary
laws into the multigrid core. Compatibility with this contract supports
later model adapters; it does not validate untested models.

## The two algorithms

Grid sequencing solves each original grid problem once, transfers the
successful numerical state upward and carries its last accepted CFL. It
does not revisit a coarse grid to correct a fine defect. Failed stages stop
the hierarchy; no independent cold retry replaces them.

Initialized FAS solves an original coarsest problem, prolongs the state,
performs a bounded number of recursive V-cycles on intermediate grids, then
uses V-cycles to converge the original finest problem. Each downward visit
restricts transported contents and constructs a new forced coarse problem:

$$
f_H=R_H(W_H^0)+\mathcal R_h^H\left[f_h-R_h(W_h)\right].
$$

Here $R$ is the signed, integrated transport residual, $f_h=0$ on the physical
finest problem, and $\mathcal R_h^H$ sums child balances. State restriction
also sums contents, followed by storage recovery and trace completion.
Boundary and cell-constraint rows receive no FAS forcing. The coarse result
supplies an additive transported-density correction; it does not overwrite
the fine solution with a homogeneous coarse solution.

Keep a genuine recursive correction cycle on unconverged visits. An optional
early check after finest pre-smoothing may skip the remaining coarse and
post-smoothing work only when all original finest convergence checks already
pass. Count that exit explicitly. Never use an unforced physical success
test to accept a recursively forced coarse problem.

## Bounded tuning controls

Begin with the existing initialized FAS reference, then vary a small number
of controls in each candidate. Avoid a large Cartesian sweep. Record complete
settings, failures and achieved work for every trial.

| Control | Question tested |
|---|---|
| Intermediate initialization cycles | Can fewer cycles suffice, or can additional cheaper intermediate cycles save fine work? |
| Pre/post smoothing | Can a one-sided cycle replace one global implicit sweep? |
| Coarsest forced visits | Can fewer accepted steps and a modest defect reduction replace unnecessary coarse work? |
| CFL start, growth and cap | Can the controller reach useful large steps with fewer retries? |
| Minimum inherited CFL | Can an explicit upward CFL floor avoid a slow fine transient after approximate coarse initialization? |
| Auxiliary coarse initialization target | How accurately must the original coarsest problem be solved before ascent? |
| Constant or limited-linear initialization | Does a more accurate conservative transfer reduce fine Newton work? |
| Coarse reconstruction | Can constant reconstruction reduce auxiliary work while retaining the chosen finest reconstruction? |
| Exact or intentionally inexact smoothing | Can bounded Newton work provide useful smoothing at lower complete-cycle cost? |
| Convergence-aware exit | Can a converged finest pre-step avoid an unnecessary descent and post-step? |

Keep candidate ranges small: one to three intermediate cycles; zero or one
pre/post sweep, with a functioning cycle retained; one, two or four coarsest
steps; and a few CFL starts/growth factors. Test auxiliary initialization
targets and inherited CFL floors explicitly. Saved configuration is
authoritative for the exact values actually tried.

The coarsest **initialization** target differs from the forced coarse
reduction target during a V-cycle. A looser auxiliary initialization target
must have its own pass flag; original residual flags remain truthful.
Finite state, admissibility and algebraic checks still apply. The original
finest transport, boundary, constraint and global targets remain $10^{-9}$.

### Conservative limited-linear initialization

Reconstruct arbitrary transported densities with minmod-limited slopes from
physical coarse-cell centers and one-sided endpoint slopes. For parent $J$,
use the volume-weighted child center $\bar x_J$:

$$
q_{h,i}=q_{H,J}+\theta s_{H,J}(x_{h,i}-\bar x_J),
\qquad
\bar x_J=\frac{\sum_{i\in\mathcal C(J)}V_{h,i}x_{h,i}}
{\sum_{i\in\mathcal C(J)}V_{h,i}}.
$$

This gives exact parent contents,

$$
\sum_{i\in\mathcal C(J)}V_{h,i}q_{h,i}=V_{H,J}q_{H,J},
$$

including unequal child volumes and noncentroid parent centers. Recover
physical states through the model's storage inverse and complete boundaries.
Reduce the scalar slope factor after a failed recovery/admissibility check,
count all trials, and allow an explicit $\theta=0$ constant-transfer fallback.
Do not clip physical variables. Keep this initial-state transfer separate
from conservative additive correction prolongation.

### Optional inexact implicit smoothing

Take one or two coupled Newton corrections with line search, then check the
actual backward-Euler residual against a configured reduction target.
Boundary completion, admissibility and algebraic targets remain explicit.
This acceptance policy does not claim native Newton convergence. A failed
attempt continues with exact Newton for the same old state, timestep and
forcing; both parts of the work are retained.

Constrained models use the exact smoothing path in this prototype. Fine
success always comes from a fresh original-equation assessment and physical
model audit. Changing the native Newton cap alone is insufficient: a capped,
unconverged native step must not silently become an accepted exact step.

## Pilot selection and final comparison

1. **Check implementation first.** Audit nested volumes, signed defect sums,
   storage round trips, zero-forcing parity, the FAS fixed point and limited-
   linear transfer. Use small flow checks before freezing timing source.
2. **Run bounded pilots.** Start with the two linear shock cases. Compare a
   compact list around the reference, then combine promising controls.
   Keep pilot measurements separate from the final timing population.
3. **Select universal settings.** Evaluate shortlisted candidates on all
   eight groups, including constant shocks and smooth/adapted cases. Select
   one configuration using convergence reliability, complete-solve work and
   timing across the suite. Do not present different per-case winners as
   one universal solver.
4. **Separate controller effects.** Retain the production sequence baseline.
   If a FAS gain accompanies a more aggressive CFL policy, run a separately
   identified sequencing diagnostic with matched CFL settings. These remain
   instances of the same two algorithms. A controller advantage must not
   be attributed solely to multigrid correction.
5. **Freeze and measure.** Record exact source/configuration identities,
   exclude one complete solve per strategy and group, then run ten measured
   complete replays in rotating order. Run CPU simulations sequentially.
   Reset fields, controllers and histories on every replay.
6. **Audit and report.** Independently recompute original residuals, branch,
   analytical error, conservation and agreement with the same finest
   discrete solution. Retain every sample; report mean, minimum, maximum,
   dispersion, actual representative trajectories and per-level work.

Keep the inherited constant-reconstruction 30% shock-position accuracy miss
visible. Do not alter the target or estimator to make a timing row eligible.
Report convergence, branch, analytical accuracy, field agreement and timing
eligibility separately. A failed or capped run remains unsuccessful and is
retained in the evidence.

## Timing, compilation and evidence

Complete solve time includes initialization, all levels, transfers, recovery,
boundary completion, monitoring, rejections and fallback work. Independent
final audits, reference generation, serialization and plotting stay outside
warm timing. Synchronize JAX before stopping the timer. Compare complete
drivers rather than a kernel against a whole solve.

Reuse compatible immutable level/operator bindings. Keep forcing, state,
timestep and damping dynamic; do not build a new compiled closure per cycle.
Reset mutable work independently for each replay. Persistent caches and
observed backend events describe executable reuse; they do not establish
shorter first-use compilation. Measure any new startup claim with separate
fresh-process/cache trials rather than shared method warmups.

Show only the active two-strategy comparison in the main report, with tuning
diagnostics identified separately. Include every sequencing grid's accepted
iterations and boundaries. A V-cycle and a pseudo-step are different work
units. Use actual recorded FAS elapsed times and sequencing stage-ending
times; do not invent sequencing per-step timestamps. Initialization is
charged before the first fine cycle and remains visible in its ledger.

Preserve raw states, settings, histories, timing samples, failures and source
archives under `outputs/`. Verify unchanged package hashes and no unexpected
warm compilation. Rebuild the standalone offline report from saved data,
then check navigation, plots, equations, themes and downloads in an actual
browser. Source, documentation and configuration must describe the measured
final algorithm even if the existing sequence remains faster.
