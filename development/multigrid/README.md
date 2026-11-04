# Initialized FAS versus grid sequencing

This study compares two implicit solution strategies for the Euler nozzle:
**FAS with coarse-to-fine initialization** and the package's **grid sequencing**.
All prototype code stays under `development/multigrid/`; package `src/` is
unchanged. The algorithms use the generic physical-model contract rather than
Euler-specific storage inverses or boundary laws.

Open [REPORT.html](REPORT.html) for the standalone offline report, or
[RESULTS.md](RESULTS.md) for the measured findings. The report embeds plots,
mathematics, runtime libraries, settings and downloadable numerical data. It
can be copied and opened alone. [PLAN.md](PLAN.md) records the implementation
and tuning strategy.

## Cases and common equations

The test set contains subsonic flow, internal shocks at 30% and 70% of the
**diverging section**, and adapted expansion. Each uses constant and linear
MUSCL Van Leer reconstruction, giving eight groups. The 30% case is a
study-local derivative of the packaged 35% case. The shock coordinate is
$x_s/L=(1+f_s)/2$. Analytical profiles are assessment data, never initial states.

Both strategies use HLL and the nested 50/100/200/400-cell hierarchy. The finest
spatial equations, physical boundary laws, EOS, precision and final stopping
criteria agree. Original spatial, boundary, global and constraint targets stay
at $10^{-9}$. Native exact implicit solves use the configured Newton criteria;
any intentionally inexact smoothing has a separate acceptance policy and is
identified explicitly in its saved FAS settings.

## The two strategies

Grid sequencing converges each grid once, prolongs its numerical state and
carries the last accepted CFL to the next grid. Failed stages stop the sequence;
no hidden cold restart replaces them.

Initialized FAS solves an original coarse problem, prolongs a conservative
state, applies a fixed number of recursive V-cycles on intermediate grids,
and then converges the original finest equations using V-cycles. At each
downward visit, transported contents are restricted conservatively and the
signed integrated defect defines the coarse forcing:

$$
f_H=R_H(W_H^0)+\mathcal R_h^H[f_h-R_h(W_h)].
$$

The additive prolonged correction is damped and recovered through the model's
storage inverse and physical boundary completion. Algebraic equations receive
no FAS forcing. A coarse correction cannot determine success: the original
unforced finest equations and physical solution audit do.

The tuning experiments examine coarse work, pre/post smoothing, intermediate
cycle counts, CFL controls, conservative limited-linear initialization and
convergence-aware cycle exit. An optional auxiliary initialization target
applies only to the coarsest original problem, with explicit auxiliary flags;
its original residual flags remain truthful. The final finest target is fixed.

Limited-linear initialization reconstructs transported densities with minmod
slopes. Volume-weighted child offsets give each parent its exact original mean.
Recovery checks admissibility and algebraic consistency; rejected slope trials
are damped and counted, with constant prolongation as an explicit fallback.

An optional inexact implicit smoother takes one or two coupled Newton
corrections, completes physical boundaries, and requires admissibility, strict
algebraic checks and a configured reduction of the actual backward-Euler
residual. It does not claim native Newton convergence. Unsuccessful attempts
continue with exact Newton for the same old state, timestep and forcing, and
all work is retained. Constrained models use the exact path in this prototype.

## Measurements and checks

Pilot trials select settings; they remain separate from the final paired timing
population. The main comparison uses one excluded complete solve and ten
measured complete solves per strategy and group. Method order rotates. Every
solve resets fields, CFLs and history while reusing immutable kernel bindings.
The CPU backend uses float64 and single-thread BLAS settings.

Total solve time includes initialization, every grid, transfers, recovery,
boundary solves, monitoring and rejected work. Setup/JIT warmup, independent
final assessment, serialization and plotting are outside the warm timer.
Success, physical branch, analytical accuracy and timing eligibility are
reported separately. The known constant-reconstruction 30% shock-position
accuracy miss remains visible; no target or estimator is adjusted to hide it.

The convergence view includes every sequencing grid's accepted steps and stage
boundaries. A pseudo-step and a V-cycle are different units of work. Sequencing
has actual stage-ending wall times but no saved per-step timestamps; the report
does not invent them. FAS initialization is charged before fine cycle one and
is recorded in a separate ledger.

Raw settings, source bytes, states, histories, rejected work and timing samples
for the current FMG study are preserved in the `outputs/fmg_*` campaigns,
including tuning trials and the CFL-control comparison. Superseded reports
and older raw datasets have been removed. Source
hashes and fresh saved-state audits check unchanged package files and agreement
with the same finest discrete root.

## Reproduce

From the repository root, using the existing environment:

```powershell
.venv/Scripts/python.exe -B -u -m development.multigrid.scripts.run --output development/multigrid/outputs/fmg_20261104 --resume
```

The default configuration is [settings_fmg.json](settings_fmg.json). For a new
campaign use a fresh `--output` directory; resume rejects changed configuration
or numerical source. Case, reconstruction, method and repeat subsets can be
selected explicitly on the command line.

```powershell
.venv/Scripts/python.exe -B -m development.multigrid.scripts.audit --algebra
.venv/Scripts/python.exe -B -m development.multigrid.scripts.audit development/multigrid/outputs/fmg_20261104
.venv/Scripts/python.exe -B -m development.multigrid.scripts.run --output development/multigrid/outputs/fmg_20261104 --assemble-only
```

Report generation reads saved results without advancing a solver. Presentation
checks cover offline packaging, every grid's residual/count data, controls,
embedded downloads, real browser plots/math and visual layout. Independent
startup measurements, if run, remain separate from warm solve timings.

## Implementation

- `scripts/core.py`: model-independent implicit smoothing, conservative recovery,
  FMG initialization and recursive FAS.
- `scripts/transfers.py`: conservative limited-linear initialization.
- `scripts/nozzle_case.py`: packaged nozzle cases and independent references.
- `scripts/run.py`: reusable bindings, sequential pilots/benchmarks and provenance.
- `scripts/audit.py`: algebra and independent saved-state checks.
- `scripts/report.py`, `scripts/report_assets/`: saved-data offline report and QA.
