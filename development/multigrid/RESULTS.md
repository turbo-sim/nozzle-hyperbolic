# Tuned initialized FAS versus grid sequencing

The active report now contains two implicit algorithms: **FAS with
coarse-to-fine initialization** and **grid sequencing**. One FAS configuration
was selected from bounded pilots and replayed unchanged across all eight
Euler-nozzle case/reconstruction groups. Package `src/` is unchanged.

The tuned FAS has a lower mean than the original sequencing reference in
six of the seven accuracy-qualified groups. However, sequencing with a more
aggressive, matched nominal CFL policy is faster in both separately tested
linear shock cases. The main improvements therefore cannot be attributed to
multigrid correction alone.

Open [REPORT.html](REPORT.html) for profiles, full grid histories, work,
settings and embedded downloads. [README.md](README.md) describes execution;
[PLAN.md](PLAN.md) explains the tuning strategy. Superseded generated reports
and their older raw datasets have been removed. The current `outputs/fmg_*`
campaigns retain the main results, tuning trials, CFL controls and source
archives used by this report.

## Primary paired timings

The main population uses the original sequencing controller to retain the
requested reference: initial CFL 0.2, growth 1.2, maximum $10^6$. FAS uses its
selected controller and cycle settings below. Both algorithms use HLL,
50/100/200/400 cells, float64, the same finest reconstruction, physical
boundary laws, EOS and final original-equation targets.

There are 160 successful measured complete solves: ten per algorithm and
group, after one excluded complete solve for each of 16 configurations.
Method order rotates. The timer includes initialization, every level,
transfers, recovery, boundary solves, monitoring, retries and rejected work.
Setup, excluded warmup, independent assessment and I/O are outside these warm
timings. All samples and their dispersion are retained in the report and
[main raw summary](outputs/fmg_20261104/summary.json).

| Case | Reconstruction | Sequence mean [s] | Initialized FAS mean [s] | FAS / sequence | Accuracy-qualified comparison |
|---|---|---:|---:|---:|---|
| Adapted | Constant | 0.13405 | 0.09007 | 0.672 | Pass |
| Adapted | Linear MUSCL | 0.12462 | 0.06925 | 0.556 | Pass |
| Shock 30% | Constant | 0.44539 | 1.22033 | 2.740 | Shock-position accuracy miss; descriptive timing |
| Shock 30% | Linear MUSCL | 0.26521 | 0.41579 | 1.568 | Pass |
| Shock 70% | Constant | 0.24050 | 0.18921 | 0.787 | Pass |
| Shock 70% | Linear MUSCL | 0.35045 | 0.22318 | 0.637 | Pass |
| Subsonic | Constant | 0.15176 | 0.08855 | 0.583 | Pass |
| Subsonic | Linear MUSCL | 0.14969 | 0.08143 | 0.544 | Pass |

A ratio below one means a lower measured mean. The six accuracy-qualified
FAS gains are approximately 21–46%. FAS remains slower for the linear 30%
shock. The constant 30% shock row retains an inherited spatial-accuracy miss
for both algorithms and is excluded from accuracy-qualified timing claims.
These results compare complete solver strategies with different CFL policies.

## Selected universal FAS settings

The same selected configuration is used for every main case. Exact settings
are in [settings_fmg.json](settings_fmg.json) and each raw record.

| Control | Selected value |
|---|---|
| Initialization | Original coarse solve, conservative limited-linear ascent, two V-cycles on each intermediate grid |
| Implicit smoothing | Exact backward-Euler/Newton steps with fresh Jacobians |
| Pre/post smoothing | One pre-step and one post-step |
| Forced coarsest visit | At most one implicit step; forced-defect reduction target 0.5 |
| Original-problem CFL | Start 1, growth 2, maximum $10^8$ |
| Warm ascent CFL floor | $10^8$, with normal retry/backtracking safeguards |
| Forced coarse CFL cap | 10 on recursive forced visits |
| Finest early exit | Stop a cycle after pre-smoothing if all original finest checks already pass |
| Coarse reconstruction | Same reconstruction as the finest grid |
| Auxiliary coarse initialization tolerance | None; the original coarse solve retains the strict target |
| Retries | Up to 32, with all rejected work retained |

The forced-coarse cap does not cap the original coarsest initialization or
the current original problem being solved during ascent. It limits recursive
forced correction visits, while the original coarse/fine solves can use large
implicit steps. The reduction value is a target within a finite work budget,
not a claim that every coarse visit achieves a 50% reduction.

The linear initialization transfer uses minmod-limited transported-density
slopes and volume-weighted child offsets, preserving parent contents.
Recovery and physical trace completion check admissibility; all rejected
slope trials and constant-transfer fallbacks would be recorded. No main
representative solve needed a constant-transfer fallback.

The finest converged pre-step avoids an unnecessary descent and post-step.
That early exit occurred in the representative linear adapted, linear
subsonic and constant 30% shock solves. Every other unconverged visit retains
the recursive FAS correction. All initialization and monitoring work remains
inside the complete-solve timer.

## What the bounded tuning established

The [saved tuning record](outputs/fmg_20261104/tuning.json) retains 33 FAS
candidate identities over three source/configuration phases. Training used
only the linear 30% and 70% shock groups. Each successful pilot has two or
three warm replays after an excluded complete solve; failed excluded solves
remain visible without a fabricated timing population. Phases keep separate
raw/source identities and are not pooled into the final ten-replay samples.

Selection used the lowest geometric mean of complete-solve means across
both training groups among candidates passing original equations, branch
and state agreement. The selected candidate was `fas_fmg_cap10`; one
configuration was then frozen for fresh all-eight-group executions. Those
executions include the two training groups, so they are not all held-out
physical cases. A bounded search does not prove a globally optimal setting.

The useful observations were:

- A modest forced-coarse CFL cap performed better in its pilot phase than
  caps of 100 or 1000. This motivates keeping forced correction work gentler
  than the large-step original solves.
- An original/fine CFL ceiling and warm floor of $10^8$ formed part of the
  selected combination. Pushing them to $10^{12}$ produced failed smoothing
  attempts in the tested shock problems; larger is not automatically better.
- Conservative linear initialization and convergence-aware exit were retained
  with two intermediate cycles and both pre/post steps. Their individual
  effects were not isolated by the final comparison.
- Post-only cycles, intentionally inexact smoothing, looser auxiliary goals
  and lower-order coarse reconstruction were not universally better. Some
  variants failed a training case; others improved one shock and worsened
  the other. The selected main configuration uses exact smoothing and no
  relaxed initialization target.

All final finest targets remain unchanged: original transport, boundary,
constraint and global defects must meet $10^{-9}$. Exact inner Newton uses
its configured $10^{-12}$ residual and $10^{-10}$ update criteria. No
analytical profile, relaxed final tolerance or per-case winning setting was
used to obtain the main timings.

## Matched-CFL sequencing diagnostic

A separate diagnostic tests whether the main difference is due to the more
aggressive controller. Both algorithms use initial CFL 1, growth 2 and a
maximum of $10^8$, with up to 32 retries. FAS retains its explicit warm floor
and forced-coarse cap; actual timesteps and controller histories are therefore
not identical step by step.

These are 40 additional successful measured complete solves: ten per
algorithm in each linear shock case, after excluded complete solves. They
are separate paired observations, not replacements for or additions to the
main sample population. See the
[matched-CFL raw summary](outputs/fmg_matched_20261104/summary.json).

| Linear reconstruction case | Matched-CFL sequence mean [s] | Initialized FAS mean [s] | FAS / sequence |
|---|---:|---:|---:|
| Shock 30% | 0.19028 | 0.44363 | 2.331 |
| Shock 70% | 0.17946 | 0.21906 | 1.221 |

Sequencing is faster in both controls. Thus the data do not establish that
this FAS cycle beats well-tuned sequencing on the shock problems. The main
six-group gain is a useful improvement over the original reference strategy,
with an important contribution from controller choices. No matched-CFL
conclusion is claimed for the six other physical/reconstruction groups.

The first 70% control attempt stopped during JSON atomic replacement with
`WinError 5`. Its partial records are preserved under
`outputs/fmg_matched_20261104/interrupted_attempt_1/`. The whole group was
rerun from fresh states with identical source/configuration. Partial and
final timings are not pooled, and this I/O interruption is not classified
as a numerical solver failure.

## Independent checks and limits

All 160 main measured solves pass native success, fresh original spatial and
algebraic equations, admissibility and physical-branch checks. Each
configuration's ten repeated states is bitwise identical. Different
algorithms agree within the preset $10^{-6}$ scaled-state threshold; the
largest main difference from sequencing is $1.365\times10^{-8}$.

Analytical accuracy passes in 140 main solves, giving 14 of 16 eligible
configurations. The remaining 20 are both methods for constant reconstruction
with the 30% shock: the first Mach crossing is $x=0.1285$ m rather than
$0.1300$ m, three cells from the reference against a two-cell target.
Normalized field L1 targets pass. The target and estimator remain unchanged.

All 40 matched-CFL solves pass the same original, branch and analytical
checks; their largest scaled difference is $1.445\times10^{-8}$. Main and
matched populations pass before/after environment probes, show bitwise
repeat consistency and issue no new warm backend compile/load requests.
Ordinary desktop/OS timing noise remains possible. Executable reuse after
warmup does not demonstrate a reduction in first-use compilation time.

The fresh saved-state audits pass for all eight main groups and both matched
groups, checking 113 package Python/YAML files without source changes. The
main [numerical audit](outputs/fmg_20261104/audit.json),
[core algebra audit](outputs/fmg_20261104/core_algebra_audit.json) and
[transfer audit](outputs/fmg_20261104/transfer_algebra_audit.json) are retained
with source archives and raw states. Algebra covers conservative nonuniform
transfers, signed forcing, native Jacobian/residual parity and the FAS fixed
point. Limited-linear transfer errors are at float64 roundoff.

The standalone report shows every sequencing grid's accepted iterations and
stage boundaries. A V-cycle and a pseudo-step are different work units.
Initialization is charged before the first fine cycle and recorded in a
separate ledger. Numerical checks and current browser/presentation evidence
are stored separately with the main campaign. Scope is the 400-cell Euler
nozzle study; other models, larger meshes and first-use startup gains have
not been established by these timings.
