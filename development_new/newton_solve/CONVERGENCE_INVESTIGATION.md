# Newton/SLSQP step control and implicit warm starts

This records the historical flux. Use `--flux-variant legacy` with the commands
below to reproduce it after the regularity corrections. Current defaults use
the corrected flux; see [the implementation report](../../docs/SOLVER_REGULARITY_IMPROVEMENTS.md).

Smaller initial steps did not resolve either shocked case from the simple
initial field. Implicit warm starts restored convergence only after substantial
pseudo-time progress. Both direct methods then reached the same discrete root
as implicit marching, but the measured total hybrid costs were higher.

These are 2026-10-26 results for the original 201-point mesh (597 unknowns),
perfect-gas backend, Stadtke FVS, Van Leer reconstruction, and unchanged model
boundaries. The common target is scaled spatial residual RMS $10^{-8}$.
No flux, limiter, boundary condition, physical residual, or Jacobian algorithm
was changed for this investigation.

## Why the original runs stalled

The original Newton failures used 200 trial attempts but accepted only 17
updates for `shock35` and 15 for `shock70`. They rejected 183 and 185 trials,
respectively. Late accepted fractions became as small as $10^{-7}$. SLSQP
exhausted 200 major iterations while its residual plateaued near
$5.0\times10^{-4}$ and $8.7\times10^{-4}$.

The saved failed fields are positive and have only a small supersonic patch
near the throat: maximum Mach numbers are approximately 1.001 to 1.022,
compared with 1.395 and 1.840 in the converged implicit solutions. Full Newton
directions from these stalled fields would produce negative pressure near the
throat. Even much smaller positive steps can increase the residual. For the
`shock70` failed Newton field, a fraction $10^{-3}$ increases the norm by
14.26 times; a fraction around $10^{-7}$ finally gives a tiny reduction.

The sampled compressed Jacobians agree with independent full AD to about
$1.6\times10^{-16}$ relative error. Dense Newton linear solves have relative
residuals around $4\times10^{-14}$ to $2.3\times10^{-12}$. Directional finite
differences approach the AD derivative as their perturbation shrinks. The
Jacobian becomes more poorly conditioned near choking, but these tests do not
indicate an inaccurate linear solve or a compressed-Jacobian implementation
error at the sampled states.

The evidence therefore points to a very narrow useful neighborhood for the
Newton linearization while forming the choked/shocked flow from the subsonic
seed. Implicit pseudo-time updates provide a different, regularized path to
that root. Hard flux/limiter/entropy-fix branches are plausible contributors;
the tests do **not** isolate a particular branch as a bug. See
[STALL_DIAGNOSIS.md](STALL_DIAGNOSIS.md) for numerical details and caveats.

The step cap also explains why a smaller *initial* fraction need not cure a
later stall. At fraction 0.25 and growth 1.5, configured caps are
0.25, 0.375, 0.5625, 0.84375, then 1.0. Newton grows this cap after accepted
moves even when backtracking shortened the actual move much further. Thus
small initial fractions mostly change the early trajectory. The additional
persistent-damping and slower-growth tests below address that limitation.

SLSQP here solves $\min 0$ subject to $R(z)=0$, with as many independent
constraints as variables. Its constraint linearization can determine the same
direction as $Jd=-R$, so Newton-like stalled trajectories are unsurprising.
It does not introduce the implicit method's pseudo-time storage term.

## Cold initial-step sweep

For each solver and shock case, the study tested fractions
1, 0.25, 0.1, 0.03, and 0.01 with growth 1.5, plus fraction 0.1 with growth
1.0 and 1.1. Every run started from the same simple field. Budgets remained
200 Newton trials or 200 SLSQP major iterations; these are different work
units. None of the 28 runs reached the target.

| Case | Lowest final Newton RMS among settings | Lowest final SLSQP RMS among settings |
|---|---:|---:|
| Shock at 35% of divergence | $5.20\times10^{-4}$ | $4.99\times10^{-4}$ |
| Shock at 70% of divergence | $8.61\times10^{-4}$ | $8.45\times10^{-4}$ |

Changing the controls changes the plateau and accepted path, but did not
provide convergence within these budgets. This does not exclude other control
schedules, starting fields, or larger budgets.

## Actual implicit-to-direct runs

Each hybrid starts again from the simple field, actually runs its implicit
prefix, and immediately switches to Newton or SLSQP. The direct schedule stays
at fraction 0.25 and growth 1.5 to isolate the warm-start effect. No analytic or
already converged field is supplied.

- `shock35`: switches after 5, 10, 20, and 40 implicit updates failed. Switching
  after 60 succeeded for both methods in all three repeats. Pure implicit
  marching takes 72 accepted updates.
- `shock70`: switches after 5, 10, 20, 40, 60, 80, 120, and 160 updates failed.
  A follow-up switch after 180 succeeded for both methods in all three repeats.
  Pure implicit marching takes 193 accepted updates.

These are the first successful **tested** counts, not minimum required counts
or optimal switching times. The follow-up requested 180, 190, and 192; the
search stopped at 180. Short warm starts of 5 to 20 updates did not solve either
case. Some later unsuccessful switches substantially increased the residual,
so convergence is not monotone with warm-start length.

Successful direct phases took six accepted updates each, ending around
$1.7\times10^{-14}$ (`shock35`) and $5.3\times10^{-14}$ (`shock70`). Their
scaled solution differences from the implicit roots were approximately
$1.6\times10^{-7}$ and $8.4\times10^{-7}$ RMS, well below the $10^{-4}$ agreement
criterion. Native termination and the analytic accuracy checks also passed.
Convergence is to the existing discretization, not an exact analytic profile.

| Case | Method after prefix | Prefix updates | Prefix [s] | Direct phase [s] | Total hybrid [s] | Pure implicit [s] |
|---|---|---:|---:|---:|---:|---:|
| `shock35` | Newton | 60 | 0.608 | 0.242 | 0.850 | 0.791 |
| `shock35` | SLSQP | 60 | 0.655 | 0.419 | 1.074 | 0.791 |
| `shock70` | Newton | 180 | 1.736 | 0.294 | 2.030 | 1.865 |
| `shock70` | SLSQP | 180 | 1.767 | 0.431 | 2.198 | 1.865 |

Total and reference times are medians of three sequential warm runs. Prefix
and direct columns describe the actual run having the median total, so their
sum agrees with that total apart from construction/handoff overhead. The total
enclosing timer includes this overhead and both stages. JAX first-call costs,
analytic reference construction, final audits, plotting, and output are
separate. BLAS threads are fixed to one. The `shock70` reference in this table
comes from the same follow-up batch as its successful hybrids; the broad sweep
had an implicit median of 1.832 s.

There is no demonstrated speed advantage here. The implicit reference uses
banded linear algebra, whereas the current PySolverView direct wrappers use
dense linear algebra and additional interfaces. Fewer accepted outer updates
do not imply less work: an implicit update also contains inner Newton steps.
For these cases, retain implicit marching as the practical baseline. The hybrid
is useful as a convergence experiment, but needs better globalization or lower
direct-solve cost before it is a demonstrated acceleration.

## Reproduction and outputs

Run from the repository root:

```powershell
# Both cases: cold controls, then warm switches (now including 180 and 190).
poetry run python -m development_new.newton_solve.investigate_convergence --no-show

# Reproduce just the late-switch follow-up.
poetry run python -m development_new.newton_solve.investigate_convergence --cases shock70 --skip-cold-sweep --warm-steps 180 190 192 --repeat 3 --no-show --output development_new/newton_solve/results_late_warm_start

# Independent saved-state operator and Newton-path diagnostics.
poetry run python -m development_new.newton_solve.diagnose_stalls
```

Use `--fractions`, `--warm-steps`, `--warm-first-step-fraction`,
`--warm-step-growth-factor`, and `--max-direct-iterations` for further
experiments. Repeating a run overwrites its selected output directory.

The recorded broad sweep is in [results_investigation](results_investigation/).
The successful late `shock70` follow-up is in
[results_late_warm_start](results_late_warm_start/). Each contains JSON with
complete stage states/histories/status, a trials CSV, accepted-history CSVs with
phase labels, and PNG/PDF convergence plots. Failed trajectories are retained.
Plots show an actual middle-duration successful repeat and mark the handoff
with a star. In the broad-sweep archive, the time axis uses accumulated stage
timers; construction/handoff overhead remains included in its enclosing total
measurements. The follow-up and the runnable script align stage clock origins
with the enclosing timer.

Verification checked every recorded trajectory's accepted indices and time
ordering, prefix/direct/total timing consistency, final residual agreement,
and independently recomputed solution differences. The successful follow-up
also exercises the recorded timer origins and the hybrid-only execution path.
Generated result directories remain ignored by Git; this report preserves the
main findings alongside the reproducible scripts.
