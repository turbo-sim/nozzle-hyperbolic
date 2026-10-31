# Adaptive CFL and Newton enhancements

**Status:** proposals for implicit steady-state solves, assessed on 31 October
2026. This note documents recommendations; it does not implement them or report
new simulations. The policy should remain model-independent for nozzle, annular,
and future compatible physical models.

## Assessment

Keep a timestep reduction factor of **0.5 for clear nonlinear failures**.
Halving is a conventional fallback: MOOSE's iteration-adaptive timestepper uses
0.5 for failed solves. The more promising improvement is to distinguish a
Newton solve that needs additional corrections from one that is stagnating or
producing invalid trials. Also, CFL should not automatically grow after an
expensive successful solve. [MOOSE documentation](https://mooseframework.inl.gov/source/timesteppers/IterationAdaptiveDT.html)

The recommended first experiment is to coordinate the Newton budget and CFL
controller, retaining strict step acceptance and the existing final steady
convergence checks. Simply replacing every factor 0.5 with 0.8 is lower priority.

## Current implementation

The relevant entry points are
[`SteadySettings` and `Scheme`](../src/nozzle_hyperbolic/solvers/fvm/config.py),
the [steady/transient drivers](../src/nozzle_hyperbolic/solvers/fvm/drivers.py),
the [Newton solver](../src/nozzle_hyperbolic/solvers/time_marching/implicit.py),
and the [operator](../src/nozzle_hyperbolic/solvers/fvm/operator.py).

| Setting | Inspected working-tree value |
|---|---|
| Initial CFL | 0.2 |
| Accepted-step CFL growth | 1.2 |
| Rejected-step CFL shrink | 0.5 |
| Maximum CFL | $10^6$ |
| Newton correction cap | 10 |
| Newton update tolerance | $10^{-10}$ |
| Newton residual tolerance | $10^{-12}$ |
| Flow Newton backtracking | Off; full undamped corrections |
| Retry budget | 14 retries after the first attempt |

These values describe the checkout inspected for this note. Its CFL growth and
maximum differ from the committed configuration at inspection, which had 1.4
and $10^7$. Committing this document does not include those existing source edits.

The steady controller currently:

1. Starts each attempt from the last accepted state.
2. Requires the inner solve to converge and remain admissible. Both configured
   Newton tests must pass; a small update alone is insufficient.
3. Checks finite/admissible spatial diagnostics before accepting the candidate.
4. Multiplies CFL by 1.2 after every accepted attempt, regardless of Newton work,
   retries, or steady residual trend.
5. Discards a failed candidate, multiplies CFL by 0.5, and retries from the
   unchanged accepted state.

Acceptance does not require the steady spatial residual to decrease at every
step. That is not inherently an error: nonlinear pseudo-time trajectories can
have temporary increases. However, the growth decision currently ignores this
information altogether.

Newton status 1 identifies iteration-cap exhaustion; status 3 identifies failed
trial acceptance. Undamped Newton permits finite, admissible corrections even
when their inner residual increases. Physical admissibility, auxiliary boundary
completion, and the final steady convergence audit must remain distinct from
the CFL adaptation logic.

For implicit integration, timestep is the requested CFL times a wave-based
timescale. At the same accepted state, halving CFL therefore halves the timestep.
With growth 1.2, four successful growth updates recover a halving:
$0.5\times1.2^4\approx1.037$. The associated rejected work and recovery steps
can still be expensive when failures recur.

## Proposed decision policy

The values below are starting points to test, not established optimal settings
or currently supported configuration options.

| Observed situation | Proposed response |
|---|---|
| Easy convergence and a productive steady residual trend | Grow CFL by approximately 1.1–1.2 |
| Converged step with substantial Newton effort or recent retries | Accept, but hold CFL |
| Newton reaches its normal budget while residuals are contracting strongly | Continue the same nonlinear solve with a limited extra budget |
| Newton stagnates, cycles, or exceeds the extended budget | Reject and reduce the timestep; retain 0.5 as the fallback |
| Nonfinite/inadmissible trial or unrecoverable solve failure | Reject promptly and reduce the timestep |

A target band around **4–6 corrections** could be an initial iteration-effort
setting, with a hold region to avoid unnecessary growth/shrink oscillations.
It is not a universal CFD optimum. The objective is minimum total time to the
verified steady solution, rather than a prescribed inner iteration count.

After an expensive accepted step, require approximately **2–3 easy, successful
steps without retries** before resuming growth. This gives the solution an
opportunity to evolve at the current CFL without immediately increasing the
nonlinear difficulty. A gentler preventive reduction, such as 0.8–0.9, could be
tested later for difficult but converged steps, separately from failure recovery.

### Additional Newton corrections before cutting the timestep

An initial policy to investigate is a normal budget of 10 corrections, optional
extensions to 20 and then 50 while convergence remains productive, and a hard
safety cap of 100. Every solve should stop as soon as its convergence tests pass;
a cap of 100 does not require 100 corrections.

Use a recent residual-contraction trend, rather than requiring every Newton
correction to decrease the residual. Do not extend indefinitely through slow
contraction, cycling, invalid states, or repeated failed trial acceptance.

An extension must preserve the original backward-Euler problem:

- Keep the last accepted state, timestep, boundary evaluation time, scaling,
  and physical equations fixed.
- Continue from the latest valid Newton iterate; do not redefine it as the old
  time level or restart the same work from the original guess.
- Require the same convergence/admissibility criteria before accepting the step.

Repeating an identical deterministic failed solve with the same initial guess,
timestep, and budget provides no new opportunity. Continuing Newton means
performing additional corrections on the same target equations.

This proposal is different from accepting any finite iterate when the cap is
reached. The earlier accept-at-cap experiment did not establish a benefit and
was reverted. It should not be silently reintroduced as a controller change.

### Diagnostics needed

The history already records final inner status, iteration count, residual norm,
rejected trials, requested CFL, actual timestep, and accepted/rejected attempts.
The proposed progress-based extension additionally needs a short inner-residual
history or an equivalent convergence-rate summary, which is not currently
exposed by the outer driver. Record which gate permitted an extension or CFL
change so its cost can be assessed independently of the physical model.

## Established approaches and alternatives

**Iteration-based adaptation** grows, holds, or shrinks the timestep according
to a target iteration band. MOOSE documents this pattern, including separate
responses to difficult converged solves and failed solves. This provides a
useful design reference, not proof of optimal thresholds for this package.
[MOOSE documentation](https://mooseframework.inl.gov/source/timesteppers/IterationAdaptiveDT.html)

**Switched evolution relaxation (SER)** adapts pseudo-time steps using reduction
of the steady spatial residual. One possible bounded form is

$$
\mathrm{CFL}_{n}=\operatorname{clip}\left[
\mathrm{CFL}_{0}\left(\frac{R_0}{R_n}\right)^p,
\mathrm{CFL}_{\min},\mathrm{CFL}_{\max}\right],
$$

where $R_n>0$ is a consistently scaled steady residual norm. A future implementation
must stop at convergence and protect the denominator against zero. The exponent,
growth limits, and response to noisy/nonmonotone residuals require tuning. A
PETSc–FUN3D study describes this family of controllers and emphasizes the
dependence on flow regime and discretization. SER is a separate controller
candidate, not part of the first Newton-budget experiment.
[PETSc–FUN3D study, section 2.4.1](https://www.mcs.anl.gov/research/projects/petsc-fun3d/Papers/sc00.pdf)

**Several convergence signals can inform CFL adaptation.** SU2's documented
controller considers limited updates, linear convergence, and stagnant or
oscillatory nonlinear residuals. Its linear-iteration criteria do not transfer
directly to this package's banded direct linear solver, but its conditional
growth policy is relevant.
[SU2 configuration template](https://raw.githubusercontent.com/su2code/SU2/master/config_template.cfg)

**Linearly implicit pseudo-transient continuation** deliberately performs a
Newton-like correction per pseudo-step instead of fully converging backward
Euler. PETSc implements this formulation and explains why multiple nonlinear
corrections change that particular algorithm. It is a valid alternative to
study separately, not a justification for arbitrary acceptance of failed
backward-Euler steps. Also, “inexact Newton” commonly refers to approximate
linear solves; it should not be used as an ambiguous label for cap exhaustion.
[PETSc pseudo-transient implementation](https://petsc.org/release/src/ts/impls/pseudo/posindep.c.html)

## Evidence already available

The retained [Newton-cap benchmark](roberto/nozzle_native_performance/CAP_BENCHMARK.md)
compared the new undamped solver at caps 10 and 100 with matched settings. These
are previous measurements, not runs performed for this note.

| New solver case | Mean runtime, cap 10 → 100 | Accepted steps | Largest actual inner count at cap 100 |
|---|---|---|---|
| Shock at 35%, MUSCL | 1.8915 → 1.2052 s | 143 → 91 | 48 |
| Subsonic, MUSCL | 0.8256 → 0.6419 s | 84 → 66 | 25 |
| Shock at 70%, constant | 1.1583 → 1.1476 s | 116 → 116 | 10 |

The shock35/MUSCL case also reduced rejected attempts from 24 to 10. In
contrast, shock70/constant retained 17 rejected attempts and 637 corrections at
both caps, with no cap-exhaustion failures. Those rejections were not caused by
insufficient iteration budget. Cap 100 was still reached in another shocked
MUSCL case, so it is not universally nonbinding.

This supports distinguishing failure reasons before tuning a universal shrink
factor. It does not establish the proposed controller's performance on annular
flow or future physical models.

## Validation sequence

1. Establish a baseline with the current equations, tolerances, initialization,
   mesh, flux, reconstruction, boundary treatment, and Newton globalization.
2. Classify existing rejections by cap exhaustion versus failed trials; measure
   accepted/rejected work and actual CFL trajectories.
3. Test conditional growth/holding and bounded Newton extensions in separately
   attributable variants before combining them.
4. Compare a gentler failure shrink factor only after that diagnosis; do not
   change the controller, cap, tolerances, and flux simultaneously.
5. Expand from a controlled difficult nozzle case to smooth/shocked nozzle and
   annular cases once the mechanism and final-solution agreement are verified.

Assess synchronized wall time, total Newton corrections, rejected work, actual
timesteps, failure reasons, and the final original spatial residual. Keep the
existing boundary/global conservation, admissibility, physical-branch, and
analytical-accuracy checks. Compare final discrete states as well as success
flags. Use excluded complete warmup solves and repeated sequential timings for
performance claims; do not infer speed from fewer outer steps alone.

A later, separate investigation could consider adaptive inner tolerances to
avoid oversolving early pseudo-steps. Inner backward-Euler residuals and outer
steady residuals have different scaling; their tolerance numbers cannot simply
be equated. Do not relax the final steady convergence requirements.

## Physical-time and explicit integration limits

These recommendations primarily concern implicit pseudo-time marching toward
steady state. The physical transient driver currently uses a requested CFL
without the steady growth policy and halves the trial timestep after failure,
with up to 16 attempts. Physical-time accuracy requires timestep refinement or
appropriate temporal error control in addition to nonlinear convergence.

Explicit integration remains subject to its wave, source, and filter stability
restrictions. Large implicit pseudo-time CFL values and Newton-budget policies
must not be copied to explicit Euler or RK3. No source changes, numerical tests,
or new timing campaigns are part of this documentation task.
