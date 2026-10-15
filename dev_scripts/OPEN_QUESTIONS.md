# Open questions — `nozzle_hyperbolic/fvm_solver/two_fluid/solver.py`

Consolidated 2026-08-04 from a repo-wide flag/comment audit. These are
questions that need **your** judgment and probably another probe run, not
things I resolved or should have silently "cleaned up". Each entry: claim →
evidence → recommended action → probe to rerun (all probes already exist,
no new ones needed).

## 1. `nc_form` pairing correctness — RESOLVED
cell is certified correct. It is possible to use wb for stability reasons, however it is a mathematical trick to improve convergence

## 2. Steady-state residual definition — `ssr_mode` is a workaround, not the fix

`solver.py:2210`. The steady-state residual is

    S_ss = t_char * R_ss / (V_cell * |V|)      t_char = dx/max(lam_max)
                                                V_cell = A*dx
         = R_ss / (A * max(lam_max) * |V|)      (the two dx cancel)

then reduced to a scalar by an RMS over all cells and all six rows
(`ssr_mode='all'`, the default, self-labeled legacy in-code) or by a
mass-fraction-weighted RMS (`ssr_mode='mass'`).

**What is right about it.** Dimensionless, so the six rows are comparable at
all (raw they are kg/m3/s vs kg/m2/s2 vs J/m3/s); relative, so no hand-tuned
per-equation constant is needed; scaled by a wave speed, so it is tied to the
physics that sets the timestep.

**What is wrong with it.** The normalisation is the *local, per-phase*
conserved variable `|V|`. A relative residual is right when `V` is a
meaningful magnitude and catastrophic when `V -> 0`: for a ghost at
alpha = 1e-3 the denominator is ~1e-4 of the mixture, so a negligible absolute
residual is reported as an enormous relative one. Measured: ghost rows owned
**99.98%** of the plain SSR2 on a converged N600 shock run, ghost/dominant
ratio **7553x**.

That is not a ghost problem. It is a normalisation problem that the ghost
merely exposes, and `ssr_mode='mass'` cancels the spurious amplification
rather than preventing it. The cost of the workaround: a genuinely diverging
ghost is suppressed too, so the convergence metric cannot warn about it
(`pin_ghost` and a mass-flow conservation check are the current cover).

Two secondary defects:

- **Not mesh-independent.** `R_ss` carries a factor of cell size, so
  `ssr_tol = 1e-8` is a different physical standard on N=420 than on N=600.
  SSR values from different grids are not comparable.
- **`t_char` uses a global `max(lam_max)`**, so cells in slow regions are
  scaled by a transit time much shorter than their own, weighting the fast
  regions more.

**Planned fix, in this order** (order matters — steps 2 and 3 make things
worse if done before step 1):

1. **Reference-scale each row against a mixture-level or global magnitude**
   instead of the local per-phase `|V|` — e.g. gas mass residual over
   `(m_g + m_l)`, momentum over the inlet momentum flux, or a running max over
   the field. A near-massless phase then contributes a near-zero residual,
   which is the honest answer.
2. **Equal weighting across rows becomes correct**, and `ssr_mode` stops
   needing to exist. Keep `'mass'` and `'all'` accepted for reproducing old
   runs.
3. **Reduce with a max over the six equations** rather than an RMS, so the
   worst-converged equation controls the stop instead of hiding behind five
   good ones. RMS over *cells*, max over *equations* — a max over cells too
   would let one shock or sonic cell in a limit cycle veto convergence
   forever, structurally the same failure as the ghost.
4. **Report all six per-equation values every `print_every`.** Worth doing
   regardless of the above: today SSR is one scalar that says nothing about
   *what* is lagging. "E_g limiting at 3e-7 while mass_l is at 2e-9" is
   actionable; "SSR = 3e-7" is not. This would also have made the ghost
   pathology visible immediately instead of requiring a dedicated probe.

**Consequences to plan for.** A max-of-six is always >= the current RMS, so
existing `ssr_tol` values become stricter and every reference case needs a
re-baselined tolerance. Archived NICFD 2026 cases record `ssr_mode: mass` in
their `settings.yaml`; those files need updating if those cases are ever
re-run under a new criterion.

**Stopgap until then:** `ssr_mode='mass'` is the right choice for any case
with a thin ghost, and is what the flashing cases use. Promoting it to the
default is still reasonable, but it is a patch — the real fix is step 1.

## 3. `w_char='auto'` — candidate for promotion

Dated 2026-07-17, certified by `debug_wref_sensitivity.py`, whose own output
recommends: *"'auto' (c·field-max) is safe as the general default."* The
actual kwarg default is still `None` (fixed two-component-calibrated floors).

**Recommended action:** consider promoting `w_char='auto'` to the default;
the certifying probe already argues for it. Rerun
`debug_wref_sensitivity.py` if you want to re-confirm before flipping.

the auto otpion should be the correct one to always have a well posed problem

## 4. `eos_rows` (8-row E3-augmented system) — open/unresolved

Self-labeled "prototype, 2026-07-15" at its definition. Fixes a measured
secular liquid-mass drain (−1.5e-4/step, kills ghost-limit runs at ~4700
steps) but `dev_scripts/config_loader.py` default stays `False`, and it's
never used in `main_two_fluid.py`'s default config — only exercised from
`debug_*.py` probes (`debug_banded_e3.py`, `debug_jac_lag.py`,
`debug_outlet_bc.py`, `debug_throat_texture.py`, `debug_wref_sensitivity.py`).

**Recommended action:** this fixes a real bug but is still "prototype" —
worth a decision on what's left before promoting it to default (the audit
didn't find an explicit blocker, just that it never got promoted).

eos_rows should be the default as said in the paper.

## 5. `eps_ghost_k2` doc/default drift — not a science question, just stale

Inline comment at the filter's use site: *"k2 (default 4): for a 2Δx mode the
4th difference is 4× the 1st difference... k2=1 was ×4 weaker in the shock
zone — held the N600 cycle at ~14% instead of draining it."* The actual kwarg
default (and `config_loader.py:215`) is `eps_ghost_k2=0.0`.

**Recommended action:** one-sentence decision needed — is 0.0 intentional
(filter off by default) or should the default be 4 per the comment's own
recommendation? If 0.0 is intentional, the comment should say so instead of
reading as an un-applied recommendation.

if we are not using it we should delete it...

---

## Already retired (2026-08-04), for reference

Two branches were downgraded from silently-selectable to a hard
`ValueError` in `solver.py`, since both had explicit textual evidence of
being dead (not a judgment call):

- `eos_project='triangular'` — comment said "FAILED certification — do not
  use"; only ever selected by `dev_scripts/debug_projection_tri.py`, which
  exists to document that failure.
- `bc_inlet='stagnation_ramped_soft'` — zero references anywhere in the
  repo, undocumented, contained an inline alternate Newton step explicitly
  marked "BROKE convergence."

Both implementations are kept commented-out in place (not deleted) as the
historical record of what was tried.


---

*Added 2026-09-10, from the van Albada eps-normalisation work
(`va_eps_mode`, solver.py `_muscl`).*

## 8. `va_eps_mode='smoothness'` diverges from a stagnation IC on coarse grids — OPEN

`results_grid_limiterfix/two_component_N150_*` (CFL 5.0, `va_eps_mode:
smoothness`) died with `NaN/Inf in state at step 1 (dt=1.16e-07)`.  The same
grid family runs fine at N=256/358/500 with the same mode and CFL 10, and N=150
is fine under `va_eps_mode: magnitude`.

**Claim:** this is a start-up problem, not a steady-state one.  The stagnation
IC is far from the solution; the smoothness gate (correctly) does not suppress
the large slopes that the IC's discontinuous start produces, where the
`magnitude` floor did — it crushed every slope on a coarse grid, so it never
saw them.

**Why it matters:** it blocks extending the convergence ladder DOWNWARD.  The
existing smoothness ladder {256, 358, 500} has N=256 under-resolved and
N=358/500 already agreed to ~0.004%, so a coarser grid is exactly what is
needed to find three grids inside one asymptotic range.

**Recommended action:** don't weaken the gate for this.  Ramp the IC instead —
start on `va_eps_mode: magnitude` (or a low CFL) for the first few hundred
steps to get off the stagnation state, then switch to `smoothness` for the
converged solve.  The converged steady state is dt- and path-independent IF it
is unique (see #9), so a two-stage start is legitimate.

**Probe:** rerun N=150 smoothness from a converged N=150 magnitude solution
(`ic.type: restart`) instead of `ic.type: stagnation`.  If it converges, the
diagnosis is confirmed and the two-stage start is the fix.

## 9. Is the converged steady state CFL-independent? — UNVERIFIED, blocks a paper claim

Every converged run in `results_grid_limiterfix` is at CFL 10; every run in
`results_grid_sesitivity` is at CFL 1-2.  So limiter mode and CFL are fully
confounded in every old-vs-new comparison made so far.

**What is proven:** the steady state cannot depend on dt *if it is unique*.
`solver.py:2216` evaluates the steady residual at `dt=1.0` and normalises by
`t_char = dx/lam_max`; and in `V_new - V_n + dt*(flux+source)`, steady state
gives `flux+source = 0` with dt cancelling.

**What is NOT proven:** uniqueness.  If the discrete system has several steady
states, a large early dt can select a different one — at CFL 10 the acoustic
waves are ~10x under-resolved and implicitly damped, and the early transient off
a stagnation IC is exactly when they run.  This system shows non-trivial
solution structure (the N=358 `magnitude` limit cycle at SSR 1.1e-6, persistent
checkerboard modes, flashing metastability), so multiplicity is plausible.

**Recommended action:** CFL sweep at N=256 (cheapest grid, 1760 steps at CFL
10), `va_eps_mode: smoothness`, CFL = 2 / 5 / 10.  Agreement establishes
CFL-independence and lets it be stated in the paper; disagreement is a much
larger finding that would affect every ladder, including the NICFD study.


## 10. two_component: the INTERIOR does not converge, the exit does — LARGELY EXPLAINED (see 10a)

Found while closing the NICFD grid study (`nicfd_figures/gci_nicfd_table.py`).
Grids {500, 351, 256, 180}, all converged to SSR < 1e-8, `va_eps_mode:
magnitude`, one shared fingerprint.

**Claim:** every interior quantity of interest fails the Celik convergence
condition on EVERY valid triplet, while the exit node passes on all four.

  station                          result
  exit node  (x2, common node)     all 4 triplets converge, p = 1.53 / 1.60 /
                                   1.65 / 1.70, phi_ext consistent to 0.02%
  inlet node (x1, common node)     values monotone but |eps21| > |eps32|, i.e.
                                   differences GROW toward the fine end (R > 1)
  5 interior stations + 3 integrals  oscillatory or diverging on all 4 triplets;
                                   the three-coarsest triplet {351,256,180}
                                   gives R = -0.16 .. -1.64 on all 8 QoIs

So it is not one bad grid — dropping the finest (N=500) does not help.

**What IS established:**

  * The differences are REAL, not noise.  At x/L = 0.80 the spread across the
    four grids is 5.1e-04 relative: 2.3e12 x machine epsilon and 5.1e4 x the
    SSR tolerance.
  * The sequence converges over the coarse end and then STALLS.  Domain L2
    difference between successive grids: 180-256 = 1.049, 256-351 = 0.630
    (shrinks 1.66x, about right for 2nd order), 351-500 = 0.841 (GREW, 0.75x).
    Against the finest: N=180 -> 1.715, N=256 -> 0.900, N=351 -> 0.841.
  * The jitter is equivalent to a position shift of 0.07 to 1.55 cells, largest
    where the gradient is steepest (x/L = 0.35, |du/dx| = 2165).

**Five mechanisms were tested and ALL ruled out.  Do not re-propose these:**

  1. Shock-position jitter (shock at x/L ~ 0.593, moves non-monotonically:
     0.59608 / 0.59104 / 0.59319).  RULED OUT: the non-monotonicity is present
     UPSTREAM of the shock too (x/L = 0.20, 0.35, 0.50), where a downstream
     shock cannot reach.
  2. Rigid translation of the downstream profile by that shift.  RULED OUT:
     predicted/observed ratios scatter 0.04 to 1.01.
  3. The 2dx checkerboard sampled at a random sub-cell phase.  RULED OUT: the
     jitter is 17-2000x the local 2dx amplitude.
  4. Machine precision or incomplete iteration.  RULED OUT by the scale above.
  5. Interpolation error from cubic sampling.  RULED OUT: N=351 and N=256 share
     interior nodes at x/L = 0.2/0.4/0.6/0.8 (gcd(350,255) = 5), and there the
     interpolated and node-read differences agree to 0.00e+00 exactly.

**Why it does not block the paper:** the two reported results do not depend on
resolving it.  two_component is quoted at the exit node (four independent
triplets agreeing, phi_ext consistent to 0.02%, GCI bands ~10x wider than the
actual extrapolation scatter), and smd 15 bar in the interior (eight QoI
definitions, p = 2.07-2.84, asym.chk = 1.000-1.002).

**Recommended action / probes, in order of cost:**

  1. Rerun the SAME grids with `va_eps_mode: smoothness`.  Under `magnitude`
     the limiter keeps only 0.2-7% of alpha's slope; under `smoothness` it keeps
     ~100% in smooth cells.  N=256/358/500 smoothness runs already exist in
     `results_grid_limiterfix`.  If the interior non-monotonicity disappears the
     cause is limiter-related; if it persists it is in the scheme or the physics.
  2. Add N ~ 700 (r = 1.40 from 500).  If the L2 difference 500-700 keeps
     shrinking, the 256-351 stall was local; if it stalls again there is a
     fixed-amplitude error floor in the interior, which would be a solver-level
     finding.

### 10a. PROBE RESULT (2026-09-15) — limiter exonerated; it is the QoI and the ladder

Probe: `probe_oq10_limiter_interior.py`, comparing the magnitude ladder
{500, 351, 256} against the smoothness ladder {500, 358, 256} (both converged to
SSR < 1e-8; the ladders are not grid-identical because magnitude's N=358
limit-cycled, hence N=351 in the study).

**The limiter is NOT the cause.**  Under `smoothness` every interior QoI still
fails.  But the FAILURE MODE changes completely:

    interior eps21 (fine pair)      magnitude        smoothness
      x/L 0.95                     -2.11e-04         2.21e-05
      x/L 0.90                     -2.62e-04         2.85e-05
      x/L 0.80                     -5.14e-04         1.03e-04

  magnitude : eps21 and eps32 comparable, signs erratic -> genuine jitter, |R| ~ 1-5
  smoothness: eps21 collapses ~10x while eps32 GROWS -> |R| ~ 0.00-0.12, i.e. the
              fine pair has converged and eps21 sits at the noise floor

So `smoothness` IMPROVED the interior convergence; it moved the problem from
"jitter" to "ladder too fine".

**N=256 is pre-asymptotic.**  Region L2 differences on the smoothness ladder:

    x/L band     |500-358|   |358-256|   ratio   (2nd order predicts 1.96)
    0.25-0.45      0.320       3.122       9.8
    0.55-0.65      1.196      11.253       9.4
    0.65-0.80      0.053       2.320      44.1
    0.80-0.99      0.006       0.404      66.5

N=358 and N=500 have converged to each other; N=256 has not entered the regime.
No three grids span one asymptotic window.  The peak difference in BOTH pairs is
at x/L = 0.599 -- the shock.

**The QoI is implicated.**  The SHOCK POSITION converges where field values do
not:

    shock position, smoothness:  0.184198 / 0.184524 / 0.185392
                                 -> PASS p = 2.91, GCI = 0.134%
    shock position, magnitude :  0.184820 / 0.184671 / 0.184176
                                 -> p = 4.0, out of band
    p, alpha_g, u_g, u_l at x/L = 0.90 and 0.70: FAIL under both limiters

Interpretation: the shock moves 0.4-0.7 cells between grids.  A field value at a
FIXED station downstream superposes two processes converging at different rates
-- the local solution converging, and the discontinuity sliding past -- which can
partially cancel and produce non-monotone eps.  The shock position contains only
one process, so it converges.  For a shocked flow a pointwise field value at a
fixed station is simply a poor QoI.

**Status:** section 10 is no longer "unexplained".  The interior failure is the
combination of (i) a pointwise QoI straddling a moving discontinuity and (ii) a
ladder whose coarse grid is pre-asymptotic.  Neither is a solver defect.

**Remaining work, and it depends on section 8:** confirming this needs a
smoothness ladder with a COARSER third grid, so that three grids sit in one
window.  N=180 and N=150 both NaN at step 1 from a stagnation IC under
`smoothness` -- exactly section 8.  The two-stage start proposed there (run
`magnitude` first, switch to `smoothness`) is the unblock.  Then re-run this
probe and check whether the shock position still gives p ~ 2.9 on two
overlapping triplets, and whether the interior field QoIs recover.

**Does NOT affect the paper:** two_component is reported at the exit node (four
triplets agreeing, phi_ext consistent to 0.02%) and smd 15 bar in the interior
(eight QoI definitions, p = 2.07-2.84).  Note also that the exit QoIs pass under
`magnitude` but go oscillatory under `smoothness` (same eps21-collapse cause), so
the study's choice of `magnitude` is the configuration in which they are
measurable.

## 11. Smooth regime blending in drag and heat transfer (smoothstep option) — TODO

**Claim.** The interphase closures switch regime with kinks at `alpha_b = 0.3`
and `alpha_d = 0.7`. `calculate_drag_force_one_component`
(`source_two_fluid.py`) is a nested `jax.lax.cond`: bubbly for alpha <= alpha_b,
droplet for alpha >= alpha_d, LINEAR blend in between. The force is continuous
there but its slope in alpha jumps. The regime-dependent HTC (`regime_htc=True`)
uses the same linear blend. A kink at a point inside the domain can keep Newton
from converging.

**Evidence (2026-10-14, CO2 design case 4, new polynomial interfacial area,
thick ghost alpha_eps 1e-3, N=400).** The stage stops on an SSR plateau
(5.1e-6). Restarting from it for 60 steps, the state drifts at constant rate;
the drifting band sits at alpha = 0.67-0.70 (x = 0.71-0.73 m), just below
alpha_d. Not proven to be the kink -- see also section 12 for the other part
of the plateau.

**Action.** Add a `blend` option (`linear` = current | `smoothstep`) to the drag
and to the regime HTC: weight w = smoothstep((alpha - alpha_b)/(alpha_d - alpha_b))
(C1, zero slope at both ends) instead of the linear w, and the bubbly/droplet
branches blended through it rather than switched by `lax.cond`. Expose it in
the settings (`sources.blend`). Default `linear` until A/B'd.

**Probe.** Restart the N400 thick-ghost case-4 solution
(`case_study/data_assimilation/results/thick_N400_thick_N400_muscl_van_albada_20261014_172944`)
with each option and compare the plateau SSR and where the remaining drift sits.

## 12. Thick-ghost stages plateau at SSR ~1e-5: the algebraic momentum pin — TODO

**Claim.** With `pin_ghost_momentum: true` and `pin_mom_tau: null` (the
algebraic blend), the solver's own comment says the steady residual cannot
reach zero where the pin weight is between 0 and 1. With alpha_eps = 1e-3 the
pin releases around pin_mom_alpha = 10*alpha_eps = 1e-2, i.e. right in the
flashing-onset range, so a wide band of cells is partially pinned.

**Evidence (2026-10-14, case 4).** Plateaus scale with the ghost: thick ghost
(alpha_eps 1e-3, void 3e-3) 1.4e-5 / 8.5e-6 / 6.4e-6 / 5.1e-6 at N = 100 /
200 / 300 / 400; thinner ghost (alpha_eps 1e-6, void 1e-4) 3.9e-8 at N = 200.
At N=100 the plateau is an EXACT fixed point: 60 steps from it change the state
by ~1e-10 (round-off) while the SSR stays at 1.419e-5.

**Action.** Rerun a thick-ghost stage with `pin_mom_tau` set (defaults.yaml
suggests 1e-6) and check whether the plateau disappears.

**Probe.** Restart from
`case_study/data_assimilation/results/verify_S1_N100_muscl_van_albada_20261014_164504`
(cheap, N=100) with `numerics.pin_mom_tau: 1.0e-6`.
