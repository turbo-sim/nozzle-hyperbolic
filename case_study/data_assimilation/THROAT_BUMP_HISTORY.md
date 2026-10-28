# Throat region of CO2 case 4: bump, jump, slip-dependent heat transfer and ṁ (2026-10-22/23)

## Summary and decision (2026-10-23)

**Decision, revised (2026-10-23, later).** For the truth and the data assimilation model:
`htc_liquid_bubbly: max` (larger of Ranz-Marshall and the Plesset-Zwick Jakob Nu) with the smooth
slip OFF, replacing u0 = 2. It is smooth on both area laws (no polynomial bump, no parabola jump)
and leaves ṁ unchanged (see "Liquid-side HTC options"). twophase-da `defaults.yaml` carries it.
The solver defaults are back to the original closure (`smooth_slip=False`,
`htc_liquid_bubbly='ranz_marshall'`), so other nozzle-hyperbolic cases reproduce as before.

**Decision (superseded).** Go on with the smooth effective slip in the Ranz-Marshall HTC (`smooth_slip` on,
u0 = 2 m/s, the solver default) for the truth and for the data assimilation model, and move to
the EnKF. With it the polynomial-law solution is smooth; the parabola-law solution (the truth)
keeps a converged compression jump at x ≈ 0.595, accepted as a feature of the model solution.

**Two different features, both between the geometric throat (x ≈ 0.540) and α ≈ 0.1-0.25:**

| | polynomial bump | parabola jump |
|---|---|---|
| where | x ≈ 0.568, α ≈ 0.08, subsonic (M 0.83 → 0.93), 8 cells before the sonic point | x ≈ 0.595, α 0.16 → 0.30, supersonic (M 1.1-1.16 → 1.04), at the wall inflection |
| shape | p drops ~2 bar over 2-3 cells, slip changes sign, σ_Mg one-cell spikes | p rises 18 → 21 bar (compression), slip +0.6 → −3 m/s |
| with `smooth_slip` off (\|du\|) | present | absent (smooth, M accelerates to 1.21 through x = 0.6) |
| with `smooth_slip` on, u0 = 2 | removed | present, converged in location and states (N400-N1000) |

**Most probable causes (ranked):**
1. *Polynomial bump — slip-dependent liquid-side HTC (high confidence, direct tests).* In the
   bubbly regime htc_l is Ranz-Marshall with |u_g − u_l|: σ_Mg ∝ 2 + c√|du|, a cusp at du = 0. The
   slip decays to zero right where the evaporation grows ~20× (α 0.003 → 0.1) and the flow
   approaches M = 1, so σ_Mg swings ~2× cell to cell (odd-even 29 %) while p, α, superheat are
   smooth. The smooth slip (u0 ≥ 2) removes it.
2. *Parabola jump — evaporation-driven compression at the wall inflection (high confidence on the
   trigger, mechanism plausible but not proven by a geometry test).* The wall divergence rate
   (1/A)dA/dx peaks at x ≈ 0.595 (7.6 1/m) and falls after it; with the parabola law the flow is
   already supersonic there and the evaporation peaks there. In supersonic flow mass addition
   decelerates and compresses; when the evaporation at the inflection is strong enough it wins
   over the weakening area expansion and the flow compresses through a jump. The smooth slip
   raises that evaporation by ~40 % (σ_Mg peak 52 000 → 72-82 000): with |du| the same case is
   smooth, with the smooth slip it jumps — independently of the numerics.
3. *Contributing: large bubbles at low α (plausible, untested).* The area laws give a_i of only
   100-1 000 1/m at α = 1e-3-1e-2, i.e. d = 6α/a_i = 30-80 µm, so the drag balance against the
   5 000-7 000 g liquid acceleration before the throat gives a slip of 2-3.4 m/s; this slip must
   decay to ~0 inside the critical α window, where the HTC is most slip-sensitive.

**Ruled out (tested):** the 4th-order filters `eps_void` (0-1.0) and `eps_ghost` (0.05/0.25), the
ghost pins and the evaporation gate (all switched by α ≈ 0.01), `lowmach_diss_bc` (0/1), the
flux slip branch `du_blend`, `nc_form: wb`, the drag law (Schiller-Naumann, no switch), the regime
blend (starts at α = 0.3), the `d_min` floor (α < ~2e-4), the vapour Nusselt number (12-2000),
the EOS table resolution (90×90), the area (C² spline, no discontinuity), the grid (N200-N1000).

**For the data assimilation:** ṁ and pressures away from x ≈ 0.59-0.61 converge to ~0.03 % /
0.1 bar; pressures inside that window depend on grid and filter (jump width, exact cell) — keep
sensors out of it or give them a larger error. Whether the jump appears depends on the
evaporation near the inflection, i.e. on the inferred area-law correction: ensemble members can
fall on either side (with / without jump), a non-Gaussian, bimodal response near x ≈ 0.6.

> **Main finding.** With the same numerics (same eigenstructure, fluxes, filters, grid) and the
> same interfacial area law, changing only the liquid-side interphase heat transfer
> coefficient in the bubbly regime moves the choked mass flow rate by **−0.3 % to −3.3 %**
> (10-100× the grid effect of 0.03 %). The interphase heat transfer near the throat is a
> first-order control of ṁ, not a detail: every comparison of ṁ between runs must state the
> HTC closure (`smooth_slip`, `smooth_slip_u0`, see below).

Case: CO2 design case 4 (`settings_case4.yaml`): polynomial area law (`d_min` 1e-5),
`regime_htc` + `smoothstep`, thin ghost at inlet void 1e-5 (`alpha_eps` 1e-8, `eps_ghost_alo`
1e-8, `pin_mom_alpha` 1e-4), CFL 4, SER β = 1, `t_end` 0.05 s, `eps_ghost` 0.05,
`lowmach_diss_bc` 1.0. Runs made with the condensed driver of the twophase-da repo
(`nozzle/run_stage.py`, bit-identical to `dev_scripts/main_two_fluid.py` on the cold ladder,
checked 2026-10-21); run folders and settings in `twophase-da/nozzle/results/` and
`results/slip_tests/` (local, not in git). Same region as the "non-unique N400 fixed point"
and the rsB odd-even limit cycle in `LADDER_HISTORY.md` (Restart ladder).

## The bump

At x ≈ 0.568 the converged N400-N700 solutions have a jump over 2-3 cells: p drops ~2 bar
(N700, `eps_void` 0.05: 20.77 → 19.48 → 18.72 bar), the slip changes sign (min −2.05 m/s),
the liquid superheat steps 27.4 → 30.3 K, σ_Mg and σ_Ql have one-cell spikes. It sits
8 cells upstream of the mixture sonic point (u_mix − a = 0 at x = 0.579), at α_g ≈ 0.08,
mixture Mach 0.83, and at the same x on every grid and inlet void:

| run | bump x | α_g | Mach | sonic x |
|---|---|---|---|---|
| N400, void 1e-3 | 0.5689 | 0.10 | 0.82 | 0.5815 |
| N400, void 1e-5 | 0.5689 | 0.09 | 0.83 | 0.5789 |
| N600, void 1e-5 | 0.5693 | 0.09 | 0.83 | 0.5793 |
| N700, void 1e-5 | 0.5680 | 0.08 | 0.83 | 0.5794 |

## What it is not

Every gate is constant across the bump cells (N700):

| gate | switches at | at the bump |
|---|---|---|
| ghost energy pin / evaporation gate `w_ev` | x = 0.486 (onset) | released (1) |
| α > 5 % clause of `evaporation_weight` | x = 0.561 | redundant: liquid 27 K superheated |
| ghost momentum pin (`pin_mom_alpha` 1e-4) | x = 0.498 | released (w ≈ 1e-12) |
| ghost filter `eps_ghost` (α < 0.003) | off from x = 0.528 | off (w ≈ 1e-5) |
| slip flux branch (\|du\| ≥ `du_blend_lo` = 5) | x = 0.691 | zero-slip branch throughout |
| regime blend (α > 0.3) | x = 0.604 | not started |
| `p_phase_change_max` (70.1 bar) | x = 0.445 | below |
| low-Mach dissipation (`lowmach_diss` 1) | — | z_ac ≡ 1, constant |
| EOS table edges | — | not reached |

Numerics tests (restarts, N400 from `eps_void` 0 unless noted):

- `du_blend_lo/hi` 1 / 4 m/s instead of 5 / 20: solution unchanged to 3e-10 of each field range.
  The zero-slip vs full flux branch plays no role.
- `nc_form: wb`: bump stays (peak 25.1 bar vs 25.6), post-bump p +0.6 bar.
- `eps_void` 0: worse — a 5-cell recompression pocket 21.3 → 25.6 bar with negative slip
  (min −5.3 m/s), closed by a one-cell drop of 4.3 bar.
- `eps_void` sweep at N700 (0, 0.05, 0.25, 0.5, 1.0): any value ≥ 0.05 removes the
  recompression peak but leaves the ~2.5 bar step and slip −1.8 m/s; 0.25-1.0 overlap
  (post-bump p within 0.07 bar), steepest one-cell drop 0.83 → 0.58 bar; 1.0 needs 3.5×
  the steps. Away from the bump the filter is harmless and consistent: upstream change
  0.086 bar at N400 → 0.015 bar at N700 (5.6×, a 4th-order filter gives ~5.4), ṁ within 0.01 %.
- Grid, `eps_void` 0.25: ṁ 38.2500 / 38.2401 / 38.2395 kg/s and p upstream converge
  (N400 / N700 / N1000), but p just behind the bump does not: p(0.60) = 18.63 / 17.59 /
  18.35 bar.

## Cause: the slip-dependent liquid-side HTC

In the bubbly regime (α < 0.3) the liquid-side HTC is Ranz-Marshall with the actual slip,
Nu = 2 + 0.6 Re^½ Pr^⅓ with Re ∝ |u_g − u_l|, so σ_Mg ∝ 2 + c·√|du| — a cusp (infinite
slope) at du = 0 and a strong √|du| dependence around it. Where the slip crosses zero the
evaporation rate swings cell to cell while the state is smooth (N700, `eps_void` 0.25,
odd-even content in x = 0.54-0.60): σ_Mg 29 %, Nu_l 36 %, slip 52 %, but p 1.1 %,
superheat 1.2 %, α 0.3 %.

| x | slip [m/s] | Nu_l | superheat [K] | σ_Mg [kg/(m³s)] |
|---|---|---|---|---|
| 0.5651 | +0.80 | 22.7 | 27.30 | 29 900 |
| 0.5665 | −0.37 | 16.2 | 27.48 | 24 100 |
| 0.5680 | −1.82 | 33.5 | 27.68 | 53 300 |
| 0.5708 | −0.28 | 14.8 | 29.81 | 27 600 |

The other slip zero crossing (x ≈ 0.82) is clean (σ_Mg odd-even 0.1 %): it is in the
droplet regime, where the liquid side uses the slip-independent internal Nu. Feedback loop:
slip → σ_Mg → α, p → dp/dx → slip (an adverse gradient decelerates the light gas first).

A floor on Re does not help unless it is large: at the bump Re = 185-1160 (d = 50-90 µm),
so Re_min ≤ 300 leaves σ_Mg unchanged; Re_min ≈ 1000 is needed (post-processing on a frozen
state).

## The option: smooth effective slip

- `source_two_fluid.ranz_marshall_htc_two_fluid(..., slip_u0)` and
  `calculate_heat_transfer_one_component(..., slip_u0)`: Re from √(du² + u0²) instead of
  |du| when `slip_u0` > 0; default 0 = the original closure for any other caller.
- `two_fluid.sources.two_fluid_closure(..., smooth_slip=True, smooth_slip_u0=2.0)`, both
  the mass-transfer and the no-mass-transfer branch. **On by default, u0 = 2 m/s** (the
  smallest value of the sweep below that removes the bump; first implemented with 1).
- `smooth_slip=False` reproduces the earlier σ_Mg bit for bit (checked). Runs before
  2026-10-22, and the NICFD 2026 results, were made with the original closure:
  `dev_scripts/main_two_fluid.py` does not pass these arguments, so it now runs with the
  smooth slip on — pass `smooth_slip=False` to reproduce older runs. The dev_scripts
  `config_loader.py` has no settings key for it yet (the twophase-da copy has
  `sources.smooth_slip` / `sources.smooth_slip_u0`).

## u0 sweep (N700, `eps_void` 0.25, restarts)

| `smooth_slip` | kink at 0.57* | slip min | σ_Mg odd-even | peak σ_Mg | ṁ [kg/s] | Δṁ | p(0.50) | p(0.55) | sonic x | outlet α |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 373× | −1.82 | 33 % | 60 025 | 38.2401 | — | 41.09 | 21.93 | 0.578 | 0.776 |
| u0 = 1 | 33× | −0.06 | 1.5 % | 70 660 | 38.1227 | −0.31 % | 41.62 | 22.77 | 0.591 | 0.782 |
| u0 = 2 | 6× | +0.23 | 0.1 % | 79 857 | 37.9125 | −0.86 % | 42.58 | 24.41 | 0.605 | 0.787 |
| u0 = 4 | 5× | −0.01 | 0.1 % | 91 829 | 37.5006 | −1.93 % | 44.43 | 27.32 | 0.631 | 0.794 |
| u0 = 8 | 19× | +0.03 | 0.5 % | 104 628 | 36.9673 | −3.33 % | 46.79 | 30.69 | 0.641 | 0.801 |

\* max |Δ²p|/p in x = 0.555-0.585 relative to its median in the smooth surroundings.
p in bar, σ_Mg in kg/(m³s), u0 in m/s. All runs reach `t_end` with SSR ≈ 9e-9.

- u0 ≥ 2 removes the bump: no jump, slip positive, σ_Mg smooth. u0 = 1 leaves a kink 10×
  weaker than without the option.
- u0 = 8 is worse again for a different reason: the evaporation is so strong that p stalls
  (30.9 → 30.2 bar over x = 0.565-0.58); with no pressure gradient drag holds the slip at
  ~0, and when p falls again the gas re-accelerates in 3 cells (slip 0.2 → 2.0 m/s, p uptick
  0.02 bar) — under-resolved on N700. σ_Mg and the superheat (~15 K) stay smooth.

## Why ṁ moves

√(du² + u0²) does two things: it removes the cusp, and it raises Nu wherever the slip is small —
between flashing onset and the throat, where the slip is 0.5-2 m/s. That second effect
dominates: peak evaporation +18 / +33 / +53 / +74 % for u0 = 1 / 2 / 4 / 8, the pressure in
the throat region rises (p(0.55) +0.8 / +2.5 / +5.4 / +8.8 bar), the sonic point moves
downstream by 13 / 27 / 53 / 63 mm and the choked ṁ drops. The void profile itself barely
moves for u0 = 1 (onset +1.4 mm, α = 0.01 and 0.1 at identical x). Any smooth Nu at du = 0
needs Re(0) > 0, i.e. extra heat transfer at zero slip, so the smoothing cannot be made
neutral for ṁ.

Consequences:
- the bubbly liquid-side HTC is a percent-level control of ṁ, at fixed eigenstructure and
  fixed interfacial area law;
- for data assimilation ṁ is informative about it: u0 (or an HTC multiplier) is a candidate
  parameter to infer rather than a value to fix by hand;
- ṁ comparisons with earlier runs (e.g. ṁ = 38.716 kg/s of the parabola ladder above, the
  NICFD results) are only like-for-like with `smooth_slip=False`.

## Parabola law (the truth): a converged compression jump

With `interfacial_area: parabola` (the truth) the smooth slip does not remove the step: p
drops to ~18 bar and jumps back to ~21 bar at x ≈ 0.60, with u0 = 1 and u0 = 2 alike. All runs
below: parabola, `smooth_slip` u0 = 2, thin ghost at inlet void 1e-5, `eps_ghost` 0.05,
restarts from the cold-ladder N400 parabola state; run folders `twophase-da/nozzle/results/par*`,
settings and figures in `results/parabola_tests/`. Only settings were changed.

| N | `eps_void` | p rise | width (10-90 %) | slip min | ṁ [kg/s] |
|---|---|---|---|---|---|
| 400 | 0.05 | 2.94 bar | 3 cells / 7.5 mm | −3.07 | 38.5543 |
| 400 | 0.25 | 2.74 bar | 5 cells / 12.5 mm | −2.48 | 38.5555 |
| 700 | 0.05 | 3.25 bar | 4 cells / 5.7 mm | −3.94 | 38.5458 |
| 700 | 0.25 | 2.98 bar | 5 cells / 7.2 mm | −3.06 | 38.5460 |
| 700 | 0.5 | 2.89 bar | 7 cells / 10.0 mm | −2.81 | 38.5463 |
| 1000 | 0.05 | 3.64 bar | 4 cells / 4.0 mm | −4.65 | 38.5419 |
| 1000 | 0.25 | 3.12 bar | 6 cells / 6.0 mm | −3.55 | 38.5420 |

- A captured discontinuity of the converged solution: at fixed `eps_void` its width stays at
  the same number of cells (so it thins in mm with N) and the rise grows with N; its location
  (x ≈ 0.595-0.600) and the states on both sides converge. Away from it p(0.55 / 0.65 / 0.70 /
  0.90) agrees within ~0.1 bar on all grids, ṁ within 0.03 % (N400 → N1000).
- `eps_void` only sets how many cells it takes. 0.05 overshoots at N700 (21.3 bar peak and a
  dip after it); 0.25 and 0.5 are monotone; 0.25 is the cleanest without widening it much.
- Mechanism: the parabola a_i is low at small α, so the flow chokes early (sonic at x = 0.579
  vs 0.605 for the polynomial law with u0 = 2) and the evaporation peak (α 0.15-0.3, σ_Mg up to
  63-90 000 kg/(m³s)) lies in supersonic flow, where mass addition decelerates and compresses.
  Across the jump M falls from its peak 1.10-1.16 toward 1.04, the adverse gradient turns the
  slip negative (+0.6 → −3 m/s) and the larger |du| raises σ_Mg by ~30 %, which feeds the
  compression. u0 = 1 makes it stronger (37× → 147× the median p curvature at N400).

N200 looks smooth only because the same jump is smeared:

| N200 | p rise | width | slip min | ṁ [kg/s] |
|---|---|---|---|---|
| `eps_void` 0.05 | 2.58 bar | 3 cells / 15 mm | −2.34 | 38.5825 |
| `eps_void` 0.25 | 2.31 bar | 3 cells / 15 mm | −1.84 | 38.5966 |
| 0.5 + `eps_void_mass_acut` 1e-3 | 1.96 bar | 4 cells / 20 mm | −1.58 | 38.5336 |
| 1.0 + `eps_void_mass_acut` 1e-3 | 1.68 bar | 5 cells / 25 mm | −1.39 | 38.5087 |

`eps_void` ≥ 0.5 at N200 needs `eps_void_mass_acut`: without it the 4th-order filter digs the
thin ghost (α ≈ 7e-6) to the α floor 1e-8 at x ≈ 0.46-0.48, just before flashing onset, and
the run fails (SSR NaN at CFL 50, and at CFL 10 for 0.5) or reaches a wrong state (1.0 at CFL
10: SSR 4e-6, ṁ +0.35 %). With it the strong filter costs ṁ −0.09 % against N1000 and ~0.1 bar
behind the jump.

For data assimilation: a truth with this jump is a legitimate model solution; ṁ and pressures
away from x ≈ 0.59-0.61 are robust, pressures inside that window depend on the grid and the
filter (jump width, exact cell) and should be left out or given a larger error. A smoothed
truth (N200, `eps_void` 1.0 + `eps_void_mass_acut`) replaces the jump by a 25 mm, 1.7 bar
compression with ṁ and the far field within ~0.1 % of N1000.

## Where the bump and the jump sit: flow transitions, not closure transitions

The closures have no switch there: the drag is Schiller-Naumann without the 0.44 cap (smooth,
C¹ through du = 0), the HTCs are smooth with `smooth_slip`, the parabola and polynomial area laws
are smooth, and their only built-in transitions are the regime blend α_b = 0.3 → α_d = 0.7
(smoothstep, C¹), the `dispersed_fraction` corner at α = 0.5 and the `d_min` floor (α < ~2e-4).

| run | feature x | α across it (±12 mm) | sonic x | M peak (x) | σ_Mg peak x | slip = 0 at |
|---|---|---|---|---|---|---|
| parabola N400 / N700 / N1000, u0 = 2 | 0.594-0.595 | 0.16 → 0.23 → 0.30 | 0.579 | 1.09-1.10 (0.592-0.594) | 0.600-0.604 | 0.592-0.595 |
| parabola N400, u0 = 1 | 0.597 | 0.18 → 0.25 → 0.30 | 0.574 | 1.16 (0.594) | 0.602 | 0.594 |
| polynomial N700, smooth slip off | 0.568 | 0.04 → 0.08 → 0.15 | 0.578-0.579 | — | 0.594 | 0.565-0.567 |
| polynomial N700, u0 = 2 | no bump | — | 0.605 | — | 0.594 | none |

- The polynomial bump (α ≈ 0.08) is far from every closure transition.
- Both features sit at a flow transition: next to the sonic point (the polynomial bump just
  before it, M 0.83 → 0.93; the parabola jump just after it, from the M peak back toward 1),
  on the steep rise of the evaporation toward its peak 5-25 mm downstream, and exactly where
  the slip changes sign. Where the u0 = 2 smooth slip keeps the polynomial slip positive, the
  bump is gone.
- One coincidence: on every grid the parabola jump ends at α ≈ 0.30 = α_b, where the bubbly →
  droplet blend starts. Inside the jump the blend weight is 0 with zero slope (smoothstep), so it
  cannot drive the jump, but it may set where the jump ends. Testing it needs α_b as a setting
  (`two_fluid_closure` has `alpha_b`, the drivers do not pass it).

## Tests of 2026-10-23: what sets the parabola jump

All parabola, N400, inlet void 1e-5, thin ghost; runs in `twophase-da/nozzle/results/` (folders
`cold_*_20261023_*`, `par_noslip_*`, `nug*`), settings in `results/nu_tests/`.
Reference: the nozzle-hyperbolic parabola ladder S4 (`ladder_S4_v1e-5_N400_..._20261016_112804`,
original \|du\| closure, `eps_void` 0.25, `eps_ghost` 0.25, `lowmach_diss_bc` 0, ṁ = 38.716 kg/s),
smooth through x = 0.6. Its settings differ from the twophase-da runs only in `smooth_slip`,
`eps_ghost`, `lowmach_diss_bc` and `ssr_tol`; same geometry CSV (byte-identical), same code
apart from the smooth slip.

**Numerics ruled out, the slip closure confirmed.** With the nozzle-hyperbolic numerics
(`eps_ghost` 0.25, `lowmach_diss_bc` 0) and `smooth_slip` on (u0 = 1), the ladder still jumps:
S3 (void 1e-3) dp/dx up to +238 bar/m vs −20 in the reference, σ_Mg peak 77 700 vs 56 200; S4
dp/dx +238 vs −11, slip at x = 0.595 −2.22 vs +0.99 m/s, ṁ 38.657 vs 38.716 kg/s. With u0 = 2 the
two numerics sets give the same solution (ṁ 38.549 vs 38.556 kg/s, p(0.61) 20.73 bar in both).
Control: the same settings with `smooth_slip` off, restarted from the jump solution, return to
the smooth reference (ṁ 38.713 vs 38.716 kg/s, M peak 1.22 at x = 0.607, max dp/dx −10 bar/m),
so the steady state is not path dependent here.

**Geometry.** Throat (min r = 10.000 mm) at x ≈ 0.540-0.545. The tabulated wall curvature
changes from convex to concave within one table interval at x ≈ 0.596 (second difference of r per
table step: +91 µm at 0.586, +15 at 0.596, −17 at 0.606): the wall inflection, where
(1/A)dA/dx peaks at 7.6 1/m and its slope switches from +163 to −58 1/m² within ~1 cm. A and
dA/dx are continuous (C² cubic spline through 100 points); the jump sits on this inflection on
every grid. Not tested: smoothing the wall curvature around x = 0.6.

**EOS tables (90×90) are adequate.** All states inside the tables with margin (h_g 364-445 kJ/kg
of 250-480, h_l 188-244 of 100-320, p 14-119 bar of 10-130); one cell is 2.9 % in p (~0.5 bar
at 18 bar) and ~2.5 kJ/kg in h, the saturation curve 1.3 % in p, so the 2-3 bar jump spans 4-6
table cells. The smooth reference and the jump runs use the same tables.

**Vapour-side Nusselt number** (new option `two_fluid_closure(nu_g_int=None)`, None = the
constants 120 / 12; twophase-da `sources.nu_g_int`):

| smooth slip | Nu_g | ṁ [kg/s] | sonic x | M peak (x) | σ_Mg peak | max dp/dx in 0.55-0.65 |
|---|---|---|---|---|---|---|
| off | 12 | 38.656 | 0.566 | 1.27 (0.604) | 49 070 | −2 (strong bump at x ≈ 0.56-0.57: p 17.5 → 14.6 bar, slip −3 m/s) |
| off | 120 | 38.713 | 0.571 | 1.22 (0.607) | 51 270 | −10 (smooth) |
| off | 1200 | 38.739 | 0.571 | 1.22 (0.609) | 52 060 | +9.5 (small plateau at x ≈ 0.56) |
| off | 2000 | 38.740 | 0.571 | 1.22 (0.609) | 52 060 | +8.0 (identical to 1200 to 0.01 bar) |
| on, u0 = 2 | 120 | 38.549 | 0.579 | 1.09 (0.591) | 77 900 | +214 (jump) |
| on, u0 = 2 | 1200 | 38.571 | 0.579 | 1.09 (0.591) | 81 700 | +222 (jump) |
| on, u0 = 2 | 2000 | 38.573 | 0.579 | 1.09 (0.591) | 82 100 | +226 (jump) |

Nu_g saturates above ~1000 (the vapour sits at saturation, its side adds no resistance) and does
not decide whether the jump appears; Nu_g = 12 destabilises the near-onset region (that run had to
restart from the Nu_g = 120 solution: from the N700 interpolation it went NaN at step 3). The
polynomial law with Nu_g = 1200 (slip off) keeps its bump (ṁ 38.268 vs 38.248 kg/s, max dp/dx +39
vs +32 bar/m). Nu_g = 120 stays.

**What changes between α = 0 and 0.1** (one fixed cell state, x = 0.559, p = 17 bar, superheat
33 K, slip held at 1 m/s; only α swept):

| α | a_i poly / para [1/m] | d poly / para [µm] | h_l·a_i poly / para [MW/(m³K)] | σ_Mg poly / para, \|du\| | same, u0 = 2 | mixture a [m/s] |
|---|---|---|---|---|---|---|
| 1e-4 | 112 / 108 | 10 / 10 (d_min) | 13 / 12 | 1 750 / 1 680 | 2 350 / 2 260 | 235 |
| 1e-3 | 222 / 180 | 30 / 37 | 14 / 10 | 1 750 / 1 240 | 2 440 / 1 740 | 234 |
| 1e-2 | 1 320 / 890 | 50 / 74 | 61 / 33 | 7 600 / 4 100 | 10 800 / 5 800 | 218 |
| 5e-2 | 6 000 / 3 880 | 55 / 85 | 263 / 135 | 32 800 / 16 500 | 46 400 / 23 600 | 179 |
| 0.1 | 11 100 / 7 260 | 59 / 91 | 468 / 244 | 58 200 / 29 800 | 82 600 / 42 700 | 156 |

The numerical gates are all switched by α ≈ 0.01 (momentum pin released by 1e-3, ghost filter
off by 1e-2; the α > 5 % evaporation clause is redundant where the liquid is superheated). Between
α = 0.01 and 0.1 only the physics changes: evaporation and drag grow ~20×, the smooth slip adds
~40 % at every α, and the mixture sound speed drops by a third, so M → 1 in this window (sonic at
α ≈ 0.1-0.13 in every run). A near-sonic flow responds to a source as ~1/(1 − M²), which
amplifies any slip dependence of σ_Mg exactly there.

**Why the slip is 2-3.4 m/s before the throat with α ~ 1e-3.** It is the drag balance of the model:
solving drag = −α(1 − ρ_g/ρ_l)dp/dx for the slip reproduces the solution (parabola, x = 0.529:
α 4e-3, d = 63 µm, dp/dx −390 bar/m = 4 900 g of liquid acceleration, slip 3.34 m/s in the
solution, 3.39 from the balance; polynomial 2.42 / 2.36 with d = 44 µm). α cancels out (drag and
pressure force both ∝ α): the slip is set by d and dp/dx only, and the area laws already give
d = 30-80 µm at α = 1e-3-1e-2. With a number-density law (e.g. Nb = 1e13 1/m³) d at α = 1e-3 would
be ~6 µm and the slip a few 0.1 m/s. Dividing d alone by a factor would break a_i·d = 6α and
multiply h_l·a_i by about that factor; the consistent change is a_i × β and d / β together
(`ai_scale`), which also multiplies the early evaporation by ~β²·(Nu ratio) — a closure change for
the data assimilation to infer, not a numerical fix.

## Liquid-side HTC options in the bubbly regime (2026-10-23)

New closure options (`two_fluid_closure`, twophase-da `sources:`), defaults = the closure above
(checked: σ_Mg identical to 0 on a frozen state):

- `htc_liquid_bubbly`: `ranz_marshall` | `jakob` | `max` | `liao`
- `htc_liquid_droplet`: `internal` (Nu 12) | `jakob` | `max`. The vapour side keeps
  Ranz-Marshall in the droplet regime (droplets in a gas stream are what RM was derived for).

Liquid Jakob number Ja = ρ_l cp_l (T_l - T_sat) / (ρ_g h_fg), local ρ_g.

| Option | Nu_l = h_l d / k_l | Source |
|---|---|---|
| `jakob` | max(2, (12/π) Ja), slip-independent | Plesset & Zwick (1954), thermal-diffusion bubble growth |
| `max` | max(Ranz-Marshall, `jakob`) | Jakob takes over where the slip collapses |
| `liao` | sqrt(12/π² Ja² + Pe / (3π)), Pe = Re Pr | Liao et al., their Eqs. 8-10 (their Ja has ΔT², a typo; used ΔT) |

Tests: N400, smooth slip OFF, Nu_g 120, same numerics as the baselines (`eps_void` 0.25, CFL 4,
SER 1). Parabola restarts from `par_noslip_N700` (as `nug120_N400`), polynomial from
`new_S4_v1e-5_N400`. Settings, logs, figures: twophase-da `nozzle/results/htc_tests/`.

| Run | ṁ [kg/s] | σ odd-even (x 0.55-0.65) | max dp/dx [bar/m] | Note |
|---|---|---|---|---|
| poly, Ranz-Marshall | 38.248 | 50 % | 31.6 | the bump |
| poly, smooth slip u0 = 2 | 37.918 (-0.86 %) | 0.7 % | 7.7 | p +2-5 bar after the throat |
| poly, `max` | **38.248 (0.00 %)** | 1.0 % | -17.7 | **smooth, upstream identical** |
| poly, `jakob` | 38.747 (+1.3 %) | 0.7 % | 21.9 | weaker onset, p plateau 0.55-0.58 |
| poly, `liao` | 38.846 (+1.6 %) | 15 % | 214 | **bump back, stronger** (18 → 19.5 bar) |
| parabola, Ranz-Marshall | 38.713 | 5.2 % | -10.1 | small kink at x ≈ 0.56 |
| parabola, smooth slip u0 = 2 | 38.549 | 2.2 % | 214 | the jump at 0.595 |
| parabola, `max` | **38.715 (+0.005 %)** | 0.8 % | 16 | **smooth, 0.2 bar plateau at 0.6, no jump** |
| parabola, `jakob` | diverges | | | 2 starts (N700 IC, `max` solution) |
| parabola, `liao` | diverges | | | 4 starts (N700 IC; `max` solution at CFL 4, at CFL 1.5 without SER, and the same with `eps_void_mass_acut` 1e-3) |

Reading:

- **`max` is the only option that is smooth on both laws without moving ṁ.** Upstream, where
  choking is set, Ranz-Marshall is larger, so nothing changes there. In the critical window the
  slip collapses and Jakob (Nu ≈ 34-35) takes over, which removes the cusp. No arbitrary u0.
- **Jakob and Liao alone weaken the onset** (σ at x = 0.50 is 0.36× / 0.61× the RM value on a
  frozen state). Less vapour reaches the throat. On the parabola the flow over-expands just past
  the throat (14-15 bar against 19.9 at x = 0.541), the slip spikes to 50-140 m/s, α drops to
  the floor at x ≈ 0.55, and the run dies. It is the same failure from every start. Whether a
  steady state exists there was not settled. On the polynomial they converge with ṁ +1.3-1.6 %.
- **`max` at N700** (restart of the N400 solutions): ṁ 38.710 parabola (-0.013 % vs N400),
  38.233 polynomial (-0.04 %); p within 0.05 bar at every tap; σ odd-even 0.5 / 0.4 %. Same
  shape, the parabola's 0.2 bar plateau at x ≈ 0.6 included. The only visible detail is a small
  slip kink (≈ 0.3 m/s) on the polynomial at x ≈ 0.57, where `max` switches from RM to Jakob.
- **Liao's Pe term does not cure the cusp.** It is linear in |du| under the root and larger than
  the Ja² term (Pe/(3π) ≈ 150 against 12/π² Ja² ≈ 100 at du = 1 m/s, Ja ≈ 9). So Nu still swings
  ~60 % between du = 0 and 1 m/s; a kink (finite slope) rather than a cusp, but enough to bring
  the bump back.

## Drag: Ishii-Zuber viscous regime (2026-10-23)

New options `drag_bubbly` / `drag_droplet`: `schiller_naumann` (default) | `ishii_zuber`.
Ishii & Zuber (1979) viscous regime only (distorted / Newton regimes left out on purpose):
C_d = 24/Re_m (1 + 0.1 Re_m^0.75), Re_m with the mixture viscosity μ_c (1 - α_d)^(-2.5 μ*),
μ* = (μ_d + 0.4 μ_c)/(μ_d + μ_c) (bubbles μ_c/(1-α), drops μ_c (1-α)^-2.5; α_d capped at 0.9
because each regime law is also evaluated, with zero weight, outside its regime). Defaults
bit-identical (frozen state, max |ΔF| = 0).

On the frozen `max` solutions F_IZ/F_SN = 0.92-1.09 before the throat (the two viscous laws
nearly coincide at Re 100-1000; only the crowding term differs), 1.07-1.15 at α 0.13-0.27,
up to 1.3-1.45 at α ≈ 0.5 and in the droplet regime.

N400, smooth slip off, HTC bubbly `max`, restarts of the `max` solutions:

| Run | ṁ [kg/s] | slip peak before throat | p@0.55 | p@0.70 | σ odd-even |
|---|---|---|---|---|---|
| parabola, SN | 38.715 | 3.37 m/s | 18.29 | 15.10 | 0.8 % |
| parabola, IZ bubbly | 38.730 (+0.04 %) | 3.25 | 18.16 | 15.01 | 0.9 % |
| parabola, IZ both | 38.730 (+0.04 %) | 3.25 | 18.16 | 15.00 | 0.9 % |
| polynomial, SN | 38.248 | 2.43 | 21.90 | 13.61 | 1.0 % |
| polynomial, IZ bubbly | 38.250 (+0.005 %) | 2.39 | 21.86 | 13.45 | 1.0 % |
| polynomial, IZ both | 38.250 (+0.005 %) | 2.39 | 21.86 | 13.44 | 1.0 % |

Reading: the viscous Ishii-Zuber drag does not change the solution (ṁ ≤ 0.04 %, p ≤ 0.2 bar,
slip before the throat -2 to -4 %). The large slip before the throat is not a drag-law effect in
the viscous regime; a real change needs a different bubble size (area law / β) or a
deformed-bubble drag (Tomiyama, Ishii-Zuber distorted), which was excluded.

## Open

- Default u0 = 2 m/s (decided 2026-10-22: smallest tested that removes the bump, at the cost
  of ṁ −0.86 % against the original closure). Values between 1 and 2 not tested.
- Grid study N400 / N700 / N1000 with the smooth slip on for the polynomial law (done for the
  parabola: converged jump).
- Geometry test of the inflection mechanism: smooth the wall curvature around x = 0.6 (or move
  the inflection) and check that the parabola jump weakens or moves with it.
- Low-α bubble size: β (`ai_scale`) > 1 only over x ≈ 0.49-0.56 to see what smaller early
  bubbles do to the slip, the bump, the jump and ṁ.
- ~~A slip-independent bubbly liquid-side Nu (Jakob)~~: done, see "Liquid-side HTC options"
  (`max` works on both laws, N700 confirmed). Still open: `max` at N1000.
- `sources.smooth_slip` / `smooth_slip_u0` keys in the dev_scripts `config_loader.py`.
- Parabola jump: does its end follow α_b (run with α_b = 0.35 / 0.4)? Does the post-jump state
  depend on `nc_form` (cell vs wb; in the polynomial case wb moved the post-bump p by 0.6 bar)?
- At N1000 (parabola) the slip also drops abruptly to ~0 at x ≈ 0.563 with a small p notch:
  a smaller polynomial-type feature, not examined.
- Truth for the data assimilation (decided 2026-10-23: parabola with the smooth slip, jump
  kept): grid of the truth (N700-N1000, or the N400 of the model; ṁ differs by 0.03 %) and the
  sensor layout around x ≈ 0.59-0.61.


# Reconstruction investigation

Muscl Van Albada, the bump is present
repo res: test_cold_start_manual_N400_muscl_van_albada_20261028_100747

Constant, the bump is present
repo res: test_cold_start_manual_N400_constant_20261028_100945

Muscl minmod, no bump
repo res: test_cold_start_manual_N400_muscl_minmod_20261028_101506

Muscl Van Leer, no bump
repo res: test_cold_start_manual_N400_muscl_van_leer_20261028_101453
