# Bump behind the throat, slip-dependent heat transfer and the mass flow rate (2026-10-22)

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

## Open

- Default u0 = 2 m/s (decided 2026-10-22: smallest tested that removes the bump, at the cost
  of ṁ −0.86 % against the original closure). Values between 1 and 2 not tested.
- Grid study N400 / N700 / N1000 with the smooth slip on: does the region behind the throat
  converge now?
- A slip-independent bubbly liquid-side Nu (e.g. Jakob-number based, heat-diffusion-controlled
  bubble growth; superheat here 15-30 K) as the physical alternative to u0.
- `sources.smooth_slip` / `smooth_slip_u0` keys in the dev_scripts `config_loader.py`.
