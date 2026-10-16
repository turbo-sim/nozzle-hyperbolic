# Cold-start ladder for design case 4: what worked (2026-10-15)

Goal: from no knowledge (the nozzlex HEM design) to the production state, N = 400,
`muscl_van_albada`, inlet ghost void 1e-5 (1e-6 later), fast enough to repeat for every
interfacial area law. Implemented in `main_cold_start.py` (case file
`settings_cold_start.yaml`). Case: CO2 120 bar / 295 K → 20 bar, parabola area law
(code-default d_min), Nb 1e12, Nd 1e10, `regime_htc` + `smoothstep` blend, `eps_void` 0.25,
`eps_ghost` 0.25, `pin_ghost_momentum`. Same physics as the hand-made chain
`results/test_cold_start_manual_*`.

## The recipe

| stage | grid | inlet void | ghost numerics | CFL | SER β | stop | steps | wall* |
|---|---|---|---|---|---|---|---|---|
| 1 HEM seed | 80 | 1e-3 | thick | 0.2 | 1 | SSR < 1e-8 | 647 | 45 s |
| 2 | 200 | 1e-3 | thick | 0.5 | 1 | SSR < 1e-8 | 876 | 123 s |
| 3 | 400 | 1e-3 | thick | 2 | 1 | SSR < 1e-8 | 387 | 77 s |
| 4 | 400 | 1e-3 → 1e-5 | thin | 4 | 1 | t = 0.05 s, SSR < 1e-6 | 1020 | 264 s |
| (5) | 400 | 1e-5 → 3e-6 | thin | 4 | 1 | t = 0.05 s, SSR < 1e-6 | 315 | 63 s |

\* `python main_cold_start.py --voids 1e-5,3e-6 --cpus 0-3,16-21`, run 2026-10-15 19:02,
folders `results/c4_cold_S1..S5_*`. That is 10 logical CPUs (6 of them E-cores) with
other runs on the machine; 9.5 min in total, ~6 min to 1e-5. The same pipeline on
8 P-core CPUs took 26 / 78 / 56 / 214 s. The hand-made chain needed 8650 + 6750 + 10676
steps for stages 1-3 (≈ 30 min) plus 4537 steps (9 min) for 1e-3 → 7e-4 alone.

- thick = `alpha_eps` 1e-6, `eps_ghost_alo` 1e-5, `pin_mom_alpha` null (code defaults)
- thin = `alpha_eps` 1e-8, `eps_ghost_alo` 1e-8, `pin_mom_alpha` 1e-4 (as in the 7e-4 run)

Checks: pipeline S3 = hand-made S3 to 6e-6 in p; S1 and S2 likewise to ~1e-6. The final
1e-5 state is identical to 1e-10 to 1e-14 whichever thinning path reached it:
1e-4 → 1e-5 at CFL 4 or 8; 3e-4 → 1e-5 at CFL 2; 7e-4 → 1e-5 or 1e-3 → 1e-5 at CFL 4;
1e-3 → 1e-5 at CFL 8. It has ṁ = 38.716 kg/s, α_min 7e-6, a ghost checkerboard of du 5.5 % and
h_g 3.5 %, and its SSR floor is 1.54e-8.

If a stage fails, the pipeline retries it. A thinning rung is split at the geometric mean
of its start and target voids (up to 2 times). Stages 1-3 are rerun once without SER,
which is the hand-made recipe. Tested with `--stages 4 --restart <1e-5 run> --voids 1e-6`:

    1e-6 diverged (step 224) → split at 3.16e-6: ok (311 steps)
    1e-6 diverged (step 133) → split at 1.78e-6: ok (274 steps)
    1e-6 diverged (step 150) → 2 splits used, sequence stopped (FAILED)

So the lowest void reached with the fast rungs (CFL 4 + SER) is **1.8e-6**.

## How a thinning rung works

1. **The cost is physical time, not the size of the jump.** After the inlet void changes,
   the void front convects through the converging section. For every rung, whatever the
   jump size or CFL, the SSR hump peaks at **t ≈ 0.015 s** (the front at the throat) and the
   SSR is on its floor by **t ≈ 0.035-0.045 s**. Earlier attempts at 5e-4 and 1e-4 that
   looked stuck at SSR ~1.2e-5 were stopped on or before the peak (t = 0.008-0.015 s); none
   of them was failing.
2. **The lever is dt.** Steps per rung: CFL 0.5 ≈ 4500-5800, CFL 2 ≈ 1600, CFL 4 ≈ 1000,
   CFL 8 ≈ 650-720 (to t = 0.08). SER helps only after the front has passed. Its reference is
   the SSR after step 1, which is small for a thinning restart, so SER falls back to the plain
   CFL during the hump.
3. **Fewer rungs.** From 1e-3, ONE rung to 1e-5 works (CFL 4 and CFL 8), so there is no need
   for 5e-4 → 1e-4 → 5e-5 → 1e-5.
4. **A low CFL is not safer.** The 7e-4 → 3e-4 rung peaked at SSR 1.5e-4 at CFL 0.5 and at
   1.3e-5 at CFL 2. The larger implicit step damps the passage of the front through the
   flashing zone.
5. **Thin-ghost runs never reach 1e-8.** The algebraic ghost momentum pin
   (`pin_mom_tau: null`) leaves a fixed residual floor that depends only on the inlet void:
   1.06e-8 at 3e-4, 7.73e-8 at 1e-4, 1.54e-8 at 1e-5, 1.23e-8 at 3e-6. Every run at the same
   void reaches the same value to 4 digits. So a rung stops at `run.t_end` and is accepted
   when SSR < 1e-6. `pin_mom_tau` is the dt-consistent alternative (OPEN_QUESTIONS 12).

## Is 1e-5 thin enough?

Converged N400 states (ṁ = mean quasi-1D mixture mass flow; p at the minimum-area cell,
which is right at the flashing front):

| inlet void | ṁ [kg/s] | vs 1e-5 | p_throat [bar] | α_outlet |
|---|---|---|---|---|
| 1e-3 | 37.964 | −1.94 % | 23.11 | 0.798 |
| 7e-4 | 38.134 | −1.51 % | 22.42 | 0.796 |
| 3e-4 | 38.425 | −0.75 % | 21.17 | 0.793 |
| 1e-4 | 38.616 | −0.26 % | 20.31 | 0.790 |
| **1e-5** | **38.716** | 0 | **19.83** | 0.789 |
| 3.2e-6 | 38.731 | +0.04 % | 19.77 | 0.789 |
| 1.8e-6 | 38.735 | +0.05 % | 19.75 | 0.789 |
| 1e-6, CFL 2 | 38.783 | +0.17 % | 19.52 | 0.788 |
| 1e-6, CFL 1 | 38.801 | +0.22 % | 19.42 | 0.788 |

The ghost lowers ṁ roughly linearly in the void (about −0.2 % per 1e-4). At 1e-5 ṁ is
within 0.05 % of the 1.8e-6 value, and p near the throat within 0.4 %. The 1e-6 points
fall off that trend and differ between CFL 1 and CFL 2 (p by up to 0.47 % near the throat):
they carry the dt-dependent pin artefact, not more physics. 1e-5 (or 3e-6 with
`--voids 1e-5,3e-6`) is therefore a good production level for the current numerics.

## Experiments (all N400 unless noted; restart → target, SER β = 1 unless noted)

| run | start → target | CFL | SER | steps | outcome |
|---|---|---|---|---|---|
| L1_plain_3e-4_c0.5 | 7e-4 → 3e-4 | 0.5 | 1 | 5791 | floor 1.06e-8, hump 1.5e-4 |
| L1_plain_3e-4_c2 | 7e-4 → 3e-4 | 2 | 1 | 1626 | floor 1.06e-8, hump 1.3e-5 |
| L1_pred_3e-4_c0.5 | 7e-4 → 3e-4, secant predictor | 0.5 | 1 | 2090 | floor 1.06e-8 |
| L1_plain_1e-4_c0.5 | 7e-4 → 1e-4 | 0.5 | 1 | >4550 | stopped at t = 0.016, SSR 2.9e-4 still rising |
| L1_pred_1e-4_* | 7e-4 → 1e-4, predictor | 0.5 / 2 | 1 | ~600 | NaN / stalled (2× extrapolated p) |
| L2_1e-4_c2_f7 | 7e-4 → 1e-4 | 2 | 1 | 1765 | floor 7.73e-8 |
| L3_1e-4_c4_f7 | 7e-4 → 1e-4 | 4 | 1 | 1050 | floor 7.73e-8 |
| L4_1e-4_c4_f10 | 1e-3 → 1e-4 | 4 | 1 | 1089 | floor 7.73e-8 |
| L2_1e-4_c2_f3 | 3e-4 → 1e-4 | 2 | 1 | 1557 | floor 7.73e-8 |
| L2_1e-4_c4_f3 | 3e-4 → 1e-4 | 4 | 1 | 943 | floor 7.73e-8 |
| L2_1e-5_c2_f3 | 3e-4 → 1e-5 | 2 | 1 | 1752 | floor 1.54e-8 |
| L3_1e-5_c4_f1 | 1e-4 → 1e-5 | 4 | 1 | 1069 | floor 1.54e-8 |
| L4_1e-5_c8_f1 | 1e-4 → 1e-5 | 8 | 1 | 671 | floor 1.54e-8 |
| L4_1e-5_c4_f7 | 7e-4 → 1e-5 | 4 | 1 | 1114 | floor 1.54e-8 |
| L4_1e-5_c4_f10 | 1e-3 → 1e-5 | 4 | 1 | 1152 | floor 1.54e-8 |
| L7_1e-5_c8_f10 | 1e-3 → 1e-5 | 8 | 1 | 718 | floor 1.54e-8 |
| L5_3e-6_c4 | 1e-5 → 3e-6 | 4 | 1 | 447 | floor 1.23e-8 |
| L4_1e-6_c4_f5 | 1e-5 → 1e-6 | 4 | 1 | 225 | NaN at t = 0.016 (h_g ×540) |
| L5_1e-6_c4n | 1e-5 → 1e-6 | 4 | 0 | 833 | NaN at t = 0.015 |
| L5_1e-6_cap8 | 1e-5 → 1e-6 | 4 | 1, CFL ≤ 8 | 430 | NaN at t = 0.015 |
| L6_1e-6_f3 | 3e-6 → 1e-6 | 4 | 1 | 137 | diverged at t = 0.016 |
| L5_1e-6_c2n | 1e-5 → 1e-6 | 2 | 0 | 8830 | survives the front; SSR stalls at 1.366e-6 |
| A_1e-6_c2 | 1.8e-6 → 1e-6 | 2 | 0 | 2750 (t = 0.025) | same: stalls at 1.366e-6 |
| L7_1e-6_c1n | 3e-6 → 1e-6 | 1 | 0 | >5700 | survives; stalls at 1.7e-7 (stopped at t = 0.026) |
| A_1e-6_c1 | 1.8e-6 → 1e-6 | 1 | 0 | 5519 (t = 0.025) | stalls at 1.7e-7; state differs from CFL 2 by 0.47 % in p |
| B_1e-6_from_c2n | the CFL-2 1e-6 state, restarted | 4 | 1 | 2 | SSR 1.4e-6 → 1.2e-2 in one step, NaN-level after 8 halvings |
| P2 pipeline | 1e-5 → 1e-6 with auto-split | 4 | 1 | | 3.16e-6 ok (311), 1.78e-6 ok (274), 1e-6 ×3 diverged |
| S1_s1 (N80) | HEM seed, void 1e-3 | 0.2 | 1 | 647 | = hand-made S1 (8650 steps) to 1e-6 |
| S2_c05s (N200) | N80 → N200 | 0.5 | 1 | 754 | = hand-made S2 (6750) to 8e-7 |
| S2_c2 (N200) | N80 → N200 | 2 | 1 | 1116 | NaN (N80 state has a 28 % du checkerboard) |
| S3_c2 (N400) | N200 → N400 | 2 | 1 | 387 | = hand-made S3 (10676) to 6e-6 |
| S3d_c05s (N400) | N80 → N400 directly | 0.5 | 1 | 1587 | converges, but costs 2× S2 + S3 |

Logs (`ladder_logs/<run>.log`) and the exact settings of every run (`ladder_cfg/<run>.yaml`)
are archived in `results/ladder_experiments_20261015/`, together with the tools:
`run_rung.py` (one rung = resolved settings of `test_cold_start_manual_N400_..._175607`
plus overrides), `ladder_status.py`, `predict_restart.py` and `exp_table.py`. The runs were
made on a copy of the code (commit 6fe085c plus the working-tree diff of 2026-10-15,
`SOURCE_PATCH_vs_6fe085c.patch` in the archive). Their solution folders were not kept,
except for the pipeline run `c4_cold_*`.

## Findings beyond the recipe

- **Secant predictor restart.** The idea: rescale α in the restart state by a log-log secant
  fitted on the last two converged rungs (S = d ln α / d ln void is 1.00 upstream of flashing
  and 0.006 at the outlet), plus a linear secant on p, u and h. For a small jump it moved the
  rung to its floor in less physical time (t ≈ 0.01 s instead of 0.02 s), but the extrapolated
  p and u start every cell off-steady (SSR 4e-5 from step 1), so at CFL 0.5 it took about as
  many steps as the plain CFL 2 rung. For 7e-4 → 1e-4 (secant factor 2, p shifted 6 % at the
  flashing front) it blew up. Not used.
- **Non-unique fixed point at 3e-4.** The CFL 2 path and the CFL 0.5 path to 3e-4 end on two
  exact fixed points (same SSR floor, same ṁ to 1e-5). p differs by up to 1.1 % in cells
  221-225, just behind the throat (x = 0.554-0.566, u ≈ 155 m/s; α 0.04-0.07), a few cells
  before the sonic point (cells 227-228). **Not the evaporation gate** (checked 2026-10-16:
  both gate clauses are identical in every cell of the two states, onset cell 194 in both,
  and the liquid is ~30 K superheated at 221-225, so the `ag > 5e-2` clause is moot). The
  only discrete difference: the CFL 2 state has extra local extrema in u_g, h_g and the slip
  at cells 222-225, where van Albada cuts the slope to zero. Branch map: 1e-3 and 7e-4 have
  the smooth shape, 3e-4 has both, and every 1e-4 / 1e-5 / 1e-6 state has the bumpy one.
  The 1e-4 state is unique to 1e-13 across 4 paths, and the 1e-5 state across 7 (the
  pipeline's, which stops at t = 0.05 instead of 0.08, to 2e-10). Relaxing the limiter
  (va_rel 1e-4, va_eps_mode range) or extending the ghost filter (eps_ghost_acut 0.03) on
  the 1e-5 state blows up exactly there within 11-385 steps: the over-limiting holds a
  weakly damped gas-phase mode. Relevant for data assimilation: near such an overlap the
  forward model can jump between branches.
- **Below 1e-5.** 3e-6 and 1.8e-6 work at CFL 4 (about 300 steps each). 1e-6 does not, and
  the cause is the level, not the jump size:
  - Every CFL 4 attempt (from 1e-5, 3e-6 and 1.8e-6, with or without SER, SER capped at 8)
    dies when the front reaches the flashing onset (t ≈ 0.015 s). The ghost h_g blows up
    even after 8 halvings (dt 7e-8).
  - At CFL ≤ 2 without SER the front passes, but the SSR then stalls on a floor that depends
    only on dt: **1.366e-6 at CFL 2** (identical from 1e-5 and from 1.8e-6) and **1.7e-7 at
    CFL 1**.
  - That CFL-2 state, restarted at CFL 4, blows up in ONE step. So the 1e-6 steady state is
    not stable under large implicit steps.

  This is the algebraic ghost momentum pin. Its fixed point depends on dt (solver.py,
  `pin_mom_tau` comment), and at 1e-6 it dominates. Next step for 1e-6: run the rung with
  `pin_mom_tau` (the dt-consistent relaxation pin) rather than a different ladder.
  Until then, 1e-6 needs CFL ≤ 1-2 and no SER for the whole rung, roughly 9000 steps at
  CFL 1.

## Other area laws through the ladder (2026-10-16, main_ladder.py, settings_ladder.yaml)

All with the ladder settings (smoothstep blend, `regime_htc: true`); tests on a copy of the
code, settings only.

- **polynomial (d_min 60 µm) and original (Nd 1e10) fail in stage 1 at t ≈ 5e-4 s**, the
  same way. Downstream of the throat the flow is vapor-continuous (α 0.6-0.87). Both laws
  give large droplets there (300-560 µm vs ~100-120 µm for the parabola), and with
  d = 6 α_disp / aᵢ the vapor heating aᵢ·k_g·Nu/d scales as 1/d²; `regime_htc: true` also
  drops the vapor Nusselt from a fixed 120 to Ranz-Marshall (17-40 there). Net: 5-35× less
  vapor heating than the parabola. The vapor expands almost adiabatically to 26 K below
  saturation (≈ 220 K at 15 bar) and reaches the bottom of the metastable vapor table
  (≈ 218 K, CO2 triple point; below h ≈ 405 kJ/kg at 15 bar T and ρ freeze, the sound speed
  jumps 207 → 255 m/s) → NaN in Newton. Linear blend, a thick ghost or `regime_htc: false`
  alone do not save it; the old recipe (regime_htc off, linear, 3e-3 ghost, alpha_eps 1e-3,
  eps_ghost 0.05, eps_void 0.2, N100, **CFL 0.5 without SER**) still reaches its SSR plateau
  of 1.42e-5 (6449 steps), which the ladder's 1e-6 acceptance would reject.
- **original, Nd sweep** (`sources.droplet_number`; Nd inside interfacial_area_params is
  overridden): stage 1 fails for 1e10 and 1e11, passes for 3e11, 1e12 and 3e12, fails for
  1e13. Nd 1e12 = Nb gives a continuous diameter through the transition (83 µm at α 0.3 and
  0.7). But **N200 stalls at SSR ~9e-4** for all three passing values, with or without SER:
  an odd-even limit cycle at α ≈ 0.05-0.2 just behind the throat (slip spike −7 m/s, a real
  pressure bump 27.31 → 27.65 → 27.33 bar), pumped by the mass-transfer source (38 % odd-even,
  peak 9.3e4 kg/m³s at the worst cell). Nd does not act there; the bubbly side (Nb 1e12) gives
  1.4-2.5× the parabola's area at α 0.01-0.1. Same region as the small kink of the parabola
  runs.
- **segments** (new law: aᵢ linear between control points, d = 6 α_disp / aᵢ). Full ladders:

| run | points α : aᵢ | result |
|---|---|---|
| SgG (code default) | 0:100, 0.05:3900, 0.25:15000, 0.7:24000, 0.9:20000, 1:100 | **converged**: N80 (SER attempt NaN, no-SER retry 8794 steps), N200 760, N400 388, 1e-5 1054 |
| SgC | 0:100, 0.05:4500, 0.35:20000, 0.7:12000, 0.9:6000, 1:100 | **converged**: N80 via no-SER retry (9246), N200 383, N400 533, 1e-5 split → 1e-4 (950) → 1e-5 (928) |
| SgD | as SgA with 22000 at 0.6, 18000 at 0.85 | N80, N200 ok; N400 stalls at SSR ~3e-6 |
| SgA | 0:100, 0.05:3900, 0.25:13000, 0.6:20000, 0.85:14000, 1:100 | N80 NaN at t ≈ 1.5e-3 s (also with corners rounded, corner 0.01) |
| SgF | SgD plus 0.1:7300, 15000 at 0.25 | N80 NaN |
| SgB, SgE | 20-60 % less area than the parabola at α ≤ 0.2 | N80 NaN at t ≈ 6e-4 s |

  Final 1e-5 states: ṁ 38.750 (SgG), 38.607 (SgC) vs 38.716 kg/s (parabola); coldest vapor
  5.0 / 2.9 K below saturation; slip odd-even at α 0.02-0.25: 0.52 / 1.73 m/s (parabola
  0.72). So far: the area up to α ≈ 0.05 must stay close to the parabola's (less fails
  early); sharp corners are not a problem; beyond that the requirements interact and are not
  yet mapped.

## Not tried

- a void-dependent CFL in the thinning rung (CFL 8 worked on both 1e-5 rungs, default kept
  at 4 for margin with other area laws)
- `pin_mom_tau` (would remove the SSR floor, let `ssr_tol` stop the rungs, and is the first
  candidate for 1e-6)
- stage 1 without SER for marginal area laws (both converged segments shapes needed the
  no-SER retry)
