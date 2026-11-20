# Two-fluid: K-free option and smooth entropy fix in `src`

Port of the [two_fluid_test_v2](../two_fluid_test_v2/README.md) prototype into the
package as **additive, default-off options** of the two-fluid model. With the defaults
the package is bit-identical to the previous K-inverse compatibility scheme.

Status (2026-11-19): steps 1-4 implemented and verified; 100-cell runs of all four
variants done; 200/400-cell runs started; pytest tests and documentation not yet
written; nothing committed.

## Options

`CompatibilityConfig` ([configuration.py](../../src/nozzle_hyperbolic/models/two_fluid/configuration.py)):

| Field | Default | Meaning |
|---|---|---|
| `k_free` | `False` | True: K-free face flux (primitive-jump viscosity) **and** straight-path nonconservative integral; `nc_form` is not used |
| `smooth` | `False` | True: continuous Harten magnitude, width `smooth_width * a_mix`, on every wave and on u1, no sonic-crossing test; applies to either flux |
| `smooth_width` | `0.02` | Harten width as a fraction of the mixture sound speed |

Mapping to the prototype arms:

| `k_free` / `smooth` | Prototype arm | Run folder name |
|---|---|---|
| off / off | `baseline` | `k_inverse` |
| off / on | `sonic_only` | `k_inverse_smooth` |
| on / off | `inverse_old_sonic` | `k_free` |
| on / on | `both` | `k_free_smooth` |

Validation: booleans required, `smooth_width` positive and finite; `k_free` with
`StabilizationConfig.mass_upwind_cutoff` is refused by the model (untested combination,
also excluded by the prototype).

## Source changes (uncommitted)

1. [flux.py](../../src/nozzle_hyperbolic/models/two_fluid/flux.py)
   - new `k_free_interface_flux(U_L, U_R, ..., smooth=False, smooth_width=0.02)`:
     `F = (F_L+F_R)/2 - J(U_bar) sum_k z_k m_k r_k (l_k . dU)/(2 D_k)`, same signed
     weights, low-Mach factor and zero-slip / cosine blend / full branches as
     `fvs_interface_flux`; magnitudes `m_k = lambda_k s_k` (default) or Harten `H(lambda_k)` (smooth).
   - new `k_free_path_nonconservative(U_a, U_b, ...)`: `int X(U) dU` on the straight
     primitive path, two-point Gauss-Legendre, no K solve.
   - `fvs_interface_flux` gains `smooth=False, smooth_width=0.02`; the default path is
     untouched, the smooth block overrides `s` and the s_12 width after the native weights.
   - `J_matrix` import added.
2. [configuration.py](../../src/nozzle_hyperbolic/models/two_fluid/configuration.py): the three fields above.
3. [operator.py](../../src/nozzle_hyperbolic/solvers/fvm/operator.py): new generic
   nonconservative mode `"path"`. `nonconservative()` calls
   `model.integrated_nonconservative(full, raw_flux, faces=reconstruct(full))`
   (same faces as the face flux; deduplicated under jit). Shape check at binding.
   The MUSCL dependency radius (+1) is unchanged and sufficient. `flux_difference` untouched.
4. [contracts.py](../../src/nozzle_hyperbolic/solvers/fvm/contracts.py): new `PathNonconservativeModel` protocol.
5. [model.py](../../src/nozzle_hyperbolic/models/two_fluid/model.py):
   `nonconservative_mode` per instance (`"path"` if `k_free`); flux selection in
   `numerical_face_flux`; `integrated_nonconservative(full, unfiltered_flux, faces=None)`
   with the K-free branch
   `1/2 A_{i+1/2} path(face i+1/2) + 1/2 A_{i-1/2} path(face i-1/2) + A_i path(within cell)`.

Geometry of the K-free NC term: each face jump is shared half/half by the neighbours,
weighted by the area of the face where the jump sits; the within-cell path (left-face to
right-face reconstructed state, zero under constant reconstruction) uses the centre area.
A uniform state gives exactly zero, so no `wb` subtraction is needed. This face weighting
is the pairing that balances a void-fraction contact at rest on varying area
(`p/2 (A+ da+ + A- da-)`); a cell-area version would leave `~ (p/4) dA da`. Native
`nc_form` default stays `cell` (as in alberto_branch); NICFD cases and the demo set `wb`.

## Verification

All numbers measured on the saved 100-cell K-inverse solutions (NICFD N2/water, Elliot nozzle).

| Check | Result |
|---|---|
| [check_k_free_kernels.py](check_k_free_kernels.py): 909 face states (3 solutions + slip x1.5/x3 copies; 37 / 119 / 753 faces in the zero-slip / blend / full branches) | K-free flux vs `inverse_old_sonic` 1.6e-16; vs `both` 3.4e-16; K-inverse flux smooth off/on vs `baseline`/`sonic_only` **0 (bit-identical)**; path NC identical; NC mass rows 0, momentum rows cancel exactly; uniform state flux = native, NC = 0; AD finite on all branches |
| Throwaway path-mode subclass vs prototype (step 3) | NC rel. 2e-17 / 9e-17 (constant / MUSCL); colored vs dense Jacobian 2e-16 |
| [check_k_free_model.py](check_k_free_model.py): residual contributions of `TwoFluidModel` vs `AblationModel`, 3 regimes x 2 reconstructions x 4 arms | baseline and sonic_only **0**; inverse_old_sonic <= 2.2e-14; both <= 5.0e-14 (W <-> U round trip of the operator faces); NC <= 2e-16; k_free+smooth MUSCL colored vs dense Jacobian 2.3e-16; mass_upwind refusal works |
| pytest | `tests/models/two_fluid` 50/50; `tests/solvers` 224/224; two-fluid regression (`test_saved_solutions`, `test_upstream_compatibility`, `test_exact_limits`) + two_fluid 62/62 |

Also checked: the source vector (`D_VE_algebric`, `D_VE_algebric_two_component`,
`D_VE_area`) is identical (AST, docstrings/comments stripped) to the current
alberto_branch; that branch only added closure options after the port commit
`faafff7` (`nu_l_int`, `drag_scale_*`, `beta_cd`/`beta_htc_*`, HTC `sum`), all
defaulting to the previous behaviour and not ported to main.

## Runs

[run.py](run.py) is `demos/two_fluid/run.py` with output/table folders changed and the
`k_free`/`smooth` switches: `poetry run python -B development/two-fluid/run.py <variant> <cells>`.
NICFD two-component setup: 20 bar / 295.15 K, quality 0.5, slip 1, Nb = Nd = 1e11,
`wb`, MUSCL Van Albada, void filter 0.05, low-Mach floor 1 (0 at boundaries), published
profile as initial guess, max 500 pseudo-steps. Outputs in `outputs/<variant>_N<cells>/`
(gitignored), shared `outputs/fluid_tables` and `outputs/jax_cache`.
[compare_runs.py](compare_runs.py) writes `outputs/comparison/{summary,differences}.csv`
and figures.

100 cells (all converged, all regimes correct):

| Variant | Accepted steps sub / shock / adapted | m_dot vs k_inverse | Mean field diff vs k_inverse |
|---|---|---|---|
| k_inverse | 36 / 35 (+1 rejected) / 37 | - | - |
| k_inverse_smooth | 36 / 35 (+1) / 37 | <= 1e-4 % | <= 1e-6 (subsonic identical to 1e-14) |
| k_free | 35 / 31 / 36 | +0.08 to +0.13 % | 0.03-0.15 % of range; p max 0.15-0.6 %; T_l max up to 2.7 % (small range) |
| k_free_smooth | 35 / 31 / 36 | as k_free | as k_free (differences ~1e-6) |

Max Mach: subsonic 0.8356 (K-inverse) vs 0.8390 (K-free); shocked 1.3316 vs 1.3323;
adapted 1.6448 vs 1.6456. Expected: K-free and K-inverse share the PDE but not the
numerical viscosity / NC discretization, so they should agree only as the grid is
refined; the 200/400-cell runs test this.

Notes:
- Wall times of the new runs (~300 s) are not comparable: three runs in parallel,
  compilation included, and an unrelated 7-worker job on the machine.
- `k_inverse_N100` results were rewritten at 17:04-17:05 by a second (warm-cache) run of
  run.py not started from this session; identical step counts, its elapsed 21-25 s vs
  96-107 s in `outputs/run_k_inverse.log` of the original run.

### Grid refinement: 200 and 400 cells (2026-11-19)

Four variants per grid run in parallel (wall times not comparable). Full tables:
`outputs/comparison/summary.csv`, `differences.csv`; profiles `profiles_<case>_N400.png`.

Convergence (accepted / rejected steps):

| Case | Cells | k_inverse | k_inverse_smooth | k_free | k_free_smooth |
|---|---|---|---|---|---|
| subsonic | 200 | 40 / 0 | 40 / 0 | 40 / 0 | 40 / 0 |
| subsonic | 400 | 43 / 0 | 43 / 0 | 44 / 0 | 44 / 0 |
| shocked | 200 | **stagnation** 69 / 59 | **stagnation** 79 / 61 | 34 / 0 | 34 / 0 |
| shocked | 400 | 47 / 2 | 50 / 3 | 47 / 2 | 43 / 1 |
| adapted | 200 | **stagnation** 91 / 64 | **stagnation** 73 / 57 | 40 / 0 | 40 / 0 |
| adapted | 400 | 44 / 0 | 44 / 0 | 44 / 0 | 44 / 0 |

- K-free converges in all 18 runs. K-inverse (with or without smooth) stagnates at
  200 cells for shocked and adapted, as in the prototype campaign (linear 200:
  baseline and sonic_only stagnate there, the K-free arms converge).
- `smooth` has no effect on the converged solutions (<= 2e-5 of range; subsonic identical to 1e-14).
- Max Mach at 400: subsonic 0.8675 (K-inverse) vs 0.8650 (K-free); shocked 1.3682 vs 1.3656;
  adapted 1.6456 vs 1.6451. m_dot difference at 400: -0.04 % subsonic, -0.007 % shocked/adapted.
- The shock sits at the same place (x ~ 0.183); profiles overlap at 400 cells.

Does K-free approach K-inverse with refinement? Mean |diff| / range on the 400 grid,
compared with how much each scheme still changes from 200 to 400 cells (interpolated):

| Case | Field | k_free - k_inverse @400 | k_inverse 400-200 | k_free 400-200 |
|---|---|---|---|---|
| subsonic | p | 9.0e-4 | 6.9e-4 | 8.8e-4 |
| subsonic | alpha_g | 2.3e-3 | 1.4e-3 | 5.2e-4 |
| shocked | p | 4.0e-4 | 1.9e-3 * | 1.6e-3 |
| adapted | p | 3.4e-4 | 6.2e-4 | 6.2e-4 |

(* K-inverse 200 not converged.)

- The cross-scheme difference is the same size as each scheme's own remaining
  discretization change between 200 and 400 cells: neither scheme is grid-converged
  at 400 (subsonic max Mach still moves 0.858 -> 0.867). The difference is not
  monotonically decreasing over 100 -> 200 -> 400 (subsonic alpha_g mean 7.7e-4 ->
  1.2e-3 -> 2.3e-3), so "same limit" is plausible but not demonstrated; that needs
  a finer grid (800 cells).
- The largest local differences are at x ~ 0.095 (alpha_g, p: the convergent-section
  contour join at 0.0994, where K-inverse shows a small alpha_g wiggle that K-free does
  not) and in the last cell for T_l (outlet boundary).

### 800 cells (k_inverse, k_free; 2026-11-19)

Both converge in all three cases (k_free 47 / 57 (+4) / 48 steps, k_inverse 47 / 75 (+9) / 63 (+4)
for subsonic / shocked / adapted). Mean |diff| / range on the 800 grid:

| Case | Field | k_free - k_inverse @800 | k_inverse 800-400 | k_free 800-400 |
|---|---|---|---|---|
| subsonic | p | 1.1e-3 | 2.9e-4 | 3.7e-4 |
| subsonic | alpha_g | 2.9e-3 | 7.4e-4 | 1.5e-4 |
| shocked | p | 2.0e-4 | 7.4e-4 | 7.6e-4 |
| shocked | alpha_g | 7.3e-4 | 7.6e-4 | 1.4e-4 |
| adapted | alpha_g | 1.7e-3 | 5.3e-4 | 1.1e-4 |

- Shocked: the difference halves from 400 to 800 (first order), so the two schemes approach each other.
- Subsonic / adapted: each scheme changes less between 400 and 800 than the two differ,
  and the difference does not shrink (subsonic m_dot difference -0.04 % -> -0.085 %):
  they converge to **different** limits locally.
- The difference is concentrated at x ~ 0.09-0.14 (subsonic 800: mean alpha_g difference
  2.5 % of range in [0.09, 0.11], 2e-5 upstream of 0.09). K-inverse has an alpha_g bump at
  x ~ 0.093 that is present at 200, 400 and 800 cells (grid-independent); K-free has none
  (`outputs/comparison/zoom_contour_join.png`).
- That location is where **det K = 0**: the equilibrated det K changes sign at x = 0.0928 in
  subsonic and adapted (adapted also at 0.1513), minimum reciprocal condition number
  2.7e-4 / 1.1e-4. Slip there is ~48 m/s (full-slip branch, not the slip blend).
- Interpretation: the K-inverse discretization (X K^-1 dF in the NC term and K-projectors
  in the flux) does not converge to the smooth solution across the det K = 0 point; it
  leaves a persistent bump and a downstream offset. K-free has no K^-1 and passes through
  it smoothly. Same mechanism as the EnKF member throat bump in twophase-da
  (K^-1 numerics at the det K = 0 point).

## Flashing water: Super Moby Dick (NICFD single_component, 2026-11-20)

[run_flashing.py](run_flashing.py) builds the unchanged published case with
`cases/NICFD2026/scripts/case.py` (published-profile initial guess, ghost-enthalpy pin,
mass transfer on) and changes only `k_free`/`smooth`. Same settings for every variant:
case CFL (2 / 1 / 5) and growth 1.2, 20 Newton corrections, 20 backtracks, at most 2000
steps, wall budget 900 s (100 cells) / 1800 s (400 cells) per regime. The 400-cell runs
start again from the published profile (no grid transfer). Outputs in
`outputs/flashing/<variant>_N<cells>/` (a first attempt that crashed while saving is kept
in `failed_first_attempt/`). Reference: published profiles (600 rows).

| Regime | Cells | k_inverse | k_inverse_smooth | k_free | k_free_smooth |
|---|---|---|---|---|---|
| subsonic | 100 | stagnation 235 / 113 | stagnation 172 / 95 | **converged** 553 / 134 | **converged** 807 / 201 |
| subsonic | 400 | stagnation 32 / 37 | stagnation 32 / 37 | **converged** 49 / 0 | **converged** 49 / 0 |
| shock | 100 | stagnation 267 / 109 | stagnation 298 / 118 | **converged** 496 / 120 | **converged** 496 / 120 |
| shock | 400 | stagnation 45 / 35 | stagnation 58 / 40 | **converged** 46 / 0 | **converged** 46 / 0 |
| adapted | 100 | stagnation 683 / 211 | stagnation 644 / 232 | **converged** 716 / 180 | **converged** 590 / 147 |
| adapted | 400 | stagnation 184 / 80 | stagnation 113 / 61 | **converged** 39 / 0 | **converged** 39 / 0 |

(accepted / rejected steps)

- K-free converges in all 12 runs and passes the case's independent residual audit;
  K-inverse (with or without smooth) stagnates in all 12, as in the NICFD diagnosis.
  At 400 cells K-free needs 39-49 steps with no rejection (about 50-110 s).
- All runs keep the intended regime (max Mach 0.62 / 1.05 with outlet 0.18 / 1.30 at 400).
- K-free at 400 vs the published profiles: mass flow 0.32 % (subsonic), 0.52 % (shock),
  0.40 % (adapted); normalized RMS p 2.9-3.4e-3, alpha_g 1.5-3.4e-3, u_g 1.6-2.8e-3.
  At 100 cells the mass flow is 14-15 % off (e.g. subsonic 20.0 vs 23.26 kg/s): strong
  grid sensitivity of the flashing mass flow; 100 cells is far from resolved.
- The stagnated K-inverse states at 400 look close to the published profile only because
  they barely moved from it (their starting point); they do not pass the residual audit.
- `smooth` changes neither convergence nor the converged solutions here.

### Grid study (k_free 100-800) and K-inverse restart (2026-11-20)

[compare_flashing.py](compare_flashing.py), outputs in `outputs/flashing/comparison/`.

K-free mass flow [kg/s] (published, 600 rows: 23.26 / 23.23 / 23.24):

| Regime | 100 | 200 | 400 | 800 |
|---|---|---|---|---|
| subsonic | 20.01 | 23.25 (stagnation) | 23.33 | 23.43 |
| shock | 19.76 | 23.22 (stagnation) | 23.35 | 23.44 |
| adapted | 20.02 | 22.61 | 23.33 | 23.43 |

- 800 cells converge in all regimes (43-54 steps, no rejection). 200 cells **stagnate**
  for subsonic and shock: Newton gets to an inner residual of 1e-7-1e-9 but its line
  search is exhausted once CFL > ~3, so the controller halves the CFL down to 2e-8 and
  stagnation triggers (the state stays finite; NaN in rejected rows only means "not
  evaluated"). The worst cells straddle alpha_b = 0.3, the kink of the linear regime
  blend; likely a closure smoothness issue, accepted as is (see TODO).
- Adapted (all grids converged): increments 2.60 -> 0.72 -> 0.10 kg/s, observed order 1.9
  then 2.8; Richardson estimate of the limit about 23.45 kg/s. 400 -> 800 changes 0.43 %
  in all regimes. Profile changes between grids shrink: 400 -> 800 about 0.05-0.27 % of
  the field range, 2-3 times smaller than 200 -> 400. Close to grid convergence at 800,
  with about 0.1 % mass flow left.
- The refined K-free solution moves **away** from the published one: mass flow +0.32 %
  (400) -> +0.76 to +0.94 % (800); limit about 0.9 % above the published value. The
  published solution was produced by the older K-inverse code on 600 rows.

K-inverse (400 cells) restarted from the converged K-free 400 solution:

| Regime | Result | Max residual start -> end | m_dot change | Mean profile drift (of range) |
|---|---|---|---|---|
| subsonic | **converged** 99 / 13 | 1.2e-3 -> 4.2e-10 | +0.04 % | 0.2-0.34 % |
| shock | stagnation 354 / 122 | 1.8e-3 -> 0.19 | -0.09 % | 0.2-0.45 % |
| adapted | wall budget 985 / 289 | 1.6e-3 -> 0.18 | +0.10 % | 0.2-0.35 % |

- From a good start, K-inverse can converge (subsonic). Its solution agrees with K-free
  within 0.04 % in mass flow and 0.2-0.34 % of range in the profiles; the local maximum
  Mach differs more (0.656 vs 0.620).
- For shock and adapted it cannot hold a steady state: the residual grows by two orders
  of magnitude while the state drifts ~0.3 % of range.

## Energy conservation issue (found 2026-11-19, model-level, not from k_free)

Gas/liquid temperatures at the outlet exceed the 295.15 K inlet (subsonic T_g 300 K,
shocked 304 K). Mixture total enthalpy per unit mass rises from inlet to outlet by 3.6 %
(subsonic), 6.8 % (shocked), 9.8 % (adapted) in both k_inverse and k_free (400 cells);
it should be constant (adiabatic, no wall work/friction). Budget of the converged
residual, sum over cells of gas+liquid energy rows, as fraction of inlet energy flux
(subsonic; shocked in brackets):

| Part | Mixture energy |
|---|---|
| nonconservative | +2.9 % (+5.8 %); gas and liquid rows do not cancel (-5.8 % / +2.9 % per phase) |
| interphase sources (closures) | +0.5 % (+0.7 %) |
| area source | 0 (gas/liquid rows cancel) |
| stabilization | 0 (1e-15) |
| outlet - inlet energy flux | +3.3 % (+6.5 %) |

Same budget for K-inverse and K-free (2.86 % vs 2.93 %): it is in the shared model
(energy rows of `X_matrix` and of the closure source vector, identical to alberto_branch),
not in the discretization. Percentages depend on the enthalpy reference; sign and size do not.

Hypothesis (2026-11-20): the NC terms are written so that the interfacial (NC) forces do
not increase the phase entropies; as a consequence total energy is not conserved, and the
imbalance is energy exchanged with the interface. In the subsonic budget the interface gives
5.8 % to the gas and takes 2.9 % from the liquid, a net +2.9 % to the mixture. The
closure-source part (+0.5-0.7 %) has the same origin: the closures conserve energy, but the
source vector on the RHS also carries inviscid (interface-related) terms; the imbalance
appears only with closures on. The effect is large and is to be acknowledged as a known
model property wherever results are reported. Verification steps in [TODO.md](TODO.md) 1c.

## Open

Remaining steps are tracked in [TODO.md](TODO.md).
