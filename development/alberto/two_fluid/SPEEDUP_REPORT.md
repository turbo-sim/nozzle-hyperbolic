# Speed-up of the two-fluid solver: report

2026-10-05/06.  Scope: `nozzle_hyperbolic/fvm_solver/two_fluid/` (Städtke two-fluid
FVS, E3-banded Newton) and its driver `dev_scripts/main_two_fluid.py`.  Test case:
SMD flashing nozzle (`settings_flashing_smd_supersonic_test.yaml`), N = 100 / 300 / 600.

**Status:** all changes are in the local working tree, not committed and not pushed.
Every change is a YAML switch.  **With all switches at their defaults the code is
bit-for-bit the git HEAD version** (verified, §5), so old settings files reproduce
old runs exactly.

**Update 2026-10-06 (settings defaults chain):** settings files now build on
defaults files through `base:` (see `config_loader.py`).  `settings/defaults.yaml`
turns ON the exact switches of §3.1–3.2 (`fast_wb_flux`, `flux_single_branch`,
`flux_local_jac`, `cell_local_jac`, `thomas: chol`, `newton_tol: 1e-8`,
`fast_increments`); SER stays off (`ser_beta: 0`).  Settings files without a
`base:` key — including the copies in old run folders — load
`settings/defaults_legacy.yaml`, i.e. the old code path, as before.

---

## 1. Headline numbers

N600 restart from N300 (same IC and settings as
`results_bubble_number_sens/smd_supersonic_N600_muscl_van_albada_20260915_081202`):

| Run | Steps | Wall | Result vs reference |
|---|---|---|---|
| Reference (git HEAD) | 3,410 | 6,984 s | — |
| Faster steps only (§3.1–3.2) | 3,410 | **487 s** | identical (1e-13 in every field) |
| Faster steps + CFL ramp (§3.3) | 259 | **64 s** | same except at the second sonic point (§6.1) |

Under equal conditions (same cores, same power state) the faster steps give
**5.1× per step** (1,070 → 210 ms) and **~7.5× per run**; the CFL ramp gives another
**~7.6×**; combined **~57×**.  The reference above ran on the slow E-cores (§6.3),
which adds ~2× to the comparison with the actual reference wall time.

---

## 2. Where the time went (profile of git HEAD)

N300, `jacobian: colored`, `eos_rows_banded: true`, `newton_iter: 8`, `jac_lag: 1`,
CFL 2: **~0.5–0.95 s per step**, every step doing 8 Newton iterations of:

| Piece | ms / iteration |
|---|---|
| Colored Jacobian (linearize + 40 JVP seeds) | 75 |
| Banded E3 normal-equations solve (block Thomas) | 18 |
| Residual | 8 |

- The Jacobian was ~75% of the step; inside it, the **face flux** dominated (~39 ms):
  6 EOS lookups, 3 eigenstructures, a 6×6 solve and an SVD per face, replayed for
  each of the 40 seeds.
- The `wb` non-conservative term re-evaluated the **full FVS flux at U_L = U_R** in
  every cell (~31% of the Jacobian).
- `lax.cond` in `fvs_interface_flux` lowers to `select` under `vmap`: **both branches
  run, and `_blend` re-runs the zero-slip branch** (two SVD-lstsq per face).
- At CFL 2 **Newton reaches roundoff in 4–5 iterations**; iterations 6–8 change W by
  ~1e-12.
- The driver's `_increments()` ran op-by-op (eager JAX): 16 ms per step.
- 70% of a typical run is the **slow tail** (SSR 1e-5 → 1e-8) at fixed CFL: dt is set by
  the liquid sound speed (~1000 m/s) while convergence waits on slow material waves and
  the near-sonic region.
- Not bottlenecks: the bicubic EOS (XLA already prunes the ~36 unused properties
  jaxprop computes), JAX compilation cache.

---

## 3. Changes

### 3.1 Solver — faster steps, same result (`numerics:` block)

| YAML key (default → recommended) | What it does | Where | Gain |
|---|---|---|---|
| `fast_wb_flux: false → true` | `wb` cell flux F_c computed directly as the physical flux F(U_i). The FVS flux is consistent, F̂(U,U) = F(U) (full branch: ρρ⁻¹F; zero-slip: y = K⁻¹·0 = 0), so the value is identical (~1e-15). **The face FVS flux is not changed.** | `solver.py` `_nc_all_interior` | Jacobian −31% |
| `flux_single_branch: false → true` | Evaluate zero-slip and full-slip flux branches once each, select with `jnp.where` (same values, same NaN masking) | `flux.py` `fvs_interface_flux(single_branch=)` | with the above: 1.6× per step |
| `flux_local_jac: false → true` | `jax.custom_jvp` on the face-flux map: each face is linearized once (exact 6×12 block from 12 tangents) and every colored seed becomes a 6×12 mat-vec | `solver.py` `_flux_faces_local` | Jacobian 75 → 24 ms |
| `cell_local_jac: false → true` | Same for the per-cell maps: NC correction, sources, filter spectral radius, ghost pin, wb F_c (`_local_jac_map`) | `solver.py` | a few % |
| `thomas: orig → chol` | Block Thomas with one Cholesky per super-block; back substitution x_i = g_i − G_i·x_{i+1} (no solves) | `solver.py` `_e3_banded_solve` | ~10% of the solve |
| `newton_tol: 0.0 → 1.0e-8` | `lax.while_loop`: stop when max\|ΔW/w_ref\| < tol, `newton_iter` stays the cap. Per-step result changes only at ~tol² | `solver.py` `step` | 8 → 3.6–6.7 iterations |

How the `custom_jvp` works: the colored Jacobian calls `jvp_fn` 40 times (8 variables ×
5-cell stencil).  Without it, each call replays the derivative of every face flux.
With it, `jax.linearize` stores the per-face 6×12 matrices once (they depend only on
the state) and the per-seed work is one small mat-vec per face: 40 → 12 flux-tangent
passes.  It is the chain rule with a different grouping, so the Jacobian is unchanged
to ~1e-15.  The rule is linear in the tangent, so reverse mode (`grad`, adjoints)
works through it automatically (§7).

### 3.2 Driver — overhead and hardware (`run:` block)

| YAML key | What it does | Gain |
|---|---|---|
| `fast_increments: false → true` | Jitted version of the per-step increment check (same formulas) | 16 → 0.1 ms per step |
| `cpu_affinity: null → "0-15"` | Pin the process to the P-cores (§6.3) | up to ~2×, depending on scheduling |

### 3.3 Fewer steps — same steady equations, different path

| YAML key | What it does | Measured |
|---|---|---|
| `time_control.ser_beta: 0.0 → 1.0`, `ser_cfl_max: 50` | SER / pseudo-transient continuation: CFL_eff = CFL·clip((SSR₁/SSR)^β, 1, cfl_max/CFL). Existing increment checks and halving remain the safety net. Steady residual uses dt = 1, dV = 0 → unchanged steady equations | N600: 3,410 → 259 steps; N300 (N420 seed): 3,846 → 363 steps |
| `bc.p_out_ramp_steps` (0 = off), `p_out_ramp_from`, `p_out_ramp_shape` | Ramped start-up: back-pressure lowered from p0 (stagnation IC) to p_out over N steps; SSR/plateau stop and SER wait for the end of the ramp. Needs the new runtime argument `step(W, dt, p_out=None)` (None = original path) | N100 from rest: 1,000 linear steps OK |
| `grid_sequence: {Nx: [...], ssr_tol: [...]}` + `python run_grid_sequence.py <yaml>` | Grid sequencing: level 1 runs the YAML as is, each finer level restarts from the previous solution (driver's `restart` IC interpolates in x), ramp off. Each level writes a normal results folder; `summary.txt` in `<repo_main>/<repo_run>_sequence_<timestamp>/` | from rest to converged N300: N100 1,067 steps / 101 s + N300 209 steps / 58 s |

Ramp length: on N100, 300 and 600 steps (linear or log) blow up at the **end** of the
ramp (p_out ≈ 1–2 bar) — the expansion needs ~0.026 s physical time to establish;
1,000 linear steps work.  Scale with 1/dt on other grids; normally ramp only on the
coarsest level.

### 3.4 Recommended settings

```yaml
numerics:
  fast_wb_flux: true
  flux_single_branch: true
  flux_local_jac: true
  cell_local_jac: true
  thomas: chol
  newton_tol: 1.0e-8
time_control:
  ser_beta: 1.0          # changes the path (§6.1); omit to reproduce old runs exactly
  ser_cfl_max: 50.0
run:
  fast_increments: true
  cpu_affinity: "0-15"
```

Use §3.1–3.2 alone to reproduce existing results (NICFD figures, GCI ladders); add
§3.3 for new studies.  Since 2026-10-06 the §3.1–3.2 values are the defaults of
`settings/defaults.yaml`, so a case file with `base: defaults_<family>.yaml` only
needs the SER and affinity lines.

---

## 4. Results

| Case | git HEAD | New | Notes |
|---|---|---|---|
| N300 from N420 seed (ref `..._20260914_225038`) | 3,820 steps, 3,138 s | 3,846 steps, 903 s (faster steps, early prototype) · 363 steps, 117 s (+ SER) | |
| N600 from N300 (ref `..._20260915_081202`) | 3,410 steps, 6,984 s | 3,410 steps, 487 s · 259 steps, 64 s (+ SER) | §1 |
| N600 Nb=1e12 (`..._20260915_081252`, stopped after 7,140 steps / 3.5 h at SSR 1.7e-6) | not converged | 258 steps, 135 s (on E-cores) | agrees with the stopped run to ~2e-4 |
| N300 Nb 1e10 → 1e12 restart (sweep) | — (no baseline) | 131 steps, 87 s (CFL up to ~2000) | shared machine |
| Cold start N100 → N300 | — | 1,276 steps, ~160 s | matches ref upstream of x ≈ 0.79 m to ~2e-6 |
| Per step, same cores, N600 | 1,070 ms | 210 ms | 5.1× |
| Per step, N420→N300 transient, `nc_form: cell` | 641 ms | 325 ms | 2.0× (no F_c to save) |
| Per step, same, `nc_form: wb` | 879 ms | 340 ms | 2.6× |

---

## 5. Verification

- **Defaults vs git HEAD:** residual, colored Jacobian and 3 marched steps
  bit-identical (`np.array_equal`), also after the runtime-`p_out` plumbing.  Driver run
  with defaults: steps 1–100 print exactly the reference log (t, dt, SSR).
- **All switches on vs HEAD:** residual 1.9e-15, Jacobian 5.6e-15, 3 steps 1.5e-14
  (relative).  Same for `nc_form: cell` and `staedtke` (residual difference exactly 0).
- **N600 end to end (faster steps):** 3,410 steps and final SSR 9.994e-9, identical to
  the reference; every SSR line printed by the reference reproduced; solution 1e-13.
- **Ported code = prototype:** SER N600 run bit-identical to the prototype run.
- **Reverse mode:** VJP of the residual with switches on vs off 7e-16; vs dense
  `jacfwd` Jacobian transposed 4.7e-13.

---

## 6. Findings worth knowing

### 6.1 Two steady states at the second (flashing) sonic point

The SMD flashing case chokes a second time in the divergent section (M_mix crosses 1 at
x ≈ 0.82 m on N300, cells 203/204; x ≈ 0.93 m on N600, cells 463/465).  There are at
least **two discrete steady states**, both with SSR < 1e-11 under the **original** code,
differing in which face triggers the hard expansion-sonic entropy-fix switch
`sonic = (λ_L < 0) & (λ_R > 0)` in `flux.py`.  Differences: up to ~0.3 bar (N300) /
~0.04 bar (N600) near the sonic point, carried downstream in the vapour velocity and
enthalpy (≤ 2.5–5% of range); mass flow identical to ~2e-6; upstream ~1e-6.  Which state
a run ends on depends on the transient path, even at roundoff level.  This is **not**
flashing onset (that is at x ≈ 0.09 m).

Implications: SSR < 1e-8 does not pin that region; expect noise there in GCI studies;
compare runs with a tolerance there.  A C¹ blend of the sonic switch is a candidate fix.

### 6.2 The production colored Jacobian is not exact

With `va_eps_mode: magnitude` the van Albada ε contains `jnp.mean(|U_h|)` over all
cells — a weak global coupling (8e-6 relative, outside the ±2-cell band) that the
coloring cannot represent: colored vs dense Jacobian differ by 3.8e-8 (single entries
up to 0.5%).  Harmless for the converged solution (only the residual matters; Newton is
slightly less than quadratic); relevant for adjoints (§7).  With
`va_eps_mode: smoothness` the coloring is exact (1e-22).

### 6.3 Hardware (i7-13850HX laptop)

- Logical CPUs 0–15 are P-cores, 16–27 E-cores (~1.9× slower single-thread).  Unpinned
  runs land on either: historical N600 runs show 1.1 vs 2.05 s/step for identical
  settings.  Use `run.cpu_affinity: "0-15"`.
- On battery every step is ~1.8× slower.  Compare timings only under the same pinning
  and power state.

### 6.4 Tried, did not help

| Idea | Outcome |
|---|---|
| Evaluate only the 4 needed EOS properties | no gain: XLA already removes unused ones |
| LAPACK banded Cholesky (`scipy.solveh_banded` via `pure_callback`) | 21 ms, slower than the JAX block Thomas (18 ms) |
| `scan` unrolling, `--xla_cpu_multi_thread_eigen=false` | no gain (sweep is per-call-overhead bound: 75 serial 32×32 blocks) |
| Frozen Jacobian within a step | linear convergence (factor 0.1–0.3 per iteration); not better than early exit |
| Just raising the CFL (10–50) from an early transient | Newton stalls at ~1e-2; needs the residual-driven ramp |
| Logarithmic back-pressure ramp | no better than linear (both fail at 300 steps) |

### 6.5 Further possible gains (not implemented)

- Zero-slip branch: SVD `lstsq` → LU solve cuts the flux derivative 39 → 29 ms, but
  drops the guard for u_i → 0 (K singular).  Would need its own switch and testing.
- Block cyclic reduction for the Thomas sweep (now ~13–17 ms, overhead bound).
- Local time stepping (per-cell dt).
- HEM solver (`fvm.py` `make_jax_stadtke_step`): same fixed 8 Newton iterations plus a
  **dense** `jnp.linalg.solve` on a banded system — the same ideas apply.
- `sensitivity_study.py` and `cases/r1233zde_inlet_sensitivity/run_simulation.py`
  build the solver themselves and do not forward the new switches yet.

---

## 7. Gradients (adjoint optimisation, data assimilation)

Compatible.  Recommended route: implicit-function theorem at the converged state
R(W*, θ) = 0 (steady rows + 2 EOS rows per cell → square system):

```
(∂R/∂W)ᵀ λ = (∂J/∂W)ᵀ            one transposed linear solve
dJ/dθ = ∂J/∂θ − λᵀ ∂R/∂θ          one jax.vjp of the residual
```

For a few parameters (Nb, Nd, T0, p_out) use the forward version
(∂R/∂W)·dW = −∂R/∂θ, one solve per parameter.  Independent of the path (CFL ramp,
early Newton exit).  Caveats:

- `newton_tol > 0` uses `lax.while_loop`, which JAX cannot reverse-differentiate: do not
  `jax.grad` through `step` with it (the IFT route does not need it).
- Use AD products or the dense Jacobian for ∂R/∂W, not the colored one (§6.2).
- Bubble/droplet numbers, BCs and geometry are baked in when the solver is built; they
  must become residual inputs to differentiate with respect to them.
- Hard switches (binary evaporation gate, sonic entropy fix, clips) make outputs only
  piecewise smooth; gradients can jump near the second sonic point (§6.1).  A smooth
  gate already exists, commented out, in `sources.py`.

---

## 8. Files changed

| File | Change |
|---|---|
| `nozzle_hyperbolic/fvm_solver/two_fluid/flux.py` | `single_branch` option |
| `nozzle_hyperbolic/fvm_solver/two_fluid/solver.py` | switches of §3.1, `_local_jac_map`, custom-JVP face flux, Cholesky Thomas, Newton early exit, runtime `p_out` (`step(W, dt, p_out=None)` → `_residual(..., p_out_t)` → `_bc_outlet(U_h, p_out_t)`) |
| `dev_scripts/config_loader.py` | new keys (defaults = old behaviour) |
| `dev_scripts/main_two_fluid.py` | forwards the switches, SER, ramp, jitted increments, CPU pinning, `[Speed-up]` log line, absolute settings paths |
| `dev_scripts/settings/settings_flashing_smd_supersonic_test.yaml` | new keys documented with **old-behaviour values**; commented `grid_sequence` example |
| `dev_scripts/run_grid_sequence.py` | new: grid-sequencing runner |
