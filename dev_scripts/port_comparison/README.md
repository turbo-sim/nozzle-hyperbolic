# Two-fluid port comparison: `alberto_branch` vs refactored `nozzle-hyperbolic`

Status at 2026-11-04. No code was changed in either repository; everything below comes from reading the code and running comparisons.

- **Reference (taken as correct):** `nozzle-hyperbolic_alberto_branch`, commit `faafff7`. Working-tree changes not committed there were not ported.
- **Port under test:**
  - `nozzle-hyperbolic/src/nozzle_hyperbolic/models/two_fluid/`, the physical model;
  - `solvers/fvm/` and `numerics/`, the model-agnostic engine.
- **Report sent to Roberto:** https://claude.ai/artifact/595GK3EYuGJYrvmXFj2tUz

---

## 1. Summary

- **Physics: identical.**
  - Code: same closures, matrices, FVS kernel, nonconservative term, area source and p·Dα interfacial pressure work.
  - Residual check: old vs new residual agrees to ≤ 3.5e-10 on two-component and flashing states.
- **What differs:** reconstruction (Van Albada), boundary formulation, removed clipping and safety floors, mesh convention, solver engine, defaults, and features that were not ported.
- **End-to-end runs:**

  | Case | Result |
  |---|---|
  | Two-component, Van Leer | ≤ 0.15 % difference |
  | Two-component, Van Albada | 0.8 % in α, 1.8 % in h_l |
  | Flashing, adapted | ṁ +0.10 %, max 0.5 % in h_g |

- **Open:**
  - Van Albada ε;
  - details of the BC assembly in the engine;
  - options never tested (only defaults were run);
  - why the NICFD flashing cases do not converge in the new code.

---

## 2. Harness (this folder)

| File | Purpose |
|---|---|
| `astcmp.py` | AST diff of the kernel functions in the two repos |
| `mkspec.py <case> <out>`, `mkspec2.py` | Write `spec.json` for a NICFD case (reference state, EOS, list of option combinations) |
| `res_old.py <out>` | Old residual via `make_two_fluid_step(..., return_debug=True)[2]` → `R_old.npz` |
| `res_new.py <out>` | New residual via `Operator(model, scheme).physical_spatial_residual` on a `Mesh` built with the **old layout** (centres = `x[1:-1]`, faces = midpoints, averaged A_face, volumes A·dx) → `R_new.npz` (including per-term contributions) |
| `cmp.py <out>`, `cmp2.py <out>` | Relative error per equation, normalised by the size of the conservative-flux term |
| `bc_new.py <cases...>` | Inserts the converged old ghosts of the NICFD references into the new trace equations |
| `pdalpha.py`, `pdalpha2.py` | Size of the p·Dα term; check that `D_VE_algebric` equals `D_VE_conservative` |
| `e2e_new.py <recon> <out>` | End-to-end run with the new solver: two-component subsonic, 198 cells |
| `flash_new.py <out> <cfl> <newton>`, `flash_cmp.py` | End-to-end run of the flashing adapted case (598 cells) and the old-vs-new comparison |
| `e2e/`, `flash/` | Old-driver YAML files (`SETTINGS_YAML=... python main_two_fluid.py`), logs, `.npz` files, figures |
| `tc_sub/`, `fl_sub/`, `fl_pin/`, `tc_sub_gc/` | Residual-test states and results |
| `report/` | HTML report |

**Environment.**

- **Interpreter:** `C:/Users/alber/AppData/Local/pypoetry/Cache/virtualenvs/nozzle-hyperbolic-LbBBi-fu-py3.12/Scripts/python.exe`, with editable jaxprop.
- **Code selection:** done through `PYTHONPATH`.
  - Old: `<alberto_branch>;<alberto_branch>/nozzle_hyperbolic/fvm_solver`
  - New: `<nozzle-hyperbolic>/src;<nozzle-hyperbolic>` (the repo root is needed for `cases.NICFD2026.scripts.case`)
- **Do not use `python -I`:** it ignores `PYTHONPATH`.
- **Old driver:** needs `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 MPLBACKEND=Agg`, otherwise it fails with a cp1252 Unicode error.
- **Tables:** all tests used the old tables in `dev_scripts/fluid_tables/*.pkl`.

---

## 3. Verified identical (no action needed)

Residual agreement ≤ 1e-10 (relative to the conservative-flux term) for:

- **Closures:** mass transfer, drag (Schiller–Naumann / Ishii–Zuber) and HTC, with `area_law="original"`.
- **Sources:** `D_VE_algebric` and `D_VE_algebric_two_component`, including **p·Dα**:
  - gas energy −p·Dα, liquid energy +p·Dα;
  - `Dα = (a₀²/ρ_s)(α_l M_g/a_l² − α_g M_l/a_g²)`;
  - with it, the algebraic sources equal `D_VE_conservative` to 1e-16.
- **Area source:** well-balanced, `dlnA = ΔA_face`.
- **Nonconservative term:** `nc_form` = `cell` / `staedtke` / `wb`.
- **FVS flux:** Harten-type entropy fix (`smoothed_sign`, `delta_factor`) and low-Mach rescaling. Tested only at `lowmach_diss=1`, `lowmach_diss_bc=0`.
- **Shared-velocity blend:** `du_blend_lo`/`du_blend_hi` 5/20.
- **4th-order JST filters:** `eps_void` (0.05, frozen speed) and `eps_ghost`.
  - With a consistent ghost, the result is identical.
  - The old filter reads the stale ghost from W^n (see D2).
- **Saturation ghost pin `pin_ghost`:** active in 40 cells of the flashing state.
- **Reconstruction:** constant and Van Leer.

---

## 4. Differences found

Each item gives the old behaviour, the new behaviour, the evidence, and whether it can change the results.

### D1. Van Albada limiter: ε removed

- **Old** (`two_fluid/solver.py`, `_muscl`, about lines 988–1050):
  - `eps = va_rel·(|U_i| + 0.1·mean|U|)`, with `va_rel = 1e-3` (mode `'magnitude'`).
  - Other modes: `'range'` (1e-4) and `'smoothness'` (1e-12).
  - `va_mean_stop_gradient = True`.
- **New** (`numerics/reconstruction.py`):
  - Generic coordinate-based `van_albada` with no ε.
  - The MC limiter is not in the JAX path.
- **Evidence:**
  - Residual difference: two-component up to 0.48 (median 3.6e-3); flashing 6.8e-2.
  - End-to-end, two-component: α 0.79 %, h_l 1.8 %.
  - Changing Van Albada to Van Leer moves α by 0.78 % in the old code but only 0.06 % in the new one. Without ε, the new Van Albada behaves almost like Van Leer.
- **Impact:** the **only interior numerical difference** that changes results at the percent level. It matters because the old default settings use Van Albada.
- **Where a fix belongs:** an ε option in the generic reconstruction (`numerics/`), **not** in the two-fluid model.

### D2. Boundary formulation: ghost cells replaced by endpoint traces

- **Old:**
  - **Location:** ghost cells at one dx outside the first and last cells (`x = linspace(x1, x2, Nx)` includes them).
  - **How they are set:** overwritten **inside** `_residual` by explicit formulas `_bc_inlet` / `_bc_outlet`, so they are not Newton unknowns with their own equations.
  - **Slope:** zero in the ghost.
- **New:**
  - **Location:** the state has N+2 rows (`full`, `points = r_[faces[0], centers, faces[-1]]`). The two end rows are **traces on the boundary faces** (x = 0, L): no volume, no balance equation.
  - **How they are set:** fixed by 6 algebraic BC equations plus 2 EOS constraints each (`models/two_fluid/boundaries.py::boundary_residual`), solved **coupled** with the cells in Newton (`boundary_treatment="coupled"`; a segregated mode also exists).
  - **Slope:** zero at the trace.
- **Consequences:**
  - **First-cell MUSCL gradient:** taken over dx/2 (to the trace) instead of dx (to the ghost).
  - **Extrapolations:**
    - Old `total_state`: `2U₁ − U₂`, to the ghost at −dx.
    - New: `U₁ + 0.5(U₁ − U₂)` (`model._left_factor`), to the face.
    - Each is linear extrapolation to its own position, so the two are consistent.
  - **JST filter at the end faces:** old uses the ghost from W^n (stale); new uses the current trace. Same at steady state, different in transients.
- **Evidence:**
  - The old converged ghosts of all six NICFD references satisfy the new trace equations: outlet to 1e-13; inlet see D3.
  - End-to-end Van Leer differences (≤ 0.15 %) sit near the ends. This fits a boundary-placement effect, expected to be O(dx).
- **Impact:** small. It should vanish as the grid is refined; **not yet shown** (T3).

### D3. Stagnation inlet: approximate vs exact isentrope

- **Old:**
  - Mixture velocity from cell 1 (zero-order).
  - 4 fixed-point iterations ρ ⇄ u ⇄ (p, h).
  - α from quality plus slip.
  - p from a **precomputed 400-point isentrope table** of the dominant phase (`jnp.interp`).
  - Everything clipped to the table bounds.
- **New:** six equations:
  - dominant-phase entropy = s₀, exactly;
  - slip;
  - mixture-velocity continuity with cell 1;
  - quality, in cross-multiplied form (no division by the velocity);
  - H_g;
  - H_l.
- **Evidence:** the inlet entropy of the old references is off by −9.2e-4 J/kg/K for the gas (about 6 Pa) and +1.0e-2 J/kg/K for the liquid (about 4 kPa at 120 bar). This is the old interpolation error from the table plus the 4 iterations.
- **Impact:** same equations; the new code is more exact. A difference of up to about 4 kPa in liquid-dominant (flashing) cases is expected.

### D4. Outlet

- **Policies:** both codes have `characteristic` (projection onto the incoming acoustic eigenvector r4) and `characteristic_old` (Wood impedance, shared velocity), with a tanh Mach blend.
- **Clipping:** the old code also clips the outlet ghost (p, h, and α in `[alpha_eps, 1 − alpha_eps]`); the new code does not.
- **NICFD references use both policies:**

  | Case | Outlet policy |
  |---|---|
  | tc subsonic | `characteristic_old` |
  | tc shock | `characteristic_old` |
  | fl subsonic | `characteristic_old` |
  | Others | `characteristic` |

  Both policies agree to 1e-13 with the old ghosts, but each case must use the right one when compared.

### D5. All clipping removed (robustness change)

- **Old clips** to the table bounds:
  - `matrices.py:67–69` (`phase_props`, p and h);
  - `sources.py:322–326` and `445–449` (closure EOS calls);
  - every BC formula (`solver.py:550–977`);
  - the MUSCL fallback to first order when a reconstructed state leaves the table ("bad" mask).
- **New:** no clipping. An out-of-table state gives NaN or a failed EOS call, and the step is rejected.
- **Impact:** identical inside the table. Cases that touch table edges now fail instead of being clamped and carrying on; for example, metastable liquid near `h_min`, or ghosts at the edge of the table. This is a **prime suspect for the NICFD flashing non-convergence**, together with the regenerated tables (D13).

### D6. FVS safety floors and the flux solve

- **Floors removed:**
  - `max(|a|, 1e-8)` in the face Mach number (`flux.py:238`, `408`);
  - `max(a², 0)` in the eigenvalues.
- **Shared-velocity flux:** `_equil_lstsq` (`flux.py:69`, least squares, `rcond = 1e-14`) is replaced by `compatible_flux_solve`. The new solve is exact, treats a phase at rest explicitly, and returns NaN if the system is incompatible.
- **Full-slip branch:** rewritten via `L_h y / D`; algebraically equal.
- **Impact:** identical in all states tested. Results differ only for a = 0, for non-hyperbolic states (a² < 0), and at exactly zero velocity.

### D7. Drag (and HTC?) Reynolds floor

- **Old** (`source_two_fluid.py:110–114`): `Re_safe = max(Re, 1e-12)` and a viscosity floor. **New:** written analytically, without the floor.
- **Impact:** same values. The Jacobian differs only at zero slip.
- **To check:** whether the Ranz–Marshall HTC floor (`source_two_fluid.py:500`) is also gone.

### D8. Precedence of `area_params`

- **Old:** keyword arguments passed at the call site override `area_params`.
- **New:** `ClosureConfig.area_params` overrides.
- **Impact:** only when both are given.

### D9. Mesh convention (NICFD case set-up only, not the engine)

| | Old driver | New `cases/NICFD2026/scripts/case.py` |
|---|---|---|
| Face area | `0.5(A_i + A_{i+1})` | Exact A(x_face) |
| Volume | `A·dx` | Gauss-integrated |
| Domain | `[dx/2, L − dx/2]` | `[0, L]` |

- **Impact:** O(dx²) on smooth geometry. It also changes the well-balanced area term `ΔA_face`.
- **Note:** all residual and end-to-end tests in this folder used the **old layout** on the new solver, so they exclude this difference.

### D10. Solver engine and time integration

- **Old:**
  - Newton on 8 rows (`eos_rows`) or with projection (`eos_project`), `w_char` scaling;
  - normal equations or lstsq;
  - adaptive dt with SER;
  - stop on SSR.
- **New:** generic implicit engine:
  - fixed scales;
  - banded Newton with backtracking;
  - CFL controller (`cfl`, `cfl_growth`);
  - convergence measured on the original residual.
- **Evidence (time):** two-component took 41 steps / 150 s (new) vs 8000 steps / 17 min (old); flashing took 55 steps / 150 s vs 2411 steps / 633 s.
- **Evidence (steady state):** the converged solutions agree (section 5).
- **Impact:** no effect on the steady state was detected. Transients were not compared (T8).

### D11. Default values

| Parameter | Old (`settings/defaults.yaml`) | New |
|---|---|---|
| `eps_void` / `void_strength` | 0.05, frozen | 0 (off) |
| `alpha_eps` / `alpha_min` | 1e-6 | 0 |
| Reconstruction | Van Albada (with ε) | constant |
| `lowmach_diss` / `lowmach_diss_bc` | 1 / – | 1.0 / None |
| `du_blend` | 5 / 20 | 5 / 20 |
| `nc_form` | cell | cell |
| `interfacial_area` / `area_law` | polynomial | polynomial |
| Dominant phase | `auto` available | must be set explicitly |
| Newton iterations | 7 | engine setting |

- **Impact:** a run with the new code's defaults is **not** the old model. Old runs must be reproduced with explicit settings.

### D12. Not ported

- **Per-cell `ai_scale` array:** only a scalar is supported. This **blocks the twophase-da β(x) injection**.
- **Inlet:** `stagnation_ramped`.
- **Dominant phase:** `dom_phase: auto`.
- **Algebraic momentum pin:** `pin_mom_tau=None`; the new code requires `momentum_time > 0`.
- **Small options:** `nc_scale`, the `va_eps_mode` options, the MC limiter.
- **Diagnostics:** `debug_newton`, `ssr_mode`.
- **Uncommitted work in the old branch:** `sum` HTC, `jakob_growth_nusselt`, `nu_l_int`.

### D13. Case and data issues (outside the solver)

- **NICFD references:** the `reference.csv` profiles come from `e185f57` (September), before p·Dα.
  - They are out of date for **both** codes; the flashing case differs from the old code by 0.8 % in p/α and 4.9 % in h_g.
  - Agreement with them therefore does not validate the port.
- **Missing setting:** `two_component/supersonic_shock/settings.yaml` has no gas `N_h`.
- **Tables:**
  - The new `protocol.json` raises the liquid `h_min` to 250 kJ/kg.
  - Tables freshly generated by jaxprop `74794ec` differ at the invalid row `h_min = 1 kJ/kg` for p > about 10 bar. This makes the PT inverse fail in `demos/two_fluid/run.py`; the old table works.
- **Old flashing solution:** has a kink one cell wide at x ≈ 0.865. The largest old-vs-new differences (u_g 0.31 %, h_g 0.53 %) are located there.

---

## 5. Results so far

| Test | Result |
|---|---|
| Residual, two-component and flashing, all options in section 3 | ≤ 3.5e-10 |
| Residual, Van Albada | up to 0.48 (two-component), 6.8e-2 (flashing) |
| BC traces vs old ghosts (6 NICFD cases) | outlet 1e-13; inlet within the isentrope-table error |
| Two-component subsonic, 198 cells, Van Leer | ≤ 0.15 % of range |
| Two-component subsonic, Van Albada | α 0.79 %, h_l 1.8 % |
| Flashing adapted, 598 cells, old tables | ṁ 23.410 vs 23.433 kg/s (+0.10 %); onset x = 0.0902 m in both; p and α 0.06 %; u_g 0.31 %; h_g 0.53 %; T_g 0.26 K |
| p·Dα size | two-component ≤ 0.2 %; flashing median 0.8 %, max 6 % of the per-phase energy source |

---

## 6. Next steps

### Priority A: decides whether the port is correct

- **T1. Jacobian equality.**
  - **What:** compare old `jax.jacfwd(_residual)` with the new `dense_jacobian` on the same state.
  - **States:** zero slip; Δu inside the blend band (5–20 m/s); α = 1e-6; flashing with the pin active.
  - **Pass:**
    - interior rows ≤ 1e-8 relative;
    - boundary rows compared separately (they differ by design);
    - if they differ only at zero slip, explain by D7.

- **T2. Van Albada ε (D1).**
  - **Step 1:** rerun the residual test with the old `va_rel → 0`. **Pass:** old and new agree to 1e-10. This would show that ε is the only difference.
  - **Step 2:** rerun the two-component end-to-end case with the old ε at three grids, to see whether the difference shrinks.
  - **Step 3:** decide whether to add ε as a generic reconstruction option in `numerics/`.

- **T3. Grid sequence (D2, D9).**
  - **What:** N = 100, 200, 400; two-component and flashing; Van Leer; both codes.
  - **Pass:** the old-vs-new difference roughly halves with each refinement (O(dx), boundary placement).
  - **If it does not:** there is a difference in the boundary implementation beyond placement, so continue with T4.

- **T4. BC implementation in depth (D2–D4).**
  - **Read:**
    - how the engine assembles the system (`solvers/fvm/operator.py`, how the boundary rows are packed);
    - the coupled and segregated paths.
  - **Compare:**
    - the BC blocks of the Jacobian;
    - the BC types `total_state`, `subsonic_inflow`, `static_state`, `characteristic_old` and the pressure outlet;
    - a ramp on `p_out` (callable `back_pressure`);
    - reverse flow at the inlet.
  - **Pass:** same converged traces as the old ghosts, within the D3 tolerance.

- **T5. Entropy fix.**
  - **States:** sonic points (transonic throat, shock cells).
  - **What:** sweep `delta_factor` = 1e-6 … 1e-1 and compare the residuals.
  - **Pass:** ≤ 1e-10 at every value.

- **T6. Low-Mach rescaling.**
  - **What:** sweep `lowmach_diss` = 0, 0.01, 0.1, 1 and `lowmach_diss_bc` = 0 and 1.
  - **States:** low-Mach states (inlet region of the two-component subsonic case).
  - **Pass:** residuals ≤ 1e-10.

- **T7. JST filter options.**
  - **What:**
    - `ucut`/`uwidth` gating;
    - `eps_void_mass_acut`;
    - `eps_ghost_k2 > 0`;
    - the `wood` speed scale;
    - the stale-ghost effect in a transient (one step from a state that is not converged).
  - **Pass:** steady-state residual ≤ 1e-10; any transient difference limited to the end faces.

- **T8. One implicit step.**
  - **What:** same state and fixed dt; the old `step` vs one new Euler-implicit step with the CFL controller off.
  - **Pass:** differences only in the boundary cells and within the solver tolerance.

### Priority B: cases

- **T9. Remaining NICFD cases on both codes.**
  - **Cases:**
    - two-component, adapted and shock;
    - flashing, subsonic and shock;
    - plus a cold start with no restart.
  - **Requirements:** use the right outlet policy for each case (D4) and add the missing gas `N_h`.
  - **Pass:** differences of the same size as in section 5.

- **T10. Why the NICFD flashing cases fail.**
  - **What:** run the new solver in four configurations:

    | Run | Tables | Mesh |
    |---|---|---|
    | a | old | old layout (known to work) |
    | b | regenerated (h_min 250k) | old layout |
    | c | old | NICFD exact-area mesh |
    | d | regenerated | NICFD exact-area mesh |

  - **Then:** find the change that breaks it, and check whether the failure comes from an out-of-table state that the old code clipped (D5).

- **T11. Edge states.**
  - **States:** equal velocities; slip changing sign; α at 1e-6 and 0; states at the table bounds.
  - **Goal:** document where the old code clips and the new one rejects (D5, D6).

### Priority C: features and data

- **T12. Missing features (D12).**
  - **Decide what to port:**
    - per-cell `ai_scale` (needed for twophase-da);
    - `sum` HTC / `nu_l_int`;
    - the algebraic momentum pin;
    - `dom_phase: auto`;
    - `stagnation_ramped`.
  - **Keep the separation between solver and models:**
    - closures and `ai_scale` belong in `models/two_fluid/` (`ClosureConfig`, closures);
    - BC types belong in `models/two_fluid/boundaries.py`;
    - limiter options belong in `numerics/`.
- **T13. Regenerate the NICFD references with p·Dα,** using the old code (`faafff7`, or the current branch) or the new code once T1–T4 pass.
- **T14. Water table generation:** use `h_min` ≳ 10–20 kJ/kg, or reuse the old `.pkl` files.
