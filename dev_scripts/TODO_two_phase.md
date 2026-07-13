# TODO — Two-Fluid (time-marching) Solver

## Correctness to verify

- [ ] **Validate `linear_solve: normal_eq` against `lstsq`** — *does the Cholesky normal-equations
      min-norm solve give the correct steady state?*
      Background: the Newton inner solve of `J·dW = −R` (`J` is rectangular `6(N−2)×8(N−2)`,
      solved for the min-norm `dW`) was replaced from dense SVD `lstsq` to the normal equations
      `dW = Jᵀ (J Jᵀ)⁻¹ (−R)` via Cholesky (+ tiny Tikhonov). **~15–18× faster/step** and finite,
      and it agrees with `lstsq` to ~1e-6 **per solve**. BUT over 5 steps the trajectories diverge
      ~15–20%, concentrated in the **momentum/void block** (`u_g, u_l, α_g, p` ~15%; enthalpies
      `h_g, h_l` agree to ~0.5%).
      - Cause (diagnosed): the **zero-slip degeneracy**. At `du = u_g − u_l = 0` (uniform IC) the
        void/acoustic wave structure degenerates → the momentum/void Jacobian block is near-singular
        (cond ~ 1/|du|). Drag & mass-transfer momentum are O(du), so the slip is a *weakly-determined
        (soft) mode*; SVD-truncation (`lstsq`) and Tikhonov (`normal_eq`) resolve that near-null-space
        differently → different transient paths. Partly physical, not just numerics.
      - **Deciding test**: run BOTH `linear_solve: lstsq` and `linear_solve: normal_eq` to steady
        state (same everything else) and compare the *converged* profiles (`p, α_g, u_g, u_l, T_g, T_l`).
        - If they match → transient divergence is benign, keep `normal_eq` for the speed.
        - If they differ → the regularization is biasing the physics; either tune the Tikhonov term,
          unify the regularization between solvers, or move to the square E3 formulation (below).
      - Convergence criterion for the comparison: mass-flux spatially constant
        `(ṁ_max − ṁ_min)/ṁ < ~0.1%` **and** SSR flat (SSR-plateau alone is unreliable in slow transients).
      - Files: `nozzle_hyperbolic/fvm_solver/two_fluid/solver.py` (`_lin_solve`), YAML `numerics.linear_solve`.

## Related root-cause work

- [x] **Zero-slip / checkerboard mode** — **SOLVED (2026-07-09), see the section below.** As predicted
      here, the cure was *dissipation on the void wave*: a 4th-order JST filter (`eps_void`) scaled by the
      spectral radius (non-vanishing as `u_g,u_l→0`). Root cause refined: it's the void-fraction (`α`)
      2Δx null mode — `α` rides the material waves `λ=u_g,u_l`, whose FVS dissipation `~|λ|·Δx` vanishes at
      the low-Mach inlet. Not a `du` slip issue per se (persisted at `du≈0.18`); the zero-slip *branch* was
      ruled out. Validated ε-independent + grid-convergent + matches nozzlex.

- [~] **E3 — square system via EOS constraints — DECIDED AGAINST (2026-07-09), do NOT implement.**
      Adding the 2 EOS closures per cell (`m_g=α_g ρ_g(p,h_g)`, `m_l=α_l ρ_l(p,h_l)`) would make it square
      `8(N−2)×8(N−2)` and unique — BUT that reintroduces a block that goes **singular exactly in the
      degenerate limits we must support**: zero slip (`du→0`) and a **vanishing (ghost) phase**. The
      RECTANGULAR min-norm (6 eqs, 8 unknowns) is INTENTIONAL — its under-determination is what keeps the
      solve well-posed in those limits. `lstsq` and `normal_eq` compute the SAME min-norm `dW` (differ only
      in conditioning near degeneracies), so the solver choice does NOT change the physics/wander. **Keep
      the rectangular min-norm.** The `p` null-space wander is inherent and tolerated by the EOS projection;
      it only turns fatal when the transient pushes `p` past the table edge (see dt-collapse item below).

## Performance (done / next)

- [x] Mixture (Städtke) wave speed in `compute_dt` and SSR `t_char` (was per-phase → ~9× larger dt).
- [x] Colored (banded) Jacobian — `jacobian: colored` (24 colors constant / 40 MUSCL), validated exact.
- [x] Normal-equations solve — `linear_solve: normal_eq` (~15× on the solve). *Correctness pending — see above.*
- [ ] **E1b / banded solve (O(N))** — current normal-eq is dense (`J Jᵀ` LU/Cholesky), still O(N³).
      Block-banded `J Jᵀ` + block-Thomas would give true O(N) scaling for large grids.
- [ ] **`adaptive_dt` controller** — implicit dt ramping past the acoustic CFL (mirror single-phase
      `adaptive_dt` flag: geometric growth + increment-based halving) to cut the number of steps to steady state.

## Two-component nozzle + low-Mach checkerboard (2026-07-09)

Case: water(liquid)+N₂(gas), no mass transfer, choked Elliot nozzle. YAML
`settings_two_fluid_nozzle_validation_two_comonent.yaml`; driver `main_two_fluid_single_phase.py`.

### Done
- [x] **Low-Mach void-fraction (`α`) checkerboard SOLVED — 4th-order JST filter.** In
      `two_fluid/solver.py` `_residual` (after nc/src, before fdiv): conservative flux
      `d_face = +eps_void·r_face·(V[j+2]−3V[j+1]+3V[j]−V[j−1])`, `r_face` = local spectral radius
      (non-vanishing as `u_g,u_l→0`), zeroed at the two boundary faces. **Sign must be `+`** (verified:
      `+` damps, `−` is anti-diffusion). `eps_void` YAML knob → `cfg.EPS_VOID` → `make_two_fluid_step`.
      `_sw` auto-widens to 2 when `eps_void>0` (colored Jacobian stays exact). 4th-order ⇒ smooth
      solution perturbed only `O(eps·Δx³)` but hits the 2Δx mode ~160× harder. `eps_void=0.1`: α
      checkerboard 87% → 0.6%.
- [x] **Validated** (`compare_nozzlex_timemarching.py`): ε-INDEPENDENCE proven (N=600 `eps=0.03/0.05/0.1`
      → identical L2: p 1.42%, u_l 2.88%, α 0.52%); grid-convergent (N=150→600: α 0.81%→0.52%); matches
      nozzlex reference (p 1.4%, u_l 2.9%, α 0.5%, T_l 0.01%). Residual diff ≈ our virtual mass (`k_vm`),
      which nozzlex/Wallis omits.
- [x] **Entropy fix** ported to strict sonic-crossing activation (single-phase parity) in `two_fluid/flux.py`
      (`s=eigvals/lam_safe`, gated on `λ_L<0<λ_R`). Not the checkerboard cause; correct + needed at throat.
- [x] **Matrices VERIFIED correct** — eig(G_h)==closed-form eigvals (2e-13), R_h/L_h eigenvectors
      (1e-10), K==dF/dU_h AD (3e-13). Biorthogonality `D~1e15` for acoustic pair is benign degeneracy
      (`u_g,u_l` doubled); not in the full-slip flux. **Don't re-suspect the matrices.**
- [x] **IC** seeded from a nozzlex pickle (`nozzlex_ic.py::ic_from_nozzlex_pickle`, auto phase-order from
      densities: Elliot pickle idx1=liquid). `ic.type: nozzlex`, `ic.pickle:` in YAML.
- [x] Driver now writes FULL two-phase `solution.csv`; `nc_scale` diagnostic knob added (keep 1.0).
- [x] **Self-contained "simple" cold-start IC (2026-07-09)** — `ic.type: simple` in the driver, so we
      don't NEED the nozzlex pickle (use nozzlex only when speed matters). Robust recipe: mild linear
      pressure ramp `P0→p_out`, uniform `α_g`, SMALL slip-consistent velocities (`u_g=S·u_l`, `u_seed_l=5`
      m/s ⇒ `du≈0`), stagnation enthalpies `H_g0/H_l0` minus dynamic head. Converges in more iterations
      but is stable. **The previous per-phase-continuity "simple" attempt gave step-1 NaNs** (finite IC but
      violent): gas continuity (ρ_g drops ~10×) and liquid continuity (ρ_l≈const) → unrelated `u_g→585`,
      `u_l→264` at the throat, `du` −2→+320 m/s (imposed slip destroyed) and mixture-SUPERSONIC (Wood
      a_mix ~tens m/s at α≈0.39) with a velocity-cap kink → first Newton step blew up. Lesson: a cold IC
      must be SUBSONIC + slip-consistent (never independent per-phase continuity); let the transient build
      the transonic flow. `ic.type: stagnation` (uniform quiescent reservoir fill) is the ultra-robust
      fallback.

- [ ] Add friction in the sources, creating a new source...

### Pending (traced)
- [ ] **Verify source terms make physical sense** (`sources.py`): `D_VE_algebric_two_component` drag/heat
      signs + magnitude (a probe showed `Fv_g ~ 3e8` — sanity-check vs tiny-bubble interfacial area),
      `two_fluid_closure` `mass_transfer=False` branch (htc, Qvi/Qli), `D_VE_area` signs. Cross-check vs
      nozzlex + Städtke. **Wallis HAS viscous drag → keep `SRC_SCALE=1`; the diff vs nozzlex is virtual mass.**
- [x] **Gate the filter to the low-velocity region — DONE (2026-07-09).** Gates on the PHASE VELOCITY,
      NOT a Mach: the α/void checkerboard is a MATERIAL-wave mode (rides λ=u_g,u_l), and in two-phase
      `a_mix` is the α-dependent Wood speed → a bad denominator (would move the gate with α, not velocity).
      Sensor `u_mat=min(|u_g|,|u_l|)` [m/s] (min ⇒ stays on if EITHER phase slows, robust under slip);
      weight `w=½(1+tanh((u0−u_mat)/Δu))` multiplies the JST face scale `r_face` in
      `two_fluid/solver.py::_residual`. `u0=eps_void_ucut` [m/s], `Δu=eps_void_uwidth` (null→u0/3).
      Velocities read from the SAME ±2-stencil cells ⇒ band not widened (`_sw` stays 2, colored Jacobian
      exact). `eps_void_ucut: null` ⇒ ungated/uniform (original). This case: `u_mat` ~3 (inlet) → ~93
      (throat) m/s; YAML `u0=15, Δu=5` (fully off by x~0.1). Wired config_loader→driver→YAML.
      **NB a per-domain velocity reference (`u/max(u)`) was REJECTED** — a global reduction couples all
      cells → breaks the colored-Jacobian band (`jac_diff` would fail); the sensor MUST be per-cell/local.
      **`lowmach_diss` was ALREADY Mach-gated** (`z_ac=clip(Ma,lowmach_diss,1)` FLOOR on the acoustic pair
      only; `z_ac→1` at `Ma≥1`) — no change. *To verify: rerun, confirm throat profile matches an ungated
      run while the inlet checkerboard stays damped.*  (Still open: optional interfacial-pressure reg...)

- [ ] **Investigate & validate the BCs against the reference** (user has a reference giving the actual
      result). Confirm the `stagnation` inlet + `characteristic` outlet reproduce the reference; compare
      the imposed-vs-extrapolated characteristic count against single-phase Städtke
      `fvm.py::_bc_inlet_real_jax`/`_bc_outlet_real_jax`. (Merges with the Städtke-BC comparison below.)

- [ ] **Water-only low-speed convergence test (ghost = gas).** IMPORTANT milestone: single-phase water at
      low speed with a ghost gas phase (α_g≈1e-6) must converge to steady state. This is the clean limit
      that isolates the liquid-dominant path (no slip physics) — get SSR→tol here before trusting the
      two-component slip case. New YAML (water liquid dominant, gas ghost), low subsonic, verify mass-flux
      constant + SSR plateau.
- [ ] **BC verification (started 2026-07-09) — OUTLET first, then INLET.**
      - [x] **Plot BC markers** — `compare_nozzlex_timemarching.py` now draws `p_out` (crimson ✕ at outlet
            + dashed line) and `p0` reservoir (navy dotted) from each run's YAML, so you can SEE if the
            outlet reached the back-pressure and how far the inlet expanded from `p0`.
      - [x] **OUTLET BUG FIXED (2026-07-09).** `characteristic` outlet was ignoring `p_out` (acting
            transmissive) while `pressure` correctly followed it. ROOT CAUSE: the regime switch
            `w=½(1+tanh((1−M_out)/0.01))` computed `M_out` from the **Wood speed**
            `a=√(1/(rmix·Σαᵢ/(ρᵢaᵢ²)))` + **mass-weighted** `u_mix`, INCONSISTENT with the scheme's actual
            characteristics (Städtke `a` incl. virtual mass `k` + slip; hat-weighted `u_mix`;
            `eigvals=[u_g,u_l,u_mix±a,…]`, matrices.py:481-497). Wood≠Städtke ⇒ `M_out`≥1 (`w≈0`→transmissive)
            though the real outgoing acoustic `u_mix−a`=`eigvals[3]`<0 (subsonic). FIX: `M_out` now from
            `eigenstructure()` (a=½(ev₂−ev₃), u_mix=½(ev₂+ev₃)); `msw` 0.01→0.05. Wood impedance KEPT for the
            LODI velocity co-move (shared with working `pressure`). Now `characteristic`→`pressure` when
            subsonic, extrapolates only when genuinely supersonic. *Verify: rerun with `outlet:
            characteristic`, confirm last cell now sits on the `p_out` ✕ marker (subsonic case).*
      - [~] **INLET was the dt-COLLAPSE BLOCKER (2026-07-09) — fixed, verify on rerun.** Symptom: pressure
            explosion (→50 bar, hitting `p_hi`) + REVERSE FLOW localized in the first ~5 cells; `dp`-gated,
            `dt`-independent, persists across pressure ratios. Cause in `stagnation` `_bc_inlet`: (1)
            `u_mix = 2·u1−u2` LINEAR extrapolation = high-pass filter → amplifies inlet noise / injects
            interior reverse flow into the ghost → sign-flipped velocity split → garbage `h_g`; (2) the
            gas-isentrope Newton `p_s += ρT(s−s0)` is unrelaxed with `ρT~6000` → ~10 bar/iter steps → with
            the bad `h_g` it overshoots to the `p` clip → the spike → feeds back through the extrapolation →
            collapse. FIX: (a) ZERO-ORDER `u_mix = u_mix(cell1)` (1st-order, stable, = Städtke's first-cell
            velocity); (b) isentrope Newton UNDER-RELAXED 0.5× and CAPPED to ≤10%·`p_s`/iter. *Rerun and
            confirm the inlet spike + reverse flow are gone and `dt` no longer collapses.* Instant
            alternative to A/B against: `bc.inlet: stagnation_ramped_soft` (minmod-limited extrap + relax).
      - [ ] **INLET (post-fix verify & document).** Our `stagnation` inlet (per-phase
            `H_g0/H_l0`, gas-only `s_g0`, imposed quality+slip, gas-isentrope `p`, extrapolate the ONE
            outgoing char `u_mix`). Compare vs the Städtke homogeneous-EQUILIBRIUM reservoir BC
            (eq 9.14-9.16): `h_in=h0−u_in²/2`, `s_in=s0`, `p_in=p(h_in,s_in)` using MIXTURE `u_in,s_in` at
            cell 1. Add a faithful Städtke-style HEM reservoir inlet as a validation reference (should
            agree with `stagnation` in the no-slip/thermal-eq limit). Cross-check vs single-phase Städtke
            `fvm.py::_bc_inlet_real_jax`/`_bc_outlet_real_jax`. Confirm inlet expands correctly below `p0`
            (navy dotted marker) and the imposed quality/slip/stagnation-T are recovered at cell 1.
      - Test cases: (1) SUBSONIC, sweep `p_out` — verify last cell→`p_out`, inlet→reservoir isentrope,
        ṁ spatially constant (exercises both BCs; `pressure` baseline then fixed `characteristic`).
        (2) CHOKED/transonic — throat M=1, supersonic outlet must IGNORE `p_out` & extrapolate (exercises
        the `characteristic` auto-switch). NB choked nozzle REQUIRES `inlet: stagnation`.
- [ ] One fresh run with the new full CSV to also get `u_g/T_g` (slip) L2 vs nozzlex (old runs were
      dominant-only).
- [ ] (Later, speed) perfect-gas(N₂)+stiffened-gas(water) analytic EOS for a fast IC solve.

## Notes

- Inlet BC caveat (from YAML): confirm the inlet should impose *static* saturated-vapour conditions,
  not total — currently `total_state` imposes stagnation enthalpies + `α_g`, extrapolates `p, u`.
- Choked nozzle ⇒ use `inlet: stagnation` (anchors `p` via reservoir isentrope); `transmissive` inlet +
  extrapolating supersonic outlet leaves the pressure *level* unanchored → it drifts.



- [ ] Need to not stop after dt floored, it might be a transition... more robustness. Just let it run for some iterations if possible
