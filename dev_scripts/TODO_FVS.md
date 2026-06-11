# TODO — FVS Newton Solver

## Bugs / Correctness

- [ ] **Shock-nozzle capture**: add CFL warm-up ramp (`CFL_init ~0.5`, ramp to full CFL once `M_throat > 0.99`) so pseudo-time marching develops the shock naturally instead of jumping to the supersonic branch
- [ ] **Mass conservation diagnostic**: slice to interior cells only `[1:-1]` for `density`, `velocity`, `A_nodes` in `main.py` mass flux check — ghost cells have inconsistent density from `EOS(h_interior, p_out)`
- [ ] **Sonic glitch mesh-dependence**: replace hard `jnp.where` sign step `s2p` with smooth `tanh(lam2 / delta_global)` so JAX computes correct Newton Jacobian gradient through the sonic transition (glitch grows with `Nx` because more cells sit near M=1)

## Validation

- [ ] **Supersonic nozzle (no shock)**: `perfect_gas=true`, `p_out=18450 Pa`, compare to `nozzle_exact2` analytical solution
- [ ] **Nozzle with normal shock**: `perfect_gas=true`, `p_out~5e4 Pa`, compare shock position and downstream state to analytical solution

## Numerical improvements

- [ ] **Non-uniform grid MUSCL**: divide differences by cell spacing before limiter, multiply by face-to-centre distances after (`muscl_van_leer_ph` and `_muscl_reconstruct`)

## Unsteady extensions

- [ ] **Unsteady option A — explicit Euler + FVS**: implement `make_jax_fvs_explicit_step` sharing `_muscl_ph` and `_jax_F_fvs_stadtke`; add `_U_from_V` EOS inversion; enable via `time_discretization: fvs_explicit` in YAML
- [ ] **Unsteady option B — dual time-stepping**: outer physical time loop (fixed small `dt_phys`, CFL~1) around existing Newton inner loop
- [ ] **Unsteady option B + BDF2**: replace backward Euler with `(3/2 V^{n+1} - 2V^n + 1/2 V^{n-1})/dt` for 2nd-order physical time accuracy

## Performance / architecture

- [ ] **Colored (banded) Jacobian** — *main per-iteration bottleneck, measured 825 ms/iter at N=100, newton_iter=5*.
      `jax.jacobian(_residual)` pushes one JVP per DOF (3·(N−2) ≈ 294 columns), each re-evaluating
      the full residual EOS chain → O(N²) EOS work. The Jacobian is block-pentadiagonal
      (MUSCL stencil = ±2 cells), so 15 colored seeds (5-cell stride × 3 vars) recover the
      **exact same matrix** with 15 JVPs instead of 294 (~20× at N=100, ~200× at N=1000).
      Keep the dense `jnp.linalg.solve` (294×294 LU ≈ 2 ms, negligible).
      **CAVEAT — sparsity assumption**: coloring is exact ONLY while the residual stays local
      (every residual row depends on cells within ±2). Holds today: MUSCL ±2 stencil; inlet ghost
      = f(cells 1,2); outlet ghost = f(cells N−2,N−3). If anything non-local is ever added
      (global integral constraint, wider stencil, BC depending on far cells, recursive filters),
      the 5-cell stride is no longer valid → silent wrong Jacobian → slow/failed Newton.
      Update the stride (or revert to full `jax.jacobian`) in that case.

- [ ] **Full JIT outer loop**: replace Python `while/for/if/break/print` with `jax.lax.while_loop`, `jnp.where`, `jax.debug.print`, `jnp.roll` carry
