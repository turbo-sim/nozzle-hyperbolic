# nozzle-hyperbolic — TODO

## Performance

- [x] **Fix JAX performance bottlenecks in `fvs_newton` solver** (234s → ~24s expected)

  Two root causes identified via `cProfile` on `dev_scripts/main.py`:

  1. ~~**`jax.debug.callback` in `jaxprop/_interp_h_p`** (~160s, 447k device↔host syncs)~~
     - Mitigated with `_DEBUG_BOUNDS = False` guard in `bicubic_interpolation.py` (52s after fix)
     - Proper fix still pending: propagate `any_oob = jnp.any(out_of_range)` as extra return from `get_state`, accumulate in `step`, check once in Python

  2. ~~**`step` and `compute_dt` not `@jax.jit`'d in `fvm.py`** (~17s, 107 XLA recompilations)~~
     - Fixed: decorators restored in `make_jax_stadtke_step` (expected ~24s after)

## Performance (next steps)

- [ ] **Reduce `newton_iter` from 20 → 5** in `make_jax_stadtke_step` (`fvm.py:1178`)
  - Newton converges quadratically; 3–5 iterations reach machine precision for a well-conditioned nozzle
  - `fori_loop` always runs the full count — cutting 20 → 5 gives ~4× speedup in the Newton inner loop
  - Make it configurable from the YAML (`numerical_settings.newton_iter`) so it can be tuned per case

- [ ] **Precompute `V_n = _V_from_U(U_n)` outside the Newton loop** in `step` (`fvm.py:1336`)
  - `_residual` calls `V_old = _V_from_U(U_n)` on the fixed time-n state every Newton iteration
  - This `get_state` is also traced into every JVP pass of `jax.jacobian`, bloating the compiled Jacobian kernel
  - Fix: compute `V_n` once in `step` and pass it as an extra argument to `_residual`/`_jac_fn` (with `argnums=0` unchanged, JAX treats `V_n` as a constant in the Jacobian)

## Solver

- [ ] **Banded Jacobian solver** — replace dense `jnp.linalg.solve` with a banded solver to allow `Nx > 500` without memory blowup. The Jacobian of `_residual` is block-tridiagonal (3×3 blocks).
  - Currently `jax.jacobian` computes `3*(N-2)` JVP passes (234 for N=80, 894 for N=300) — scales O(N²)
  - With graph-coloring (3 stencil neighbors → 9 JVP passes regardless of N) and a Thomas-algorithm solve, cost becomes O(N)

- [ ] **Perfect-gas initialization via `nozzleh`** — replace the bisection-based `stadtke_puh_initialization` with an area-Mach analytical init using `nozzleh`'s `exact_isentropic` pattern. Faster and no table needed.
