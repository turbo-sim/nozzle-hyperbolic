# Plan: shock capture in the diverging section

## Root cause summary

The FVS Newton solver fails to converge to the shocked steady state because:

1. **FVS is fully upwinded for M > 1**: F⁻ = 0 when supersonic → ghost cell state
   has zero influence on the flux → p_out cannot reach supersonic interior cells.

2. **Hard BC switch disconnects p_out**: `where(M < 1, p_out, p_prev)` — once the
   outlet cell goes supersonic, p_out is permanently lost.  The `jnp.where` also
   gives zero JAX gradient at M=1 → Newton blind to the BC transition.

3. **s2p hard sign step in FVS flux**: same zero-gradient issue for the sonic kink
   inside the flux — Newton Jacobian wrong near M=1 cells.

Both 2 and 3 combine: even if the shock forms during the transient, Newton cannot
reduce the residual below ~5e-3 because the Jacobian misses the shock sensitivity.

---

## Strategy A — Shock-placement initialisation  *(recommended first try)*

Instead of discovering the shock from a subsonic start, initialise with the shock
**already placed** at an approximate location x_shock.

### Construction
1. For cells 0 → i_shock-1: use subsonic isentropic solution (existing `stadtke_puh_initialization_subsonic`)
   but only up to the throat, then supersonic isentropic from throat to i_shock.
2. At i_shock: apply Rankine-Hugoniot jump relations for a normal shock
   - M1 = M_supersonic(x_shock)  →  M2, p2, T2, h2 from RH
3. For cells i_shock+1 → N-1: isentropic subsonic expansion from (p2, h2, s2)
   back to outlet, matching p_out approximately.

### Why this helps
Newton only needs to **refine** the shock position — not create it from scratch
during a transient.  The post-shock region is subsonic from the start, so p_out
is correctly imposed throughout the Newton iterations.

### Parameters
- x_shock: initial guess — bisect between throat and outlet, e.g. 75% of the way.
  Alternatively compute the exact location analytically for perfect gas first.
- New YAML key: `initialization: shocked` with optional `x_shock_fraction: 0.75`

---

## Strategy B — Always impose p_out at the outlet

Change the outlet BC to **always** set `p_ghost = p_out`, regardless of Mach number.

```python
p_ghost = pout_j   # remove the Mach switch entirely
```

### Why this is safe with FVS
When M_interior > 1: FVS gives F⁻(U_ghost) = 0, so the ghost pressure does not
affect the computed flux.  The BC is physically "over-specified" but numerically
harmless for the flux.

### Why it helps Newton
Newton's Jacobian ∂R/∂U always sees p_ghost = p_out as a fixed value.
`∂p_ghost/∂U_interior = 0` (constant), which is correct — it stops Newton from
chasing spurious p_ghost sensitivity through the Mach switch.

### Caveat
For a purely supersonic nozzle (no shock, p_out below supersonic design pressure)
this would impose a wrong ghost pressure.  Could add a flag
`always_impose_outlet_pressure: true` only for the shocked-nozzle case.

---

## Strategy C — Smooth s2p in FVS flux  *(fixes Newton Jacobian at sonic cells)*

Replace the hard sign step in `_jax_F_fvs_stadtke`:

```python
# current — zero JAX gradient at lam2 = 0:
s2p = jnp.where(lam2 >= 0, 1.0, 0.0)

# proposed — smooth tanh, correct gradient:
s2p = 0.5 * (1.0 + jnp.tanh(lam2 / delta_global))
```

This eliminates the sonic glitch and gives Newton a correct Jacobian through the
M=1 transition inside the flux.  Combine with Strategy A or B.

### Note
`delta_global = delta_factor * (|u| + a)` — with `delta_factor = 1e-6` the
transition is very sharp (width ~ 7e-4 m/s in eigenvalue space), essentially
identical to the hard step away from M=1 but differentiable.

---

## Strategy D — Smooth outlet BC tanh  *(current state, partial fix)*

Already implemented (sigma=0.004, blend from M≈0.99 to M≈1.01).
Only helps during the brief moment the outlet cell is near M=1.
Necessary but not sufficient on its own.

---

## Recommended implementation order

1. **Strategy A** (shock-placement init): highest impact, no change to flux or BC.
   Try with existing smooth-tanh BC (Strategy D already in place).
   Expected outcome: Newton converges from a good initial guess without needing
   to discover the shock dynamically.

2. **Strategy C** (smooth s2p): add on top of A.  Eliminates the sonic glitch
   that causes SSR to stall at 5e-3 even when the shock is placed correctly.

3. **Strategy B** (always impose p_out): fallback if A+C still fails.  Simple
   one-line change, test independently.

---

## Analytical shock position (perfect gas, γ=1.4, A/A*=2)

For the validation case (p0=2 bar, p_out=0.5845 bar):

- p_out / p0 = 0.2923
- Fully supersonic M≈2.2: p/p0 ≈ 0.0942 (too low)
- Subsonic M≈0.3: p/p0 ≈ 0.94 (too high)
- Shock at A/A* where RH + downstream isentropic gives p_out/p0 = 0.2923
  → analytical x_shock ≈ 0.176 m  (≈ 88% of nozzle length)

Use this to set `x_shock_fraction ≈ 0.88` for the initialisation.
