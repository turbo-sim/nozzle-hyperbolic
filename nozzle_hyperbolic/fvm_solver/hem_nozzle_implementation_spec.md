# Implementation Spec: HEM Quasi-1D Nozzle Solver (Städtke FVS Method)

## For Claude Code

This document specifies how to implement a **homogeneous-equilibrium-model
(HEM) quasi-one-dimensional compressible flow solver with variable cross
section**, using Städtke's Flux Vector Splitting methodology, MUSCL
reconstruction with Van Leer limiter, and implicit Newton time-marching
with JAX automatic differentiation.

**Companion document**: `stadtke_single_phase_theory.md` contains the full
mathematical derivation. This document is the implementation plan. Read the
theory document first for all matrix definitions and derivations.

**Key design principle**: The solver is EOS-agnostic. The same code handles
pure single-phase gas (ideal or real) and HEM two-phase mixtures — only the
EOS calls differ. HEM is "just an EOS" where, in the two-phase region,
JAXProp/CoolProp returns mixture density and the (small) HEM mixture sound
speed via the lever rule on the saturation curve.

---

## 0. Technology Stack

- **Language**: Python 3.11+
- **Core numerics**: JAX (jax, jaxlib) — for autodiff Jacobians and JIT
- **EOS**: JAXProp (CoolProp wrapper) — for `(p, h) -> rho, T, a, partials`
- **Linear algebra**: jax.numpy.linalg for dense blocks; scipy.sparse or a
  banded solver for the global system
- **Testing**: pytest

Install: `pip install jax jaxlib jaxprop pytest --break-system-packages`

Use double precision throughout: set `jax.config.update("jax_enable_x64", True)`
at module load. Single precision breaks the sonic entropy fix and Newton
convergence.

---

## 1. State Variables and Conventions

### 1.1 Primitive variables (the working variables)

$$\mathbf U = (p, u, h)^T$$

- `p` — pressure [Pa]
- `u` — velocity [m/s]
- `h` — specific enthalpy [J/kg]

Store as a JAX array of shape `(N, 3)` for N cells, or flattened `(3N,)`
for the Newton solve.

### 1.2 Conservative variables (what is updated/conserved)

$$\mathbf V = (\rho, \rho u, \rho E)^T = (\rho, \rho u, \rho H - p)^T$$

where `H = h + u^2/2` is specific total enthalpy and `rho E = rho*H - p`.

### 1.3 Physical flux

$$\mathbf F = (\rho u, \rho u^2 + p, u\rho H)^T$$

(The third component uses `u*rho*H`, since `rho E + p = rho H`.)

### 1.4 Why enthalpy form

- JAXProp natural lookup is `(p, h)`.
- Coefficient matrix has the clean `a^2` coupling (book eq. 2.26).
- Correct shock capturing because conservative update uses total energy.
- See theory doc Section 1.2.

---

## 2. EOS Module (implement first)

Create `eos.py` wrapping JAXProp. The interface must provide, given `(p, h)`:

```
state_ph(p, h) -> (rho, T, a, rho_p, rho_h)
```

where:
- `rho`   = density [kg/m^3]
- `T`     = temperature [K]
- `a`     = sound speed [m/s]  (HEM mixture sound speed in two-phase region)
- `rho_p` = (drho/dp)_h   [partial at constant h]
- `rho_h` = (drho/dh)_p   [partial at constant p]

### 2.1 Requirements

- Must work transparently in single-phase AND two-phase (HEM) regions.
  CoolProp returns mixture properties in the two-phase dome automatically.
- The partials `rho_p`, `rho_h` are obtained from CoolProp first-derivative
  calls (`d(D)/d(P)|H`, `d(D)/d(H)|P`) or via JAX autodiff if JAXProp
  supports differentiable property evaluation.
- Vectorize over cells (accept arrays of `p`, `h`).

### 2.2 Sanity test (required)

The thermodynamic identity must hold:

$$\frac{1}{a^2} = \rho_p + \frac{\rho_h}{\rho}$$

```
test: |1/a^2 - (rho_p + rho_h/rho)| / (1/a^2) < 1e-6
```

Test at several states: superheated gas, compressed liquid, and inside
the two-phase dome. If it fails in the dome, the HEM partials are
inconsistent — investigate before proceeding.

### 2.3 Fallback EOS for early development

Implement an `IdealGasEOS` class with the same interface, using closed forms:
- `rho = gamma*p / ((gamma-1)*h)`
- `a^2 = gamma*p/rho`
- `rho_p = rho/p`, `rho_h = -rho/h`

Use ideal gas to validate the solver on Sod/Toro shock tubes before
switching to the real HEM EOS. Default `gamma=1.4`, `R=287`.

---

## 3. Geometry Module

Create `geometry.py` to hold the nozzle geometry. Given a grid of cell
centers `x_i` and faces `x_{i+1/2}`:

```
A_cell[i]     = A(x_i)            # centroid area, shape (N,)
A_face[j]     = A(x_{j-1/2})      # face areas, shape (N+1,)
dx[i]         = x_{i+1/2} - x_{i-1/2}   # cell widths, shape (N,)
dA_dx[i]      = dA/dx at x_i      # area derivative at centroid, shape (N,)
```

### 3.1 Requirements

- The area profile `A(x)` is supplied as an analytical function or a
  smooth spline. NEVER compute `dA_dx` by finite differencing `A_cell`.
  Use the analytical derivative or the spline derivative.
- Cell volume: `V_cell[i] = A_cell[i] * dx[i]`.
- For boundary cells, extend the area profile by one ghost cell on each
  side (mirror or constant) so `dA_dx` is centered everywhere.
- Area must be smooth (C^1 at least). Kinks in `A(x)` cause spurious
  entropy generation.

### 3.2 Test geometries

- **Constant area** (`A = const`, `dA_dx = 0`): for shock-tube validation.
- **Converging-diverging (Laval)**: e.g. `A(x) = 1 + k*(x - x_throat)^2`.

---

## 4. Coefficient and Jacobian Matrices

Create `matrices.py`. Implement closed-form builders (from theory doc
Sections 2–4). These are used to construct the FVS projection matrices.

### 4.1 Coefficient matrix G (theory Section 2.4)

```
G(p, u, h, eos) -> 3x3 matrix
  [[u,        rho*a^2,  0  ],
   [1/rho,    u,        0  ],
   [0,        a^2,      u  ]]
```

### 4.2 Flux Jacobian K = dF/dU (theory Section 2.3)

```
K(p, u, h, eos) -> 3x3 matrix
  with H = h + u^2/2:
  [[u*rho_p,         rho,            u*rho_h        ],
   [u^2*rho_p + 1,   2*rho*u,        u^2*rho_h      ],
   [u*rho_p*H,       rho*(H + u^2),  u*(rho_h*H+rho)]]
```

### 4.3 Conservative Jacobian J = dV/dU (theory Section 2.2)

```
J(p, u, h, eos) -> 3x3 matrix
  [[rho_p,          0,       rho_h         ],
   [u*rho_p,        rho,     u*rho_h       ],
   [rho_p*H - 1,    rho*u,   rho_h*H + rho ]]
```

### 4.4 Split matrices G_1, G_2, G_3 (theory Section 4.2)

```
split_matrices(p, u, h, eos) -> (G1, G2, G3)
```

G1 = (u+a) * [[1/2,        rho*a/2,  0],
              [1/(2*rho*a), 1/2,     0],
              [1/(2*rho),   a/2,     0]]

G2 = (u-a) * [[1/2,         -rho*a/2, 0],
              [-1/(2*rho*a), 1/2,     0],
              [1/(2*rho),    -a/2,    0]]

G3 = u *     [[0,        0, 0],
              [0,        0, 0],
              [-1/rho,   0, 1]]

### 4.5 Required unit tests

1. `sum(G1, G2, G3) == G`  (partition identity, atol 1e-12)
2. `J @ G == K`            (consistency, atol 1e-10)
3. Eigenvalues of G are `{u+a, u-a, u}` (atol 1e-10)

Note: These closed forms can alternatively be replaced entirely by JAX
autodiff (compute J, K by differentiating V(U), F(U)). Keep the closed
forms for unit-testing the autodiff versions against each other.

---

## 5. FVS Interface Flux

Create `flux.py`. This is the core numerical scheme (theory Section 5).

### 5.1 Projection matrices

```
projection_matrices(U_avg, eos, delta_factor=0.1) -> (R_plus, R_minus)
```

Steps:
1. Compute `G1, G2, G3` and `K` and eigenvalues at `U_avg`.
2. For each wave k: solve `K @ X = G_k` -> `R_k = K @ X`  (= K G_k K^{-1}).
   Use `jnp.linalg.solve(K, G_k)`, NOT explicit inverse.
3. Sonic entropy fix: `lam_safe = sign(lam) * max(|lam|, delta)` with
   `delta = delta_factor * (|u| + a)`.
4. `R_tilde_k = R_k / lam_safe`.
5. `R_plus = sum of R_tilde_k for lam_k >= 0`
   `R_minus = sum of R_tilde_k for lam_k < 0`
   (using TRUE eigenvalue sign, not safe one)

### 5.2 Interface flux

```
interface_flux(U_L, U_R, eos, delta_factor=0.1) -> F_hat (3-vector)
```

Steps:
1. `U_avg = 0.5 * (U_L + U_R)`
2. `R_plus, R_minus = projection_matrices(U_avg, eos, delta_factor)`
3. `F_L = physical_flux(U_L, eos)`  (from ACTUAL left state)
4. `F_R = physical_flux(U_R, eos)`  (from ACTUAL right state)
5. `F_hat = R_plus @ F_L + R_minus @ F_R`

CRITICAL: projection matrices use the AVERAGED state; physical fluxes use
the ACTUAL L/R states. Do not confuse these. (See theory Section 5.5.)

### 5.3 Physical flux

```
physical_flux(U, eos) -> 3-vector
  p, u, h = U
  rho = eos density at (p, h)
  H = h + 0.5*u^2
  return [rho*u, rho*u^2 + p, u*rho*H]
```

### 5.4 Required unit tests

1. Partition of unity: `R_plus + R_minus == I` (atol 1e-10).
2. Uniform flow: if `U_L == U_R`, then `F_hat == physical_flux(U_L)`.
3. Supersonic flow (`u > a > 0`): `F_hat == F_L` (R_plus = I).

---

## 6. MUSCL Reconstruction

Create `reconstruction.py` (theory Section 6).

### 6.1 Van Leer limiter

```
van_leer(sigma_L, sigma_R, eps=1e-12) -> sigma
  s = sigma_L * sigma_R
  return where(s > 0, 2*s/(sigma_L + sigma_R + eps), 0.0)
```

Applied componentwise on `(p, u, h)`.

### 6.2 Slope computation

```
limited_slopes(U_cells, dx) -> sigma  # shape (N, 3)
```

For interior cells:
- `sigma_L = (U[i] - U[i-1]) / dx[i]`
- `sigma_R = (U[i+1] - U[i]) / dx[i]`
- `sigma[i] = van_leer(sigma_L, sigma_R)`

Boundary cells: set `sigma = 0` (first-order at boundary).

### 6.3 Face reconstruction

```
reconstruct_faces(U_cells, sigma, dx) -> (U_L, U_R)  # each (N-1, 3)
  U_L = U[:-1] + sigma[:-1] * dx[:-1]/2
  U_R = U[1:]  - sigma[1:]  * dx[1:]/2
```

### 6.4 Positivity safeguard

After reconstruction, where `p_L <= 0` or `p_R <= 0` (or any state outside
the EOS valid region), fall back to first-order at that face:
`U_L = U[i]`, `U_R = U[i+1]`.

For HEM, also guard against enthalpy values outside the EOS table range.

---

## 7. Source Term (Area Variation)

Create the source as part of `residual.py` (theory Sections 1.1, 7.3).

With area variation only (no friction, no heat, no body force), the
conservative-form source is **momentum-only**:

```
C_area(U, geometry) -> array (N, 3)
  p = U[:, 0]
  C = zeros((N, 3))
  C[:, 1] = p * dA_dx / A_cell      # lateral wall pressure force
  return C
```

CRITICAL: In conservative form, the area source appears ONLY in the
momentum equation. The mass and energy equations get their area effects
through the area-weighting of the fluxes (A_face factors). Do NOT add
area sources to mass or energy equations — that would double-count.
(See theory Section 1.1 and 7.3.)

---

## 8. Residual Assembly

Create `residual.py` (theory Section 7).

```
residual(U_flat, U_n_flat, dt, geometry, eos, delta_factor=0.1) -> R_flat
```

This is the central function. JAX will autodiff it for the Jacobian.

Steps:
1. Reshape `U_flat` to `(N, 3)`.
2. Compute `V_new = V_from_U(U)` and `V_old = V_from_U(U_n)`.
3. MUSCL reconstruct face states `U_L, U_R` (Section 6).
4. Positivity safeguard.
5. Compute interior interface fluxes `F_hat[1:N]` via `interface_flux`
   (vectorize with `jax.vmap`).
6. Compute boundary fluxes `F_hat[0]`, `F_hat[N]` (Section 9).
7. Compute area source `C_area` (Section 7).
8. Assemble residual per cell:

```
V_cell = A_cell * dx
time_term   = V_cell[:, None] * (V_new - V_old)
flux_div    = A_face[1:, None]*F_hat[1:] - A_face[:-1, None]*F_hat[:-1]
source_term = V_cell[:, None] * C_area
R = time_term + dt * flux_div - dt * source_term
return R.flatten()
```

### 8.1 Important details

- All cell-level quantities scaled by cell volume `V_cell = A_cell*dx`.
- Fluxes scaled by face areas `A_face`.
- The time term uses conservative variables `V`, NOT primitive.
- Make the whole function `@jax.jit`-able (no Python branches on traced
  values; use `jnp.where` for the positivity fallback).

---

## 9. Boundary Conditions

Create `boundary.py` (theory Section 8).

Implement for the nozzle case:

### 9.1 Reservoir inlet (left)

Given stagnation pressure `p0` and stagnation enthalpy `h0`:
- Extrapolate velocity `u` from the first interior cell.
- Static enthalpy: `h = h0 - 0.5*u^2`.
- Static pressure: from isentropic relation between stagnation and static
  state (use EOS: find `p` such that entropy matches the reservoir entropy
  at the computed `h`). For ideal gas, use the isentropic `p/p0` formula.
- Build ghost state, compute boundary flux.

### 9.2 Pressure outlet (right)

Given back-pressure `p_back`:
- If subsonic outflow (`u < a` at last cell): set `p = p_back` in ghost,
  zero-gradient extrapolate `u`, `h`.
- If supersonic outflow (`u >= a`): zero-gradient extrapolate all
  primitives (pressure cannot propagate upstream).
- Compute boundary flux from ghost state.

### 9.3 Wall (for shock-tube closed end)

- `u_ghost = -u_interior`, `p` and `h` unchanged.
- Boundary flux: `[0, p, 0]`.

---

## 10. Implicit Newton Time-Marching

Create `solver.py` (theory Section 7).

### 10.1 Newton step

```
newton_step(U_flat, U_n_flat, dt, geometry, eos) -> (U_new_flat, increment_norm)
  R = residual(U_flat, U_n_flat, dt, geometry, eos)
  Jac = jax.jacfwd(residual, argnums=0)(U_flat, U_n_flat, dt, geometry, eos)
  dU = jnp.linalg.solve(Jac, -R)
  return U_flat + dU, max(|dU| / (|U_flat| + eps))
```

- Use `jax.jacfwd` (forward-mode) for the Jacobian — efficient for this
  problem size where input and output dims are equal.
- For large N, replace dense `solve` with a banded/sparse solver. The
  Jacobian is block-pentadiagonal (bandwidth 5 blocks of 3x3). For N up to
  ~2000, dense solve is acceptable for first implementation.

### 10.2 One time step

```
advance_one_step(U_n_flat, dt, geometry, eos, max_iter=5, tol=1e-8)
  U_k = U_n_flat
  for k in range(max_iter):
      U_k, inc = newton_step(U_k, U_n_flat, dt, geometry, eos)
      if inc < tol: break
  return U_k
```

Typically converges in 2-3 iterations.

### 10.3 Time-step controller

```
accept_step(U_old, U_new, dt, geometry, eos, C_max=1.0) -> (accepted, dt_next)
```

Three criteria (theory Section 7.8):
1. CFL: `dt * max((|u| + a)/dx) < C_max`. Start C_max=1, raise to 5-10
   once stable.
2. Mass-error: relative density change vs Taylor extrapolation < 1e-3.
3. Increment bounds: `|dp|/p < 0.2`, `|du|/(|u|+a) < 0.2`, `|dh|/h < 0.1`.

If any fails: halve dt, restore U_new = U_old, retry.
If all pass and Newton converged in <= 2 iters: dt_next = min(1.2*dt, dt_max).

### 10.4 Main loop

```
simulate(U_init, dt_init, geometry, eos, t_final)
  U, t, dt = U_init, 0, dt_init
  while t < t_final:
      U_new = advance_one_step(U, dt, geometry, eos)
      accepted, dt_next = accept_step(U, U_new, dt, geometry, eos)
      if accepted:
          U, t = U_new, t + dt
          dt = min(dt_next, t_final - t)
      else:
          dt = dt / 2
  return U
```

---

## 11. Implementation Order (do NOT skip steps)

Implement and validate in this strict sequence. Each stage must pass its
tests before moving on.

### Phase 1 — Foundations
1. `eos.py` with `IdealGasEOS`. Test the sound-speed identity.
2. `matrices.py`: G, J, K, split matrices. Test partition identity
   `sum(Gk)=G`, consistency `J@G=K`, eigenvalues.
3. `flux.py`: projection matrices and interface flux. Test partition of
   unity `R+ + R- = I`, uniform-flow consistency, supersonic upwinding.

### Phase 2 — Explicit solver (debugging aid)
4. Implement a SIMPLE first-order EXPLICIT solver (forward Euler, no MUSCL,
   no Newton). Constant area. This validates the spatial discretization
   independently of the implicit machinery.
5. Run Sod shock tube (ideal gas, constant area). Compare with exact
   Riemann solution. Must capture shock/contact/rarefaction qualitatively.

### Phase 3 — Second order + implicit
6. Add MUSCL reconstruction (`reconstruction.py`) with Van Leer. Re-run Sod;
   shocks should sharpen.
7. Add implicit Newton time-marching (`solver.py`). Re-run Sod; should match
   the explicit result but allow larger time steps.
8. Run Toro tests 1-5. Test 2 (123 problem) validates the sonic entropy fix
   — if negative density/pressure appears, increase delta_factor.

### Phase 4 — Area variation
9. Add geometry module and area source. Validate constant-area limit gives
   identical results to Phase 3.
10. Steady subsonic nozzle: compare with isentropic-flow analytical solution.
    Check `rho*u*A = const` along the nozzle to 1e-6.
11. Steady Laval nozzle (converging-diverging): choked subsonic-supersonic
    flow. Throat Mach = 1.0 +/- 0.01. Test multiple back-pressures
    (subsonic, choked, design, normal-shock-in-divergent).

### Phase 5 — HEM
12. Switch EOS from `IdealGasEOS` to real HEM EOS (JAXProp `Water`).
    Re-run the sound-speed identity test in the two-phase dome.
13. Steam expansion through nozzle: superheated inlet, expansion into the
    wet region. Verify the solution passes smoothly through the saturation
    curve and the HEM sound-speed dip is handled (CFL adjusts automatically).
14. Compare with HEM critical-flow data if available (e.g. Moby-Dick,
    Deich nozzle from the book Chapter 9).

---

## 12. Required Unit Tests (collect in `test_*.py`)

| Test | What it checks | Tolerance |
|---|---|---|
| `test_sound_speed_identity` | `1/a^2 = rho_p + rho_h/rho` | 1e-6 rel |
| `test_partition_G` | `G1+G2+G3 = G` | 1e-12 |
| `test_consistency_JGK` | `J@G = K` | 1e-10 |
| `test_eigenvalues` | eigvals(G) = {u+a, u-a, u} | 1e-10 |
| `test_partition_unity` | `R+ + R- = I` | 1e-10 |
| `test_uniform_flow` | `U_L=U_R => F_hat = F(U_L)` | 1e-12 |
| `test_supersonic` | `u>a => F_hat = F_L` | 1e-12 |
| `test_const_area_limit` | area code with A=const matches no-area | 1e-10 |
| `test_steady_mass_flux` | `rho*u*A = const` at steady state | 1e-6 |
| `test_jacobian_autodiff` | autodiff J vs closed-form J | 1e-7 |

---

## 13. Engineering Notes

- **Double precision mandatory** (`jax_enable_x64`).
- **JIT everything** for performance. First call compiles (slow), then fast.
- **Start with explicit + ideal gas** before implicit + HEM. Do not try to
  debug all components at once.
- **The sonic entropy fix matters more in HEM** than single-phase gas,
  because the HEM sound speed is small and sonic points are easy to reach.
  Tune `delta_factor` in [0.05, 0.2] on the Toro 123 test.
- **Area source sign trap**: when the nozzle converges (`dA_dx < 0`), the
  momentum source `p*dA_dx/A` is negative. Net effect with the area-weighted
  flux divergence should be flow acceleration. If velocity decreases through
  a converging section, the area handling has a sign error.
- **Verify the constant-area limit first** when adding area variation:
  set A=const and confirm bit-identical results to the no-area solver.
- **HEM EOS partials are discontinuous** at the saturation curve. CoolProp
  returns correct values on each side, but Newton may need an extra iteration
  in cells straddling the curve. This is expected, not a bug.
- **For nozzle steady state**: initialize with an analytical isentropic
  profile rather than uniform stagnant gas. Reduces transient settling by
  an order of magnitude.
- **The Jacobian is block-pentadiagonal** (MUSCL 5-cell stencil). For large
  N, exploit this with a sparse/banded solver instead of dense `jnp.linalg.solve`.

---

## 14. File Structure

```
hem_nozzle_solver/
├── eos.py              # EOS interface (IdealGasEOS + JAXProp wrapper)
├── geometry.py         # Nozzle geometry, area profiles, dA/dx
├── matrices.py         # G, J, K, split matrices (closed form)
├── flux.py             # FVS interface flux, projection matrices
├── reconstruction.py   # MUSCL + Van Leer limiter
├── boundary.py         # Inlet, outlet, wall BCs
├── residual.py         # Residual assembly + area source
├── solver.py           # Newton step, time-step control, main loop
├── tests/
│   ├── test_eos.py
│   ├── test_matrices.py
│   ├── test_flux.py
│   ├── test_shock_tube.py
│   └── test_nozzle.py
└── run_*.py            # Driver scripts for each benchmark
```

---

## 15. References

- Städtke, H., *Gasdynamic Aspects of Two-Phase Flow*, Wiley-VCH, 2006.
  Chapter 2 (single-phase), Chapter 5 (two-phase), Chapter 8 (numerics),
  Appendix B (matrices). Especially eqs. 2.26, 2.39, 8.28, 8.38-8.40.
- Toro, E.F., *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  Springer, 2009. Section 1.6.2 (area variation), Chapter 4 (Riemann
  problem test cases), Chapter 14 (MUSCL/limiters).
- Companion theory document: `stadtke_single_phase_theory.md`.

---

End of implementation spec.
