# Städtke Newton Solver — Practical Implementation Guide

This document explains **how the Städtke code actually runs**, step by step,
from the moment `fvm_euler_1d` is called with `time_method = 'fvs_newton'`
to the final converged solution.

---

## 1. Entry Point and Setup

When `time_method = 'fvs_newton'` the solver takes a separate code path in
`fvm_euler_1d`. Three things happen before any time-stepping:

### 1a. Initialization — `stadtke_puh_initialization`

The solver works in **primitive variables** `U = (p, u, h)` (pressure, velocity,
specific enthalpy). The initialization builds an isentropic field directly in
this space — no conservative variables involved.

**Algorithm:**

1. Get stagnation state `(h0, s0)` from `(p_stag, T_stag)` via CoolProp.
2. Sweep `h` from `h_lo` to `h0` (300 points); for each `h` find `p` such
   that `s(h, p) = s0` using 50 iterations of bisection on `HmassP_INPUTS`.
   This traces the **isentrope** from inlet to throat in (h, p) space.
3. Compute mass flux along the isentrope: `G(h) = ρ(h, p_iso) · √(2(h0 − h))`.
4. Throat is at `G_max`; compute `ṁ = G_max · A_throat`.
5. For each cell `i`: find `h_i` on the subsonic/supersonic branch such that
   `G(h_i) = ṁ / A_i`, then `p_i = p_iso(h_i)`, `u_i = √(2(h0 − h_i))`.

Result: `U0` of shape `(N, 3)` with `(p, u, h)` at each cell.
All values come directly from `HmassP_INPUTS` — self-consistent with the EOS.

### 1b. Step function — `make_jax_stadtke_step`

This factory is called once and returns a `@jax.jit`-compiled `step(U_n)` 
function. During the factory call, several things are **precomputed and frozen
into the closure**:

| Quantity | Shape | Description |
|---|---|---|
| `A_j` | (N,) | Cell-centre areas |
| `D_h_j` | (N,) | Hydraulic diameters |
| `A_face` | (N-1,) | Face areas (arithmetic mean) |
| `V_cell` | (N,) | Cell volumes = A·dx |
| `dAdx_A` | (N,) | `(1/A) dA/dx` at each cell, centred differences |
| `h_lo, h_hi, p_lo, p_hi` | scalars | EOS table bounds |
| `h0_j, s0_j, pout_j` | scalars | BCs baked into the closure |
| `_jac_fn` | function | `jax.jacobian(_residual)` — compiled once |

### 1c. Main loop

The same `jax.lax.while_loop` structure as the existing JAX path is used.
Each iteration calls `step(U_n)` and checks the SSR convergence metric.

---

## 2. One Time Step — `step(U_n)`

`step` is a `@jax.jit`-compiled function. Everything inside runs on XLA;
no Python is executed between time steps.

```
U_n (N,3) ──► [1. dt from CFL]
              [2. Newton loop × 3]
                  ├─ compute residual R
                  ├─ compute Jacobian J = dR/dU
                  └─ update U ← U + solve(J, -R)
              [3. rebuild BCs on ghost cells]
              [4. compute SSR]
──► U_new (N,3), dt, SSR
```

### Step 1 — Time step

```python
dt = CFL · dx / max_i(|u_i| + a_i)
```

A single EOS call `HmassP_INPUTS` gives `a` for all cells. `dt` scales with
the fastest wave anywhere in the domain.

### Step 2 — Newton loop (unrolled 3×)

The Newton unknowns are the **interior cells only**: `U_int` of shape
`(N-2, 3)`, flattened to `((N-2)·3,)`. Ghost cells 0 and N-1 are set by BCs
inside the residual and are not part of the Newton system.

Each Newton iteration:

```python
R = _residual(U_k, U_n_flat, dt)          # shape ((N-2)·3,)
J = jax.jacobian(_residual)(U_k, ...)     # shape ((N-2)·3, (N-2)·3)
U_k = U_k + jnp.linalg.solve(J, -R)
```

The Jacobian is **exact** (machine precision) — JAX traces through every
operation in `_residual` including EOS calls, MUSCL slopes, and matrix solves.

**Limitation**: `J` is a dense `(3(N-2)) × (3(N-2))` matrix. For `N = 50`
cells this is `144 × 144` (fine). For `N = 500` it is `1494 × 1494` (~18 MB,
still OK). Above ~1000 cells a banded solver would be needed.

---

## 3. The Residual — `_residual(U_int_flat, U_n_flat, dt)`

This is the function JAX differentiates. It assembles the fully-implicit
conservation-law residual for interior cells.

### 3a. Boundary conditions

Ghost cells are set from the interior before any fluxes are computed.

**Inlet (cell 0)**:
- Extrapolate velocity: `u_in = 2·U[1,1] − U[2,1]`
- Stagnation enthalpy: `h_s = clip(h0 − u_in²/2, h_lo, h_hi)`
- Find static pressure by bisecting on entropy: 12-iter bisection finds `p_s`
  such that `s(h_s, p_s) = s0` (isentropic from reservoir).
- Ghost cell: `U[0] = (p_s, u_in, h_s)`

**Outlet (cell N-1)**:
- Zero-gradient extrapolate `u` and `h` from cell N-2.
- Fix pressure: `p_ghost = p_out` if subsonic (`|u| < a`), else extrapolate.
- Ghost cell: `U[-1] = (p_ghost, u_out, h_out)`

### 3b. Conservative variables from primitives

```python
V = (ρ, ρu, ρH − p)    where H = h + u²/2
```

Two sets are computed:
- `V_new` from the current Newton iterate `U` (at time n+1)
- `V_old` from `U_n` (at time n, fixed)

Both use `HmassP_INPUTS` to get `ρ = ρ(h, p)`.

### 3c. MUSCL reconstruction in (p, u, h)

Van Leer slope limiter applied **component-wise on (p, u, h)**:

```
σ_i = VanLeer(U_i - U_{i-1}, U_{i+1} - U_i)   for i = 1 … N-2
σ_0 = σ_{N-1} = 0  (first-order at boundaries)

U_L_{i+1/2} = U_i + σ_i/2
U_R_{i+1/2} = U_{i+1} − σ_{i+1}/2
```

Pressure positivity is enforced: `p_L, p_R ≥ 1 Pa`.

### 3d. Städtke FVS interface flux — `_jax_F_fvs_stadtke`

For each of the N-1 faces, a **3×3 upwind matrix** is built from the local
wave structure. Inputs: `U_L, U_R` (each `(N-1, 3)` arrays).

**Step 1 — EOS at averaged state**

```python
U_avg = (U_L + U_R) / 2
ρ, a, G = HmassP_INPUTS(h_avg, p_avg)   → density, speed of sound, Grüneisen
ρ_p = (1 + G) / a²    # (∂ρ/∂p)_h
ρ_h = −ρG / a²        # (∂ρ/∂h)_p
```

**Step 2 — Eigenvalues**

```
λ1 = u + a  (right acoustic)
λ2 = u − a  (left acoustic)
λ3 = u      (convective)
```

**Step 3 — Split matrices G_k = λ_k (v_k ⊗ l_k)**

Each `G_k` is a rank-1 matrix built from the right and left eigenvectors of
the coefficient matrix G (theory §4.2). All three are `(N-1, 3, 3)` arrays.

**Step 4 — Flux Jacobian K = ∂F/∂U**

```
K = [[u ρ_p,          ρ,        u ρ_h         ],
     [u²ρ_p + 1,      2ρu,      u²ρ_h         ],
     [u ρ_p H,        ρ(H+u²),  u(ρ_h H + ρ)  ]]
```

Shape: `(N-1, 3, 3)`.

**Step 5 — Projection matrices R̃_k = K G_k K⁻¹ / λ_k^safe**

Computed via batched linear solve (no explicit inverse):

```python
R_k K = K G_k  →  K^T R_k^T = (K G_k)^T
R_k^T = jnp.linalg.solve(K^T, (K @ G_k)^T)
R̃_k  = R_k / λ_k^safe
```

The **entropy fix** prevents division by zero when λ_k → 0 (sonic point):

```python
λ_safe = sign(λ) · max(|λ|, δ)   with δ = 0.1(|u| + a)
sign(λ) = +1 if λ ≥ 0, −1 otherwise   ← never zero
```

**Step 6 — Upwind sum**

```python
R̃⁺ = Σ_{λk ≥ 0} R̃_k      (right-going waves)
R̃⁻ = Σ_{λk <  0} R̃_k      (left-going waves)
```

Note: `R̃⁺ + R̃⁻ = I` (partition of unity — consistency check).

**Step 7 — Interface flux**

Physical fluxes `F = (ρu, ρu² + p, uρH)` are computed from the **actual L/R
states** (not the average), using `HmassP_INPUTS` for density.

```python
F̂_{i+1/2} = R̃⁺ · F_L + R̃⁻ · F_R
```

### 3e. Source term (momentum only)

```python
S_momentum = p · (dA/dx)/A  −  (C_f/8) · ρ · u² · 4/D_h
```

- Area term: computed from precomputed `dAdx_A`.
- Friction: Haaland friction factor `C_f(Re, ε/D)` evaluated at each cell
  using viscosity from `HmassP_INPUTS`.

Mass and energy source terms are **zero** (adiabatic no-slip walls).

### 3f. Residual assembly

For each interior cell `i = 1 … N-2`:

```
R_i = V_i · (V_new_i − V_old_i)
    + dt · [A_{i+1/2} · F̂_{i+1/2} − A_{i-1/2} · F̂_{i-1/2}]
    − dt · V_i · S_i
```

where `V_i = A_i · dx` is the cell volume. Setting `R = 0` is the discrete
conservation law at time `n+1`.

---

## 4. Convergence Metric — SSR

After each time step:

```python
SSR = max_{i,j} |U_new[i,j] − U_n[i,j]| / (|U_n[i,j]| + ε)
```

This is the **maximum relative change in any primitive variable** across all
interior cells. Convergence is declared when `SSR < residuals` (from settings).

A plateau detector also stops the loop if SSR has not decreased by more than
50% over the last 5 iterations.

---

## 5. Output

After the while-loop, `U_final` (N, 3) is converted back to conservative
`Q` (3, N) via `_puh_to_conservative` for consistency with the rest of the
post-processing pipeline (`density`, `velocity`, `pressure`, etc. in `main.py`).

---

## 6. How to Run

In `settings.yaml`:

```yaml
time_discretization: fvs_newton

# Flux method is ignored — FVS is built into the scheme.
# Reconstruction is ignored — MUSCL Van Leer in (p,u,h) is always used.
# Warmup block is ignored.

# CFL can be raised to 5–10 once the solution is stable.
CFL: 1.0

# Keep Nx ≤ 500 (dense Jacobian). For larger grids, a banded solver is needed.
Nx: 50
```

---

## 7. Key Files

| File | What it contains |
|---|---|
| `fvm.py` | `stadtke_puh_initialization`, `make_jax_stadtke_step`, `_jax_F_fvs_stadtke` |
| `stadtke_single_phase_theory.md` | Full mathematical derivation (matrices, eigenvectors, FVS) |
| `hem_nozzle_implementation_spec.md` | Implementation spec (data structures, algorithm order) |
| `settings.yaml` | Runtime configuration |
