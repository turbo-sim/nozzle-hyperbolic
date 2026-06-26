# Hand-off Briefing for Claude Code: Two-Fluid FVS Solver

## Project context

We are extending an existing **JAX-based quasi-1D compressible nozzle solver** from single-phase real-gas (working and validated) to **two-phase flow** using Stadtke's six-equation, single-pressure, conditionally-hyperbolic two-fluid model (Stadtke, Wiley 2006). The existing solver provides:

- JAXProp/CoolProp real-gas EOS wrappers (`get_state` returning $\varrho, a, \Gamma, T$ per phase)
- Implicit Newton time-marching framework with `jax.jacfwd` for Jacobians
- Van Leer slope limiter, JIT/vmap infrastructure, boundary conditions, area-variation geometry

**The new work is purely the LHS algebra** of the two-fluid system: closed-form matrices, flux-vector-splitting kernel, non-conservative correction, source-term transformation. The Newton iteration, EOS interface, and overall time loop are reused.

**Frame of choice:** total-energy conservative state $V$ in the equation, but enthalpy primitives $U_h = (p, u_g, u_l, \alpha_g, h_g, h_l)$ for all closed-form matrices. Newton iterates on Stadtke's $W$ basis (8 components) to avoid Jacobian conditioning singularities; the 8th and 7th components are $p$ and $\alpha_g$, closed by two EOS consistency equations.

## Reference materials (read in this order)

All three are markdown documents in the project. Read each fully before coding.

1. **`stadtke_enthalpy_frame_derivation.md`** — the algebra. Closed forms for $G_h, \Phi, J, K, X, H^{nc}, R_h, L_h, D$. All matrices verified to machine-zero residuals in rational arithmetic at the nominal water/steam state. Use as the algebraic source of truth for every matrix.

2. **`stadtke_twofluid_implementation_pipeline.md`** — the build plan. Per-section breakdown of what each function does, with pseudocode. Appendix A gives the explicit primitive-frame source vector $D_s$ for code transcription.

3. **`stadtke_twofluid_verification_plan.md`** — the test plan. 42 tests across 9 stages. Each stage gates the next; don't advance past a failing stage.

## Implementation sequence

This is the order to build. Each step has a deliverable function or module, a unit test, and a pass criterion. The verification plan section numbers are listed in brackets.

### Step 1 — EOS layer extension [Verif §1]

**File:** `eos.py` (extend existing single-phase wrapper)

**Deliverable:** function `eos_phase(p, h)` returning `(rho, a, Gamma, T, rho_p, rho_h)` for one phase. The last two are computed algebraically:
```
rho_p = (1 + Gamma) / a**2
rho_h = -rho * Gamma / a**2
```

**Critical:** all six leaves must come from the same EOS call. Separate table lookups for $\varrho$ and $a$ break the closure $1/a^2 = \varrho_p + \varrho_h/\varrho$ and silently corrupt downstream matrices.

**Test (1.1):** $|1/a^2 - \varrho_p - \varrho_h/\varrho| < 10^{-12}/a^2$ at 20 random states.

### Step 2 — State conversion [Verif §2]

**File:** `state.py`

**Deliverables:**
- `primitives_to_conserved(U_h)` → $V$. Direct algebraic.
- `conserved_to_primitives(V)` → $U_h$. Inner Newton on 3 unknowns $(p, h_g, h_l)$ since $\alpha_g$ follows from the density constraint.

**Test (2.3):** round-trip $U_h \to V \to U_h'$ at 20 random states, $\|U_h' - U_h\|/\|U_h\| < 10^{-10}$.

### Step 3 — Closed-form matrices [Verif §3]

**File:** `matrices.py`

**Deliverables (each a pure function of `(U_h, eos_state)`):**
- `Phi_inv(state)` — entropy-to-enthalpy thermo transform (§3.2 of derivation doc)
- `G_h(state)` — quasi-linear coefficient matrix (§4 of derivation doc)
- `J(state)` — conservative bridge $\partial V/\partial U_h$ (§8)
- `K(state)` — flux Jacobian $\partial F/\partial U_h$ (§9)
- `X(state)` — non-conservative defect $JG_h - K$ (§11)
- `eigenstructure(state)` — returns `(R_h, L_h, D, eigvals)` with all the abbreviations (§13.1-13.4)

All entries are closed-form from the derivation doc. **Do not derive them again** — transcribe directly. Use the abbreviations $\hat\varrho_g, \hat\varrho_l, \hat\varrho_s, \hat\varrho^2, \varrho_s, a_0^2, \tilde a^2, \Delta a^2, a^2, a_1^2, a_2^2, \delta_a, \hat\alpha_g, \hat\alpha_l, u_{\text{mix}}$ as defined in §13.1.

**Critical typo to apply:** in $a_1^2$, use $\hat\varrho_l^{\,2}$ (squared), not the first power printed in Stadtke's book. See §13.1 note in the derivation doc.

**Tests (3.1-3.8):** all eight tests in verification plan §3 must pass at machine tier before moving on. The eigenstructure identities (3.5-3.8) are the most stringent — they confirm the full algebra is correctly transcribed.

### Step 4 — Flux-split kernel [Verif §4]

**File:** `flux.py`

**Deliverables:**
- `flux_vector(U_h, eos_state)` — physical flux $F$, 6-vector, from eq. 5.90 (with $E_i = h_i - p/\varrho_i + u_i^2/2$).
- `smoothed_sign(lam, eps)` — Harten entropy fix (linear interp through $|\lambda|<\epsilon$).
- `fvs_interface_flux(U_h_L, U_h_R, eos_L, eos_R)` — the per-face FVS kernel.

The FVS kernel structure (see pipeline §2.5):
```python
U_h_bar = 0.5 * (U_h_L + U_h_R)
eos_bar = eos_phase_average(eos_L, eos_R)
R_h, L_h, D, eigvals = eigenstructure(U_h_bar, eos_bar)
K_mat = K(U_h_bar, eos_bar)
rho = K_mat @ R_h                              # flux-space right vecs
rho_inv = jnp.linalg.solve(rho, jnp.eye(6))    # ONE LU per face
ell = jnp.diag(D) @ rho_inv                    # flux-space left vecs (rows)
eps_fix = 0.1 * mixture_sound_speed(U_h_bar)
s = smoothed_sign(eigvals, eps_fix)
w_plus = 0.5 * (1.0 + s); w_minus = 0.5 * (1.0 - s)
beta_L = (ell @ F_L) / D; beta_R = (ell @ F_R) / D
beta = w_plus * beta_L + w_minus * beta_R
F_hat = rho @ beta
```

**Tests (4.1-4.4):** projector-sum identity, smoothed-sign continuity, single-phase reduction, no-slip homogeneous reduction.

### Step 5 — Non-conservative correction [Verif §5]

**File:** `flux.py` (continued)

**Deliverable:** `nonconservative_correction(U_h_cell, dF)`:
```python
K_mat = K(U_h_cell, eos_state)
X_mat = X(U_h_cell, eos_state)
y = jnp.linalg.solve(K_mat, dF)
return X_mat @ y
```

Never form $H^{nc}$ dense. The $K$-LU per face produces both this correction and the flux-split dual basis from Step 4.

**Singular limit handling:** at $u_g$ or $u_l \to 0$, $\det K \to 0$. Fall back to `jnp.linalg.lstsq` with small `rcond` if pivoting in `solve` fails. Document this as a known soft spot.

**Tests (5.1-5.3).**

### Step 6 — Source assembly [Verif §6]

**File:** `sources.py`

**Deliverable:** `D_VE(U_h, eos_state, sources, geometry)` — the conservative-frame total-energy source.

Implementation is the pseudocode in **Appendix A** of the pipeline document. Two functions:
- `D_s_primitive(U_h, eos_state, sources, geometry, k)` — Stadtke Table B.46 with the $k \to k\varrho$ correction applied throughout. Code skeleton in pipeline Appendix A.5.
- `D_VE(U_h, eos_state, sources, geometry, k)` — applies $\Phi^{-1}$ then $J$ to $D_s$.

**Critical typo correction:** every occurrence of "$k$" multiplying or appearing next to a single density in Stadtke's Table B.46 must be read as "$k\varrho$" (mixture density). Without this, momentum conservation fails by ~65%.

**Known issue (flag, don't block):** energy conservation residual $\approx -F^v_g\,\Delta u$ at the ~1% relative level. Source unidentified — likely a second typo in B.46 rows 2-3 affecting the $F^v$ coefficient, or a kinetic-energy decomposition mismatch in eq. 4.50. **Should not impair steady-state nozzle solves** where $F^v_g$ is small. Document this in code comments.

**Tests (6.1-6.8):** row-by-row check, conservation sums for mass (must pass), momentum (must pass after typo correction), energy (log residual; pass if $\sim F^v_g \Delta u$, fail if larger).

### Step 7 — Residual function [Verif §7]

**File:** `residual.py`

**Deliverable:** `residual_W(W_new, W_old, dt, geometry)` returning an 8-vector per cell:
- 6 components: conservation residual $\frac{V(W) - V_n}{\Delta t} + \mathcal F(W) + \mathcal N(W) - D_{V_E}(W)$
- 2 components: EOS constraints $\mathcal C_g, \mathcal C_l$ (from pipeline §3 of W-basis discussion)

Use `state_from_W(W)` to compute $V(W)$ — polynomial substitution, EOS-free.

```python
def state_from_W(W):
    # W = (p, alpha_g, alpha_g*rho_g, u_g, alpha_l*rho_l, u_l, alpha_g*rho_g*e_g, alpha_l*rho_l*e_l)
    return jnp.array([
        W[2], W[4],                                  # masses
        W[2] * W[3], W[4] * W[5],                    # momenta
        W[6] + 0.5 * W[2] * W[3]**2,                 # gas total energy
        W[7] + 0.5 * W[4] * W[5]**2,                 # liquid total energy
    ])
```

**EOS constraint:** $\mathcal C_g(W) = W_3 - W_2\,\varrho_g^{\mathrm{EOS}}(W_1, h_g(W))$ where $h_g(W) = (W_7 + W_1 W_2)/W_3$. Same for liquid.

**Test (7.1):** stationary uniform state with no sources → residual zero to machine precision at every cell. This is the gating integration test — failure here means one of steps 3-6 has a bug.

### Step 8 — Newton iteration [Verif §8]

**File:** `newton.py` (extend existing single-phase Newton driver)

**Deliverable:** per-timestep Newton solver.

```python
def newton_step(W_old, dt, geometry):
    V_old = vmap(state_from_W)(W_old)
    W = W_old  # initial guess
    for k in range(max_newton_iters):
        R = vmap(residual_W)(W, W_old, V_old, dt, geometry)
        if jnp.linalg.norm(R) < newton_tol:
            break
        J_R = vmap(jax.jacfwd(residual_W))(W, W_old, V_old, dt, geometry)
        dW = vmap(jnp.linalg.solve)(J_R, -R)        # 8x8 per cell
        W = W + dW
    return W
```

Use `jax.jacfwd` for the Jacobian — exact, no transcription risk, polynomial in $V(W), F(W)$ and EOS-mediated for $D(W)$ and $\mathcal C(W)$.

**Test (8.1):** quadratic convergence near steady state. Plot $\log\|R^{k+1}\|$ vs $\log\|R^k\|$, slope $\approx 2$.

### Step 9 — Physics validation [Verif §9]

Once steps 1-8 pass, run the physics benchmarks:
1. Single-phase gas limit ($\alpha_g = 1 - 10^{-6}$) — match existing single-phase solver
2. Single-phase liquid limit — symmetric
3. No-slip homogeneous transient — preserve $u_g = u_l, h_g = h_l$
4. Stadtke shock tube benchmark
5. Subsonic / choked / supersonic nozzle

## Known issues and design decisions

| Issue | Decision |
|-------|----------|
| Table B.46 prints "$k$" alone next to densities | Apply $k \to k\varrho$ correction throughout (mandatory for momentum conservation) |
| Energy conservation residual $\sim F^v_g \Delta u$ | Flag and document; do not block; investigate after main solver works |
| $a_1^2$ in Stadtke book has $\hat\varrho_l$ first power | Use **squared** $\hat\varrho_l^{\,2}$ as in derivation doc §13.1 |
| Newton variable choice | Use Stadtke's 8-component $W$, not 6-component $V$, for Jacobian conditioning |
| $K^{-1}$ representation | Never assemble dense; LU per face, reused for flux-split and $H^{nc}$ correction |
| Entropy fix | **Mandatory**, not optional. Required for both physical entropy condition and Newton $C^1$ regularity. $\epsilon \sim 0.1\,a$ |
| Slope limiter | Van Leer (preferred over Stadtke's hybrid for $C^1$ smoothness with Newton) |

## What to defer

These can wait until after the core solver works:
- Two-pressure model (single-pressure is the validated intermediate step)
- More sophisticated interfacial closure models (use Stadtke's simple constants $C_1, C_2, C_3$ from 1997 paper §3 for first cuts)
- The energy conservation residual investigation
- Performance tuning beyond what JIT + `vmap` give automatically

## File structure

```
two_fluid/
├── eos.py          # Step 1: real-gas EOS, per-phase wrapper
├── state.py        # Step 2: U_h <-> V converters
├── matrices.py     # Step 3: G_h, Phi_inv, J, K, X, eigenstructure
├── flux.py         # Steps 4-5: FVS kernel, H^nc correction
├── sources.py      # Step 6: D_s primitive, D_VE conservative
├── residual.py     # Step 7: 8-component W-residual
├── newton.py       # Step 8: implicit time step
├── tests/          # Verification plan: one test file per stage
│   ├── test_eos.py
│   ├── test_state.py
│   ├── test_matrices.py
│   ├── test_flux.py
│   ├── test_sources.py
│   ├── test_residual.py
│   └── test_newton.py
└── benchmarks/     # Step 9: physics validation
```

## Discipline

- **Each step must pass its verification tests before the next is started.** The verification plan §10 is explicit: a stage-4 bug that surfaces only at stage 7 is debug hell.
- **Transcribe the closed forms; don't re-derive.** The derivation doc has the verified algebra; the role of code is to evaluate it, not to re-prove it.
- **Use `jax.jacfwd` for Jacobians.** Hand-coded derivatives invite transcription errors. The polynomial structure of $V(W), F(W)$ makes autodiff exact.
- **Comment the known issues at the relevant line of code.** Future-you (or me) will forget the $k \to k\varrho$ correction otherwise.

Start with Step 1 and work down. Pause and check in after Step 3 (closed-form matrices) — that's the largest single deliverable and the most error-prone, so worth a sanity review before pushing on to flux assembly.
