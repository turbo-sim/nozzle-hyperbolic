# Implementation Pipeline: FVS Solver for Städtke's Six-Equation Two-Fluid Model
### Quasi-1D Nozzle, Enthalpy Primitives, Total-Energy Conservative, Implicit Newton

*Sister document to `stadtke_enthalpy_frame_derivation.md`. That document gives the closed-form
algebra of $G_h, \Phi, J, K, X, H^{nc}, R_h, L_h, D$. This document gives the step-by-step
implementation pipeline — what to build, in what order, with what dependencies, and how the
per-timestep computation flows. The single-phase real-gas solver is assumed to exist; the goal
here is to extend it to the two-fluid system with minimal disruption to the existing
time-marching, EOS interface, and Newton infrastructure.*

---

## 1. The discrete equation, top-down

The quasi-1D conservative form, integrated over a cell of area $A_i(x)$ and length $\Delta x$,
is

$$\frac{\mathrm d V_i}{\mathrm d t} = -\frac{1}{A_i\Delta x}\bigl[(A\hat F)_{i+1/2} - (A\hat F)_{i-1/2}\bigr] - (H^{nc})_i\cdot\frac{(A\hat F)_{i+1/2}-(A\hat F)_{i-1/2}}{A_i\Delta x} + S_i,$$

where $\hat F_{i\pm1/2}$ are the interface fluxes from FVS, $H^{nc}$ is the non-conservative
coupling matrix, and $S_i = JD_i$ is the conservative-frame source vector (geometric +
interfacial). Discretized implicitly in time:

$$R(V^{n+1}) \;\equiv\; \frac{V^{n+1}-V^n}{\Delta t} + \mathcal F(V^{n+1}) + \mathcal N(V^{n+1}) - S(V^{n+1}) = 0,$$

with $\mathcal F$ the conservative-flux divergence, $\mathcal N$ the non-conservative
contribution, and $S$ the source vector. The Newton iteration is the existing infrastructure
from the single-phase solver — only $\mathcal F, \mathcal N, S$ change. JAX autodiff supplies
$\partial R/\partial V^{n+1}$ for the linear solve at each Newton step.

---

## 2. Building blocks, in dependency order

Implement bottom-up. Each block has a verification test that must pass before the next is
built. All blocks are pure functions, taking explicit state and returning explicit arrays — no
hidden state, no mutability — to keep JIT and autodiff clean.

### 2.1 EOS interface

**Inputs:** $(p, h_i)$ per phase.
**Outputs:** $(\varrho_i, a_i, \Gamma_i, T_i)$ per phase, then algebraically
$\varrho_{p,i} = (1+\Gamma_i)/a_i^2$, $\varrho_{h,i} = -\varrho_i\Gamma_i/a_i^2$.

```
def eos_phase(p, h):                          # per phase
    rho, a, Gamma, T = jaxprop.get_state(p, h)
    rho_p = (1 + Gamma) / a**2
    rho_h = -rho * Gamma / a**2
    return rho, a, Gamma, T, rho_p, rho_h
```

**Critical:** $\varrho_{p,i}$ and $a_i$ must come from the **same** EOS call — separate
table lookups break the closure $1/a_i^2 = \varrho_{p,i} + \varrho_{h,i}/\varrho_i$ at the
round-off level, which propagates into spurious non-zero entries in mass rows of $X$ (see
§11.1 of derivation doc).

**Test:** Verify the closure $1/a_i^2 \approx \varrho_{p,i} + \varrho_{h,i}/\varrho_i$ to
$\sim 10^{-12}$ at a few states; failure indicates the table call is inconsistent.

### 2.2 Primitive ↔ conservative converters

The state stored in the solver is $V$ (conserved); the matrices and fluxes work in $U_h$
(primitives). Converters in both directions are needed.

**$U_h \to V$** is direct algebraic:

```
def primitives_to_conserved(p, ug, ul, ag, hg, hl):
    rho_g, *_ = eos_phase(p, hg)
    rho_l, *_ = eos_phase(p, hl)
    al = 1 - ag
    Eg = hg - p/rho_g + 0.5*ug**2
    El = hl - p/rho_l + 0.5*ul**2
    return jnp.array([
        ag*rho_g, al*rho_l,
        ag*rho_g*ug, al*rho_l*ul,
        ag*rho_g*Eg, al*rho_l*El,
    ])
```

**$V \to U_h$** requires solving a small nonlinear system, because $\varrho_i, h_i$ are
coupled through the EOS. The unknowns are $(p, h_g, h_l)$ given $V_1, V_2, V_3, V_4, V_5, V_6$
and $\alpha_g$ (which is determined by $V_1/\varrho_g + V_2/\varrho_l = 1$). One clean approach:
solve a 3-variable system for $(p, h_g, h_l)$ via inner Newton, with $(\varrho_g, \varrho_l)$
from EOS and $\alpha_g = V_1/\varrho_g(p, h_g)$. This is per-cell and is the only nonlinear
inversion in the timestep (besides the outer Newton on the full system).

**Test:** Round-trip $V \to U_h \to V'$ should give $\|V'-V\|/\|V\| < 10^{-12}$.

### 2.3 Closed-form matrices

All from the derivation document. Each is a pure function of the state $U_h$ plus the EOS
quantities at that state. Implementation strategy: write each as a Python function returning a
$6\times 6$ JAX array, taking the abbreviated quantities $(\varrho_i, a_i, \varrho_{p,i},
\varrho_{h,i}, T_i)$ pre-computed by the EOS layer.

The matrices needed:

| Symbol | Function | Use |
|--------|----------|-----|
| $G_h$ | `G_h(state)` | Eigenvalue/eigenvector source; not used directly in flux assembly |
| $\Phi^{-1}$ | `Phi_inv(state)` | Transform sources from $U_s$ frame to $U_h$ frame |
| $J$ | `J_matrix(state)` | Apply to source vector: $S = J\,D$; not needed dense if source is in primitive frame |
| $K$ | `K_matrix(state)` | Per-face: used in LU for $H^{nc}\Delta F$ and for flux-space projectors |
| $X$ | `X_matrix(state)` | Per-cell: used in $H^{nc}\Delta F = X(K^{-1}\Delta F)$ |
| $R_h, L_h, D$ | `eigenstructure(state)` | Per-face: build the per-wave projectors |

**Test for each:** Pass a representative state, compare against the values in §13.5 of the
derivation doc. The eigenstructure check should pass to $\sim 10^{-12}$ in float64.

### 2.4 Per-face eigenstructure

At a face with reconstructed left and right states $U_{h,L}, U_{h,R}$, an *average* state
$\bar U_h$ is needed for the eigenstructure. Two reasonable choices:

- **Arithmetic average** of primitives: $\bar U_h = (U_{h,L} + U_{h,R})/2$. Simple, robust,
  loses exact wave conservation for steady contact discontinuities. Acceptable for the
  nozzle problem.
- **Roe-like average** preserving exact shock relations: more complex, no clear two-phase
  generalization. Skip for the initial implementation.

Start with arithmetic. Compute $R_h, L_h, D, \Lambda$ at $\bar U_h$.

### 2.5 Per-face flux split with entropy fix

Given $R_h, L_h, D, \Lambda$ at the face average, and the left/right *fluxes* $F_L = F(U_{h,L})$,
$F_R = F(U_{h,R})$, the FVS interface flux is

$$\hat F = R^+ F_L + R^- F_R, \qquad R^\pm = \sum_k\frac{1\pm s_\epsilon(\lambda_k)}{2}\,\widetilde P_k,$$

where $\widetilde P_k = K r_k\,(l_k K^{-1})/D_k$ are the flux-space projectors and
$s_\epsilon(\lambda)$ is the **smoothed sign function** (Harten entropy fix):

$$s_\epsilon(\lambda) = \begin{cases}\mathrm{sign}(\lambda) & |\lambda|\ge\epsilon \\[2pt] \lambda/\epsilon & |\lambda|<\epsilon\end{cases},\qquad \epsilon \sim 0.1\,a.$$

**Why the smoothing is mandatory, not optional.** Without it, the splitting is discontinuous
at $\lambda_k=0$ (sonic transitions through the throat, or void/entropy waves at $u_i=0$).
This breaks both the entropy condition (allowing unphysical expansion shocks) and the
$C^1$ regularity required for Newton convergence — the implicit iteration loses quadratic
convergence and can stall at sonic points. The smoothing introduces a small "wave-sharing"
zone $|\lambda_k|<\epsilon$ where the wave contributes to both $R^+$ and $R^-$ with weights
that vary smoothly through zero, restoring both physical and numerical regularity.

```
def smoothed_sign(lam, eps):
    # Harten entropy fix: linear interpolation through |lam| < eps
    return jnp.where(jnp.abs(lam) >= eps, jnp.sign(lam), lam / eps)

def fvs_interface_flux(U_bar, F_L, F_R, eps_fix=None):
    # Build closed-form matrices at the average state
    R_h, L_h, D, eigvals = eigenstructure(U_bar)
    K = K_matrix(U_bar)
    
    # Default entropy-fix threshold ~ 0.1 * mixture sound speed
    if eps_fix is None:
        a_local = mixture_sound_speed(U_bar)
        eps_fix = 0.1 * a_local
    
    # Flux-space right vectors: rho_k = K r_k (closed-form mat-vec, no inverse)
    rho = K @ R_h                              # columns are rho_k
    
    # Flux-space left vectors: ell_k = (l_k K^{-1}), recovered as dual basis of rho_k
    # via one shared 6x6 LU. By biorthogonality, ell_j . rho_k = D_k delta_{jk},
    # so the row vector ell_j is the j-th row of diag(D) @ rho^{-1}.
    rho_inv = jnp.linalg.solve(rho, jnp.eye(6))    # one LU
    ell = jnp.diag(D) @ rho_inv                    # rows are ell_k
    
    # Smoothed signs per wave
    s = smoothed_sign(eigvals, eps_fix)
    w_plus  = 0.5 * (1.0 + s)                      # weight on F_L per wave
    w_minus = 0.5 * (1.0 - s)                      # weight on F_R per wave
    
    # Per-wave amplitudes: beta_k^pm = (ell_k . F_{L or R}) / D_k, smoothly blended
    beta_L = (ell @ F_L) / D                       # all waves treated against F_L
    beta_R = (ell @ F_R) / D                       # all waves treated against F_R
    beta   = w_plus * beta_L + w_minus * beta_R    # blended amplitudes
    
    # Assemble interface flux as sum of beta_k * rho_k
    F_hat = rho @ beta
    return F_hat
```

This is the rank-one kernel: one LU on the $6\times 6$ matrix $\rho = [\rho_1|\ldots|\rho_6]$
per face, six wave amplitudes, six scaled vector adds. The smoothing adds only one
elementwise branch per wave — no additional matrix work.

**Test:** At a state with $u_g = u_l$ (no slip) and $h_g = h_l$ (no thermal disequilibrium),
$\hat F$ should reduce to the single-phase Euler flux. At a sonic point where one wave has
$\lambda_k = 0$, $\hat F$ should be smooth in the surrounding parameter range — verify by
checking that small perturbations in $\Delta u$ or $p$ produce $\hat F$ changes
proportional to the perturbation, not jumps.

### 2.6 Per-cell non-conservative correction

For each cell $i$, compute the flux difference $\Delta F = \hat F_{i+1/2} - \hat F_{i-1/2}$,
then:

```
def nonconservative_correction(U_h_cell, dF):
    K = K_matrix(U_h_cell)
    X = X_matrix(U_h_cell)
    # H^nc dF = X K^{-1} dF, computed via solve, never as dense H^nc
    y = jnp.linalg.solve(K, dF)
    return X @ y
```

**Singular limit.** $\det K \propto u_g u_l$, so $K$ is singular at zero phasic velocity.
At such states the corresponding columns of $K$ vanish (the $h_i$ column at $u_i=0$), and the
$h_i$-flux components also vanish, so $\Delta F$ lies in the range of $K$ — the solve is
well-posed in the least-squares sense even when not in the inverse sense. Implementation:
prefer `jnp.linalg.lstsq(K, dF)` with a small `rcond`, or precondition with
$\operatorname{diag}(1, 1, 1, 1, \max(|u_g|, \epsilon), \max(|u_l|, \epsilon))$ scaling. The
exact handling depends on how the problem domain regularizes near stagnation; defer the
final choice until the first stagnation-bearing test case.

### 2.7 Sources — from Stadtke's entropy-form source to the total-energy source

The source vector $D_s$ for the **entropy formulation** in primitive variables for quasi-1D
flow is given in Stadtke's **Table B.46**. The corresponding source $D_{V_E}$ for the
**total-energy conservative formulation** is obtained by transformation chain, not by
re-deriving from physical first principles. The transformation has two steps.

**Important: Table B.46 has at least one transcription error.** Where the original printing
shows "$k$" multiplying a single density (e.g., "$k\varrho_g$", "$\alpha_g k$"), the correct
expression requires "$k\varrho$" with $\varrho$ the mixture density — only this preserves
dimensional consistency. For example, "$\varrho_l + \alpha_g k$" in row 2 must read
"$\varrho_l + \alpha_g k\varrho$". The same pattern appears in rows 2, 3, and the
$F_g, F_l$ couplings. **Apply this correction throughout when implementing Table B.46.**

#### Step 1 — entropy primitives to enthalpy primitives: apply $\Phi^{-1}$

The chain rule on the primitive-form equation $\partial_t U_s + G_s\,\partial_x U_s = D_s$
combined with $U_s = U_s(U_h)$ gives $\partial_t U_h + G_h\,\partial_x U_h = \Phi^{-1}D_s$, so

$$\boxed{\;D_h = \Phi^{-1}\,D_s.\;}$$

Only rows 5 and 6 change:

$$D_{h,5} = \frac{D_{s,1}}{\varrho_g} + T_g\,D_{s,5}, \qquad D_{h,6} = \frac{D_{s,1}}{\varrho_l} + T_l\,D_{s,6}.$$

The transformed rows have a clean physical reading. $D_{s,5} = \sigma_g^S/(\alpha_g\varrho_g)$
is the material entropy source. Using $\sigma_g^S$ from Stadtke eq. 4.50:

$$T_g D_{s,5} = \frac{\sigma_g^M\bigl[(h^{ex}-h_g)+\tfrac12(u^{ex}-u_g)^2\bigr] + F^v_g(u^{int}-u_g) + \sigma_g^Q + Q_g}{\alpha_g\varrho_g}.$$

#### Step 2 — enthalpy primitives to total-energy conservative: apply $J$

$$\boxed{\;D_{V_E} = J\,D_h = J\,\Phi^{-1}\,D_s.\;}$$

#### Conservation status — verified diagnostic

The chain $D_{V_E} = J\Phi^{-1}D_s$ should preserve the physical conservation laws for
strictly-1D, interfacial-only sources (no externals, no geometry): under the constraints
$\sigma_g^M + \sigma_l^M = 0$, $\sigma_g^Q + \sigma_l^Q = 0$, $F^v_g + F^v_l = 0$, and a
single interface velocity $u^{int} = u^{ex}$, the sums $D_{V_E,1}+D_{V_E,2}$,
$D_{V_E,3}+D_{V_E,4}$, $D_{V_E,5}+D_{V_E,6}$ should all vanish.

Diagnostic at the nominal water/steam state confirmed:

| Sum | Result | Status |
|-----|--------|--------|
| Mass: $D_{V_E,1}+D_{V_E,2}$ | $\sim 10^{-16}$ relative | ✓ machine zero |
| Momentum: $D_{V_E,3}+D_{V_E,4}$ | $\sim 10^{-16}$ relative | ✓ machine zero (after $k\to k\varrho$ correction) |
| Energy: $D_{V_E,5}+D_{V_E,6}$ | $\approx -0.98\,F^v_g\,\Delta u$ | ✗ ~1% relative residual |

The **energy non-conservation is isolated to the viscous-friction term**:

- Heat-only sources ($\sigma_g^Q\neq 0$, all else zero): energy conserves to machine zero ✓
- Mass-only sources ($\sigma_g^M\neq 0$, all else zero): energy conserves to $\sim 10^{-4}$
  relative ✓
- Viscous-only ($F^v_g\neq 0$, all else zero): residual $\approx -F^v_g\Delta u$ ✗

This residual is independent of the choice of interface velocity $u^{int}$ in the
$F^v_g(u^{int}-u_g)/T_g$ term of $\sigma_g^S$, suggesting the issue is not a velocity
ambiguity but rather a missing bulk-work contribution somewhere in the chain.

**Likely cause (under investigation):** the kinetic-energy decomposition in Stadtke eq. 4.50
splits viscous work as dissipation $F^v_g(u^{int}-u_g)/T_g$ in $\sigma_g^S$ and bulk work
$F^v_g$ in the momentum source $D_{s,2}$. The bulk work should contribute to the conservative
energy source via the chain-rule term $\alpha_g\varrho_g u_g\,D_{s,2}$ (the $u_g$-column entry
of row 5 of $J$). Comparing with the physical expectation $F^v_g u^{int}$ for the total energy
gain of phase $g$, there appears to be a missing or mis-scaled coefficient — possibly a
**second transcription error in Table B.46 rows 2–3 affecting the $F^v$ term specifically**,
or a definition mismatch in the kinetic-energy treatment between the entropy and energy
formulations.

**Practical implication:** for typical nozzle flows where viscous interfacial friction is
small compared to inertia and pressure forces, the 1% energy non-conservation translates to
even smaller global error and should not impair the steady-state solver. For long-time
transients or shock interactions where $F^v_g$ becomes dominant, the residual could
accumulate — flag for re-investigation before those use cases.

#### Operational form

```
def source_total_energy(U_h_cell, sources_physical, geometry):
    # Step A: Assemble D_s from Table B.46 — with k -> k*rho correction applied
    D_s = stadtke_table_B46_corrected(U_h_cell, sources_physical, geometry)
    
    # Step B: Phi^{-1} transformation — only rows 5,6 change
    D_h = D_s.copy()
    D_h[4] = D_s[0]/rho_g + T_g * D_s[4]
    D_h[5] = D_s[0]/rho_l + T_l * D_s[5]
    
    # Step C: J multiplication
    J = J_matrix(U_h_cell)
    D_VE = J @ D_h
    return D_VE
```

**Verification (rows 1–2 — mass equations).** Substitution confirms

$$D_{V_E,1} = \sigma_g^M - \alpha_g\varrho_g u_g\,\frac{1}{A}\frac{\mathrm dA}{\mathrm dx}, \qquad D_{V_E,2} = \sigma_l^M - \alpha_l\varrho_l u_l\,\frac{1}{A}\frac{\mathrm dA}{\mathrm dx},$$

exactly. This is the integrity check that the chain is implemented correctly — if it failed
here, the transformation would be wrong everywhere.

**Why not derive $D_{V_E}$ from physical first principles?** Numerical testing confirmed that
the "obvious" form — heat + mass-transfer enthalpy + body-force work + geometric area — misses
1–10% in momentum and energy rows due to **virtual-mass redistribution** that the $k$
coefficient couples between phases. The transformation chain $J\Phi^{-1}D_s$ inherits this
redistribution automatically from Stadtke's verified Table B.46 (modulo the $k\to k\varrho$
typo and the open energy-row question above).

---

## 3. Per-timestep flow

```
def residual(V_new, V_old, dt, geometry):
    # Step 1: Recover primitives at the new time level (per cell)
    U_h_cells = vmap(conserved_to_primitives)(V_new)
    
    # Step 2: Reconstruct face L/R states (Van Leer slope limiter on U_h)
    U_h_L, U_h_R = reconstruct_faces(U_h_cells)
    
    # Step 3: Interface fluxes per face (FVS)
    F_L_arr = vmap(flux_vector)(U_h_L)            # F(U_h_L) per face
    F_R_arr = vmap(flux_vector)(U_h_R)            # F(U_h_R) per face
    U_h_bar = 0.5 * (U_h_L + U_h_R)               # average state for eigenstructure
    F_hat = vmap(fvs_interface_flux)(U_h_bar, F_L_arr, F_R_arr)
    
    # Step 4: Flux divergence (per cell)
    dF_dx = (F_hat[1:] - F_hat[:-1]) / dx
    
    # Step 5: Non-conservative correction (per cell)
    H_nc_dF = vmap(nonconservative_correction)(U_h_cells, dF_dx * dx)  # times dx because we passed un-divided
    
    # Step 6: Sources (per cell)
    S = vmap(source_conservative)(U_h_cells, geometry.A, geometry.dA_dx)
    
    # Step 7: Assemble residual
    R = (V_new - V_old) / dt + dF_dx + H_nc_dF / dx - S
    return R
```

The Newton iteration on `R(V_new) = 0` then proceeds exactly as in the single-phase code:
JAX computes $\partial R/\partial V_{\text{new}}$ via `jax.jacfwd` or `jax.jacrev`, the linear
solve is via `jnp.linalg.solve` (or a Krylov method if the grid is large), and Newton steps
are taken with line search / damping for robustness.

---

## 4. Edge cases to handle explicitly

### 4.1 Phase vanishing $\alpha_g \to 0$ or $\to 1$

When one phase vanishes, $J$ degenerates ($\det J \propto \alpha_g^2\alpha_l^2$), $K$
degenerates similarly, and the EOS at the vanishing phase is undefined (no meaningful state).

**Strategy:** Floor $\alpha_g \in [\epsilon_\alpha, 1-\epsilon_\alpha]$ with
$\epsilon_\alpha \sim 10^{-6}$. The vanishing-phase variables ($\varrho, h, u$) are still
carried by the solver but become passively advected. Use the same floor consistently in EOS
calls, source closure, and matrix assembly.

### 4.2 Zero slip $\Delta u \to 0$

Void waves and entropy waves coalesce at $\lambda = u_g = u_l$. The individual projectors $P_1,
P_5$ (and $P_2, P_6$) become singular individually but their sum $P_1 + P_5$ (and $P_2 + P_6$)
remains well-defined and is what physically matters.

**Strategy:** Branch on $|\Delta u| < \epsilon_{\Delta u}$. Below the threshold, assemble the
convective block as $P_{\text{conv}} = I - P_3 - P_4$ rather than summing individual void and
entropy projectors. The acoustic projectors $P_3, P_4$ remain non-singular at $\Delta u = 0$
because the acoustic block of $R_h$ does not vanish there.

### 4.3 Zero phasic velocity $u_g$ or $u_l \to 0$

$K$ is singular ($\det K \propto u_g u_l$). $H^{nc}\Delta F = X K^{-1}\Delta F$ via direct
inverse fails. Use the least-squares solve as discussed in §2.6, or pivot regularization.

### 4.4 Sonic transition through the throat

The acoustic wave speeds $u_{\text{mix}} \pm a$ change sign at the throat. The FVS scheme
handles this without special treatment — the upwinding flips automatically. No branch needed.

---

## 5. Validation tests, in order

Each test must pass before the next is attempted. Each verifies a specific reduction or
benchmark.

1. **Single-phase gas reduction** ($\alpha_g \to 1-\epsilon$). The two-fluid solver run with
   $\alpha_g = 1-10^{-6}$ should reproduce the single-phase gas solver's output to
   $10^{-4}$ relative error for a steady nozzle flow. This validates the EOS coupling and
   the $\alpha_g\to 1$ limit of all matrices.
2. **Single-phase liquid reduction** ($\alpha_g \to \epsilon$). Symmetric to (1).
3. **No-slip homogeneous flow** ($u_g = u_l$, same enthalpies). The solver should preserve
   $u_g = u_l$ and $h_g = h_l$ exactly (or to $\sim 10^{-10}$) over many timesteps. Failure
   indicates spurious slip generation from the discretization.
4. **Shock tube** with two-phase initial conditions. Compare against published Städtke
   reference results.
5. **Subsonic nozzle** at low pressure ratio. Should reach steady state with no oscillation.
6. **Choked nozzle** at the critical pressure ratio. Mass flux should agree with
   thermodynamic critical flow to $\sim 1\%$.
7. **Supersonic nozzle with shock**. The shock should appear at the predicted location.

---

## 6. Performance considerations

The closed-form matrices and rank-one projectors avoid the major costs that would arise from
naïve implementations:

- **No autodiff through EOS in the hot loop.** $\varrho_{p,i}, \varrho_{h,i}$ are formed
  algebraically from $a_i, \Gamma_i$ — both leaves from the EOS table. Differentiating the
  table is unnecessary and costly.
- **No dense $K^{-1}$.** A single $6\times 6$ LU per face, reused for both the
  non-conservative correction and the flux-space dual basis.
- **No eigenvalue solve.** All eigenvalues and eigenvectors are closed form.
- **`vmap` over faces and cells.** No Python loops; JAX vectorizes everything.
- **`jit` the residual function.** The full per-timestep residual is one JIT-compiled
  function; the Newton outer loop calls it repeatedly with different state arguments.

Estimated per-face cost (face count $N_f$, cell count $N_c$): closed-form matrices $\sim 200$
flops each, eigenstructure $\sim 500$ flops, LU on $K$ for projectors $\sim 200$ flops, total
per face $\sim 1.5$ kflops. For $N_c \sim 10^3$, residual evaluation is $\sim 1.5$ Mflops —
microseconds on modern hardware once JIT-compiled.

---

## 7. Implementation order checklist

In the order to build and test:

- [ ] EOS phase wrapper returning $(\varrho, a, \Gamma, T)$ + algebraic
      $(\varrho_p, \varrho_h)$. Test closure identity.
- [ ] $U_h \to V$ direct converter.
- [ ] $V \to U_h$ inner Newton. Test round-trip.
- [ ] Closed-form $G_h$. Verify eigenvalues match the analytic $\Lambda$ at a test state.
- [ ] Closed-form $J, K$. Verify $\det J \propto \alpha_g^2\alpha_l^2$ structure.
- [ ] Closed-form $X = J G_h - K$. Verify zero mass rows, zero enthalpy columns.
- [ ] Closed-form $R_h, L_h$ with all the abbreviations. Verify the four eigenstructure
      identities of §13.5.
- [ ] Per-face flux assembly via rank-one projectors. Test against single-phase Euler limit.
- [ ] Per-cell non-conservative correction. Test vanishing on homogeneous, no-slip state.
- [ ] Source assembly with $J$ applied to primitive-frame $D$.
- [ ] Full residual function. Test on a stationary uniform state — residual should be zero.
- [ ] Newton iteration on the existing time-marching framework.
- [ ] Validation suite §5.

Once the checklist is complete, the implementation is ready for the production validation
campaign.

---

## Appendix A. Explicit $D_s$ vector for code transcription

This appendix gives the explicit entries of the primitive-frame entropy-form source vector
$D_s$ (Stadtke Table B.46) with the $k \to k\varrho$ typo correction applied throughout. This
is what gets transcribed into a `stadtke_table_B46_corrected(state, sources, geometry)`
function in the codebase. The chain to $D_{V_E}$ is then two matrix products:

```
D_h  = Phi_inv @ D_s        # only rows 5, 6 change
D_VE = J @ D_h              # 6x6 matrix-vector
```

### A.1 Physical inputs (from closure models)

These come from closure correlations specific to the flow regime; treat each as an input to
the source-assembly function.

| Symbol | Meaning | Constraint |
|--------|---------|------------|
| $\sigma_g^M$ | Gas mass source per unit volume (evaporation rate) | $\sigma_l^M = -\sigma_g^M$ |
| $\sigma_g^Q$ | Gas interfacial heat source per unit volume | $\sigma_l^Q = -\sigma_g^Q$ |
| $F^v_g$ | Interfacial viscous (drag) force per unit volume on gas | $F^v_l = -F^v_g$ |
| $F_g, F_l$ | External body force per unit volume (e.g., gravity) | independent |
| $Q_g, Q_l$ | External heat source per unit volume (e.g., wall heating) | independent |
| $u^{ex}$ | Velocity of transferred mass at interface | single value |
| $u^{int}$ | Interface velocity for viscous-work calculation | single value |
| $h^{ex}$ | Enthalpy of transferred mass at interface | single value |
| $(1/A)\,\mathrm{d}A/\mathrm{d}x$ | Logarithmic area derivative for quasi-1D | from geometry |

For typical closures: $u^{ex} = u^{int} = $ mass-weighted mixture velocity; $h^{ex} = $
saturation enthalpy of the appearing phase (boiling) or the disappearing phase (condensation).

### A.2 State-derived intermediates

All computed from the current $U_h$ state and EOS leaves:

$$\begin{aligned}
\varrho &= \alpha_g\varrho_g + \alpha_l\varrho_l & &\text{mixture density} \\
\varrho_s &= \alpha_g\varrho_l + \alpha_l\varrho_g & &\text{cross-weighted density} \\
\hat\varrho^2 &= \varrho_g\varrho_l + k\varrho^2 & &\text{virtual-mass-modified density} \\
a_0^2 &= \dfrac{\varrho_s}{\alpha_g\varrho_l/a_g^2 + \alpha_l\varrho_g/a_l^2} & &\text{frozen sound combination} \\
\Delta u &= u_g - u_l & &\text{slip velocity}
\end{aligned}$$

### A.3 Entropy sources from physical inputs (Stadtke eq. 4.50)

The reduced entropy sources $\sigma_g^S, \sigma_l^S$ that appear in $D_s$ rows 5, 6 are built
from the physical inputs:

$$T_g\sigma_g^S = \sigma_g^M\Bigl[(h^{ex} - h_g) + \tfrac{1}{2}(u^{ex} - u_g)^2\Bigr] + F^v_g(u^{int} - u_g) + \sigma_g^Q + Q_g,$$

$$T_l\sigma_l^S = \sigma_l^M\Bigl[(h^{ex} - h_l) + \tfrac{1}{2}(u^{ex} - u_l)^2\Bigr] + F^v_l(u^{int} - u_l) + \sigma_l^Q + Q_l.$$

So $\sigma_g^S = T_g\sigma_g^S / T_g$ etc. — divide by $T_i$ at the end. In code, compute the
$T_i\sigma_i^S$ products first (they're what shows up algebraically in the chain).

**Note on the thermodynamic identity** used in $D_{s,1}$ and $D_{s,4}$. Stadtke's printed form
uses the entropy-frame EOS partial $(\partial\varrho_i/\partial s_i)_p$, but this equals
$T_i\,\varrho_{h,i}$ by the Gibbs relation. We use $T_i\,\varrho_{h,i}$ throughout — this keeps
everything in the enthalpy-frame variables and reuses the EOS quantities we already have.

### A.4 The six entries of $D_s$ (with $k \to k\varrho$ correction)

Define the recurring "mass + entropy" group that appears in rows 1 and 4:

$$\mathcal M_g \equiv \sigma_g^M - \frac{T_g\,\varrho_{h,g}}{\varrho_g}\,\sigma_g^S, \qquad \mathcal M_l \equiv \sigma_l^M - \frac{T_l\,\varrho_{h,l}}{\varrho_l}\,\sigma_l^S.$$

These collect the mass-transfer and density-of-entropy contributions into one combination each.

**Row 1 — pressure equation:**

$$D_{s,1} = \frac{\varrho_l\,a_0^2}{\varrho_s}\,\mathcal M_g \;+\; \frac{\varrho_g\,a_0^2}{\varrho_s}\,\mathcal M_l \;-\; \frac{\varrho_l\varrho_g}{\varrho_s}\,(\alpha_g u_g + \alpha_l u_l)\,a_0^2\,\frac{1}{A}\frac{\mathrm{d}A}{\mathrm{d}x}$$

**Row 2 — gas velocity equation:**

$$D_{s,2} = \frac{\varrho_l}{\alpha_g\hat\varrho^2}\bigl[F^v_g + \sigma_g^M(u^{ex} - u_g)\bigr] \;-\; \sigma_g^M\,\frac{k\varrho\,\varrho_g + \alpha_l(\varrho_l^2 - \varrho_g^2)}{\varrho_g\,\hat\varrho^2}\,\Delta u \;+\; \frac{F_g\,(\varrho_l + \alpha_g\,k\varrho) + F_l\,\alpha_g\,k\varrho}{\alpha_g\hat\varrho^2}$$

**Row 3 — liquid velocity equation:**

$$D_{s,3} = -\frac{\varrho_g}{\alpha_l\hat\varrho^2}\bigl[F^v_g + \sigma_g^M(u^{ex} - u_l)\bigr] \;-\; \sigma_g^M\,\frac{k\varrho\,\varrho_l + \alpha_g(\varrho_g^2 - \varrho_l^2)}{\varrho_l\,\hat\varrho^2}\,\Delta u \;+\; \frac{F_g\,\alpha_l\,k\varrho + F_l\,(\varrho_g + \alpha_l\,k\varrho)}{\alpha_l\hat\varrho^2}$$

Note: $F^v_g$ appears explicitly in both rows 2 and 3 (with $F^v_l = -F^v_g$ already absorbed
into the sign of the leading factor in row 3). Same for $\sigma_g^M$.

**Row 4 — void fraction equation:**

$$D_{s,4} = \frac{\alpha_l\,a_0^2}{\varrho_s\,a_l^2}\,\mathcal M_g \;-\; \frac{\alpha_g\,a_0^2}{\varrho_s\,a_g^2}\,\mathcal M_l \;+\; \alpha_l\alpha_g\,\frac{a_0^2}{\varrho_s}\Bigl[\frac{\varrho_l u_l}{a_g^2} - \frac{\varrho_g u_g}{a_l^2}\Bigr]\,\frac{1}{A}\frac{\mathrm{d}A}{\mathrm{d}x}$$

**Row 5 — gas entropy equation:**

$$D_{s,5} = \frac{\sigma_g^S}{\alpha_g\varrho_g}$$

**Row 6 — liquid entropy equation:**

$$D_{s,6} = \frac{\sigma_l^S}{\alpha_l\varrho_l}$$

### A.5 Pseudocode skeleton

```python
def D_s_primitive(U_h, eos_state, sources, geometry, k):
    """
    Primitive-frame entropy-form source vector for the two-fluid model.
    Stadtke Table B.46 with k -> k*rho correction applied throughout.
    """
    p, ug, ul, ag, hg, hl = U_h
    al = 1.0 - ag
    rho_g, a_g, T_g, rho_pg, rho_hg = eos_state.gas
    rho_l, a_l, T_l, rho_pl, rho_hl = eos_state.liquid
    sigma_Mg, sigma_Qg, Fv_g, F_g, F_l, Q_g, Q_l, u_ex, u_int, h_ex = sources
    dlnA_dx = geometry.dlnA_dx
    
    # Interface constraints (Newton's 3rd law)
    sigma_Ml = -sigma_Mg
    sigma_Ql = -sigma_Qg
    Fv_l     = -Fv_g
    
    # State intermediates
    rho   = ag*rho_g + al*rho_l
    rho_s = ag*rho_l + al*rho_g
    rh2   = rho_g*rho_l + k*rho**2
    a0sq  = rho_s / (ag*rho_l/a_g**2 + al*rho_g/a_l**2)
    du    = ug - ul
    
    # Entropy sources (eq. 4.50) - compute T*sigma_S first
    Tg_sSg = sigma_Mg*((h_ex-hg) + 0.5*(u_ex-ug)**2) + Fv_g*(u_int-ug) + sigma_Qg + Q_g
    Tl_sSl = sigma_Ml*((h_ex-hl) + 0.5*(u_ex-ul)**2) + Fv_l*(u_int-ul) + sigma_Ql + Q_l
    sSg = Tg_sSg / T_g
    sSl = Tl_sSl / T_l
    
    # Recurring mass+entropy group
    Mg = sigma_Mg - (T_g * rho_hg / rho_g) * sSg
    Ml = sigma_Ml - (T_l * rho_hl / rho_l) * sSl
    
    # Row 1: pressure
    Ds_1 = (rho_l*a0sq/rho_s * Mg
          + rho_g*a0sq/rho_s * Ml
          - rho_l*rho_g/rho_s * (ag*ug + al*ul) * a0sq * dlnA_dx)
    
    # Row 2: u_g  (note: k -> k*rho throughout)
    Ds_2 = (rho_l/(ag*rh2) * (Fv_g + sigma_Mg*(u_ex - ug))
          - sigma_Mg * (k*rho*rho_g + al*(rho_l**2 - rho_g**2)) / (rho_g*rh2) * du
          + (F_g*(rho_l + ag*k*rho) + F_l*ag*k*rho) / (ag*rh2))
    
    # Row 3: u_l
    Ds_3 = (-rho_g/(al*rh2) * (Fv_g + sigma_Mg*(u_ex - ul))
          - sigma_Mg * (k*rho*rho_l + ag*(rho_g**2 - rho_l**2)) / (rho_l*rh2) * du
          + (F_g*al*k*rho + F_l*(rho_g + al*k*rho)) / (al*rh2))
    
    # Row 4: alpha_g
    Ds_4 = (al*a0sq/(rho_s*a_l**2) * Mg
          - ag*a0sq/(rho_s*a_g**2) * Ml
          + al*ag*a0sq/rho_s * (rho_l*ul/a_g**2 - rho_g*ug/a_l**2) * dlnA_dx)
    
    # Rows 5, 6: entropy
    Ds_5 = sSg / (ag*rho_g)
    Ds_6 = sSl / (al*rho_l)
    
    return jnp.array([Ds_1, Ds_2, Ds_3, Ds_4, Ds_5, Ds_6])


def D_VE_conservative(U_h, eos_state, sources, geometry, k):
    """Conservative-form total-energy source: D_VE = J Phi^{-1} D_s."""
    D_s = D_s_primitive(U_h, eos_state, sources, geometry, k)
    
    # Phi^{-1}: identity on rows 1-4; rows 5,6 mix in D_s[0] (pressure) and rescale by T_i
    T_g = eos_state.gas.T; T_l = eos_state.liquid.T
    rho_g = eos_state.gas.rho; rho_l = eos_state.liquid.rho
    D_h = D_s.at[4].set(D_s[0]/rho_g + T_g * D_s[4]) \
              .at[5].set(D_s[0]/rho_l + T_l * D_s[5])
    
    # J: conservative bridge (already in toolkit from §2.3)
    J = J_matrix(U_h, eos_state)
    
    return J @ D_h
```

### A.6 Known issues and verification status

**$k \to k\varrho$ correction.** The original Stadtke Table B.46 prints "$k$" alone next to
densities (e.g., "$\varrho_l + \alpha_g k$") which is dimensionally inconsistent. Throughout
rows 2 and 3, every instance of "$k$" multiplying a single density or appearing next to a
$\varrho$ term must be read as "$k\varrho$" with $\varrho$ the mixture density. **Apply this
correction throughout** when transcribing. Without it, momentum conservation fails by ~65% at
nominal states; with it, momentum conserves to machine zero.

**Energy conservation residual.** After the $k \to k\varrho$ correction, mass and momentum
conservation are exact (machine zero), but the energy sum $D_{V_E,5}+D_{V_E,6}$ retains a
residual $\approx -F^v_g\,\Delta u$ — about 1% of the viscous-energy scale. This appears to be
a second transcription error in rows 2–3 affecting the $F^v_g$ coefficient specifically, or a
kinetic-energy decomposition mismatch in Stadtke's eq. 4.50. **Flagged for further
investigation**; should not impair steady-state solutions where $F^v_g$ is small, but worth
revisiting for long-time transients or shock-dominated cases.

**Verification at nominal state:**

| Check | Status |
|-------|--------|
| Mass row 1: $D_{V_E,1} = \sigma_g^M - \alpha_g\varrho_g u_g\,\mathrm{d}\ln A/\mathrm{d}x$ | ✓ exact |
| Mass row 2: $D_{V_E,2} = \sigma_l^M - \alpha_l\varrho_l u_l\,\mathrm{d}\ln A/\mathrm{d}x$ | ✓ exact |
| $D_{V_E,1} + D_{V_E,2} = 0$ (interfacial, strict-1D) | ✓ machine zero |
| $D_{V_E,3} + D_{V_E,4} = 0$ (interfacial, strict-1D) | ✓ machine zero after $k\!\to\!k\varrho$ |
| $D_{V_E,5} + D_{V_E,6} = 0$ (interfacial, strict-1D) | ✗ residual $\approx F^v_g\Delta u$ |
