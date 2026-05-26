# Nozzle-hyperbolic: governing equations and numerical method

## 1. Governing equations

### 1.1 Quasi-1D Euler equations

The compressible flow in the nozzle is governed by the one-dimensional Euler equations in conservative form. For a channel with variable cross-sectional area $A(x)$, the natural form is obtained by integrating over the cross-section, giving fluxes multiplied by area:

$$\frac{\partial (\mathbf{Q}A)}{\partial t} + \frac{\partial (\mathbf{F}A)}{\partial x} = \mathbf{S}_A$$

with conservative variables, fluxes, and source terms:

$$\mathbf{Q} = \begin{bmatrix} \rho \\ \rho u \\ \rho E \end{bmatrix}, \qquad \mathbf{F} = \begin{bmatrix} \rho u \\ \rho u^2 + p \\ (\rho E + p)u \end{bmatrix}, \qquad \mathbf{S}_A = \begin{bmatrix} 0 \\ p\dfrac{dA}{dx} - \tau_w\dfrac{4A}{D_h} \\ 0 \end{bmatrix}$$

The momentum source $p\,dA/dx$ arises from the pressure force on the variable-area walls. The friction term $\tau_w\,4A/D_h$ decelerates the flow. The energy source is zero: at a no-slip adiabatic wall $u_{wall}=0$, so friction does no work on the fluid boundary, and total enthalpy flux $(\rho H u A)$ is conserved. This is consistent with Fanno flow: $H = h + u^2/2 = \mathrm{const}$ for adiabatic flow with friction.

### 1.2 Thermodynamic relations

**Perfect gas:**

- $p = \rho R T$, $R = c_p - c_v$, $\gamma = c_p/c_v$
- $p = (\gamma-1)\!\left(\rho E - \frac{(\rho u)^2}{2\rho}\right)$
- $c = \sqrt{\gamma p/\rho}$

**Real gas (via jaxprop / CoolProp):**

- State $(p, c, s, T, h, e)$ is computed from $(\rho,\, e)$ where $e = E - u^2/2$
- `jaxprop.Fluid`: CoolProp-backed, supports `DmassUmass_INPUTS` directly
- `jaxprop.FluidBicubic`: bicubic-interpolated table in $(h, \log p)$; only `HmassP_INPUTS` and `DmassP_INPUTS` are direct lookups. The inversion $(\rho, e) \to (p, c)$ requires a nested bisection (15 iterations, vectorised with JAX vmap). $2^{-15}\approx 3\times10^{-5}$ relative precision — adequate for solver tolerances $\sim10^{-4}$.
- Selected via `fluid.use_lookup_table: true/false` in `settings.yaml`. When `false`, `jxp.Fluid` (direct HEOS) is used instead — no table build required but slower.

### 1.3 Derived quantities

- Velocity: $u = \rho u / \rho$
- Total enthalpy: $H = E + p/\rho$
- Specific internal energy: $e = E - u^2/2$
- Mach number: $M = u/c$
- Mass flux: $\dot{m} = \rho u A$ (constant at steady state — key diagnostic)

---

## 2. Finite volume discretization

### 2.1 Spatial discretization

The domain is divided into $N_x$ cells with uniform width $\Delta x$. Cell $i$ has centre $x_i$, interfaces at $x_{i\pm1/2}$, and cross-sectional area $A_i$. Two ghost cells (indices 0 and $N-1$) hold boundary conditions.

Integrating the governing equations over cell $i$ and dividing by $A_i\Delta x$:

$$\frac{d\mathbf{Q}_i}{dt} = -\frac{(F\,A)_{i+1/2} - (F\,A)_{i-1/2}}{A_i\,\Delta x} + \frac{\mathbf{S}_{A,i}}{A_i}$$

**Face area.** Interface areas are taken as the arithmetic mean of adjacent cell areas:

$$A_{i+1/2} = \tfrac{1}{2}(A_i + A_{i+1})$$

This is the only consistent choice: using $A_i$ for the left state and $A_{i+1}$ for the right state inside a Riemann solver introduces a spurious non-zero source in the mass equation even at the exact steady state (because $\rho A$ varies even when $\rho u A$ is constant), leading to O(1) mass-flux deviations across the convergent section.

The per-unit-volume source term is:

$$\frac{\mathbf{S}_{A,i}}{A_i} = \begin{bmatrix} 0 \\[4pt] \dfrac{p_i}{A_i}\dfrac{dA}{dx}\bigg|_i - \dfrac{4\tau_w}{D_h}\bigg|_i \\[8pt] 0 \end{bmatrix}$$

The area gradient is discretised with central differences for interior cells:

$$\left.\frac{dA}{dx}\right|_i \approx \frac{A_{i+1} - A_{i-1}}{2\,\Delta x}$$

and one-sided differences at the boundaries.

### 2.2 Flux schemes

All flux functions return the area-weighted interface flux $(F\,A)_{i+1/2}$. The Riemann problem is solved using states $\mathbf{Q}_L, \mathbf{Q}_R$ reconstructed at the interface, and the result is scaled by $A_{i+1/2}$:

$$(F\,A)_{i+1/2} = \mathbf{F}_{num}(\mathbf{Q}_L, \mathbf{Q}_R) \cdot A_{i+1/2}$$

#### Rusanov (local Lax-Friedrichs)

$$\mathbf{F}_{Rus} = \tfrac{1}{2}\bigl[\mathbf{F}(\mathbf{Q}_L) + \mathbf{F}(\mathbf{Q}_R) - a_{\max}(\mathbf{Q}_R - \mathbf{Q}_L)\bigr]$$

where $a_{\max} = \max(|u_L|+c_L,\,|u_R|+c_R)$.

#### HLL

$$\mathbf{F}_{HLL} = \begin{cases} \mathbf{F}_L & S_L \ge 0 \\ \dfrac{S_R\,\mathbf{F}_L - S_L\,\mathbf{F}_R + S_L S_R(\mathbf{Q}_R - \mathbf{Q}_L)}{S_R - S_L} & S_L < 0 < S_R \\ \mathbf{F}_R & S_R \le 0 \end{cases}$$

Wave speeds:
$S_L = \min(u_L - c_L,\; u_R - c_R)$, $\quad S_R = \max(u_L + c_L,\; u_R + c_R)$.

#### HLLC

Three-wave solver that resolves the contact discontinuity. The contact speed is:

$$S_M = \frac{p_R - p_L + \rho_L u_L(S_L - u_L) - \rho_R u_R(S_R - u_R)}{\rho_L(S_L - u_L) - \rho_R(S_R - u_R)}$$

Star states:
$$\rho_{L/R}^* = \rho_{L/R}\frac{S_{L/R} - u_{L/R}}{S_{L/R} - S_M}, \quad p_{L/R}^* = p_{L/R} + \rho_{L/R}(S_{L/R} - u_{L/R})(S_M - u_{L/R})$$

For real gas, the star-state internal energy is obtained from $(\rho^*, p^*)$ via EOS.  
The flux selector:

$$\mathbf{F}_{HLLC} = \begin{cases} \mathbf{F}_L & S_L > 0 \\ \mathbf{F}_L + S_L(\mathbf{Q}_L^* - \mathbf{Q}_L) & S_L \le 0,\; S_M \ge 0 \\ \mathbf{F}_R + S_R(\mathbf{Q}_R^* - \mathbf{Q}_R) & S_M < 0,\; S_R \ge 0 \\ \mathbf{F}_R & S_R < 0 \end{cases}$$

#### Roe

$$\mathbf{F}_{Roe} = \tfrac{1}{2}\bigl[\mathbf{F}(\mathbf{Q}_L) + \mathbf{F}(\mathbf{Q}_R) - \mathbf{A}_{Roe}(\mathbf{Q}_R - \mathbf{Q}_L)\bigr]$$

Roe-averaged state (weighted by $\sqrt{\rho}$):

$$\bar{u} = \frac{\sqrt{\rho_L}\,u_L + \sqrt{\rho_R}\,u_R}{\sqrt{\rho_L}+\sqrt{\rho_R}}, \qquad \bar{H} = \frac{\sqrt{\rho_L}\,H_L + \sqrt{\rho_R}\,H_R}{\sqrt{\rho_L}+\sqrt{\rho_R}}, \qquad \bar{c} = \sqrt{(\gamma-1)\!\left(\bar{H}-\bar{u}^2/2\right)}$$

Eigenvalues: $\lambda_{1,3} = \bar{u}\mp\bar{c}$, $\lambda_2 = \bar{u}$.

Entropy fix (Harten):
$$|\lambda|_{\varepsilon} = \begin{cases} |\lambda| & |\lambda|\ge\delta \\ \frac{|\lambda|^2+\delta^2}{2\delta} & \text{otherwise}\end{cases}, \qquad \delta = \max\!\bigl(|\lambda_R-\lambda_L|,\;0.01\bar{c}\bigr)$$

Note: Roe uses a scalar cell-by-cell loop and is not JIT-compiled; use HLL/HLLC/Rusanov for the fast JAX path.

#### Lax-Friedrichs

$$\mathbf{F}_{LF} = \tfrac{1}{2}\bigl[\mathbf{F}_L + \mathbf{F}_R - (\Delta x/\Delta t)(\mathbf{Q}_R - \mathbf{Q}_L)\bigr]$$

---

## 3. Time integration

### 3.1 CFL time step

$$\Delta t = \mathrm{CFL}\cdot\frac{\Delta x}{\max_i(|u_i|+c_i)}$$

For explicit schemes: CFL $\le 1$ (MUSCL + Euler requires CFL $\le 0.5$).  
For the diagonal implicit scheme: CFL can be $\gg 1$.

### 3.2 Forward Euler

$$\mathbf{Q}_i^{n+1} = \mathbf{Q}_i^n + \Delta t\!\left[\mathcal{R}(\mathbf{Q}^n)_i + \frac{\mathbf{S}_{A,i}}{A_i}\right]$$

where $\mathcal{R}_i = -\bigl[(F\,A)_{i+1/2} - (F\,A)_{i-1/2}\bigr]\,/\,(A_i\Delta x)$.

### 3.3 SSP Runge-Kutta 3 (Shu-Osher)

$$\mathbf{Q}^{(1)} = \mathbf{Q}^n + \Delta t\,\mathcal{L}(\mathbf{Q}^n)$$

$$\mathbf{Q}^{(2)} = \tfrac{3}{4}\mathbf{Q}^n + \tfrac{1}{4}\mathbf{Q}^{(1)} + \tfrac{1}{4}\Delta t\,\mathcal{L}(\mathbf{Q}^{(1)})$$

$$\mathbf{Q}^{n+1} = \tfrac{1}{3}\mathbf{Q}^n + \tfrac{2}{3}\mathbf{Q}^{(2)} + \tfrac{2}{3}\Delta t\,\mathcal{L}(\mathbf{Q}^{(2)})$$

where $\mathcal{L} = \mathcal{R} + \mathbf{S}_A/A$.

### 3.4 Classic Runge-Kutta 4

$$\mathbf{Q}^{n+1} = \mathbf{Q}^n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

with $k_1 = \mathcal{L}(\mathbf{Q}^n)$, $k_2 = \mathcal{L}(\mathbf{Q}^n + \tfrac{1}{2}\Delta t\,k_1)$, etc.

### 3.5 Diagonal implicit Euler

For accelerated convergence to steady state, a point-implicit correction is applied to the explicit Euler update. Defining the local spectral radius $\lambda_i = |u_i| + c_i$ and face-maximum $\lambda_{i+1/2} = \max(\lambda_i, \lambda_{i+1})$, the stiffness denominator is:

$$d_i = 1 + \frac{\Delta t_i}{\Delta x}\bigl(\lambda_{i-1/2} + \lambda_{i+1/2}\bigr)$$

The update becomes:

$$\mathbf{Q}_i^{n+1} = \mathbf{Q}_i^n + \frac{\Delta t_i}{d_i}\!\left[\mathcal{R}_i + \frac{\mathbf{S}_{A,i}}{A_i}\right]$$

This is a first-order-accurate diagonal approximation to the full implicit system (no matrix inversion). It allows CFL $\gg 1$ (typically 10–100) and drastically reduces iteration count to steady state. Set `time_discretization: implicit` and raise `CFL` in `settings.yaml`.

### 3.6 Local Time Stepping (LTS)

Instead of advancing all cells with the global minimum time step $\Delta t = \mathrm{CFL}\cdot\Delta x/\max_i\lambda_i$, each cell is updated with its own local step:

$$\Delta t_i = \mathrm{CFL}\cdot\frac{\Delta x}{\lambda_i}, \qquad \lambda_i = |u_i| + c_i$$

Subsonic cells far from the throat use a much larger $\Delta t_i$, removing the throat bottleneck and accelerating convergence to steady state. The final steady-state solution is unchanged — LTS only affects the transient path.

**Note for diagonal implicit:** For large CFL the denominator $d_i \approx \Delta t_i(\lambda_{i-1/2}+\lambda_{i+1/2})/\Delta x$, so the effective update is $\Delta t_i / d_i \approx \Delta x / (\lambda_{i-1/2}+\lambda_{i+1/2})$, which is independent of $\Delta t_i$. LTS therefore has negligible effect on the implicit scheme but is beneficial for explicit schemes (Euler, RK3, RK4).

---

## 4. Reconstruction schemes

**Constant (piecewise-constant, 1st order):** $\mathbf{Q}_L = \mathbf{Q}_i$, $\mathbf{Q}_R = \mathbf{Q}_{i+1}$.

**MUSCL (piecewise-linear, 2nd order):**

$$\mathbf{Q}_{L,i+1/2} = \mathbf{Q}_i + \tfrac{1}{2}\sigma_i, \qquad \mathbf{Q}_{R,i+1/2} = \mathbf{Q}_{i+1} - \tfrac{1}{2}\sigma_{i+1}$$

where $\sigma_i$ is the TVD-limited slope. Ghost cells use zero slope to pass BC values unchanged.

**Limiter choices** (`reconstruction` key in `settings.yaml`):

| Key | Description |
| --- | --- |
| `constant` | No reconstruction — 1st-order Godunov |
| `muscl_minmod` | Minmod: most dissipative, picks the smaller of the two adjacent slopes |
| `muscl_van_leer` | Van Leer: smooth harmonic mean, good compromise between accuracy and monotonicity |
| `muscl_superbee` | Superbee: least dissipative, can sharpen contact waves but may overshoot |

For the real-gas JAX path, reconstruction is performed in **primitive variables** $(\rho, u, p)$ rather than conservative variables. After limiting, `DmassP_INPUTS` is called once per interface to recover $e$ and $c$, avoiding the expensive $(\rho,e)\to p$ inversion at each interface.

---

## 5. Boundary conditions

### 5.1 Inlet (subsonic, stagnation conditions given)

Velocity is linearly extrapolated from the two nearest interior cells: $u_{in} = 2u_1 - u_2$.

**Perfect gas:**
$$T_s = T_0 - \frac{u_{in}^2}{2c_p}, \quad p_s = \frac{p_0}{\left(1+\frac{\gamma-1}{2}M_{in}^2\right)^{\gamma/(\gamma-1)}}, \quad \rho_s = \frac{p_s}{R T_s}$$

**Real gas:**
$$h_s = h_0 - \tfrac{1}{2}u_{in}^2, \quad (\rho_s,\,e_s) = \mathrm{EOS}(h_s,\,s_0)$$

The isentropic state at $(h_s, s_0)$ is found by a **12-iteration bisection** over $p$ using `HmassP_INPUTS` (vectorised as a 1-element array so the JAX vmap path is exercised). $2^{-12}\approx 2\times10^{-4}$ relative precision; entropy is monotone decreasing in $p$ at constant $h$, so the bracket always converges.

Ghost cell: $\mathbf{Q}_0 = [\rho_s,\; \rho_s u_{in},\; \rho_s(e_s + u_{in}^2/2)]^T$.

### 5.2 Outlet (subsonic, static pressure given)

Density and momentum flux $\rho A$ are extrapolated (area-weighted linear):
$$(\rho A)_{ghost} = 2(\rho A)_{N-2} - (\rho A)_{N-3}$$

which is equivalent to $\rho_{ghost} = 2\rho_{N-2}A_{N-2}/A_{N-1} - \rho_{N-3}A_{N-3}/A_{N-1}$.  
This extrapolation of $\rho A$ (not $\rho$) enforces a consistent mass flux across the outlet face.

Total energy is set to enforce $p_{out}$:
- Perfect gas: $\rho E = p_{out}/(\gamma-1) + \tfrac{1}{2}(\rho u)^2/\rho$
- Real gas: $e_{out}$ from EOS at $(\rho_{out}, p_{out})$, then $\rho E = \rho_{out}(e_{out}+u_{out}^2/2)$

For supersonic outlet ($M\ge 1$), all variables are extrapolated without pressure fixing.

---

## 6. Real-gas EOS path (JAX)

For `perfect_gas: false` with a non-Roe flux method, the solver uses a fully `@jax.jit`-compiled step and a `jax.lax.while_loop` convergence loop — zero Python↔XLA synchronisation between iterations.

### 6.1 Per-step operations

Each compiled step executes the following in order:

1. **BCs** inside JIT (inlet: 12-iter $p$-bisection at fixed $h_s, s_0$; outlet: `DmassP_INPUTS`)
2. **EOS inversion** $(\rho, e) \to (p, c, \mu)$: **15-iteration** JAX bisection over $p$, bracket tightened with table $h$-range ($2^{-15}\approx 3\times10^{-5}$ relative error). All $N_x$ cells vectorised in one XLA kernel.
3. **Geometry source**: $\mathbf{S}_A$ uses $(\mathrm{d}A/\mathrm{d}x)/A$ precomputed once at solver construction — the array is a closure constant inside the compiled step, so no allocation or computation occurs at run time.
4. **Friction**: Haaland equation in `jnp` ops from $(Re,\,\varepsilon/D_h)$.
5. **Flux**: Rusanov / HLL / HLLC / LF, all in JAX.
6. **LTS update**: each cell uses its own $\Delta t_i = \mathrm{CFL}\cdot\Delta x/\lambda_i$ (see §3.6).

### 6.2 Compiled convergence loop

The outer iteration is a `jax.lax.while_loop` — the entire convergence loop stays on-device with no Python round-trips:

- **Convergence metric (SSR):** Steady-State Residual, computed free-of-charge each step as a by-product of the already-evaluated RHS:

$$\mathrm{SSR} = \sqrt{\left\langle\left(\frac{t_{\min}\,[\mathcal{R}(\mathbf{Q}) + \mathbf{S}/A]}{|\mathbf{Q}|}\right)^2\right\rangle_{\text{interior}}}, \qquad t_{\min} = \frac{\Delta x}{\max_i\lambda_i}$$

Multiplying by $t_{\min}$ makes the residual dimensionless and CFL-independent: SSR$\to 0$ exactly at steady state, regardless of how large CFL is. Convergence is declared when SSR $< \varepsilon_{tol}$.

- **Plateau detection:** if the SSR history over the last 5 iterations spans less than $2\times$ and SSR $< 10\,\varepsilon_{tol}$, the loop exits to avoid stalling on a plateau.

- **Progress printing:** `jax.debug.print` + `jax.lax.cond` inside the loop; host sync occurs only at print frequency (every 1 / 10 / 50 iterations for early / mid / late), not every step.

### 6.3 Warmup phase

An optional warmup phase runs $N_{wu}$ iterations with cheaper settings (typically Rusanov + constant reconstruction) before the main loop, using `jax.lax.fori_loop` for zero Python overhead. This establishes a smooth initial field that dramatically reduces the iteration count of the more expensive main flux (HLLC, MUSCL, etc.). Configured in `settings.yaml` under `numerical_settings.warmup`.

`jax.config.update("jax_enable_x64", True)` must be called before any JAX or jaxprop import to ensure 64-bit precision throughout.

---

## 7. Isentropic initialisation

### Perfect gas

Analytical subsonic isentropic solution with mass flow set by the critical (throat) state. A `brentq` root-find per cell inverts $\dot{m}(u) = \rho(u)\,u\,A$.

### Real gas

1. Build an isentropic table: sweep $h$ from $h_{lo}$ to $h_0$ (stagnation enthalpy); for each $h$ find $p$ such that $s(h,p) = s_0$ via 50-iteration vectorised bisection.
2. Compute $G = \rho\,u = \rho\sqrt{2(h_0-h)}$ along the isentrope; the throat is at $G_{max}$.
3. For each cell, interpolate $h$ and $p$ from the isentropic table at the target $G = \dot{m}/A_i$, using the subsonic or supersonic branch depending on position relative to the throat.
4. One batch `HmassP_INPUTS` call retrieves all cell states.

---

## 8. Friction model

Wall shear stress:

$$\tau_w = \frac{C_f}{8}\,\rho u^2, \qquad C_f = f_{Haaland}(Re,\,\varepsilon/D_h)$$

Haaland equation for the Darcy friction factor:

$$\frac{1}{\sqrt{f}} = -1.81\log_{10}\!\left[\left(\frac{\varepsilon/D_h}{3.7}\right)^{1.11} + \frac{6.9}{Re}\right]$$

Hydraulic diameter for planar nozzle of half-height $h$ and width $w$:

$$D_h = \frac{4A}{P} = \frac{4\cdot 2hw}{2(w+2h)}$$

Reynolds number: $Re = \rho\,u\,D_h/\mu$.

**Two-phase flow (HEM, Beattie viscosity):**

$$\mu_{mix} = \mu_L(1-y)(1+2.5y) + \mu_v\,y, \qquad y = \frac{\rho_L x}{\rho_L x + \rho_v(1-x)}$$

where $x$ is vapour quality and $y$ is void fraction.

---

## 9. Nozzle geometry

The profile $r(x)$ (half-height for planar, radius for axisymmetric) is fitted from experimental data using a spline or polynomial.  A piecewise definition handles the constant inlet section:

$$r(x) = \begin{cases} r_{in} & x < x_{throat,\,inlet} \\ r_{spline}(x) & x \ge x_{throat,\,inlet} \end{cases}$$

**Planar** (rectangular channel of depth $w$):
$$A = 2\,r\,w, \qquad D_h = \frac{4A}{2(w+2r)}$$

**Axisymmetric** (conical/circular cross-section):
$$A = \pi r^2, \qquad D_h = 2r$$

For the Lettieri nozzle $w = 6.48\;\mathrm{mm}$; selected via `nozzle_parameters.type: planar | axisymmetric` in `settings.yaml`.

---

## 10. Convergence and spatial residuals

### Steady-State Residual (SSR) — stopping criterion

The convergence metric used in the JAX path is the normalised steady-state residual:

$$\mathrm{SSR} = \sqrt{\frac{1}{3(N_x-2)}\sum_{k=1}^{3}\sum_{i=1}^{N_x-2} \left(\frac{t_{\min}\,[\mathcal{R}_i(\mathbf{Q}) + S_{k,i}/A_i]}{|Q_{k,i}|}\right)^2}$$

where $t_{\min} = \Delta x / \max_i(|u_i|+c_i)$ is the physical advection time across one cell at the throat. The factor $t_{\min}$ makes the residual dimensionless and **CFL-independent**: SSR$\to 0$ exactly at steady state for any CFL. Ghost cells (BC cells, indices 0 and $N_x-1$) are excluded — their values are set by BCs, not by flux conservation. Convergence is declared when SSR $<\varepsilon_{tol}$ (`residuals` in `settings.yaml`).

### Spatial conservation residuals (post-processing diagnostic)

At steady state three quantities should be uniform across the domain:

| Quantity | Expression | Physical meaning |
| --- | --- | --- |
| Mass flux | $\dot{m}(x) = \rho u A$ | Continuity |
| Total enthalpy | $H(x) = h + u^2/2$ | Adiabatic energy conservation |
| Entropy (real gas) | $s(x)$ | Isentropic flow (rises only across shocks/friction) |

Reported as max percentage deviation from the spatial mean over **interior cells only** (indices 1 to $N_x-2$), and plotted as deviation profiles. Ghost cells are excluded because their mass flow is set by BCs, not by flux conservation. Large deviations indicate insufficient grid resolution or unresolved BC transients, not a scheme error (provided the face-area fix is applied).

---

## 11. Variable definitions

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\rho$ | density | kg/m³ |
| $u$ | velocity | m/s |
| $p$ | pressure | Pa |
| $T$ | temperature | K |
| $e$ | specific internal energy | J/kg |
| $E$ | specific total energy ($e + u^2/2$) | J/kg |
| $H$ | specific total enthalpy ($E + p/\rho$) | J/kg |
| $h$ | specific static enthalpy ($e + p/\rho$) | J/kg |
| $s$ | specific entropy | J/(kg·K) |
| $c$ | speed of sound | m/s |
| $M$ | Mach number | — |
| $x$ | vapour quality | — |
| $\gamma$ | heat capacity ratio | — |
| $A$ | cross-sectional area | m² |
| $D_h$ | hydraulic diameter | m |
| $\tau_w$ | wall shear stress | Pa |
| $C_f$ | skin friction coefficient | — |
| $\varepsilon$ | wall roughness | m |
| $\dot{m}$ | mass flow rate | kg/s |

---

## 12. Settings reference (`settings.yaml`)

### Numerical settings

| Key | Options / type | Notes |
| --- | --- | --- |
| `flux_method` | `hll`, `hllc`, `rusanov`, `lf`, `roe` | `roe` bypasses JAX path |
| `time_discretization` | `euler`, `RK3`, `RK4`, `implicit` | `implicit` allows CFL ≫ 1 |
| `reconstruction` | `constant`, `muscl_minmod`, `muscl_van_leer`, `muscl_superbee` | MUSCL + Euler: CFL ≤ 0.5 |
| `CFL` | float | ~0.2 explicit; 10–100 implicit |
| `residuals` | float | SSR convergence threshold (e.g. `1e-4`) |
| `max_iterations` | int | Hard iteration cap |

### Warmup phase (`numerical_settings.warmup`)

Optional block; omit entirely to skip warmup. When present, a `jax.lax.fori_loop` runs the specified number of iterations with cheap settings before the main while_loop.

| Key | Default | Notes |
| --- | --- | --- |
| `n_iterations` | 200 | Number of warmup iterations |
| `flux_method` | `rusanov` | Cheap flux for warmup |
| `reconstruction` | `constant` | Reconstruction during warmup |
| `CFL` | (main CFL) | Can be equal to main CFL |

### Fluid settings (`fluid`)

| Key | Options / type | Notes |
| --- | --- | --- |
| `perfect_gas` | `true`/`false` | Selects EOS and code path |
| `fluid_name` | CoolProp string | e.g. `CO2`, `Water`, `Nitrogen` |
| `use_lookup_table` | `true`/`false` | `true` → `FluidBicubic` (fast, needs table build); `false` → `jxp.Fluid` (direct HEOS, always available) |
| `lut.h_min/h_max` | float (J/kg) | Enthalpy range of the bicubic table |
| `lut.p_min/p_max` | float (Pa) | Pressure range of the bicubic table |
| `lut.N_h`, `N_p` | int | Table resolution (200×200 recommended) |
| `lut.table_dir` | path string | Directory where table binaries are cached |

---

## 13. References

- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
- Roe, P. L. (1981). Approximate Riemann solvers, parameter vectors, and difference schemes. *JCP*, 43, 357–372.
- Gottlieb, S., Shu, C.-W., & Tadmor, E. (2001). Strong stability-preserving high-order time discretization methods. *SIAM Review*, 43(1), 89–112.
- Haaland, S. E. (1983). Simple and explicit formulas for the friction factor in turbulent pipe flow. *J. Fluids Eng.*, 105(1), 89–90.
- Beattie, D. R. H., & Whalley, P. B. (1982). A simple two-phase frictional pressure drop calculation method. *Int. J. Multiphase Flow*, 8(1), 83–87.
