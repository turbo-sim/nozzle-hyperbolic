# Städtke-Style Implicit FVS Scheme for Quasi-1D Compressible Flow with Variable Area

## Theoretical Framework and Mathematical Derivation

This document develops the full mathematical framework for solving the
quasi-one-dimensional compressible Euler equations with variable cross
section, using:

- **Primitive variables**: $(p, u, h)$ — pressure, velocity, specific enthalpy
- **Conservative variables**: $(\rho, \rho u, \rho E)$ — for the discrete update
- **Flux Vector Splitting** based on a linearized Riemann problem in flux variables (Städtke 2006, §7.5–8.2)
- **MUSCL reconstruction** with Van Leer slope limiter
- **Implicit time integration** via Newton-Raphson with JAX automatic differentiation
- **Equation of state**: any single-phase or homogeneous-equilibrium-mixture (HEM) EOS via JAXProp/CoolProp

The framework generalizes seamlessly from pure single-phase gases (ideal or real) to HEM two-phase mixtures with no code changes — only EOS calls differ.

References to Städtke's textbook *Gasdynamic Aspects of Two-Phase Flow* (Wiley-VCH, 2006) are given throughout as "book §X" or "book eq. Y.Z".

---

## 1. Governing Equations

### 1.1 Conservation form (quasi-1D with variable area)

For a duct of slowly-varying cross-section $A(x)$, integrating the 3D Euler equations over the cross-section gives (book §2.2, eqs. 2.30–2.32; Toro 2009 §1.6.2, eqs. 1.95–1.98):

$$
\frac{\partial (\rho A)}{\partial t} + \frac{\partial (\rho u A)}{\partial x} = 0
$$

$$
\frac{\partial (\rho u A)}{\partial t} + \frac{\partial \big[(\rho u^2 + p) A\big]}{\partial x} = p\,\frac{dA}{dx}
$$

$$
\frac{\partial (\rho E A)}{\partial t} + \frac{\partial \big[u(\rho E + p) A\big]}{\partial x} = 0
$$

where $\rho E = \rho(e + u^2/2)$ is the total energy per unit volume and $e$ is specific internal energy. The term $p\,dA/dx$ is the lateral-wall pressure force per unit length — it represents the pressure exerted by the slanted walls on the fluid.

In compact vector form:

$$
\frac{\partial (A\mathbf V)}{\partial t} + \frac{\partial (A\mathbf F)}{\partial x} = A\,\mathbf C^{\text{area}}
$$

with conservative state vector, flux, and area source:

$$
\mathbf V = \begin{pmatrix} \rho \\ \rho u \\ \rho E \end{pmatrix}, \qquad
\mathbf F = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u(\rho E + p) \end{pmatrix}, \qquad
\mathbf C^{\text{area}} = \begin{pmatrix} 0 \\ \dfrac{p}{A}\dfrac{dA}{dx} \\ 0 \end{pmatrix}
$$

**Critical observation**: in the conservative form, the area-variation source appears **only in the momentum equation**, as the lateral pressure $(p/A)\,dA/dx$. The mass and energy equations have **no additional source from area variation** — the area weighting of the fluxes through $A_{i\pm 1/2}$ accounts for all geometric effects on mass and energy transport.

### 1.2 Primitive variables and the choice of $(p, u, h)$

We introduce the primitive state vector:

$$
\mathbf U = \begin{pmatrix} p \\ u \\ h \end{pmatrix}
$$

The choice of enthalpy $h$ over internal energy $e$ or entropy $s$ is motivated by:

1. **Direct EOS interface**: JAXProp / CoolProp's natural lookup direction is $(p, h) \to \rho, T, a$.
2. **Cleanest matrix structure**: the coefficient matrix $\mathbf G$ in $(p, u, h)$ form (book eq. 2.26) has $a^2$ in position $(3, 2)$ — a single, EOS-derived quantity.
3. **Stagnation enthalpy** $H = h + u^2/2$ has direct physical meaning at inlet boundaries.
4. **Correct shock capturing**: like the energy form $(p, u, e)$, the enthalpy form gives correct Rankine–Hugoniot jump conditions because the conservative update uses total energy $\rho E$ (whereas entropy form does not).

Related quantities used throughout:

$$
H = h + u^2/2 \quad\text{(specific total enthalpy)}
$$

$$
\rho E = \rho(e + u^2/2) = \rho h - p + \rho u^2/2 = \rho H - p
$$

So in $(p, u, h)$ primitives, the third conservative variable is $V_3 = \rho H - p$ and the third flux component simplifies:

$$
F_3 = u(\rho E + p) = u(\rho H - p + p) = u\rho H
$$

### 1.3 EOS interface

The equation of state provides, given $(p, h)$:

- $\rho(p, h)$ — density
- $T(p, h)$ — temperature
- $a(p, h)$ — speed of sound
- $\hat\rho_p \equiv (\partial \rho/\partial p)_h$ — EOS partial
- $\hat\rho_h \equiv (\partial \rho/\partial h)_p$ — EOS partial

The thermodynamic identity relating these (derived from the Gibbs relation $dh = T\,ds + dp/\rho$ at $ds = 0$):

$$
\boxed{\;\frac{1}{a^2} = \hat\rho_p + \frac{\hat\rho_h}{\rho}\;}
$$

This identity holds for any single-phase fluid and for HEM mixtures in the two-phase region.

**HEM extension**: For two-phase flow at saturation, the same EOS interface applies — JAXProp returns mixture properties (density, sound speed, partials) using the lever rule. The numerical scheme is unchanged.

---

## 2. Coefficient Matrix in Primitive Form

### 2.1 Derivation strategy

The conservation form $\partial_t \mathbf V + \partial_x \mathbf F = \mathbf{(source)}$ is transformed to primitive form by chain rule. Defining the Jacobians:

$$
\mathbf J \equiv \frac{\partial \mathbf V}{\partial \mathbf U}, \qquad \mathbf K \equiv \frac{\partial \mathbf F}{\partial \mathbf U}
$$

we have:

$$
\mathbf J\,\frac{\partial \mathbf U}{\partial t} + \mathbf K\,\frac{\partial \mathbf U}{\partial x} = (\text{source})
$$

Pre-multiplying by $\mathbf J^{-1}$:

$$
\frac{\partial \mathbf U}{\partial t} + \mathbf G\,\frac{\partial \mathbf U}{\partial x} = \mathbf D
$$

with:

$$
\mathbf G = \mathbf J^{-1}\mathbf K, \qquad \mathbf D = \mathbf J^{-1}\mathbf C^{\text{area}}
$$

### 2.2 The Jacobian $\mathbf J = \partial \mathbf V/\partial \mathbf U$

Computing partial derivatives row by row, treating $\rho = \rho(p, h)$ and $H = h + u^2/2$:

$$
\boxed{\;\mathbf J = \begin{pmatrix} \hat\rho_p & 0 & \hat\rho_h \\ u\hat\rho_p & \rho & u\hat\rho_h \\ \hat\rho_p H - 1 & \rho u & \hat\rho_h H + \rho \end{pmatrix}\;}
$$

**Determinant** (computed by cofactor expansion):

$$
\det\mathbf J = \rho(\rho\hat\rho_p + \hat\rho_h) = \frac{\rho}{a^2}
$$

This is strictly positive for any physically valid state, so $\mathbf J$ is always invertible.

### 2.3 The Jacobian $\mathbf K = \partial \mathbf F/\partial \mathbf U$

$$
\boxed{\;\mathbf K = \begin{pmatrix} u\hat\rho_p & \rho & u\hat\rho_h \\ u^2\hat\rho_p + 1 & 2\rho u & u^2\hat\rho_h \\ u\hat\rho_p H & \rho(H + u^2) & u(\hat\rho_h H + \rho) \end{pmatrix}\;}
$$

Note the structural simplification in row 3: because $F_3 = u\rho H$ in enthalpy form (no explicit $up$ term), $\partial F_3/\partial p = u\hat\rho_p H$ — no isolated "$+u$" term as would appear in energy-form derivatives.

### 2.4 The coefficient matrix $\mathbf G$

Computing $\mathbf G = \mathbf J^{-1}\mathbf K$ symbolically (or numerically at runtime) yields the result of book eq. 2.26:

$$
\boxed{\;\mathbf G = \begin{pmatrix} u & \rho a^2 & 0 \\ 1/\rho & u & 0 \\ 0 & a^2 & u \end{pmatrix}\;}
$$

Physical interpretation of each row as a PDE:

- **Row 1 (pressure)**: $\partial_t p + u\,\partial_x p + \rho a^2\,\partial_x u = 0$
  Standard pressure equation: pressure advects at $u$, changes from velocity gradients weighted by acoustic impedance squared.

- **Row 2 (velocity)**: $\partial_t u + (1/\rho)\,\partial_x p + u\,\partial_x u = 0$
  Standard momentum equation.

- **Row 3 (enthalpy)**: $\partial_t h + a^2\,\partial_x u + u\,\partial_x h = 0$
  Compression changes enthalpy by $a^2\,\partial_x u$.

The off-diagonal coupling in row 3 ($a^2$ in position (3, 2)) is what makes enthalpy form differ from entropy form — entropy is conserved across acoustic waves (no coupling), but enthalpy changes by the compression work.

### 2.5 The source vector $\mathbf D$ in primitive form

Transforming $\mathbf C^{\text{area}} = (0, p/A\cdot dA/dx, 0)^T$ via $\mathbf D = \mathbf J^{-1}\mathbf C^{\text{area}}$ — or equivalently working from the conservation form directly — yields the result of book eq. 2.39 (entropy form) transformed by $\mathbf J_1$ from eq. 2.25:

$$
\boxed{\;\mathbf D^{\text{area}} = -u\,\frac{1}{A}\frac{dA}{dx}\begin{pmatrix} \rho a^2 \\ 0 \\ a^2 \end{pmatrix}\;}
$$

In primitive form, the area variation generates source terms in **both row 1 (pressure) and row 3 (enthalpy)** — but **not** in row 2 (velocity). This is the result of the similarity transformation $\mathbf J_1$ applied to the conservative-form source (which is in row 2 only).

**Important note for implementation**: The conservative-form source ($p\,dA/dx$ in momentum only) and the primitive-form source (in pressure and enthalpy only) describe the **same physics** in different representations. For the implicit Newton scheme described in Section 7 below, we work in conservative form for the residual — the area source appears only in row 2.

### 2.6 HEM compatibility

The coefficient matrix $\mathbf G$, the Jacobians $\mathbf J$, $\mathbf K$, and the source $\mathbf C^{\text{area}}$ have the **same algebraic form** for HEM as for single-phase. The only differences are the numerical values of $\rho$, $a$, $\hat\rho_p$, $\hat\rho_h$ returned by the EOS — which are continuous through the saturation curve (with discontinuous derivatives at the curve, but this is handled by CoolProp/JAXProp automatically).

The sound speed $a$ can become small in the two-phase region (HEM mixture sound speed at modest void fractions is often 30–100 m/s), but the matrices remain well-conditioned.

---

## 3. Eigenstructure of $\mathbf G$

### 3.1 Eigenvalues

The characteristic equation $\det(\mathbf G - \lambda\mathbf I) = 0$ factors cleanly:

$$
(u - \lambda)\big[(u - \lambda)^2 - a^2\big] = 0
$$

giving:

$$
\boxed{\;\lambda_1 = u + a, \qquad \lambda_2 = u - a, \qquad \lambda_3 = u\;}
$$

Physical interpretation:

- $\lambda_1$: right-going acoustic wave (pressure/density wave)
- $\lambda_2$: left-going acoustic wave
- $\lambda_3$: convective wave (entropy/enthalpy/quality, advected with the flow)

### 3.2 Right eigenvectors

Solving $(\mathbf G - \lambda_k\mathbf I)\mathbf v_k = \mathbf 0$ for each eigenvalue:

$$
\mathbf v_1 = \begin{pmatrix} \rho a \\ 1 \\ a \end{pmatrix}, \qquad
\mathbf v_2 = \begin{pmatrix} -\rho a \\ 1 \\ -a \end{pmatrix}, \qquad
\mathbf v_3 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}
$$

The eigenvector matrix:

$$
\boxed{\;\mathbf T = (\mathbf v_1\ \mathbf v_2\ \mathbf v_3) = \begin{pmatrix} \rho a & -\rho a & 0 \\ 1 & 1 & 0 \\ a & -a & 1 \end{pmatrix}\;}
$$

with determinant $\det\mathbf T = 2\rho a > 0$ for physical states.

### 3.3 Left eigenvectors

The inverse $\mathbf T^{-1}$ is:

$$
\boxed{\;\mathbf T^{-1} = \begin{pmatrix} 1/(2\rho a) & 1/2 & 0 \\ -1/(2\rho a) & 1/2 & 0 \\ -1/\rho & 0 & 1 \end{pmatrix}\;}
$$

The rows are the left eigenvectors $\mathbf l_k$ (such that $\mathbf l_k \mathbf G = \lambda_k \mathbf l_k$).

### 3.4 Physical interpretation of the eigenvectors

For acoustic waves ($\lambda_{1,2}$), the eigenvector structure encodes the **acoustic impedance** relationship: $\delta p = \pm\rho a\,\delta u$, and $\delta h = \pm a\,\delta u$. Density, velocity, and enthalpy change together in fixed proportions across an isentropic acoustic wave.

For the convective wave ($\lambda_3$), only enthalpy changes — $\delta p = 0$, $\delta u = 0$. This represents the **contact discontinuity** or "entropy wave" — material is transported but pressure and velocity are continuous.

In HEM at saturation, the convective wave transports vapor quality (since enthalpy at saturation parametrizes the mixture composition).

---

## 4. Split Matrices $\mathbf G_k$

### 4.1 Construction

The coefficient matrix $\mathbf G$ is decomposed into wave-by-wave contributions:

$$
\mathbf G = \sum_{k=1}^{3} \mathbf G_k, \qquad \mathbf G_k = \mathbf T\,\boldsymbol\Lambda_k\,\mathbf T^{-1} = \lambda_k\,\mathbf v_k\,\mathbf l_k
$$

where $\boldsymbol\Lambda_k$ retains only the $k$-th eigenvalue (zeros elsewhere). Each $\mathbf G_k$ is a **rank-1 matrix** expressing the action of the $k$-th wave.

### 4.2 Explicit expressions

Computing each outer product:

**Right-going acoustic wave** ($\lambda_1 = u + a$):

$$
\mathbf G_1 = (u+a)\begin{pmatrix} 1/2 & \rho a/2 & 0 \\ 1/(2\rho a) & 1/2 & 0 \\ 1/(2\rho) & a/2 & 0 \end{pmatrix}
$$

**Left-going acoustic wave** ($\lambda_2 = u - a$):

$$
\mathbf G_2 = (u-a)\begin{pmatrix} 1/2 & -\rho a/2 & 0 \\ -1/(2\rho a) & 1/2 & 0 \\ 1/(2\rho) & -a/2 & 0 \end{pmatrix}
$$

**Convective wave** ($\lambda_3 = u$):

$$
\mathbf G_3 = u\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ -1/\rho & 0 & 1 \end{pmatrix}
$$

### 4.3 Partition identity (verification)

Direct computation entry-by-entry shows:

$$
\boxed{\;\sum_{k=1}^{3} \mathbf G_k = \mathbf G\;}
$$

This identity must hold to machine precision in any implementation — it is the most fundamental sanity check.

### 4.4 Physical content

- $\mathbf G_1$ and $\mathbf G_2$ encode the acoustic coupling between pressure, velocity, and enthalpy (the nonzero entries in the third row are the work done by pressure on the fluid as the acoustic wave passes).
- $\mathbf G_3$ has nonzero entries only in the third row — the convective wave acts only on enthalpy/material content.

---

## 5. Flux Vector Splitting Based on the Linearized Riemann Problem

This is the core of Städtke's numerical method (book §7.5, eqs. 7.69–7.77).

### 5.1 Transformation to flux variables

The primitive-form equation $\partial_t \mathbf U + \mathbf G \partial_x \mathbf U = \mathbf 0$ can be transformed to a system in flux variables. Pre-multiplying by $\mathbf K$ and using $\partial_t\mathbf F = \mathbf K\partial_t\mathbf U$ and $\partial_x\mathbf U = \mathbf K^{-1}\partial_x \mathbf F$:

$$
\frac{\partial \mathbf F}{\partial t} + \mathbf R\,\frac{\partial \mathbf F}{\partial x} = \mathbf 0, \qquad \mathbf R = \mathbf K\,\mathbf G\,\mathbf K^{-1}
$$

The matrix $\mathbf R$ has:

- **Same eigenvalues** as $\mathbf G$ (similarity transformation preserves spectrum).
- **Different eigenvectors** (transformed by $\mathbf K$).
- **Same wave structure** but expressed in flux space.

### 5.2 Flux-space split matrices

The decomposition $\mathbf G = \sum \mathbf G_k$ induces:

$$
\mathbf R = \sum_{k=1}^{3} \mathbf R_k, \qquad \mathbf R_k = \mathbf K\,\mathbf G_k\,\mathbf K^{-1}
$$

Normalizing by the eigenvalue:

$$
\tilde{\mathbf R}_k = \frac{\mathbf R_k}{\lambda_k}
$$

These $\tilde{\mathbf R}_k$ are **rank-1 projection matrices in flux space**: they project any flux vector onto the eigenspace associated with the $k$-th wave.

### 5.3 Partition of unity

A key property — derivable from the eigenvector decomposition — is:

$$
\boxed{\;\sum_{k=1}^{3} \tilde{\mathbf R}_k = \mathbf I\;}
$$

**Derivation**: Using $\mathbf G_k = \lambda_k \mathbf v_k \mathbf l_k$, we have $\mathbf G_k/\lambda_k = \mathbf v_k \mathbf l_k$. Summing:

$$
\sum_k \frac{\mathbf G_k}{\lambda_k} = \sum_k \mathbf v_k \mathbf l_k = \mathbf T \mathbf T^{-1} = \mathbf I
$$

Then:

$$
\sum_k \tilde{\mathbf R}_k = \mathbf K\left(\sum_k \frac{\mathbf G_k}{\lambda_k}\right)\mathbf K^{-1} = \mathbf K\,\mathbf I\,\mathbf K^{-1} = \mathbf I
$$

This identity is the foundation of the FVS upwinding — it guarantees consistency in smooth flow.

### 5.4 Sonic-point entropy fix

When $\lambda_k$ approaches zero (sonic point in a rarefaction), division by $\lambda_k$ becomes problematic and naive FVS produces unphysical expansion shocks. The remedy is **Harten's entropy fix**:

$$
\lambda_k^{\text{safe}} = \text{sign}(\lambda_k)\cdot\max\big(|\lambda_k|, \delta\big), \qquad \delta = \delta_{\text{factor}}\cdot(|u| + a)
$$

with $\delta_{\text{factor}} \in [0.05, 0.20]$ (default 0.1). Use $\lambda_k^{\text{safe}}$ only in the **denominator** when forming $\tilde{\mathbf R}_k$; the eigenvalue inside $\mathbf G_k$ retains its true value.

### 5.5 The interface flux

For each cell face $x_{i+1/2}$:

**Step 1**: Compute the averaged state (book eq. 7.37):

$$
\bar{\mathbf U}_{i+1/2} = \frac{1}{2}\big(\mathbf U_{i+1/2}^L + \mathbf U_{i+1/2}^R\big)
$$

**Step 2**: Evaluate $\rho, a, \mathbf K, \lambda_k, \mathbf G_k$ at $\bar{\mathbf U}_{i+1/2}$.

**Step 3**: Build the projection matrices:

$$
\mathbf R_k = \mathbf K\,\mathbf G_k\,\mathbf K^{-1}, \qquad \tilde{\mathbf R}_k = \mathbf R_k / \lambda_k^{\text{safe}}
$$

**Step 4**: Sum by sign of (true) eigenvalue:

$$
\tilde{\mathbf R}^+ = \sum_{k:\,\lambda_k \geq 0} \tilde{\mathbf R}_k, \qquad \tilde{\mathbf R}^- = \sum_{k:\,\lambda_k < 0} \tilde{\mathbf R}_k
$$

These satisfy $\tilde{\mathbf R}^+ + \tilde{\mathbf R}^- = \mathbf I$.

**Step 5**: Compute physical fluxes from the **actual reconstructed left/right states** (not from the averaged state):

$$
\mathbf F^L = \mathbf F(\mathbf U_{i+1/2}^L), \qquad \mathbf F^R = \mathbf F(\mathbf U_{i+1/2}^R)
$$

**Step 6**: Apply the upwind interface flux formula (book eq. 7.75):

$$
\boxed{\;\hat{\mathbf F}_{i+1/2} = \tilde{\mathbf R}^+\,\mathbf F^L + \tilde{\mathbf R}^-\,\mathbf F^R\;}
$$

### 5.6 Behavior

- **Subsonic flow** ($|u| < a$): both $\tilde{\mathbf R}^+$ and $\tilde{\mathbf R}^-$ are nontrivial; the flux is a mix of $\mathbf F^L$ and $\mathbf F^R$.
- **Supersonic flow** ($u > a > 0$): all eigenvalues positive, $\tilde{\mathbf R}^+ = \mathbf I$, $\tilde{\mathbf R}^- = \mathbf 0$, so $\hat{\mathbf F} = \mathbf F^L$ (full upwinding).
- **Smooth flow** ($\mathbf U^L \approx \mathbf U^R$): $\mathbf F^L \approx \mathbf F^R \approx \mathbf F$ and partition of unity gives $\hat{\mathbf F} \approx \mathbf F$ — consistent.

### 5.7 Practical computation

In implementation, the matrices $\tilde{\mathbf R}_k$ are **not computed in closed form** but assembled numerically at each face using:
- Closed-form $\mathbf G_k$ (from Section 4)
- Closed-form $\mathbf K$ (from Section 2.3)
- LAPACK/JAX solve for $\mathbf K^{-1}\mathbf G_k$

This is cheap (3×3 systems) and avoids tedious algebra.

---

## 6. MUSCL Reconstruction with Van Leer Limiter

### 6.1 Piecewise linear reconstruction

Within each cell $i$, the primitive state is assumed to vary linearly (book eq. 8.36):

$$
\mathbf U_i(x) = \mathbf U_i + \boldsymbol\sigma_i\,(x - x_i)
$$

where $\boldsymbol\sigma_i$ is the limited slope vector and $x_i$ is the cell centroid. This gives **second-order accuracy** in smooth regions.

### 6.2 Reconstructed face states

At face $x_{i+1/2}$:

$$
\mathbf U_{i+1/2}^L = \mathbf U_i + \boldsymbol\sigma_i\,\frac{\Delta x_i}{2}
$$

$$
\mathbf U_{i+1/2}^R = \mathbf U_{i+1} - \boldsymbol\sigma_{i+1}\,\frac{\Delta x_{i+1}}{2}
$$

### 6.3 Van Leer slope limiter

For each cell, the one-sided slopes are:

$$
\sigma^L = \frac{\mathbf U_i - \mathbf U_{i-1}}{\Delta x_i}, \qquad \sigma^R = \frac{\mathbf U_{i+1} - \mathbf U_i}{\Delta x_i}
$$

The Van Leer limited slope (applied componentwise on $(p, u, h)$):

$$
\boxed{\;\sigma^{\text{VL}} = \begin{cases}
\dfrac{2\,\sigma^L\,\sigma^R}{\sigma^L + \sigma^R} & \text{if } \sigma^L \cdot \sigma^R > 0 \\
0 & \text{otherwise}
\end{cases}\;}
$$

This is the **harmonic mean** of left and right slopes when they have the same sign, zero at local extrema.

### 6.4 Properties of Van Leer

- **TVD** (Total Variation Diminishing): no spurious oscillations at shocks.
- **Second-order accurate** in smooth regions where the limiter is not active.
- **$C^1$ smooth**: differentiable everywhere (no kinks), unlike minmod or superbee — important for Newton convergence in implicit schemes.
- **Sharper than minmod, less compressive than superbee**: a sensible engineering compromise.
- **No tunable parameters**.

### 6.5 Positivity safeguard

After reconstruction, if any face state has $p \leq 0$ or other unphysical values, the scheme falls back to **first-order at that face** ($\boldsymbol\sigma_i = 0$, so $\mathbf U^L = \mathbf U_i$, $\mathbf U^R = \mathbf U_{i+1}$). This rarely activates but provides robustness near strong shocks.

### 6.6 Boundary cells

At the first and last cells, the slope is set to zero (first-order at the boundary). This does not affect interior accuracy and avoids issues with ghost-cell extrapolation.

---

## 7. Implicit Time Integration via Newton-Raphson

### 7.1 Discrete conservation equation

For cell $i$ with cell volume $V_i = A_i\Delta x_i$ (where $A_i$ is the centroid area):

$$
V_i\,\mathbf V(\mathbf U_i^{n+1}) = V_i\,\mathbf V(\mathbf U_i^n) - \Delta t\,\big[A_{i+1/2}\hat{\mathbf F}_{i+1/2}^{n+1} - A_{i-1/2}\hat{\mathbf F}_{i-1/2}^{n+1}\big] + \Delta t\,V_i\,\mathbf C^{\text{area}}(\mathbf U_i^{n+1})
$$

**All terms at $n+1$ are evaluated implicitly** — the fluxes and the source. This is the key feature of Städtke's scheme: full implicit coupling between convection and sources.

### 7.2 Residual form

Define the residual:

$$
\boxed{\;\mathbf R_i(\mathbf U^{n+1}) = V_i\big[\mathbf V(\mathbf U_i^{n+1}) - \mathbf V(\mathbf U_i^n)\big] + \Delta t\,\big[A_{i+1/2}\hat{\mathbf F}_{i+1/2}^{n+1} - A_{i-1/2}\hat{\mathbf F}_{i-1/2}^{n+1}\big] - \Delta t\,V_i\,\mathbf C^{\text{area}}(\mathbf U_i^{n+1})\;}
$$

Setting $\mathbf R = \mathbf 0$ gives a system of $3N$ nonlinear equations in $3N$ unknowns $\{\mathbf U_i^{n+1}\}$.

### 7.3 Source term explicit form

With area variation only (no friction, no body force, no heat source), the conservative source is:

$$
\mathbf C^{\text{area}}(\mathbf U) = \begin{pmatrix} 0 \\ p\,(dA/dx)/A \\ 0 \end{pmatrix}
$$

Note: $A$ and $dA/dx$ are **geometric quantities** (functions of $x$ only), specified at code initialization. Only $p$ depends on the state $\mathbf U$, so the source Jacobian is sparse and trivial.

### 7.4 Newton iteration in primitive variables

For single-phase, the base vector $\mathbf W$ in Städtke's notation (book eq. 8.39) coincides with the primitive vector $\mathbf U$:

$$
\mathbf W = \mathbf U = (p, u, h)^T
$$

(For two-phase, $\mathbf W$ would be chosen differently to handle phase-disappearance singularities — not needed here.)

The Newton iteration starts with $\mathbf U^{(0)} = \mathbf U^n$ and iterates:

$$
\frac{\partial \mathbf R}{\partial \mathbf U}\bigg|^{(k)}\,\Delta\mathbf U^{(k)} = -\mathbf R(\mathbf U^{(k)})
$$

$$
\mathbf U^{(k+1)} = \mathbf U^{(k)} + \Delta\mathbf U^{(k)}
$$

Iteration continues until $\|\Delta\mathbf U^{(k)}\|/\|\mathbf U^{(k)}\| < \text{tol}_{\text{Newton}}$ (typically $10^{-8}$). Per Städtke (§8.2.4), **2–3 iterations** are typically sufficient.

### 7.5 The Jacobian via JAX autodiff

The Jacobian $\partial\mathbf R/\partial\mathbf U$ is a $(3N) \times (3N)$ sparse matrix. Rather than deriving it analytically, we use **JAX automatic differentiation**:

$$
\frac{\partial \mathbf R}{\partial \mathbf U} = \text{jax.jacobian}(\mathbf R)
$$

JAX traces through the entire residual computation — including MUSCL reconstruction, FVS interface flux, area-weighted flux divergence, and source terms — to produce the **exact Jacobian** to machine precision.

### 7.6 Sparsity structure

The residual $\mathbf R_i$ depends on:

- $\mathbf U_i$ directly (through $\mathbf V_i$, $\mathbf C^{\text{area}}_i$, and via $\mathbf U^L_{i+1/2}$, $\mathbf U^R_{i-1/2}$)
- $\mathbf U_{i\pm 1}$ through neighboring flux contributions
- $\mathbf U_{i\pm 2}$ through MUSCL slope reconstruction (which uses 3-cell stencils)

Thus the Jacobian is **block-pentadiagonal** with $3\times 3$ blocks:

$$
\mathbf J_{i,j} = \frac{\partial \mathbf R_i}{\partial \mathbf U_j}, \quad j \in \{i-2, i-1, i, i+1, i+2\}
$$

Bandwidth: 5 blocks (15 scalars). Each row has 15 nonzero entries out of $3N$. Sparse linear algebra solvers handle this efficiently.

### 7.7 Linear system solve

The linear system $\mathbf J\,\Delta\mathbf U = -\mathbf R$ is block-pentadiagonal. Options for the solve:

- **Banded direct solver** (LAPACK `gbsv` or equivalent): efficient for $N \lesssim 10^4$.
- **Sparse direct solver** (SuperLU, UMFPACK): for larger 1D problems or 2D extensions.
- **Krylov solver** (GMRES, BiCGSTAB): for very large problems, possibly with JVP-based matrix-free formulation.

For the 1D single-phase problem with $N \sim 10^3$ cells, banded direct is fastest.

### 7.8 Time-step controller

After Newton convergence, three criteria are checked (book §8.2.4):

**Criterion 1 — Courant number:**

$$
\text{CFL} = \Delta t\,\max_i\frac{|u_i| + a_i}{\Delta x_i} < C_{\max}
$$

For implicit schemes, $C_{\max}$ can be up to ~10. Start conservatively at $C_{\max} = 1$.

**Criterion 2 — Mass conservation error:**

The relative error in $\rho$ between the Newton-converged value and a Taylor extrapolation from $t^n$ should be below a threshold (default $10^{-3}$).

**Criterion 3 — Primitive-variable increment bounds:**

$$
\max_i \frac{|\Delta p|}{p} < 0.2, \quad \max_i \frac{|\Delta u|}{|u| + a} < 0.2, \quad \max_i \frac{|\Delta h|}{h} < 0.1
$$

If any criterion fails: $\Delta t \leftarrow \Delta t/2$, restore $\mathbf U^{n+1} = \mathbf U^n$, retry. If all pass and Newton converged in $\leq 2$ iterations, $\Delta t$ can be gradually increased (factor 1.2) up to a user-defined maximum.

### 7.9 Mapping to Städtke's notation

Städtke's eqs. 8.38–8.40 correspond exactly to what is described above:

- **Eq. 8.38** (first-order Taylor expansion of $\mathbf V^{n+1}$, $\mathbf F^{n+1}$, $\mathbf D^{n+1}$ about $\mathbf W^n$): this is the **first Newton iteration**, with base vector $\mathbf W$.
- **Eq. 8.39** (choice of $\mathbf W$): for single-phase, $\mathbf W = \mathbf U = (p, u, h)$.
- **Eq. 8.40** (update of primitives via $\partial\mathbf U/\partial\mathbf W$): for us $\mathbf U = \mathbf W$, so this is trivially $\mathbf U^{n+1} = \mathbf U^n + \Delta\mathbf W$.

Subsequent Newton iterations (re-linearizing around $\mathbf U^{(k)}$ rather than $\mathbf U^n$) extend the basic Städtke recipe to full Newton-Raphson convergence — needed for strong shocks or stiff sources.

---

## 8. Boundary Conditions

### 8.1 Reservoir inlet (stagnation conditions)

For a nozzle inlet with prescribed stagnation pressure $p_0$ and stagnation enthalpy $h_0$:

- The Mach number at the boundary is extrapolated from the interior.
- Isentropic relations give the static pressure and static enthalpy at the boundary:

$$
h = h_0 - u^2/2, \qquad p = p_{\text{isentropic}}(h_0, h)
$$

In practice: the ghost-cell state is set from the (extrapolated) boundary velocity together with the stagnation conditions.

### 8.2 Pressure outlet

For subsonic outflow with prescribed back-pressure $p_{\text{back}}$:

- Pressure is set to $p_{\text{back}}$ at the ghost cell.
- Other primitives are zero-gradient extrapolated.

For supersonic outflow (Mach number > 1 at the boundary): zero-gradient extrapolation of all primitives (pressure outlet inactive — flow is determined entirely by upstream conditions).

### 8.3 Reflective wall

For a closed end (e.g., closed-pipe blowdown):

- Velocity in the ghost cell: $u_{\text{ghost}} = -u_{\text{interior}}$.
- Pressure and enthalpy: same as interior.

This gives zero mass flux and twice the interior pressure as momentum flux at the wall.

### 8.4 Implementation in the residual

Boundary fluxes $\hat{\mathbf F}_{1/2}$ and $\hat{\mathbf F}_{N+1/2}$ are computed from the prescribed boundary states (via FVS or directly from boundary type) and inserted into the residual at cells 1 and $N$. JAX autodiff handles the Jacobian contributions automatically.

---

## 9. Algorithmic Structure

The full algorithm for one time step proceeds as follows:

1. **Begin iteration**: set Newton iterate $\mathbf U^{(0)} = \mathbf U^n$.
2. **For each Newton iteration**:
   a. **MUSCL reconstruction** of $\mathbf U_{i+1/2}^L$, $\mathbf U_{i+1/2}^R$ at every face using Van Leer limiter on $(p, u, h)$.
   b. **Positivity check**: fall back to first-order at any face where reconstruction gives unphysical values.
   c. **Average state computation** $\bar{\mathbf U}_{i+1/2} = (\mathbf U^L + \mathbf U^R)/2$.
   d. **EOS calls** at averaged states and at reconstructed L/R states.
   e. **Build projection matrices**: $\tilde{\mathbf R}^+, \tilde{\mathbf R}^-$ at each face (using $\mathbf G_k$, $\mathbf K$, sonic fix).
   f. **Compute physical fluxes** $\mathbf F^L$, $\mathbf F^R$ from L/R states.
   g. **FVS interface flux**: $\hat{\mathbf F}_{i+1/2} = \tilde{\mathbf R}^+ \mathbf F^L + \tilde{\mathbf R}^- \mathbf F^R$.
   h. **Boundary fluxes** at $i = 0$ and $i = N$.
   i. **Source term** $\mathbf C^{\text{area}}$ at each cell.
   j. **Assemble residual** $\mathbf R_i$ using area-weighted FV update.
   k. **Convergence check**: if $\|\mathbf R\|$ small enough, exit.
   l. **Build Jacobian** via JAX autodiff.
   m. **Solve linear system** $\mathbf J\,\Delta\mathbf U = -\mathbf R$.
   n. **Update**: $\mathbf U^{(k+1)} = \mathbf U^{(k)} + \Delta\mathbf U$.
   o. **Recompute EOS-dependent quantities** for the next iteration.
3. **Time-step acceptance check** via three criteria; halve $\Delta t$ and retry if any fails.
4. **Advance time**: $\mathbf U^{n+1} \leftarrow \mathbf U^{(k_{\text{final}})}$, $t \leftarrow t + \Delta t$.

---

## 10. Summary Table of Key Equations

| Quantity | Symbol | Expression |
|---|---|---|
| Primitive state | $\mathbf U$ | $(p, u, h)^T$ |
| Conservative state | $\mathbf V$ | $(\rho, \rho u, \rho H - p)^T$ |
| Flux | $\mathbf F$ | $(\rho u, \rho u^2 + p, u\rho H)^T$ |
| Coefficient matrix | $\mathbf G$ | $\begin{pmatrix} u & \rho a^2 & 0 \\ 1/\rho & u & 0 \\ 0 & a^2 & u \end{pmatrix}$ |
| Eigenvalues | $\lambda_{1,2,3}$ | $u+a,\ u-a,\ u$ |
| Right eigenvectors | $\mathbf T$ | $\begin{pmatrix} \rho a & -\rho a & 0 \\ 1 & 1 & 0 \\ a & -a & 1 \end{pmatrix}$ |
| Sound speed | $a^2$ | $1/(\hat\rho_p + \hat\rho_h/\rho)$ |
| Slope limiter | $\sigma^{\text{VL}}$ | Van Leer (harmonic mean) |
| Sonic fix | $\lambda^{\text{safe}}$ | $\text{sign}(\lambda)\max(|\lambda|, 0.1(|u|+a))$ |
| Conservative source | $\mathbf C^{\text{area}}$ | $(0,\ p\,dA/dx\,/A,\ 0)^T$ |
| Interface flux | $\hat{\mathbf F}$ | $\tilde{\mathbf R}^+\mathbf F^L + \tilde{\mathbf R}^-\mathbf F^R$ |

---

## 11. Verification Checks

Required unit tests for any implementation:

1. **Partition identity for $\mathbf G$**: $\sum_k \mathbf G_k = \mathbf G$ to machine precision.
2. **Partition of unity for projection matrices**: $\sum_k \tilde{\mathbf R}_k = \mathbf I$ to machine precision.
3. **Jacobian consistency**: $\mathbf J\,\mathbf G = \mathbf K$ where $\mathbf J = \partial\mathbf V/\partial\mathbf U$, $\mathbf K = \partial\mathbf F/\partial\mathbf U$.
4. **Sound speed identity**: $1/a^2 = \hat\rho_p + \hat\rho_h/\rho$ matches EOS value of $a$.
5. **Eigenvalue ordering**: $\lambda_2 < \lambda_3 < \lambda_1$ for any physical state.

Validation benchmarks:

1. **Sod shock tube** (constant area): exact Riemann solution reference.
2. **Toro 123 problem**: near-vacuum expansion — tests sonic entropy fix.
3. **Steady isentropic nozzle flow** (subsonic): compare with analytical isentropic-flow tables.
4. **Steady Laval-nozzle flow**: choked subsonic-supersonic transition, multiple back-pressure regimes.
5. **Edwards' pipe blowdown** (single-phase analog): unsteady depressurization through a pipe exit.
6. **HEM steam expansion**: nozzle flow with phase transition from superheated vapor to wet steam.

---

## 12. Extension to Two-Phase Flow

The framework presented here generalizes to Städtke's full two-fluid model with minimal structural changes:

1. **Primitive vector** expands from 3 components to 6 (or 8 with interfacial area transport): $\mathbf U = (p, u_g, u_l, \alpha_g, h_g, h_l)^T$ for the enthalpy form.
2. **Coefficient matrix** becomes $6 \times 6$ (book Table B.29 in entropy form; similarity transformation gives enthalpy form).
3. **Non-conservative term** $\mathbf H^{nc}\hat{\mathbf F}$ appears as an additional face quantity (book eq. 8.28) — needed because the separated phase momentum equations cannot be put in fully conservative form.
4. **Source vector $\mathbf D$** includes interfacial drag, mass transfer, heat transfer — typically stiff, requiring the implicit treatment for tractable time steps.
5. **Same FVS-on-flux architecture**: build $\mathbf G_k$, transform to $\mathbf R_k$, sum by eigenvalue sign, apply $\hat{\mathbf F} = \tilde{\mathbf R}^+\mathbf F^L + \tilde{\mathbf R}^-\mathbf F^R$.
6. **Same Newton-Raphson implicit time-marching**, with bigger blocks and the $\mathbf H^{nc}\hat{\mathbf F}$ contribution added to the residual.

The single-phase code developed here serves as the foundation: ~70% of the code (FVS interface flux, MUSCL reconstruction, Newton machinery, sparse solver) carries over directly to the two-phase version.

---

End of document.
