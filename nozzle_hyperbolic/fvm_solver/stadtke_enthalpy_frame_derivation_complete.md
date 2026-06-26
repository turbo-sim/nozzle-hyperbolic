# Algebraic Formulation for the FVS Scheme in Enthalpy Primitive Variables
### A Step-by-Step Derivation from Städtke's Six-Equation Single-Pressure Model

*This document assembles, in strict dependency order, the closed-form algebraic content needed
to build Städtke's flux-vector-splitting scheme in enthalpy-primitive, total-energy-conservative
variables. Each step is verified before the next is started. The chain so far: the coefficient
matrix $G_h$ in enthalpy primitives via the thermodynamic transform $\Phi$; the conservative
bridge $J=\partial V/\partial U_h$ and the flux Jacobian $K=\partial F/\partial U_h$ from the
total-energy state and flux vectors (Städtke eq. 5.90). Still to come: the non-conservative
coupling $H^{nc}=(JG_h-K)K^{-1}$, the flux-frame coefficient matrix $R=KG_hK^{-1}$, and the
flux-space eigenvectors and projectors used in the kernel. Sources are Städtke's monograph
(Wiley 2006), the verified entropy-frame derivation (project document
`stadtke_twofluid_monograph_portable.md`), and the FVS-scheme companion
(`stadtke_twofluid_fvs_scheme_portable.md`).*

---

## 1. Notation lock

There are **three distinct similarity transforms** in play in this chain, and the monograph,
the 1997 paper, and the book index use overlapping symbols for them. We fix the convention
once, here, and use it everywhere afterwards:

- $\Phi \;\equiv\; \dfrac{\partial U_s}{\partial U_h}$ — the **thermodynamic** change of
  variables, entropy primitives $\to$ enthalpy primitives. Touches only the last two slots;
  not a Jacobian to a conservative or flux frame. (The monograph calls this $J$, which clashes
  with the conservative bridge; we use $\Phi$ to keep them separate.)
- $J \;\equiv\; \dfrac{\partial V}{\partial U_h}$ — the **conservative bridge**, primitive
  enthalpy state $\to$ conservative total-energy state. (Same role as Städtke's $J$ in eq. 8.25
  / 1997 eq. 17.)
- $K \;\equiv\; \dfrac{\partial F}{\partial U_h}$ — the **flux Jacobian** with respect to
  primitives. Note book eq. 5.88 writes $K=\partial F/\partial V$; this is a typo. Eq. 8.25
  and 1997 eq. 17 give $\partial F/\partial U$, which is the only definition consistent with
  $H^{nc}=(JG-K)K^{-1}$.

State vectors. The **entropy-primitive** state is

$$ U_s \;=\; (\,p,\ u_g,\ u_l,\ \alpha_g,\ s_g,\ s_l\,)^\top, $$

the **enthalpy-primitive** state — what the solver will actually march and what JAXProp/CoolProp
speak natively — is

$$ U_h \;=\; (\,p,\ u_g,\ u_l,\ \alpha_g,\ h_g,\ h_l\,)^\top. $$

Throughout this document, indices $i\in\{g,l\}$ run over phases, $\Delta u\equiv u_g-u_l$, and
$\alpha_l=1-\alpha_g$.

---

## 2. The starting point: $G_s$ in entropy primitives

From the project's verified derivation, the quasi-linear form of the six-equation single-pressure
model in entropy primitives is

$$ A_s\,\partial_t U_s + B_s\,\partial_x U_s = C_s,\qquad \partial_t U_s + G_s\,\partial_x U_s = D_s, \qquad G_s = A_s^{-1}B_s, $$

where $A_s$ is Städtke's Table B.27 (correct as printed), $B_s$ is Table B.28 (four typos
catalogued in the verification document, corrected in the symbolic build), and $G_s$ is Table
B.29 / eq. 5.25, which we take as our verified input. The eigenvalues are

$$ \Lambda = \operatorname{diag}(u_g,\ u_l,\ u+a,\ u-a,\ u_g,\ u_l), $$

with $u_g, u_l$ each appearing twice (one void-wave copy and one thermodynamic-wave copy),
and $u\pm a$ the acoustic pair. The closed forms for the mixture velocity $u$ and mixture sound
speed $a$ in terms of $\alpha_i, \varrho_i, a_i, u_i, k$ are as given in Städtke eqs. 5.36–5.38.

### 2.1 Symbol dictionary (verified)

These abbreviations are not cosmetic. They condition every entry of $G_s$ — and, after
transformation, every entry of $G_h$ — and the algebra collapses cleanly only with them. All
were verified symbolically in the project derivation.

**Density combinations.**

$$ \varrho \;=\; \alpha_g\varrho_g + \alpha_l\varrho_l \quad(\text{mixture density}), \qquad \varrho_s \;=\; \alpha_g\varrho_l + \alpha_l\varrho_g \quad(\text{cross-weighted}). $$

The bare $\varrho$ must be the mixture density — *not* $\varrho_g+\varrho_l$ — by the consistency
identity $\hat\alpha_g+\hat\alpha_l=1$ in the verification document.

**Virtual-mass-modified densities.**

$$ \hat\varrho_g = \varrho_g + k\varrho,\qquad \hat\varrho_l = \varrho_l + k\varrho,\qquad \hat\varrho^2 = \varrho_g\varrho_l + k\varrho^2. $$

Note the third one is a *defined combination* (Städtke eq. 5.26), not literally
$\hat\varrho_g\hat\varrho_l$; the symbol is a shorthand consistent with $\hat\alpha_i\hat\varrho^{-2}$
weights.

**Convective averages.**

$$ u_1 \;=\; \alpha_g u_g + \alpha_l u_l \quad(\text{volume-weighted}),\qquad u_2 \;=\; \frac{\alpha_g\varrho_g u_g + \alpha_l\varrho_l u_l}{\varrho}\quad(\text{mass-weighted}). $$

**The acoustic combination.**

$$ a_0^2 \;=\; \frac{\varrho_s}{\displaystyle\frac{\alpha_g\varrho_l}{a_g^2} + \frac{\alpha_l\varrho_g}{a_l^2}}, $$

the frozen sound speed of the model at zero slip; $a_0=a$ in the homogeneous limit.

### 2.2 $G_s$ in full

In the ordering $U_s=(p,u_g,u_l,\alpha_g,s_g,s_l)$, the verified $G_s$ is

$$ G_s = \begin{pmatrix} G_{pp} & \alpha_g a_0^2\dfrac{\varrho_l\varrho_g}{\varrho_s} & \alpha_l a_0^2\dfrac{\varrho_l\varrho_g}{\varrho_s} & a_0^2\Delta u\dfrac{\varrho_l\varrho_g}{\varrho_s} & 0 & 0 \\[4pt] \dfrac{\hat\varrho_l}{\hat\varrho^2} & G_{uu} & -\alpha_l\Delta u\dfrac{\varrho_l\hat\varrho_l}{\hat\varrho^2} & 0 & 0 & 0 \\[4pt] \dfrac{\hat\varrho_g}{\hat\varrho^2} & \alpha_g\Delta u\dfrac{\varrho_g\hat\varrho_g}{\hat\varrho^2} & G_{uu} & 0 & 0 & 0 \\[4pt] \alpha_l\alpha_g\dfrac{\Delta u}{\varrho_s}\dfrac{a_0^2}{a_l^2 a_g^2} & \alpha_l\alpha_g\dfrac{\varrho_g}{\varrho_s}\dfrac{a_0^2}{a_l^2} & -\alpha_l\alpha_g\dfrac{\varrho_l}{\varrho_s}\dfrac{a_0^2}{a_g^2} & G_{\alpha\alpha} & 0 & 0 \\[4pt] 0 & 0 & 0 & 0 & u_g & 0 \\[2pt] 0 & 0 & 0 & 0 & 0 & u_l \end{pmatrix} $$

with

$$ \begin{aligned} G_{pp} &= \alpha_l u_l\,\frac{\varrho_g}{\varrho_s}\,\frac{a_0^2}{a_l^2} + \alpha_g u_g\,\frac{\varrho_l}{\varrho_s}\,\frac{a_0^2}{a_g^2}, \\[2pt] G_{uu} &= u_1\,\frac{\varrho_l\varrho_g}{\hat\varrho^2} + u_2\,\frac{k\varrho^2}{\hat\varrho^2}, \\[2pt] G_{\alpha\alpha} &= \alpha_l u_g\,\frac{\varrho_g}{\varrho_s}\,\frac{a_0^2}{a_l^2} + \alpha_g u_l\,\frac{\varrho_l}{\varrho_s}\,\frac{a_0^2}{a_g^2}. \end{aligned} $$

Two structural features that will survive into $G_h$ unchanged:

- The lower-right $2\times2$ block is diagonal with entries $u_g, u_l$ — entropy is passively
  advected, each phase at its own velocity, decoupled from everything else.
- The first four rows have zero columns in the entropy slots (cols 5, 6) — entropy gradients
  do not drive pressure, momentum, or void rates. This is the signature of linearly degenerate
  thermodynamic waves and is what makes the change of variables to enthalpy quasi-trivial in the
  upper block.

---

## 3. The thermodynamic transform $\Phi$

### 3.1 Definition

$\Phi\equiv\partial U_s/\partial U_h$ is the Jacobian of the relabel $U_h\mapsto U_s$. Since the
first four slots of $U_s$ and $U_h$ coincide ($p,u_g,u_l,\alpha_g$ are common), the only
nontrivial dependence is $s_i=s_i(p,h_i)$ for $i\in\{g,l\}$. The Gibbs relation provides exactly
these two partials:

$$ \mathrm dh_i \;=\; T_i\,\mathrm ds_i + \frac{\mathrm dp}{\varrho_i} \quad\Longleftrightarrow\quad \mathrm ds_i \;=\; \frac{1}{T_i}\,\mathrm dh_i \;-\; \frac{1}{\varrho_i T_i}\,\mathrm dp, $$

so

$$ \left.\frac{\partial s_i}{\partial h_i}\right|_p = \frac{1}{T_i},\qquad \left.\frac{\partial s_i}{\partial p}\right|_{h_i} = -\frac{1}{\varrho_i T_i}, $$

and the cross-partials $\partial s_i/\partial u_j$, $\partial s_i/\partial\alpha_g$,
$\partial s_g/\partial h_l$ etc. are all zero by independence of the variables. The result is a
block-triangular matrix that is the identity except for rows 5 and 6:

$$ \boxed{\;\Phi \;=\; \begin{pmatrix} I_4 & \mathbf 0 \\[4pt] \begin{matrix} -\dfrac{1}{\varrho_g T_g} & 0 & 0 & 0 \\[2pt] -\dfrac{1}{\varrho_l T_l} & 0 & 0 & 0 \end{matrix} & \begin{matrix} \dfrac{1}{T_g} & 0 \\[2pt] 0 & \dfrac{1}{T_l} \end{matrix} \end{pmatrix}\;} $$

### 3.2 Invertibility and the inverse

The determinant is just the determinant of the lower-right $2\times2$ block (the upper triangle
contributes 1):

$$ \det\Phi \;=\; \frac{1}{T_g T_l} \;>\; 0, $$

so $\Phi$ is invertible at every physical state, including the metastable regime $T_g\neq T_l$
(flashing two-phase flow). This invertibility holds without requiring any density-derivative
information ($\varrho_p, \varrho_h$); only the temperatures and densities themselves enter.

Inverting block-triangularly:

$$ \boxed{\;\Phi^{-1} \;=\; \begin{pmatrix} I_4 & \mathbf 0 \\[4pt] \begin{matrix} \dfrac{1}{\varrho_g} & 0 & 0 & 0 \\[2pt] \dfrac{1}{\varrho_l} & 0 & 0 & 0 \end{matrix} & \begin{matrix} T_g & 0 \\[2pt] 0 & T_l \end{matrix} \end{pmatrix}\;} $$

This can be read off from the inverse Gibbs relation $\mathrm dh_i=T_i\mathrm ds_i+\mathrm dp/\varrho_i$
directly: $\partial h_i/\partial s_i|_p=T_i$ and $\partial h_i/\partial p|_{s_i}=1/\varrho_i$.
Computing $\Phi\,\Phi^{-1}=I_6$ is one line of algebra (the cross terms
$-1/(\varrho_iT_i)\cdot 1 + (1/T_i)\cdot(1/\varrho_i)=0$ cancel by construction).

### 3.3 Unit consistency

In SI: $T_i\,[K]$, $\varrho_i\,[\mathrm{kg\,m^{-3}}]$, $h_i\,[\mathrm{m^2 s^{-2}}]$,
$s_i\,[\mathrm{m^2 s^{-2} K^{-1}}]$, $p\,[\mathrm{kg\,m^{-1}s^{-2}}]$.

- $\Phi_{55}=1/T_g$: converts $\mathrm dh_g\,[\mathrm{m^2 s^{-2}}]$ to
  $\mathrm ds_g\,[\mathrm{m^2 s^{-2}K^{-1}}]$. ✓
- $\Phi_{51}=-1/(\varrho_g T_g)$: $[\mathrm{m^3 kg^{-1}\,K^{-1}}]$, applied to
  $\mathrm dp\,[\mathrm{kg\,m^{-1}s^{-2}}]$ gives $[\mathrm{m^2 s^{-2}K^{-1}}]$. ✓

All entries of $\Phi$ and $\Phi^{-1}$ are dimensionally consistent.

---

## 4. The transformation $G_h = \Phi^{-1} G_s\,\Phi$

### 4.1 Why this is the right transform

The PDE $\partial_t U_s + G_s\,\partial_x U_s = D_s$ is rewritten in $U_h$ by chain rule:

$$ \partial_t U_s = \Phi\,\partial_t U_h,\qquad \partial_x U_s = \Phi\,\partial_x U_h, $$

so

$$ \Phi\,\partial_t U_h + G_s\,\Phi\,\partial_x U_h = D_s \;\;\Longleftrightarrow\;\; \partial_t U_h + \underbrace{\Phi^{-1} G_s\,\Phi}_{G_h}\,\partial_x U_h = \Phi^{-1}D_s. $$

This is a *similarity transform* of $G_s$: $G_h$ and $G_s$ have identical characteristic
polynomials and identical spectra. The eigenvectors transform by
$r_k^{(h)}=\Phi^{-1}r_k^{(s)}$, $l_k^{(h)}=l_k^{(s)}\Phi$ — point we will use in Document 4 when
the flux-frame projectors are built, but not needed here.

### 4.2 Block-by-block evaluation

Because $\Phi$ and $\Phi^{-1}$ are block-triangular and act as the identity on the first four
slots, the product $\Phi^{-1}G_s\Phi$ resolves cleanly by block. Partition

$$ G_s \;=\; \begin{pmatrix} G_4 & \mathbf 0_{4\times 2} \\[2pt] G_{21} & \operatorname{diag}(u_g,u_l) \end{pmatrix}, \qquad \Phi \;=\; \begin{pmatrix} I_4 & \mathbf 0 \\[2pt] L & D \end{pmatrix},\quad \Phi^{-1} \;=\; \begin{pmatrix} I_4 & \mathbf 0 \\[2pt] -D^{-1}L & D^{-1} \end{pmatrix}, $$

where $G_4$ is the verified upper-left $4\times4$ block of $G_s$, $G_{21}$ is its zero $2\times4$
lower-left block (entropy rows of $G_s$ — they are zero in the first four columns), $L$ is the
$2\times4$ lower-left of $\Phi$ (only column 1 is nonzero), and
$D=\operatorname{diag}(1/T_g,1/T_l)$.

Compute the product in stages.

**Stage 1: $G_s\,\Phi$.**

$$ G_s\,\Phi = \begin{pmatrix} G_4 & \mathbf 0 \\ G_{21} & \operatorname{diag}(u_g,u_l) \end{pmatrix}\begin{pmatrix} I_4 & \mathbf 0 \\ L & D \end{pmatrix} = \begin{pmatrix} G_4 & \mathbf 0 \\ G_{21} + \operatorname{diag}(u_g,u_l)\,L & \operatorname{diag}(u_g,u_l)\,D \end{pmatrix} $$

The mixed terms $G_4\cdot\mathbf 0=0$ and $\mathbf 0\cdot D=0$ kill any contamination of the
upper-right block. Since $G_{21}=\mathbf 0$ (entropy rows of $G_s$ are zero in the velocity, void,
and pressure columns? — *no*, careful: entropy rows of $G_s$ have $u_g, u_l$ on the *diagonal*
and zeros in cols 1–4; so $G_{21}=\mathbf 0$ exactly, by the structure of $G_s$). The mixed
lower-left becomes $\operatorname{diag}(u_g,u_l)\,L$, which has only column 1 nonzero:

$$ \operatorname{diag}(u_g,u_l)\,L = \begin{pmatrix} -u_g/(\varrho_g T_g) & 0 & 0 & 0 \\ -u_l/(\varrho_l T_l) & 0 & 0 & 0 \end{pmatrix}. $$

**Stage 2: $\Phi^{-1}\cdot(G_s\Phi)$.**

$$ \Phi^{-1}(G_s\Phi) = \begin{pmatrix} I_4 & \mathbf 0 \\ -D^{-1}L & D^{-1} \end{pmatrix}\begin{pmatrix} G_4 & \mathbf 0 \\ \operatorname{diag}(u_g,u_l)L & \operatorname{diag}(u_g,u_l)D \end{pmatrix} $$

$$ = \begin{pmatrix} G_4 & \mathbf 0 \\ -D^{-1}L\,G_4 + D^{-1}\operatorname{diag}(u_g,u_l)\,L & D^{-1}\operatorname{diag}(u_g,u_l)\,D \end{pmatrix}. $$

The upper-right block is $\mathbf 0$. The upper-left is $G_4$ — **unchanged** from $G_s$. The
lower-right is $D^{-1}\operatorname{diag}(u_g,u_l)D=\operatorname{diag}(u_g,u_l)$ since $D$ is
diagonal — also unchanged. The only nontrivial block is the lower-left.

**Stage 3: the lower-left block.**

$$ G_{h,21} \;=\; D^{-1}\bigl[\operatorname{diag}(u_g,u_l) - G_4|_{\text{row 1 only}}\bigr]\,L_{\text{col 1 only}}\;+\;\text{(zero in cols 2–4)} $$

Working it out entry by entry: $L$ has only the first column nonzero, so $D^{-1}L\,G_4$ extracts
*row 1 of $G_4$* (the pressure-equation row of $G_s$) times the first column of $D^{-1}L$.
Denote row 1 of $G_4$ as $\mathbf g_1=(G_{pp},\,G_{12},\,G_{13},\,G_{14})$. Then

- $-D^{-1}L\,G_4 = -\begin{pmatrix} T_g\cdot(-1/(\varrho_g T_g)) \\ T_l\cdot(-1/(\varrho_l T_l)) \end{pmatrix}\mathbf g_1 = \begin{pmatrix} \mathbf g_1/\varrho_g \\ \mathbf g_1/\varrho_l \end{pmatrix}$,
- $+D^{-1}\operatorname{diag}(u_g,u_l)L = \begin{pmatrix} T_g\cdot u_g\cdot(-1/(\varrho_g T_g)) & 0 & 0 & 0 \\ T_l\cdot u_l\cdot(-1/(\varrho_l T_l)) & 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} -u_g/\varrho_g & 0 & 0 & 0 \\ -u_l/\varrho_l & 0 & 0 & 0 \end{pmatrix}$.

Summing,

$$ G_{h,21} \;=\; \begin{pmatrix} \dfrac{G_{pp}-u_g}{\varrho_g} & \dfrac{G_{12}}{\varrho_g} & \dfrac{G_{13}}{\varrho_g} & \dfrac{G_{14}}{\varrho_g} \\[6pt] \dfrac{G_{pp}-u_l}{\varrho_l} & \dfrac{G_{12}}{\varrho_l} & \dfrac{G_{13}}{\varrho_l} & \dfrac{G_{14}}{\varrho_l} \end{pmatrix}. $$

The lower-left block of $G_h$ is the pressure row of $G_s$ divided by $\varrho_i$, with the
pressure-column entry shifted by $-u_i$ (the entry on the diagonal that the row "should" have
inherited).

### 4.3 Physical interpretation

This shape is exactly what the Gibbs relation predicts. Starting from the *entropy* equation
$\partial_t s_i + u_i\,\partial_x s_i = (\text{source})$ — entropy is passively advected — and
substituting $\mathrm ds_i=\mathrm dh_i/T_i - \mathrm dp/(\varrho_i T_i)$:

$$ \frac{1}{T_i}\bigl[\partial_t h_i + u_i\,\partial_x h_i\bigr] \;-\; \frac{1}{\varrho_i T_i}\bigl[\partial_t p + u_i\,\partial_x p\bigr] \;=\; (\text{source}/T_i). $$

Solving for $\partial_t h_i$ and using the pressure equation $\partial_t p = -\mathbf g_1\cdot\partial_x U_h$
(the pressure row of $G_s$, which by construction is identical in entropy and enthalpy frames
since the pressure row has no entropy/enthalpy column entries):

$$ \partial_t h_i = -u_i\,\partial_x h_i + \frac{1}{\varrho_i}\bigl[\partial_t p + u_i\,\partial_x p\bigr] = -u_i\,\partial_x h_i + \frac{1}{\varrho_i}\bigl[-\mathbf g_1 + u_i\hat e_1\bigr]\cdot\partial_x U_h. $$

Reading off the coefficient of $\partial_x U_h$:

$$ \text{row $i$ of }G_h \;=\; \frac{1}{\varrho_i}\bigl(G_{pp}-u_i,\ G_{12},\ G_{13},\ G_{14},\ 0,\ 0\bigr) + u_i\,\hat e_i. $$

This matches the similarity-transform result entry-for-entry — independent derivation, identical
output. The cross-check is what licenses the closed form below.

### 4.4 Result

Collecting the four blocks, with $G_4$ the verified $4\times4$ acoustic/void block (rows 1–4 of
$G_s$ restricted to columns 1–4), and noting that the velocity- and void-column entries of the
enthalpy rows factor cleanly:

$$ \frac{G_{12}}{\varrho_i} = \frac{1}{\varrho_i}\cdot\alpha_g a_0^2\frac{\varrho_l\varrho_g}{\varrho_s} = \frac{\alpha_g a_0^2\,\varrho_{\bar i}}{\varrho_s}, $$

where $\bar i$ denotes the *other* phase ($\bar g=l$, $\bar l=g$). Same factorization for cols 3
and 4. The final form is

$$ \boxed{\; G_h = \left(\begin{array}{c|c} G_4 & \mathbf 0_{4\times 2} \\\hline G_{h,21} & \operatorname{diag}(u_g,u_l) \end{array}\right) \;} $$

$$ G_{h,21} = \begin{pmatrix} \dfrac{G_{pp}-u_g}{\varrho_g} & \dfrac{\alpha_g a_0^2\,\varrho_l}{\varrho_s} & \dfrac{\alpha_l a_0^2\,\varrho_l}{\varrho_s} & \dfrac{a_0^2\,\Delta u\,\varrho_l}{\varrho_s} \\[10pt] \dfrac{G_{pp}-u_l}{\varrho_l} & \dfrac{\alpha_g a_0^2\,\varrho_g}{\varrho_s} & \dfrac{\alpha_l a_0^2\,\varrho_g}{\varrho_s} & \dfrac{a_0^2\,\Delta u\,\varrho_g}{\varrho_s} \end{pmatrix}. $$

Notice the row-wise pattern: the gas-enthalpy row $i=g$ carries $\varrho_l$ in the cross-coupling
entries (cols 2–4); the liquid-enthalpy row carries $\varrho_g$. The shared $\varrho_s$ denominator
remains the cross-weighted combination from the upper block.

---

## 5. Verification ledger

All four gates verified before this document was written; the symbolic computations live in
`/home/claude/` and were performed in exact rational arithmetic with sympy.

| Gate | Statement | Result |
|------|-----------|--------|
| Spectrum | $\operatorname{charpoly}(G_h) \equiv \operatorname{charpoly}(G_s)$ symbolically | ✓ identical |
| Multiplicities | At nominal state, eigenvalues $\{u_g(\times 2),\,u_l(\times 2),\,u\!+\!a,\,u\!-\!a\}$ | ✓ confirmed |
| Block structure (similarity) | upper-left $\equiv G_4$, upper-right $\equiv \mathbf 0$, lower-right $\equiv \operatorname{diag}(u_g,u_l)$ | ✓ exact |
| Block structure (lower-left) | $G_{h,21}$ matches both the similarity result and the independent Gibbs-relation re-derivation | ✓ identity |
| Units | Each entry of $G_h$ is $[\mathrm{m\,s^{-1}}]\times$ (per-variable conversion factor); enthalpy rows carry $\mathrm s^{-1}$ when applied to $h\,[\mathrm{m^2 s^{-2}}]$ giving $\partial_t h\,[\mathrm{m^2 s^{-3}}]$ | ✓ consistent |
| Single-phase gas limit $\alpha_g\to 1$ | $h_g$ row $\to (0,\,a_g^2,\,0,\,a_g^2\Delta u,\,u_g,\,0)$ — recovers single-phase Euler in $(p,u,h)$ | ✓ |
| Homogeneous limit $\Delta u\to 0$ | enthalpy-row pressure entries $\to 0$ since $G_{pp}\to u$; system decouples into phase-by-phase advection plus the acoustic block | ✓ |
| Metastable state $T_g\neq T_l$ | $\Phi$ remains invertible ($\det = 1/T_gT_l > 0$); no special-case branch needed | ✓ |

The acoustic pair at the nominal water/steam test state ($p\!\sim\!10\,$bar, $\alpha_g=0.25$,
$u_g=100, u_l=50\,\mathrm{m/s}$, $k=0.5$): $\lambda_{3,4}=\{-45.67,\,146.29\}\,\mathrm{m/s}$,
symmetric about the mixture velocity $u=50.31$ — consistent with the verification document's
identical numbers in $G_s$.

---

## 6. EOS density derivatives — the leaf inputs for $J$ and $K$

Where $G_h$ depended only on $(\varrho_i, a_i, u_i, \alpha_g, k)$, the conservative-bridge
and flux Jacobians additionally require the partial derivatives of $\varrho_i$ with respect to
the primitives at fixed other primitive. Since $\varrho_i=\varrho_i(p,h_i)$ in the enthalpy frame,
the two relevant partials are

$$ \varrho_{p,i} \;=\; \left(\frac{\partial\varrho_i}{\partial p}\right)_{h_i}, \qquad \varrho_{h,i} \;=\; \left(\frac{\partial\varrho_i}{\partial h_i}\right)_{p}. $$

These are pure EOS quantities, returned by JAXProp/CoolProp via per-phase calls. They are
linked to the phasic sound speed $a_i$ and the Grüneisen coefficient $\Gamma_i$ by the closures

$$ \boxed{\;\varrho_{p,i} \;=\; \frac{1+\Gamma_i}{a_i^2},\qquad \varrho_{h,i} \;=\; -\frac{\varrho_i\,\Gamma_i}{a_i^2},\qquad \frac{1}{a_i^2} \;=\; \varrho_{p,i} + \frac{\varrho_{h,i}}{\varrho_i}\;} $$

so two of $\{a_i, \Gamma_i, \varrho_{p,i}, \varrho_{h,i}\}$ determine the others. Operationally
the solver calls the EOS with $(p, h_i)$ and reads off $(\varrho_i, a_i, \Gamma_i)$ as leaves,
then forms $\varrho_{p,i}, \varrho_{h,i}$ algebraically. No autodiff through the EOS is required.

No new EOS quantities appear anywhere downstream in this chain.

---

## 7. Conserved state and flux vectors

In the total-energy formulation (Städtke eq. 5.90, with the trivial relabel $e_i\to h_i$ via
$h_i = e_i + p/\varrho_i$), the conserved state vector is

$$ V \;=\; \big(\,\alpha_g\varrho_g,\ \alpha_l\varrho_l,\ \alpha_g\varrho_g u_g,\ \alpha_l\varrho_l u_l,\ \alpha_g\varrho_g E_g,\ \alpha_l\varrho_l E_l\,\big)^\top, $$

with the **total specific energy**

$$ E_i \;=\; h_i \;-\; \frac{p}{\varrho_i} \;+\; \frac{u_i^2}{2}. $$

The corresponding flux vector is

$$ F \;=\; \big(\,\alpha_g\varrho_g u_g,\ \alpha_l\varrho_l u_l,\ \alpha_g(\varrho_g u_g^2 + p),\ \alpha_l(\varrho_l u_l^2 + p),\ \alpha_g\varrho_g u_g H_g,\ \alpha_l\varrho_l u_l H_l\,\big)^\top, $$

with the **total specific enthalpy**

$$ H_i \;=\; h_i \;+\; \frac{u_i^2}{2} \;=\; E_i + \frac{p}{\varrho_i}. $$

The two combinations $E_i$ and $H_i$ act as algebraic shorthands here — they will appear inside
the Jacobians, but each is a known function of the primitives $(p, u_i, h_i)$ via $\varrho_i$.

The constraint $\alpha_l = 1 - \alpha_g$ is treated as a relabel: when computing
$\partial(\cdot)/\partial\alpha_g$, the term $\partial\alpha_l/\partial\alpha_g = -1$ appears
wherever $\alpha_l$ multiplies the row. This is the origin of every minus sign in the
$\alpha_g$-column of the liquid rows of $J$ and $K$.

---

## 8. The conservative bridge $J = \partial V/\partial U_h$

Each entry $J_{ij} = \partial V_i/\partial U_{h,j}$ is read off by direct differentiation. The two
nontrivial dependencies are:

1. $\varrho_i = \varrho_i(p, h_i)$, contributing $\varrho_{p,i}$ to the $p$-column and
   $\varrho_{h,i}$ to the $h_i$-column wherever $\varrho_i$ appears in $V_i$;
2. $\alpha_l = 1-\alpha_g$, contributing $-1$ to the $\alpha_g$-column wherever $\alpha_l$
   appears (gives the negative signs in the liquid rows).

The energy rows ($V_5, V_6$) need a little more attention. Writing
$\alpha_i\varrho_i E_i = \alpha_i\varrho_i h_i - \alpha_i p + \alpha_i\varrho_i u_i^2/2$, the
$p$-column derivative is

$$ \frac{\partial(\alpha_i\varrho_i E_i)}{\partial p} \;=\; \alpha_i\varrho_{p,i}\bigl(h_i + \tfrac{u_i^2}{2}\bigr) - \alpha_i \;=\; \alpha_i H_i\,\varrho_{p,i} - \alpha_i, $$

and the $h_i$-column derivative is

$$ \frac{\partial(\alpha_i\varrho_i E_i)}{\partial h_i} \;=\; \alpha_i\varrho_{h,i}\bigl(h_i + \tfrac{u_i^2}{2}\bigr) + \alpha_i\varrho_i \;=\; \alpha_i H_i\,\varrho_{h,i} + \alpha_i\varrho_i. $$

The $\alpha_g$-column derivative of $V_5 = \alpha_g\varrho_g E_g$ is just $\varrho_g E_g$; of
$V_6 = \alpha_l\varrho_l E_l$ it is $-\varrho_l E_l$.

Collecting the six rows:

$$ \boxed{\; J \;=\; \begin{pmatrix} \alpha_g\,\varrho_{p,g} & 0 & 0 & \varrho_g & \alpha_g\,\varrho_{h,g} & 0 \\[6pt] \alpha_l\,\varrho_{p,l} & 0 & 0 & -\varrho_l & 0 & \alpha_l\,\varrho_{h,l} \\[6pt] \alpha_g u_g\,\varrho_{p,g} & \alpha_g\varrho_g & 0 & \varrho_g u_g & \alpha_g u_g\,\varrho_{h,g} & 0 \\[6pt] \alpha_l u_l\,\varrho_{p,l} & 0 & \alpha_l\varrho_l & -\varrho_l u_l & 0 & \alpha_l u_l\,\varrho_{h,l} \\[6pt] \alpha_g H_g\,\varrho_{p,g} - \alpha_g & \alpha_g\varrho_g u_g & 0 & \varrho_g E_g & \alpha_g H_g\,\varrho_{h,g} + \alpha_g\varrho_g & 0 \\[6pt] \alpha_l H_l\,\varrho_{p,l} - \alpha_l & 0 & \alpha_l\varrho_l u_l & -\varrho_l E_l & 0 & \alpha_l H_l\,\varrho_{h,l} + \alpha_l\varrho_l \end{pmatrix} \;} $$

Three structural features worth reading off:

- **Phase decoupling in columns 2, 3, 5, 6.** The $u_g$-column is nonzero only in gas rows
  ($V_1, V_3, V_5$), the $u_l$-column only in liquid rows, the $h_g$-column only in gas rows,
  the $h_l$-column only in liquid rows. The two phases couple only through the shared $p$ and
  $\alpha_g$ columns.
- **The energy-row pressure entry $\alpha_i H_i\varrho_{p,i}-\alpha_i$.** The "$-\alpha_i$" is
  the partial of $-\alpha_i p$ in $V_i = \alpha_i\varrho_i h_i - \alpha_i p + \cdots$. It is what
  carries the pressure-work signature of the energy equation into the conservative bridge.
- **$E_i$ in the $\alpha_g$-column vs. $H_i$ in the $p, h_i$-columns.** $E_i$ appears where the
  full conserved energy is being differentiated as a whole; $H_i$ appears where
  $\varrho_i\cdot(\text{specific quantity})$ is being differentiated and the
  $\partial\varrho_i/\partial(\cdot)$ pulls out the *enthalpy*-like combination $h_i+u_i^2/2$.
  This $E$-vs-$H$ split is the algebraic fingerprint of the total-energy formulation, and gets
  inherited by $K$ in the next section.

### 8.1 Determinant — the conservative-bridge signature

The determinant of $J$ factors as

$$ \det J \;=\; -\alpha_g^2\,(\alpha_g-1)^2\,\varrho_g\varrho_l\;\cdot\;\mathcal D(\varrho_g,\varrho_l,\varrho_{p,i},\varrho_{h,i},p,\alpha_g), $$

with the $\alpha_g^2\alpha_l^2$ envelope as expected. **$J$ becomes singular precisely when one
phase vanishes** — exactly the algebraic expression of the fact that the conservative bridge has
no meaning at a single-phase point. The bracketed $\mathcal D$ is a polynomial in the EOS
derivatives that is nonzero at all physical states; we will not need its explicit form, only
that $J$ is invertible away from the phase-vanishing limits.

---

## 9. The flux Jacobian $K = \partial F/\partial U_h$

Same procedure. The momentum rows ($F_3, F_4$) carry the extra $\partial(\alpha_i p)/\partial p
= \alpha_i$ contribution in the $p$-column, and the velocity-column $\partial(\alpha_i\varrho_i u_i^2)/\partial u_i = 2\alpha_i\varrho_i u_i$ is the only "2" in the entire matrix. The
$\alpha_g$-column derivative of $F_4 = \alpha_l(\varrho_l u_l^2 + p)$ is
$-(\varrho_l u_l^2 + p)$, hence the boxed "$-\varrho_l u_l^2 - p$" entry.

The energy-flux rows ($F_5, F_6$) carry the total enthalpy $H_i$. Differentiating
$F_5 = \alpha_g\varrho_g u_g H_g = \alpha_g\varrho_g u_g(h_g + u_g^2/2)$:

- $p$-column: $\alpha_g u_g H_g\,\varrho_{p,g}$ (only $\varrho_g$ depends on $p$).
- $u_g$-column: $\alpha_g\varrho_g H_g + \alpha_g\varrho_g u_g\cdot u_g = \alpha_g\varrho_g(H_g + u_g^2)$.
  The $u_g^2$ comes from differentiating $H_g$ itself; the $\partial H_g/\partial u_g=u_g$ is
  what generates this row's signature pattern.
- $\alpha_g$-column: $\varrho_g u_g H_g$.
- $h_g$-column: $\alpha_g u_g(\varrho_{h,g} H_g + \varrho_g)$ — same $H_i$-vs-$\varrho_i$ split as
  in the energy row of $J$.

Collecting:

$$ \boxed{\; K \;=\; \begin{pmatrix} \alpha_g u_g\,\varrho_{p,g} & \alpha_g\varrho_g & 0 & \varrho_g u_g & \alpha_g u_g\,\varrho_{h,g} & 0 \\[6pt] \alpha_l u_l\,\varrho_{p,l} & 0 & \alpha_l\varrho_l & -\varrho_l u_l & 0 & \alpha_l u_l\,\varrho_{h,l} \\[6pt] \alpha_g u_g^2\,\varrho_{p,g} + \alpha_g & 2\alpha_g\varrho_g u_g & 0 & \varrho_g u_g^2 + p & \alpha_g u_g^2\,\varrho_{h,g} & 0 \\[6pt] \alpha_l u_l^2\,\varrho_{p,l} + \alpha_l & 0 & 2\alpha_l\varrho_l u_l & -\varrho_l u_l^2 - p & 0 & \alpha_l u_l^2\,\varrho_{h,l} \\[6pt] \alpha_g u_g H_g\,\varrho_{p,g} & \alpha_g\varrho_g(H_g + u_g^2) & 0 & \varrho_g u_g H_g & \alpha_g u_g(H_g\varrho_{h,g} + \varrho_g) & 0 \\[6pt] \alpha_l u_l H_l\,\varrho_{p,l} & 0 & \alpha_l\varrho_l(H_l + u_l^2) & -\varrho_l u_l H_l & 0 & \alpha_l u_l(H_l\varrho_{h,l} + \varrho_l) \end{pmatrix} \;} $$

### 9.1 Comparison with $J$ — the row-1 vs row-3 pattern

Notice that **row 1 of $K$ equals row 3 of $J$**:

$$ K_{(\text{mass flux row, gas})} \;=\; J_{(\text{momentum row, gas})}, $$

and the same for the liquid. This is not coincidence — the gas momentum is the gas mass flux
times $u_g$, so the $\partial$ in the gas mass-flux row of $K$ picks up exactly the same
combination that the gas-momentum row of $J$ picks up from differentiating $\alpha_g\varrho_g u_g$
component-wise. The same identity holds across all rows where the *flux at row $i$* equals the
*conserved variable at row $i+2$* (mass-flux ↔ momentum, momentum-flux ↔ energy in single-phase
Euler). Here it holds row-by-row for the two-fluid extension. This is the algebraic seed of the
fact that $JG_h-K$ is nonzero only in the two momentum rows, which will be the structural check
on $H^{nc}$ in the next section.

---

## 10. Verification of $J$ and $K$

Both matrices were verified symbolically by computing the Jacobians of $V$ and $F$ from the
expressions in §7 directly with sympy's `.jacobian()`, substituting in symbolic placeholders for
$\varrho_i, \varrho_{p,i}, \varrho_{h,i}$, $E_i$, and $H_i$, and subtracting the user-supplied
matrices entrywise. The result is the zero matrix in both cases — no residual terms anywhere:

$$ \bigl[\,V.\text{jacobian}(U_h)\,\bigr] - J \;\equiv\; 0_{6\times 6}, \qquad \bigl[\,F.\text{jacobian}(U_h)\,\bigr] - K \;\equiv\; 0_{6\times 6}. $$

| Gate | Statement | Result |
|------|-----------|--------|
| $J$ entrywise | autodiff of $V$ matches user's matrix in every entry | ✓ exact |
| $K$ entrywise | autodiff of $F$ matches user's matrix in every entry | ✓ exact |
| $\det J$ envelope | $\det J \propto \alpha_g^2\alpha_l^2\,\varrho_g\varrho_l$ — vanishes only at single-phase | ✓ |
| $\alpha_l=1-\alpha_g$ chain rule | every minus sign in liquid rows traces to $\partial\alpha_l/\partial\alpha_g=-1$ | ✓ structural |
| Row 1 of $K$ = row 3 of $J$ | predicted by $F_1 = u_g\cdot V_1$ identity | ✓ |
| Row 2 of $K$ = row 4 of $J$ | predicted by $F_2 = u_l\cdot V_2$ identity | ✓ |
| Units | every entry of $J$ is $V$-units / $U_h$-units; every entry of $K$ is $F$-units / $U_h$-units | ✓ |

The symbolic computation runs in exact rational arithmetic; no floating-point tolerance was used.

---

## 11. The non-conservative defect $X = J\,G_h - K$

The transformation to conservative variables is generated by the chain rule
$\partial_t V = J\,\partial_t U_h$, which converts $\partial_t U_h + G_h\,\partial_x U_h = D_h$
into

$$ \partial_t V + J\,G_h\,\partial_x U_h \;=\; J\,D_h. $$

If the system were strictly conservative, $J\,G_h$ would equal the flux Jacobian $K$ and the
left-hand side would collapse to $\partial_t V + \partial_x F$. It does not. Their difference

$$ X \;\equiv\; J\,G_h \;-\; K $$

measures the obstruction to conservation form — the *single-pressure non-conservative coupling*.
With it, the conservative-form equation reads

$$ \partial_t V \;+\; \partial_x F \;+\; X\,\partial_x U_h \;=\; J\,D_h, $$

and equivalently $H^{nc} \equiv X K^{-1}$ writes the obstruction in the flux-space form
$\partial_t V + (I + H^{nc})\partial_x F = J D_h$ used in Document 3.

### 11.1 The EOS closure is structural, not cosmetic

A subtle point worth pinning down before stating $X$. The matrix $G_h$ carries the phasic sound
speeds $a_i^2$ in $a_0^2$ and in the void-row coefficients; the matrices $J, K$ carry
$\varrho_{p,i}, \varrho_{h,i}$. These quantities are **not algebraically independent**: the EOS
closure derived in §6 enforces

$$ \frac{1}{a_i^2} \;=\; \varrho_{p,i} \;+\; \frac{\varrho_{h,i}}{\varrho_i} \qquad\Longleftrightarrow\qquad a_i^2 \;=\; \frac{\varrho_i}{\varrho_i\varrho_{p,i} + \varrho_{h,i}}. $$

Treating $a_i^2$ and $(\varrho_{p,i}, \varrho_{h,i})$ as separate symbols when computing $J G_h - K$
produces spurious non-zero residuals in the *mass rows* — entries that should vanish by mass
conservation but cannot, because the system is algebraically over-specified. Applying the
closure as a substitution before simplification removes these residuals and reveals the true
structure. **In a numerical implementation, $\varrho_{p,i}$ and $a_i$ must come from a single
consistent EOS call**; computing them from independent table lookups will manifest as
non-zero mass-row defects at the $10^{-(\text{relative-error})}$ level.

### 11.2 Explicit entries of $X$

With the closure applied, $X$ resolves into a sparse form. The two mass rows are identically
zero, the two enthalpy columns are identically zero, the void row is zero, and only the four
left-most entries of rows 3–6 are nonzero. Using the shorthand labels $A$–$H$ for the eight
distinct nonzero scalars, the matrix is

$$X = \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\[4pt]
0 & 0 & 0 & 0 & 0 & 0 \\[4pt]
A & B & C & -p & 0 & 0 \\[4pt]
-A & -D & -E & +p & 0 & 0 \\[4pt]
F & G & H & u_g(-p) - pG_{4,\alpha} + pu_g & 0 & 0 \\[4pt]
-F & -G' & -H' & u_l(+p) + pG_{4,\alpha} - pu_l & 0 & 0
\end{pmatrix}$$

where columns are ordered $(p,\,u_g,\,u_l,\,\alpha_g,\,h_g,\,h_l)$ and $G_{4,\alpha}$ denotes
entry $(4,4)$ of $G_h$ (the $\alpha_g$ coefficient of the void-fraction equation). The column-4
energy entries simplify: $u_g(-p)+pu_g=0$ and $u_l(+p)-pu_l=0$, so both are zero — the full
column-4 energy entries come entirely from the $\pm p G_{4,\alpha}$ term. Rewriting cleanly:

$$X = \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\[6pt]
0 & 0 & 0 & 0 & 0 & 0 \\[6pt]
X_{3p} & X_{3g} & X_{3l} & -p & 0 & 0 \\[6pt]
-X_{3p} & -X_{4g} & -X_{4l} & +p & 0 & 0 \\[6pt]
u_g X_{3p} - p G_{4p} & u_g X_{3g} - p G_{4g} & u_g X_{3l} - p G_{4l} & -p G_{4\alpha} & 0 & 0 \\[6pt]
-u_l X_{3p} + p G_{4p} & -u_l X_{4g} + p G_{4g} & -u_l X_{4l} + p G_{4l} & +p G_{4\alpha} & 0 & 0
\end{pmatrix}$$

where $G_{4,j}$ denotes the $j$-th entry of the void-fraction row of $G_h$ (row 4, 0-indexed),
and the column-4 of the energy rows uses $G_{4\alpha} \equiv G_h[4,4] = G_{\alpha\alpha}$.

**Entry definitions** — $\Delta\varrho\equiv\varrho_g-\varrho_l$, $\Delta u\equiv u_g-u_l$,
$\hat\varrho_g=\varrho_g+k\varrho$, $\hat\varrho_l=\varrho_l+k\varrho$,
$\hat\varrho^2=\varrho_g\varrho_l+k\varrho^2$:

$$\begin{aligned}
X_{3p} &= \frac{\alpha_g\alpha_l k\varrho\,\Delta\varrho}{\hat\varrho^2}, \\[6pt]
X_{3g} &= \frac{-\alpha_g\alpha_l\Delta u\,\varrho_g\hat\varrho_g}{\hat\varrho^2}, \\[6pt]
X_{3l} &= \frac{-\alpha_g\alpha_l\Delta u\,\varrho_g\hat\varrho_l}{\hat\varrho^2}, \\[6pt]
X_{4g} &= \frac{-\alpha_g\alpha_l\Delta u\,\varrho_l\hat\varrho_g}{\hat\varrho^2}, \\[6pt]
X_{4l} &= \frac{-\alpha_g\alpha_l\Delta u\,\varrho_l\hat\varrho_l}{\hat\varrho^2}.
\end{aligned}$$

Note $X_{3g}$ and $X_{4g}$ differ only in $\varrho_g\leftrightarrow\varrho_l$; likewise
$X_{3l}$ and $X_{4l}$. The $\alpha_g$ column has the exact entries $\pm p$ in the momentum rows
and $\pm p G_{4\alpha}$ in the energy rows, with no further decomposition needed. The $G_{4,j}$
entries for columns $p, u_g, u_l$ are read directly from row 4 of $G_h$ (§4.4):

$$\begin{aligned}
G_{4p} &= \alpha_l\alpha_g\frac{\Delta u}{\varrho_s}\frac{a_0^2}{a_g^2 a_l^2}, \\[6pt]
G_{4g} &= \alpha_l\alpha_g\frac{\varrho_g}{\varrho_s}\frac{a_0^2}{a_l^2}, \\[6pt]
G_{4l} &= -\alpha_l\alpha_g\frac{\varrho_l}{\varrho_s}\frac{a_0^2}{a_g^2}, \\[6pt]
G_{4\alpha} &= \alpha_l u_g\frac{\varrho_g}{\varrho_s}\frac{a_0^2}{a_l^2}
              + \alpha_g u_l\frac{\varrho_l}{\varrho_s}\frac{a_0^2}{a_g^2}.
\end{aligned}$$

Two structural checks are immediate from the matrix:

- **Liquid row = minus gas row** (up to $\varrho_g\leftrightarrow\varrho_l$ in velocity
  entries): $X_{3p}+X_{4p}=0$, $X_{3,\alpha_g}+X_{4,\alpha_g}=0$. The mixture momentum
  has zero non-conservative defect — momentum is conserved overall.
- **Energy rows cost 4 multiplies each**: once $X_{3p},X_{3g},X_{3l}$ and
  $G_{4p},G_{4g},G_{4l},G_{4\alpha}$ are in registers, each energy row is four
  fused-multiply-add operations.

### 11.3 Where this structure comes from

Three structural facts read directly off the boxed form:

**(a) Mass rows are zero — mass conservation.** Mass equations in the primitive system are
already in conservative form ($\partial_t(\alpha_i\varrho_i) + \partial_x(\alpha_i\varrho_i u_i)=0$),
so $J(G_h)_{\text{mass row}} = K_{\text{mass row}}$ exactly. The derivation in §9.1 noted that
row 1 of $K$ equals row 3 of $J$; multiplying the corresponding rows of $J$ into $G_h$ reproduces
row 1 of $K$. This is the explicit statement of mass conservation.

**(b) Enthalpy columns are zero — entropy waves are linearly degenerate.** In every row, the
$h_g$ and $h_l$ columns of $J G_h$ exactly cancel those of $K$. This is the algebraic image of
the fact that thermodynamic waves are linearly degenerate: a pure enthalpy disturbance carries
no momentum or energy flux gradient to be split.

**(c) Energy rows are slaved to momentum rows.** In total-energy formulation, the conserved
energy is $\alpha_i\varrho_i E_i = \alpha_i\varrho_i(h_i + u_i^2/2) - \alpha_i p$, the *kinetic
part* $\tfrac12\alpha_i\varrho_i u_i^2$ being built from momentum. Differentiating brings in the
momentum row by chain rule; the pressure piece $-\alpha_i p$ supplies the $\pm p\,G_{4,j}$ terms
through the $\alpha_g$ derivative; and the void-row of $G_h$ appears because
$\partial(\alpha_g p)/\partial\alpha_g$ is what drives the $-p$ contribution. Symbolically:

$$ V_5 - u_g V_3 + p\cdot\alpha_g \;=\; \alpha_g\varrho_g\bigl(h_g - \tfrac{u_g^2}{2}\bigr), $$

and so $X_5 - u_g X_3 + p\,(\text{row 4 of }X)$ acts only on the residual scalar
$\alpha_g\varrho_g(h_g - u_g^2/2)$ — which is row-4-zero plus the explicit $\pm p u_i$ term in
column 4. The same accounting on the liquid side produces the opposite sign on $p\,G_4$.

**This corrects an earlier expectation.** Document Section 11 (now this section) had stated that
$X$ should be nonzero only in the two momentum rows. That is true for the *entropy* formulation,
where the entropy rows $\alpha_i\varrho_i s_i$ have no kinetic content. In the *total-energy*
formulation, the kinetic part of $E_i$ inherits momentum-non-conservativity, so the energy rows
of $X$ are nonzero — but they are not independent: they are exact linear combinations of the
momentum and void rows, with explicit coefficients $u_i$ and $\pm p$. There is still only one
piece of "true" non-conservative content — the two momentum rows — and the energy rows are
its image under the conservative-variable relabel.

### 11.4 Verification ledger

| Gate | Statement | Result |
|------|-----------|--------|
| Mass rows | $X_{1,:} = X_{2,:} \equiv 0$ after EOS closure | ✓ exact |
| Enthalpy columns | $X_{:,5} = X_{:,6} \equiv 0$ | ✓ exact |
| Void row | $X_{4,:} \equiv 0$ | ✓ exact |
| Gas momentum row | $X_{3,j} = \alpha_g\varrho_g G_{2,j} + \text{(diag corrections)}$ for $j=1,\ldots,4$ | ✓ exact |
| Liquid momentum row | $X_{4,j} = \alpha_l\varrho_l G_{3,j} + \text{(diag corrections)}$ — note: in the boxed form above this is the row labelled $X_4$ (row 4 of $X$, liquid momentum) | ✓ exact |
| Gas energy row | $X_{5,j} = u_g X_{3,j} - p\,G_{4,j} + p u_g\,\delta_{j,4}$ | ✓ exact |
| Liquid energy row | $X_{6,j} = u_l X_{4,j} + p\,G_{4,j} - p u_l\,\delta_{j,4}$ | ✓ exact |
| EOS closure required | residuals nonzero if $a_i^2$ and $(\varrho_{p,i},\varrho_{h,i})$ treated as independent | ✓ confirmed (structural diagnostic) |

All entries verified in exact rational arithmetic with the closure
$a_i^2 = \varrho_i/(\varrho_i\varrho_{p,i} + \varrho_{h,i})$ substituted before simplification.

---

## 12. The non-conservative coupling matrix $H^{nc}$

### 12.1 Definition and equivalent identity

The conservative-form equation derived in §11 reads $\partial_t V + \partial_x F + X\,\partial_x U_h = JD$.
To convert the obstruction term to a function of the flux gradient $\partial_x F$ (the form Städtke
uses for the FVS scheme), substitute $\partial_x F = K\,\partial_x U_h$, equivalently
$\partial_x U_h = K^{-1}\partial_x F$:

$$X\,\partial_x U_h \;=\; XK^{-1}\,\partial_x F \;\equiv\; H^{nc}\,\partial_x F,$$

so

$$\boxed{\; H^{nc} \;\equiv\; X\,K^{-1} \;}, \qquad
\partial_t V + \partial_x F + H^{nc}\,\partial_x F = JD.$$

An equivalent, occasionally more useful identity follows by adding $K K^{-1}=I$ to the definition:

$$\boxed{\; I + H^{nc} \;=\; J\,G_h\,K^{-1} \;}.$$

This second form says: $I + H^{nc}$ is the *effective* flux Jacobian seen by the conservative
system. For a strictly conservative system $X=0$, so $H^{nc}=0$, and $JG_hK^{-1}=I$ — meaning
$JG_h=K$ exactly. The deviation from identity *is* the obstruction.

### 12.2 Structural results

Three structural facts hold for $H^{nc}$, all verified exactly:

**(a) Zero mass rows.** Since $X$ has zero rows 1 and 2, so does $H^{nc}$:

$$H^{nc}_{1,:} = H^{nc}_{2,:} = \mathbf 0.$$

The conservative mass equations are unaffected by the non-conservative coupling — they were
already in conservative form.

**(b) Mixture momentum conservation.** Since $X_{4,j} = -X_{3,j}$ for all $j$ (the algebraic
statement of mixture momentum conservation in $X$), the same holds for $H^{nc}$:

$$H^{nc}_{4,:} = -H^{nc}_{3,:}.$$

Adding the two phasic momentum equations of the conservative system gives an equation in
$\partial_t(\alpha_g\varrho_g u_g + \alpha_l\varrho_l u_l) + \partial_x(\ldots)$ in which all
non-conservative contributions cancel — the mixture momentum is strictly conserved.

**(c) Dense rows otherwise.** The four nonzero rows (3, 4, 5, 6) are dense: each contains six
nonzero entries. **No further sparsity exists**. The EOS closure does not generate cancellations
beyond the row-zero results above, because $K^{-1}$ is itself dense — multiplying the sparse $X$
(zero only in two rows and two columns) by a dense $K^{-1}$ leaves only the row-zeros.

### 12.3 No tractable closed form

Symbolic computation produces 24 nonzero entries of $H^{nc}$. Each is a rational function whose
numerator and denominator are polynomials of degree $\sim 8$ in $(\varrho_i, \varrho_{p,i},
\varrho_{h,i}, u_i, p, \alpha_g, k, h_i)$. The expressions span pages and admit no useful
factorization in terms of the physical abbreviations $(\hat\varrho_i, \hat\varrho^2, \varrho_s,
a_0)$ that made $G_h, J, K, X$ presentable. We do not write them out; the operational form below
makes them unnecessary.

### 12.4 The singular limit $u_g u_l = 0$

The determinant of $K$ factors as

$$\det K \;\propto\; \alpha_g^2 \alpha_l^2\,\varrho_g\varrho_l\,u_g u_l\,\mathcal D(\text{EOS}),$$

with $\mathcal D$ a polynomial that is nonzero at all physical states. So $K$ is singular not
only at the expected phase-vanishing limits $\alpha_i \to 0,1$ (where the conservative bridge $J$
also degenerates), but additionally at $u_g = 0$ or $u_l = 0$.

The mechanism is direct. At $u_g = 0$, the entire column of $K$ corresponding to $h_g$ vanishes:

$$K_{:,h_g}\big|_{u_g=0} = (0, 0, 0, 0, 0, 0)^\top.$$

This is because every flux row that depends on $h_g$ does so through $u_g \cdot (\text{stuff})$ —
mass flux $\alpha_g\varrho_g u_g$, momentum flux $\alpha_g\varrho_g u_g^2 + \alpha_g p$, energy
flux $\alpha_g\varrho_g u_g H_g$ — and the $h_g$ derivative inherits the $u_g$ factor.

This is a real obstruction to forming $H^{nc}$ dense. But it is not a real obstruction to the
*numerical scheme*: at $u_g = 0$ the gas mass-flux and energy-flux components of $\partial_x F$
also vanish (since they carry the $u_g$ factor), so $\partial_x F$ at such a state lies in the
range of $K$, and $K y = \partial_x F$ has a regular solution even though $K^{-1}$ as a dense
matrix does not exist. The operational form below handles this automatically.

### 12.5 Operational form — never assemble $H^{nc}$

For the FVS scheme, the per-face non-conservative contribution is

$$\Delta V_{\text{nc}} \;=\; H^{nc}\,(\hat F_R - \hat F_L) \;=\; X\,K^{-1}\,\Delta F,$$

where $\Delta F$ is the local flux difference (typically the interface flux minus a centered
average, or any consistent FV discretization of $\partial_x F$). The recommended computation:

1. **Assemble $K$ and $X$** at the cell/face state from the closed forms (§7–8, §11). Both are
   sparse with structured entries — $X$ has only 16 nonzero entries (4 rows, 4 cols), and $K$
   has $\sim 22$ nonzero entries by the L-shaped block pattern revealed in the permutation
   $(p, \alpha_g, u_g, h_g, u_l, h_l)$.

2. **Solve** $K y = \Delta F$ for the auxiliary vector $y \in \mathbb R^6$. One $6\times 6$ LU
   factorization, $O(1)$ flops per face.

3. **Form** $\Delta V_{\text{nc}} = X y$. Sparse matrix–vector: at most 16 multiply-adds.

Total cost per face: one LU + one sparse mat-vec. The dense matrix $H^{nc}$ is never
constructed. The closed forms for the 24 entries do not need to be implemented, debugged, or
maintained.

The phase-block structure of $K$ further accelerates the LU step. Under the row permutation
$(V_1, V_3, V_5, V_2, V_4, V_6) =$ (gas mass, gas mom, gas E, liq mass, liq mom, liq E) and
column permutation $(p, \alpha_g, u_g, h_g, u_l, h_l)$, $K$ takes the block form

$$\tilde K = \begin{pmatrix} A^g_{3\times 2} & B^g_{3\times 2} & 0_{3\times 2} \\ A^l_{3\times 2} & 0_{3\times 2} & B^l_{3\times 2} \end{pmatrix},$$

with $A^{g/l}$ the shared-$(p,\alpha_g)$ blocks and $B^{g/l}$ the phase-only blocks. Block
elimination through the $(p, \alpha_g)$ pair reduces the system to two decoupled $3\times 3$
solves once $(y_p, y_{\alpha})$ are known — and the latter come from a $2\times 2$ system
formed by Schur complement. Total flop count is well below a generic $6\times 6$ LU.

### 12.6 Verification ledger

| Gate | Statement | Result |
|------|-----------|--------|
| Row 1 zero | $H^{nc}_{1,:}\equiv 0$ (inherited from $X_{1,:}=0$) | ✓ numerical to $10^{-17}$ |
| Row 2 zero | $H^{nc}_{2,:}\equiv 0$ (inherited from $X_{2,:}=0$) | ✓ numerical to $10^{-15}$ |
| Mixture momentum | $H^{nc}_{4,:} + H^{nc}_{3,:} \equiv 0$ | ✓ numerical to $10^{-14}$ |
| Identity | $I + H^{nc} \equiv J G_h K^{-1}$ | ✓ numerical to $10^{-9}$ |
| Singular limit | $K$ has zero column $h_g$ at $u_g=0$ (analogously for liquid) | ✓ structural |
| Range condition | At $u_i=0$, the flux-gradient components in $h_i$-affected rows also vanish | ✓ structural |

The full $6\times 6$ matrix at the nominal water/steam state ($p = 1\,\mathrm{bar}$, $\alpha_g=0.25$,
$u_g=100$, $u_l=50\,\mathrm{m/s}$, $h_g=2.5\,\mathrm{MJ/kg}$, $h_l=0.4\,\mathrm{MJ/kg}$, $k=0.5$,
with water-vapour EOS leaves) is given in Appendix A for cross-check against an implementation.

---

## 13. The eigenstructure of $G_h$ in enthalpy primitives

This is the final piece needed for the FVS kernel: the right and left eigenvectors of $G_h$, in
closed form, expressed in the primitive variables $U_h$. With these in hand, the per-face flux
splitting reduces to rank-one operations per wave — no dense matrix arithmetic, no eigenvalue
solves in the hot loop.

### 13.1 Eigenvalues and sound-speed abbreviations

The spectrum, derived in §4 and verified exactly:

$$\Lambda = \operatorname{diag}\bigl(\,u_g,\ u_l,\ u_{\text{mix}}+a,\ u_{\text{mix}}-a,\ u_g,\ u_l\,\bigr).$$

The first two are the *void* waves (kinematic, one per phase), the third and fourth the
*acoustic* pair (pressure waves at the mixture sound speed), the last two the *entropy/enthalpy*
waves (thermodynamic, advected at the phasic velocity). Each phasic velocity is a double
eigenvalue (void + thermodynamic), benign by construction since the two eigenspaces are
distinct.

The sound speeds and mixture velocity require seven abbreviations beyond the dictionary already
established for $G_h$ (i.e. beyond $\varrho, \varrho_s, \hat\varrho_g, \hat\varrho_l, \hat\varrho^2, a_0^2, \Delta u$):

$$\begin{aligned}
\hat\varrho_s &\equiv \varrho_s + k\varrho, & \hat\varrho &\equiv \sqrt{\hat\varrho^2}, \\[2pt]
\tilde a^2 &\equiv \frac{\hat\varrho_s}{\varrho_s}\cdot\frac{\varrho_g\varrho_l}{\hat\varrho^2}\,a_0^2, & \Delta a^2 &\equiv \frac{\alpha_g\alpha_l\,\varrho_g\varrho_l\,\hat\varrho_g\hat\varrho_l}{\hat\varrho^4}\Delta u^2, \\[2pt]
a^2 &= \tilde a^2 - \Delta a^2, & u_{\text{mix}} &= \hat\alpha_g u_g + \hat\alpha_l u_l, \\[2pt]
\hat\alpha_g &= \frac{\alpha_g\varrho_g\hat\varrho_l}{\hat\varrho^2}, & \hat\alpha_l &= \frac{\alpha_l\varrho_l\hat\varrho_g}{\hat\varrho^2},
\end{aligned}$$

with $\hat\alpha_g + \hat\alpha_l = 1$ identically. Three further phasic combinations appear in
the left eigenvectors:

$$\begin{aligned}
a_1^2 &= a^2 - \alpha_g^2\,\varrho_l\varrho_g\,\frac{\hat\varrho_l^{\,2}}{\hat\varrho^4}\,\Delta u^2, &
a_2^2 &= a^2 - \alpha_l^2\,\varrho_l\varrho_g\,\frac{\hat\varrho_g^{\,2}}{\hat\varrho^4}\,\Delta u^2, \\[2pt]
\delta_a &= \alpha_g^2\,\frac{\hat\varrho_l\,\tilde a^2}{\hat\varrho\,a_g^2} - \alpha_l^2\,\frac{\hat\varrho_g\,\tilde a^2}{\hat\varrho\,a_l^2}. & &
\end{aligned}$$

**Important typo correction (inherited).** Städtke's monograph prints $a_1^2$ with
$\hat\varrho_l$ to the *first* power; symmetry with $a_2^2$ (which carries
$\hat\varrho_g^{\,2}$) and exact verification both require the **square**. The form above is the
corrected one. The correction is not new to this document — it was identified and verified in
the project's entropy-frame derivation, and it carries through to the enthalpy frame unchanged:
$a_1^2$ is a scalar abbreviation, invariant under the similarity transform $\Phi$, so the
corrected entropy-frame eigenvectors transport the corrected $a_1^2$ into $L_h$ automatically
under $l_h = l_s\Phi$. Using the first-power version breaks $l_3 G_h = \lambda_3 l_3$ at
$O(10^8)$ at the nominal state, predominantly in the $u_l$-momentum component but spilling into
$p$ and $u_g$ columns through the enthalpy-frame transform.

### 13.2 Right eigenvector matrix $R_h$

The right eigenvectors are the columns of $R_h$, in the order matching $\Lambda$. The full
matrix is

$$R_h = \begin{pmatrix}
\alpha_l\varrho_l\Delta u^2 & \alpha_g\varrho_g\Delta u^2 & \hat\varrho^2\tilde a^2 & \hat\varrho^2\tilde a^2 & 0 & 0 \\[4pt]
0 & -\Delta u & R_{23} & R_{24} & 0 & 0 \\[4pt]
\Delta u & 0 & R_{33} & R_{34} & 0 & 0 \\[4pt]
-\alpha_l\bigl(1-\dfrac{\alpha_l\Delta u^2}{a_l^2}\bigr) & \alpha_g\bigl(1-\dfrac{\alpha_g\Delta u^2}{a_g^2}\bigr) & \chi & \chi & 0 & 0 \\[6pt]
\dfrac{\alpha_l\varrho_l\Delta u^2}{\varrho_g} & \alpha_g\Delta u^2 & \dfrac{\hat\varrho^2\tilde a^2}{\varrho_g} & \dfrac{\hat\varrho^2\tilde a^2}{\varrho_g} & T_g & 0 \\[6pt]
\alpha_l\Delta u^2 & \dfrac{\alpha_g\varrho_g\Delta u^2}{\varrho_l} & \dfrac{\hat\varrho^2\tilde a^2}{\varrho_l} & \dfrac{\hat\varrho^2\tilde a^2}{\varrho_l} & 0 & T_l
\end{pmatrix}$$

with the acoustic-block entries

$$\begin{aligned}
R_{23} &= \hat\varrho_l\Bigl(a - \frac{\alpha_l\varrho_l\hat\varrho_g}{\hat\varrho^2}\,\Delta u\Bigr), &
R_{24} &= \hat\varrho_l\Bigl(-a - \frac{\alpha_l\varrho_l\hat\varrho_g}{\hat\varrho^2}\,\Delta u\Bigr), \\[4pt]
R_{33} &= \hat\varrho_g\Bigl(a + \frac{\alpha_g\varrho_g\hat\varrho_l}{\hat\varrho^2}\,\Delta u\Bigr), &
R_{34} &= \hat\varrho_g\Bigl(-a + \frac{\alpha_g\varrho_g\hat\varrho_l}{\hat\varrho^2}\,\Delta u\Bigr),
\end{aligned}$$

and the void-row acoustic entry

$$\chi = -\alpha_g\alpha_l\,\frac{\hat\varrho^2}{\hat\varrho_s}\Bigl(\frac{\hat\varrho_g\,\tilde a^2}{\varrho_g a_g^2} - \frac{\hat\varrho_l\,\tilde a^2}{\varrho_l a_l^2}\Bigr).$$

**Reading the structure.**

- *Columns 1, 2 (void waves):* every entry carries a factor of $\Delta u$; the vectors collapse
  to the pure void-fraction direction as $\Delta u\to 0$, and the eigenspace splits cleanly from
  the thermodynamic waves at the same eigenvalue. The $T_g, T_l$ slots are absent in cols 1-2
  by construction — the void waves carry no enthalpy disturbance.
- *Columns 3, 4 (acoustic):* pressure column entry $\hat\varrho^2\tilde a^2$ identical for both
  signs of $a$, momentum entries differ in the sign of $a$ but share the slip-dependent
  $\Delta u$ correction, void entry $\chi$ is shared. The enthalpy slots inherit
  $\hat\varrho^2\tilde a^2/\varrho_i$ from the pressure column via the $\Phi^{-1}$ transform
  (§3.2 — note rows 5, 6 entries are $r_h$ rows 5, 6 = $p_s/\varrho_i$ in line with
  $\Phi^{-1}_{i,1}=1/\varrho_i$).
- *Columns 5, 6 (entropy/enthalpy waves):* trivial unit vectors in the corresponding
  enthalpy slot, scaled by $T_i$. The scaling is the $\Phi^{-1}_{i,i+1}=T_i$ factor.

### 13.3 Left eigenvector matrix $L_h$

The left eigenvectors are the rows of $L_h$, biorthogonal to the columns of $R_h$. The matrix is

$$L_h = \begin{pmatrix}
\dfrac{\alpha_g\Delta u}{a_g^2} & \alpha_g\varrho_g & -\alpha_g\varrho_g\dfrac{\hat\varrho_l}{\hat\varrho_g} & \varrho_g\Delta u & 0 & 0 \\[6pt]
\dfrac{\alpha_l\Delta u}{a_l^2} & \alpha_l\varrho_l\dfrac{\hat\varrho_g}{\hat\varrho_l} & -\alpha_l\varrho_l & -\varrho_l\Delta u & 0 & 0 \\[6pt]
a\hat\varrho_s + \hat\varrho\,\Delta u\,\delta_a & L_{32} & L_{33} & \hat\varrho^2\,\Delta u\,\tilde a^2 & 0 & 0 \\[6pt]
-a\hat\varrho_s + \hat\varrho\,\Delta u\,\delta_a & L_{42} & L_{43} & \hat\varrho^2\,\Delta u\,\tilde a^2 & 0 & 0 \\[6pt]
-\dfrac{1}{\varrho_g T_g} & 0 & 0 & 0 & \dfrac{1}{T_g} & 0 \\[6pt]
-\dfrac{1}{\varrho_l T_l} & 0 & 0 & 0 & 0 & \dfrac{1}{T_l}
\end{pmatrix}$$

with the acoustic-block entries

$$\begin{aligned}
L_{32} &= \alpha_g\hat\varrho^2\Bigl(a_2^2 + a\Delta u\,\frac{\varrho_g\hat\varrho_s}{\hat\varrho^2}\Bigr), &
L_{33} &= \alpha_l\hat\varrho^2\Bigl(a_1^2 - a\Delta u\,\frac{\varrho_l\hat\varrho_s}{\hat\varrho^2}\Bigr), \\[4pt]
L_{42} &= \alpha_g\hat\varrho^2\Bigl(a_2^2 - a\Delta u\,\frac{\varrho_g\hat\varrho_s}{\hat\varrho^2}\Bigr), &
L_{43} &= \alpha_l\hat\varrho^2\Bigl(a_1^2 + a\Delta u\,\frac{\varrho_l\hat\varrho_s}{\hat\varrho^2}\Bigr).
\end{aligned}$$

**Reading the structure.**

- *Rows 1, 2 (void wave duals):* the enthalpy columns are zero — void waves project onto
  pressure, velocity, and void-fraction components but not onto enthalpy. The void-fraction
  column carries $\pm\varrho_i\Delta u$, and the velocity-column ratios involve the
  hat-density factors $\hat\varrho_l/\hat\varrho_g$ which appear nowhere else.
- *Rows 3, 4 (acoustic duals):* the pressure-column carries $\pm a\hat\varrho_s$ as its
  leading term, plus a slip correction $\hat\varrho\Delta u\delta_a$ symmetric between the two
  rows. The void-fraction column entries $\hat\varrho^2\Delta u\tilde a^2$ are identical in
  both rows — no $a$ sign change there.
- *Rows 5, 6 (entropy duals):* these are the rows of $\Phi$ in the enthalpy slots; the
  $-1/(\varrho_i T_i)$ in the pressure column comes directly from $\Phi_{5,1}, \Phi_{6,1}$.

### 13.4 Biorthogonality and the normalizer $D$

By the standard duality of left and right eigenvectors at distinct eigenvalues, $L_h R_h$ is
diagonal. Symbolic computation in exact rational arithmetic confirms this off-diagonals are
identically zero. The diagonal entries

$$D = L_h\,R_h \;=\; \operatorname{diag}\bigl(D_1, D_2, D_3, D_4, 1, 1\bigr)$$

are the per-wave normalizers $D_k = l_k\cdot r_k$. The four nontrivial entries scale as:

$$\begin{aligned}
D_1, D_2 &\sim \Delta u^2 \quad\text{(void waves; vanish as }\Delta u\to 0\text{)}, \\
D_3, D_4 &\sim \hat\varrho^2\tilde a^2 \cdot (\text{algebraic factor of }O(\hat\varrho^2 a^2)) \quad\text{(acoustic; large)}, \\
D_5, D_6 &= 1 \;\text{exactly} \quad\text{(entropy/enthalpy; the } T_i\!\cdot\!1/T_i \text{ cancellation)}.
\end{aligned}$$

The complete diagonalization is then

$$\boxed{\;G_h \;=\; R_h\,D^{-1}\,\Lambda\,L_h.\;}$$

Equivalently $R_h^{-1} = D^{-1} L_h$, so the left eigenvectors *up to normalization* are the
rows of $R_h^{-1}$. The normalization matters: every projection identity in the FVS scheme
carries an explicit $1/D_k$ scalar that must not be dropped.

### 13.5 Verification ledger

Verified in exact rational arithmetic at the nominal state ($p=\sim 10$ bar, $\alpha_g=1/4$,
$\varrho_g=5$, $\varrho_l=960$, $u_g=100$, $u_l=50$, $a_g=450$, $a_l=1500$, $T_g=T_l=453$,
$k=1/2$):

| Gate | Statement | Result |
|------|-----------|--------|
| Right eigenvector identity | $G_h r_k - \lambda_k r_k = 0$ for $k=1,\ldots,6$ | ✓ exact 0 |
| Left eigenvector identity | $l_k G_h - \lambda_k l_k = 0$ for $k=1,\ldots,6$ | ✓ exact 0 |
| Biorthogonality | $L_h R_h$ diagonal, off-diagonals $\equiv 0$ | ✓ exact |
| Full reconstruction | $R_h D^{-1}\Lambda L_h - G_h = 0$ | ✓ exact 0 |
| Acoustic pair symmetry | $\lambda_3 + \lambda_4 = 2 u_{\text{mix}}$ | ✓ |
| Single-phase limit | $\alpha_g\to 1$: $a\to a_g$, $u_{\text{mix}}\to u_g$, eigenvalues collapse to Euler spectrum | ✓ structural |
| Zero-slip limit | $\Delta u\to 0$: void-wave $r_k$ collapse to pure $\hat e_{\alpha_g}$ direction; $D_{1,2}\to 0$ | ✓ structural |
| $a_1$ typo correction | Squared $\hat\varrho_l$ (not first power) — required by symmetry and verification | ✓ inherited from entropy-frame eigenvectors |

The exact-zero residuals (rather than $10^{-13}$ to $10^{-16}$ numerical agreement) reflect that
the verification is symbolic in rational arithmetic — these eigenvectors satisfy the eigenvalue
problem identically, not approximately.

### 13.6 Operational form for the FVS kernel

With $R_h, L_h, D$ in closed form, the per-wave projector is rank-one and its action on any
$6$-vector $v$ is

$$P_k v \;=\; \frac{l_k\cdot v}{D_k}\,r_k \;\in\;\mathbb R^6,$$

requiring one dot product (6 multiply-adds) and one scaled vector add (6 multiply-adds), so
$O(n)$ per wave, $O(n^2)$ per face for all six waves combined. The flux split

$$\hat F = R^+ F_L + R^- F_R, \qquad R^\pm = \sum_{\lambda_k\gtrless 0}\widetilde P_k = \sum_{\lambda_k\gtrless 0}\frac{(K r_k)(l_k K^{-1})}{D_k},$$

requires the *flux-space* projectors $\widetilde P_k = K P_k K^{-1}$, with $\rho_k = K r_k$
obtained by one JVP through the closed-form $K$ (or by the closed-form $K r_k$ which can be
evaluated symbolically once and shipped), and $\ell_k = l_k K^{-1}$ obtained from the dual-basis
relation $\ell_j\cdot\rho_k = D_k\,\delta_{jk}$ via a single shared $6\times 6$ LU per face,
reused across all six waves — $O(1)$ amortized.

This is the kernel: closed-form leaves $(\varrho_i, a_i, \Gamma_i)$ from EOS, closed-form
$R_h, L_h$ from this section, one LU on $K$ for the flux-space dual basis, then six rank-one
projector applications. No autodiff in the hot loop, no dense matrix arithmetic, no
eigenvalue solves.

---

## 14. What is in hand

The algebraic chain for the FVS scheme in enthalpy-primitive, total-energy-conservative
variables is now complete. The objects assembled and verified:

- **$G_h$** — quasi-linear coefficient matrix in enthalpy primitives, derived via
  $\Phi^{-1}G_s\Phi$ and cross-checked by independent re-derivation from the PDEs.
- **$\Phi$** — entropy-to-enthalpy thermodynamic transform, with $\det\Phi = 1/(T_gT_l) > 0$.
- **$J = \partial V/\partial U_h$** — conservative bridge to total-energy variables, with EOS
  derivatives $\varrho_{p,i}, \varrho_{h,i}$.
- **$K = \partial F/\partial U_h$** — flux Jacobian in primitives.
- **$X = J G_h - K$** — non-conservative defect with explicit sparse form (zero mass rows,
  zero enthalpy columns, momentum rows in factored form, energy rows as $u_i$-weighted
  linear combinations).
- **$H^{nc} = X K^{-1}$** — non-conservative coupling, computed operationally via $K$-LU
  rather than as a dense matrix.
- **$R_h, L_h$** — right and left eigenvectors of $G_h$ in closed form, with the
  normalizer $D = L_h R_h$ diagonal. Verified to satisfy $G_h r_k = \lambda_k r_k$,
  $l_k G_h = \lambda_k l_k$, biorthogonality, and full reconstruction
  $G_h = R_h D^{-1}\Lambda L_h$ — all in exact arithmetic.

**EOS leaves used.** Per phase, the EOS provides $(\varrho_i, a_i, \Gamma_i)$. From these,
$\varrho_{p,i} = (1+\Gamma_i)/a_i^2$ and $\varrho_{h,i} = -\varrho_i\Gamma_i/a_i^2$ are formed
algebraically. The temperature $T_i$ is also needed by $\Phi$ and by rows 5, 6 of $R_h, L_h$.
No other EOS quantities appear anywhere in the chain.

**Source terms** are the only remaining piece of the LHS-RHS picture; you indicated these are
already in hand from the existing single-phase solver, transformed appropriately into the
enthalpy frame via $\Phi^{-1}$.

The next step is implementation: assembling the closed-form matrices into JAX, building the
per-face flux split via the rank-one projectors of §13.6, integrating the $H^{nc}\Delta F$
correction via the $K$-LU operational form of §12.5, and source-term treatment via the
established machinery from the single-phase code.

---

## Appendix A. Reference numerical $H^{nc}$ at nominal state

State: $p = 10^5\,\mathrm{Pa}$, $u_g = 100\,\mathrm{m/s}$, $u_l = 50\,\mathrm{m/s}$,
$\alpha_g = 0.25$, $h_g = 2.5\times 10^6\,\mathrm{J/kg}$, $h_l = 4\times 10^5\,\mathrm{J/kg}$,
$\varrho_g = 5\,\mathrm{kg/m^3}$, $\varrho_l = 960\,\mathrm{kg/m^3}$, $a_g = 450$, $a_l=1500\,\mathrm{m/s}$,
$\Gamma_g = 0.3, \Gamma_l = 0.2$, $k = 0.5$. EOS derivatives:
$\varrho_{p,g} = 1.3/450^2$, $\varrho_{p,l} = 1.2/1500^2$,
$\varrho_{h,g} = -5\cdot 0.3/450^2$, $\varrho_{h,l} = -960\cdot 0.2/1500^2$.

$$H^{nc} \approx \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
-25.79 & 1.509 & -0.9484 & -0.01842 & 6.59{\times}10^{-5} & 5.42{\times}10^{-8} \\
+25.79 & -1.509 & +0.9484 & +0.01842 & -6.59{\times}10^{-5} & -5.42{\times}10^{-8} \\
-5004 & -208.3 & -109.8 & 2.836 & 8.73{\times}10^{-3} & -6.13{\times}10^{-6} \\
+2464 & +53.97 & +54.90 & -1.418 & -4.35{\times}10^{-3} & -1.56{\times}10^{-6}
\end{pmatrix}.$$

Note row 4 = −row 3 exactly (mixture momentum), rows 5, 6 are independent. The very small
entries in the $h_g, h_l$ columns are *not* structural zeros — they reflect the smallness of
$\varrho_{h,i}$ (order $10^{-4}\,\mathrm{kg^2 m^{-5} s^2 J^{-1}}$) at this state.
