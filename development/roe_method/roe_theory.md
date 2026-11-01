# Roe flux: the implemented general-EOS formulation

This note describes the production method selected by `Scheme(flux="roe")`.
It is a description of the implemented algorithm, including its floating-point
policy, rather than an implementation proposal. The same construction supports
nozzle Euler and annular Euler with swirl. Its measured verification and
remaining limits are in [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md).

## 1. What Roe linearizes

At a fixed face, Roe replaces the nonlinear homogeneous flux relation by an
interface matrix $\widetilde A$ satisfying, in exact arithmetic,

$$
\widetilde A\,\Delta U=\Delta F,\qquad
\Delta f=f_R-f_L.
$$

The matrix has a real characteristic decomposition and approaches the physical
flux Jacobian when the two states coincide. These are stronger requirements than
evaluating the physical Jacobian at an arbitrary mean pressure and enthalpy.
The matrix supplies numerical dissipation; Newton differentiates the complete
discrete residual in the model's iteration variables, not this matrix alone.

For nozzle, the reconstruction variables are $W=(p,u,h)$ and the unweighted
conservative state and physical flux are

$$
U_N=\begin{pmatrix}\rho\\\rho u\\\rho E\end{pmatrix},
\qquad
F_N=\begin{pmatrix}\rho u\\\rho u^2+p\\\rho uH\end{pmatrix}.
$$

For annular flow, $W=(p,u,\ell,h)$, where $u$ is meridional velocity and
$v=\ell/r_f$ is transverse velocity at the common face radius $r_f$. Roe works
in the transverse-momentum frame:

$$
U_T=\begin{pmatrix}\rho\\\rho u\\\rho v\\\rho E\end{pmatrix},
\qquad
F_T=\begin{pmatrix}\rho u\\\rho u^2+p\\\rho uv\\\rho uH\end{pmatrix}.
$$

Here $e=h-p/\rho$, $K=(u^2+v^2)/2$, $E=e+K$ and $H=h+K$; set $v=0$ for
nozzle. Area variation, curvature, centrifugal terms and wall closures remain
in the model's integrated source discretization. They are not inserted into
the homogeneous Roe matrix. Geometry is fixed in time.

## 2. Thermodynamics uses the existing PH property calls

For each reconstructed side state, the physical model obtains

```python
state = fluid.get_state(jaxprop.HmassP_INPUTS, h, p)
rho, a, G = state["rho"], state["a"], state["G"]
```

The API argument order is **enthalpy, then pressure**. No mean-state EOS flash,
density/internal-energy inversion or perfect-gas formula is required. Model
initialization may use a prescribed reservoir PT state; the Roe numerical path
uses the PH side properties and reuses them through the fused face callback.

For a locally smooth, thermodynamically consistent EOS,

$$
G=\frac{1}{\rho}\left(\frac{\partial p}{\partial e}\right)_\rho,
\qquad
\left(\frac{\partial\rho}{\partial p}\right)_h
=\frac{1+G}{a^2},
\qquad
\left(\frac{\partial\rho}{\partial h}\right)_p
=-\frac{\rho G}{a^2}.
$$

Let $\mathcal I=\rho e=\rho h-p$ be internal-energy density. Inverting the
PH differential map gives

$$
dp=\chi\,d\rho+\kappa\,d\mathcal I,
\qquad
\kappa=G,
\qquad
\chi=a^2-Gh.
$$

Thus $\chi$ holds **internal-energy density** fixed, not specific internal
energy. The distinction is essential to the conservative Euler matrix.

## 3. Mechanical averages and the thermodynamic secant

Write $z_s=\sqrt{\rho_s}$. For any quantity $f$ used below, define

$$
\langle f\rangle
=\frac{z_Lf_L+z_Rf_R}{z_L+z_R}.
$$

The implemented mechanical averages are
$\widetilde u=\langle u\rangle$,
$\widetilde v=\langle v\rangle$ and
$\widetilde H=\langle H\rangle$, with
$\widetilde K=(\widetilde u^2+\widetilde v^2)/2$.
For these averages, the exact product identity is

$$
\Delta(\rho fg)
=\langle f\rangle\Delta(\rho g)
+\langle g\rangle\Delta(\rho f)
-\langle f\rangle\langle g\rangle\Delta\rho.
$$

It gives the mechanical rows of the Roe jump relation. The remaining
thermodynamic requirement is

$$
\Delta p=\widetilde\chi\,\Delta\rho
+\widetilde\kappa\,\Delta\mathcal I.
$$

This is one equation for two coefficients. The package chooses a centered,
dimensionless minimum-distance projection to fix that freedom. It is the
package's specified averaging rule, not a claim to reproduce a particular
published real-gas averaging prescription.

### Fixed reference and energy centering

Each model fixes positive reference values $(\rho_*,p_*,a_*,h_*)$ from its
reservoir/inlet reference state. Define

$$
e_*=h_*-\frac{p_*}{\rho_*},\qquad
S=\rho_*a_*^2,\qquad
\mathcal I_c=\mathcal I-e_*\rho,
\qquad
\chi_c=\chi+\kappa e_*.
$$

The provisional coefficients are square-root-density averages of the physical
side derivatives:

$$
\chi_c^0=\left\langle a^2-G(h-e_*)\right\rangle,
\qquad
\kappa^0=\langle G\rangle.
$$

The internal-energy jump is evaluated with the symmetric product identity

$$
\Delta\mathcal I_c
=\frac{h_L+h_R-2e_*}{2}\Delta\rho
+\frac{\rho_L+\rho_R}{2}\Delta h-\Delta p,
$$

which avoids directly subtracting two large energy densities. Introduce

$$
x=\frac{\Delta\rho}{\rho_*},\qquad
y=\frac{\Delta\mathcal I_c}{S},\qquad
d=\frac{\Delta p}{p_*},\qquad
g_0=\frac{\rho_*\chi_c^0}{p_*},\qquad
b_0=\frac{S\kappa^0}{p_*}.
$$

Let $\eta=d-g_0x-b_0y$ and $q=x^2+y^2$. Away from roundoff-sized jumps,

$$
g=g_0+\frac{\eta x}{q},\qquad
b=b_0+\frac{\eta y}{q},\qquad
\widetilde\chi_c=\frac{p_*}{\rho_*}g,\qquad
\widetilde\kappa=\frac{p_*}{S}b.
$$

This is the orthogonal projection of $(g_0,b_0)$ onto $gx+by=d$. Both
coordinates are dimensionless, so the projection does not mix incompatible
units. Its fixed reference scales are part of the numerical method.

Centering also preserves an arbitrary consistent energy-reference shift
$h'=h+C$, $e_*'=e_*+C$: $\widetilde\chi_c$ and $\widetilde\kappa$ stay
unchanged, while the uncentered coefficient transforms as
$\widetilde\chi'=\widetilde\chi-\widetilde\kappa C$.

### The implemented small-jump policy

The exact projection divides a small EOS/subtraction defect by a small jump.
The implementation instead uses a $C^2$ quintic transition. With machine
precision $\epsilon$ and $\tau=\epsilon^{1/3}$, define

$$
t=\operatorname{clip}\!\left(\frac{q/\tau^2-1}{3},0,1\right),
\qquad
\beta=t^3(10-15t+6t^2).
$$

The actual correction is

$$
c=\begin{cases}
0,&q\le\tau^2,\\
\beta\eta/q,&q>\tau^2,
\end{cases}
\qquad
g=g_0+cx,\qquad b=b_0+cy.
$$

Thus provisional derivatives apply below $\sqrt q=\tau$, the full projection
applies above $\sqrt q=2\tau$, and the intervening blend has continuous first
and second derivatives at its endpoints. Inactive branches use safe
denominators, including coincident and subnormal jumps.

Below the full-projection threshold, an exact finite secant is **not** claimed.
The dimensionless remaining pressure defect
$r_p=d-gx-by$ must satisfy the implemented mixed audit

$$
|r_p|\le4096\epsilon
\left(1+|d|+|gx|+|by|\right).
$$

The factor accommodates finite resolution of the existing host PH inversions;
it does not change flow-solver convergence tolerances. For smooth consistent
EOS states the provisional secant defect decreases cubically with state
separation. An inconsistent property backend can still fail the audit.

## 4. Matrices, sound speed and grouped eigenspaces

The code constructs its matrix in the centered-energy frame. In the following
matrices only, write $u=\widetilde u$, $v=\widetilde v$,
$K=\widetilde K$, $H_c=\widetilde H-e_*$,
$\chi_c=\widetilde\chi_c$ and $\kappa=\widetilde\kappa$.

For nozzle,

$$
A_{N,c}=\begin{pmatrix}
0&1&0\\
\chi_c+\kappa K-u^2&(2-\kappa)u&\kappa\\
u(\chi_c+\kappa K-H_c)&H_c-\kappa u^2&(1+\kappa)u
\end{pmatrix}.
$$

For annular flow in the transverse-momentum frame,

$$
A_{T,c}=\begin{pmatrix}
0&1&0&0\\
\chi_c+\kappa K-u^2&(2-\kappa)u&-\kappa v&\kappa\\
-uv&v&u&0\\
u(\chi_c+\kappa K-H_c)&H_c-\kappa u^2&-\kappa uv&(1+\kappa)u
\end{pmatrix}.
$$

The interface sound speed is determined by the secant coefficients:

$$
\widetilde a^2=\chi_c+\kappa(H_c-K).
$$

It is not the EOS sound speed at an invented averaged state. Positive physical
side sound speeds do not guarantee positive $\widetilde a^2$ for every finite
jump of an arbitrary EOS.

For either matrix, set $B=A_c-uI$. The characteristic polynomial implies
$B^3=\widetilde a^2B$, giving polynomial spectral projectors

$$
P_{\pm,c}=\frac{B^2\pm\widetilde aB}{2\widetilde a^2},
\qquad
P_{0,c}=I-\frac{B^2}{\widetilde a^2}.
$$

They sum to the identity. The code uses three groups in order
$(u+\widetilde a,u-\widetilde a,u)$: two acoustic groups and one convective
group. The convective projector has rank one for nozzle and rank two for
annular contact/entropy plus transverse shear. Grouping avoids choosing an
arbitrary eigenvector basis inside the repeated annular eigenspace.
There is no division by $u$, $u\pm\widetilde a$ or $G$.

Let $T_e$ be the identity with the last-row/first-column entry equal to $e_*$,
so $U=T_eU_c$. The returned physical-energy matrix and projectors are

$$
\widetilde A=T_eA_cT_e^{-1},\qquad
P_g=T_eP_{g,c}T_e^{-1}.
$$

The reported uncentered derivative is
$\widetilde\chi=\widetilde\chi_c-\widetilde\kappa e_*$.
The finite-jump matrix defect follows directly from the pressure defect. With
dimensional $d_p=p_*r_p$, it is
$(0,-d_p,-u d_p)^T$ for nozzle and
$(0,-d_p,0,-u d_p)^T$ for annular. This supplies an independent numerical check
of the construction.

### Annular angular momentum and geometry

Both reconstructed annular states use the same $r_f$ to obtain $v=\ell/r_f$.
After Roe computes its transverse-frame flux, the model transforms the third
component to angular-momentum flux by multiplying by $r_f$. Equivalently,
with $T_f=\operatorname{diag}(1,1,r_f,1)$,
$U_J=T_fU_T$ and $A_J=T_fA_TT_f^{-1}$.
The shared operator applies face area once. Neither radius nor area is applied
inside the generic Roe contraction; evaluating left and right velocities with
different cell radii would violate this common-frame construction.

## 5. Numerical flux and acoustic entropy correction

The shared numerical kernel evaluates

$$
\widehat F
=\frac{F_L+F_R}{2}
-\frac12\sum_{g\in\{+, -,0\}}\psi(\lambda_g,\delta_g)P_g\Delta U,
$$

with the Harten absolute value

$$
\psi(\lambda,\delta)=\begin{cases}
(\lambda^2+\delta^2)/(2\delta),&\delta>0\text{ and }|\lambda|<\delta,\\
|\lambda|,&\text{otherwise}.
\end{cases}
$$

Let $\delta_0=\texttt{delta\_factor}\,\widetilde a$ and use the physical side
acoustic speeds $u_s\pm a_s$ to define

$$
s_\pm=(u_R\pm a_R)-(u_L\pm a_L),\qquad
\delta_\pm=\delta_0+
\frac14\left(s_\pm+\sqrt{s_\pm^2+\delta_0^2}\right),
\qquad \delta_{\mathrm{conv}}=0.
$$

The convective factor is exactly $|u|$, preserving an isolated
stationary contact/shear without an artificial convective entropy width.
The zero-width branch is safe under JAX differentiation, although $|u|$ retains
its mathematical kink at zero. The full flux is therefore not globally $C^1$.

An entropy correction changes dissipation near corrected waves; it leaves the
underlying matrix identity intact. It is not a proof of general nonlinear
entropy stability or positivity for all real-gas states.

## 6. Admissibility, time integration and source ownership

The physical callback marks Roe data invalid when side density/sound speed,
coefficient finiteness, the pressure-secant audit, positive interface sound
speed squared or finite projectors fail. The shared operator also verifies
finite eigenvalues, finite nonnegative widths and the declared validity mask.
Safe arithmetic in invalid branches prevents numerical contamination; it does
not make those states admissible. Invalid trials are rejected. There is no
sound-speed clipping or hidden HLL fallback.

For explicit Euler and SSPRK3, the usual cell-based timestep estimate is also
bounded using the actual reconstructed interface dissipation speeds:

$$
s_f=\max_g\psi(\lambda_{g,f},\delta_{g,f}),\qquad
\Delta t\le\mathrm{CFL}\min_i
\frac{\Delta x_i}{\max(s_{i-1/2},s_{i+1/2})}.
$$

Existing source and boundary bounds may further reduce the step. This matters
for strong annular transverse shear, where Roe interface speeds can exceed the
physical cell wave-speed estimate. The implicit CFL controller keeps its
existing definition. Roe does not replace nonlinear boundary completion or
Newton globalization: explicit stages solve segregated traces; backward Euler
supports coupled or segregated boundary treatment through the common engine.

The production ownership is deliberate:

| Component | Responsibility |
|---|---|
| [Euler physical construction](../../src/nozzle_hyperbolic/models/euler_roe.py) | PH-dependent coefficients, reference convention, matrix, projectors, entropy widths and physical validity |
| [Nozzle model](../../src/nozzle_hyperbolic/models/euler_nozzle/model.py) / [annular model](../../src/nozzle_hyperbolic/models/euler_annular/model.py) | EOS calls, geometry/frame maps, sources, closures and physical boundary laws |
| [Generic Roe kernel](../../src/nozzle_hyperbolic/numerics/roe.py) | Conservative flux contraction from supplied projectors, speeds and widths; no Euler indices or EOS |
| [Shared operator](../../src/nozzle_hyperbolic/solvers/fvm/operator.py) | Reconstruction, frame/area application, residuals, numerical validity, timestep and integration assembly |
| [Optional model protocols](../../src/nozzle_hyperbolic/solvers/fvm/contracts.py) | Direct or fused Roe capability with independent wave and variable counts |

Another physical model may provide its own compatible Roe data without
changing the numerical kernel. Existence of suitable data for a future
two-fluid model is a separate physical/mathematical task, not guaranteed by
this interface.

## 7. Verification meaning and limits

The retained checks cover the finite-jump identity, coincident physical
Jacobian, projector algebra, reference-energy invariance, contacts/shear,
small/subnormal jumps, forward/reverse AD, a nonlinear nonideal analytic EOS,
actual CO₂ states and all supported time/BC combinations. Classical
perfect-gas Roe is recovered as a limiting test without a perfect-gas branch
in this production path. Flow verification additionally checks original
spatial residuals, conservation, physical branch and grid refinement.

The current evidence supports locally smooth, consistent single-fluid EOS
branches in the tested domain. It does not establish unconditional
hyperbolicity/positivity of the averaged matrix, arbitrary nonconvex-EOS wave
patterns, phase switches or two-fluid applicability. Independently interpolated
$\rho,a,G$ tables need their own consistency audit. The actual host-fluid AD
accuracy is limited by JAXProp's existing finite-difference custom-JVP bridge;
the algebraic matrix limit and the backend's approximate derivative are
distinct checks. See the [measured assessment](VERIFICATION_REPORT.md) for
coarse-grid accuracy misses, incomplete real-gas flow attempts and the absence
of a validated performance ranking.

The method records the policy identifiers
`centered_secant_projection_v1`, `c2_roundoff_blend_v1`, `acoustic_harten`
and `reject`, together with the fixed reference values and factor 4096.
These conventions are part of the reproducible numerical definition.
