# Two-fluid model

The active two-fluid model describes two interpenetrating compressible phases
at a **common pressure**, with separate velocities and enthalpies. It includes
virtual mass, interphase drag, heat exchange and optional phase change. It is
the constrained port of Alberto's physical formulation, pinned in the source to
revision `faafff704e54fd866abff95fcb88fc0a5217405c`. Its model callbacks use the
shared numerical engine described in the [solver chapters](../numerics/index.md).

This page states the actual matrix model and its source transformation. A
generic pair of independent Euler systems would omit its pressure/void and
virtual-mass coupling. Its discrete nonconservative action also matters at
discontinuities; it uses the explicit `model_compatibility` flux described below.

## Variables and thermodynamic constraints

Write $\alpha=\alpha_g$, $\beta=1-\alpha$, with phase densities $\rho_g,\rho_l$.
Phase mass per mixture volume is $m_g=\alpha\rho_g$, $m_l=\beta\rho_l$.
The eight iteration variables are

$$
W=(p,\alpha,m_g,u_g,m_l,u_l,I_g,I_l),\qquad I_k=m_ke_k.
$$

The constitutive and reconstruction variables are the six enthalpy primitives

$$
U=(p,u_g,u_l,\alpha,h_g,h_l),\qquad
h_g=\frac{I_g+p\alpha}{m_g},\quad
h_l=\frac{I_l+p\beta}{m_l}.
$$

Six transported densities and their physical fluxes are

$$
Q=\begin{bmatrix}
m_g\\m_l\\m_gu_g\\m_lu_l\\
I_g+\tfrac12m_gu_g^2\\I_l+\tfrac12m_lu_l^2
\end{bmatrix},\qquad
F=\begin{bmatrix}
m_gu_g\\m_lu_l\\m_gu_g^2+\alpha p\\m_lu_l^2+\beta p\\
m_gu_gH_g\\m_lu_lH_l
\end{bmatrix},\quad H_k=h_k+\tfrac12u_k^2.
$$

Two intrinsic-density equations complete the state:

$$
g_g(W)=m_g/\alpha-\rho_g(p,h_g)=0,\qquad
g_l(W)=m_l/\beta-\rho_l(p,h_l)=0.
$$

These are algebraic rows with zero storage, rather than two additional
transport equations. They use fixed reference-density scales, so a small phase
fraction cannot hide a large EOS error. The representation requires
$0<\alpha<1$, positive phase masses and admissible phase property states.
Exactly pure phases require a reduced model. The optional `alpha_min` narrows
this open domain; it does not clip a state back into it.

Each phase fluid supplies $\rho_k,a_k,T_k,s_k,\Gamma_k$ at $(p,h_k)$, with

$$
\rho_{k,p}|_h=(1+\Gamma_k)/a_k^2,\qquad
\rho_{k,h}|_p=-\rho_k\Gamma_k/a_k^2,\qquad
dh_k=T_kds_k+dp/\rho_k.
$$

Separate phase states need not be in thermal or chemical equilibrium. Equating
their temperatures or replacing them by a homogeneous equilibrium EOS would
change the model.

## Primitive formulation and virtual mass

The source uses an entropy-to-enthalpy transformation of a six-variable
quasilinear model. The following compact form specifies its complete homogeneous
matrix. Let $k\geq0$ be the dimensionless virtual-mass coefficient, default 0.5,
and define

$$
\rho_m=\alpha\rho_g+\beta\rho_l,\quad
\rho_s=\alpha\rho_l+\beta\rho_g,\quad
D=\rho_g\rho_l+k\rho_m^2,
$$

$$
\widehat\rho_g=\rho_g+k\rho_m,\quad
\widehat\rho_l=\rho_l+k\rho_m,\quad
a_0^2=\frac{\rho_s}{\alpha\rho_l/a_g^2+\beta\rho_g/a_l^2},\quad
d=u_g-u_l,
$$

$$
u_1=\alpha u_g+\beta u_l,\quad
u_2=\frac{\alpha\rho_gu_g+\beta\rho_lu_l}{\rho_m},
$$

$$
b_p=\frac{a_0^2}{\rho_s}
\left(\frac{\beta u_l\rho_g}{a_l^2}+\frac{\alpha u_g\rho_l}{a_g^2}\right),\quad
b_u=\frac{u_1\rho_g\rho_l+u_2k\rho_m^2}{D},\quad
b_\alpha=\frac{a_0^2}{\rho_s}
\left(\frac{\beta u_g\rho_g}{a_l^2}+\frac{\alpha u_l\rho_l}{a_g^2}\right).
$$

For $z=(p,u_g,u_l,\alpha)$ the upper block is

$$
B=\begin{bmatrix}
b_p & \alpha a_0^2\rho_l\rho_g/\rho_s & \beta a_0^2\rho_l\rho_g/\rho_s & a_0^2d\rho_l\rho_g/\rho_s\\
\widehat\rho_l/D & b_u & -\beta d\rho_l\widehat\rho_l/D & 0\\
\widehat\rho_g/D & \alpha d\rho_g\widehat\rho_g/D & b_u & 0\\
\alpha\beta d a_0^2/(\rho_sa_g^2a_l^2)&
\alpha\beta\rho_ga_0^2/(\rho_sa_l^2)&
-\alpha\beta\rho_la_0^2/(\rho_sa_g^2)&b_\alpha
\end{bmatrix}.
$$

Let $e_p^T=(1,0,0,0)$ and $B_{p,:}$ be the pressure row of $B$. Applying
$dh_k=T_kds_k+dp/\rho_k$ gives the final enthalpy-primitive matrix

$$
U_t+G_hU_x=D_h,\qquad
G_h=\begin{bmatrix}
B&0_{4\times2}\\
(B_{p,:}-u_ge_p^T)/\rho_g&u_g&0\\
(B_{p,:}-u_le_p^T)/\rho_l&0&u_l
\end{bmatrix}.
$$

This also explains the last two rows: subtract each phase material pressure
derivative from its enthalpy equation and the remaining homogeneous quantity
is the transported phase entropy. The virtual-mass terms modify both velocity
coupling and the acoustic spectrum, rather than adding an independent variable.
The formula is implemented in {mod}`nozzle_hyperbolic.models.two_fluid.matrices`.

## From primitive equations to six balances

Differentiate the storage and flux maps:

$$
J=\frac{\partial Q}{\partial U},\qquad
K=\frac{\partial F}{\partial U},\qquad X=JG_h-K.
$$

Multiplying the primitive system by $J$ yields its total-energy balance form

$$
Q_t+F_x+XU_x=D_Q,\qquad D_Q=JD_h.
$$

The first two rows of $X$ are zero, so phase masses have ordinary conservative
transport. Its momentum rows are opposite, so their nonconservative contributions
cancel in the mixture momentum equation. The energy rows contain pressure/void
work and virtual-mass coupling; at finite slip their sum is generally nonzero.
The name “total-energy variables” does not imply that each phase energy or
their sum has a zero nonconservative contribution.

On a fixed varying-area duct the implemented form is

$$
\partial_t(AQ)+\partial_x(AF)+AXU_x
=A D_{Q,\mathrm{exchange}}+S_{\mathrm{area}}.
$$

The area-weighted flux already produces convective geometric terms. The
remaining geometric source is

$$
S_{\mathrm{area}}=
\begin{bmatrix}0\\0\\\alpha p A'\\\beta p A'\\-p d_\alpha A'\\+p d_\alpha A'\end{bmatrix},
\qquad
d_\alpha=\frac{\alpha\beta a_0^2}{\rho_s}
\left(\frac{\rho_lu_l}{a_g^2}-\frac{\rho_gu_g}{a_l^2}\right).
$$

The opposite energy signs are the interfacial pressure work associated with the
geometrically driven void-fraction change. Discretely, `D_VE_area` uses the
same $\Delta A=A_{i+1/2}-A_{i-1/2}$ as the area-weighted divergence. It must
not be added a second time through a $-F A'/A$ source.

## Exchange-source derivation

Let $\sigma_g=-\sigma_l=\sigma$ be mass transfer into gas,
$F_{v,g}=-F_{v,l}=F_v$ the interphase force, $u_{\rm ex}$ the transfer velocity,
$u_{\rm int}$ the interfacial velocity and $h_{{\rm ex},k}$ the transferred
enthalpies. Let $q_k^I$ be the implemented phase interfacial heat source. The
entropy production combinations are

$$
\Theta_k=T_k\sigma_k^S
=\sigma_k\left[h_{{\rm ex},k}-h_k+\tfrac12(u_{\rm ex}-u_k)^2\right]
+F_{v,k}(u_{\rm int}-u_k)+q_k^I,
\qquad
M_k=\sigma_k-\frac{\rho_{k,h}}{\rho_k}\Theta_k.
$$

The source-driven void change is

$$
D_\alpha=\frac{\beta a_0^2}{\rho_s a_l^2}M_g
-\frac{\alpha a_0^2}{\rho_s a_g^2}M_l.
$$

It follows either by combining the two mass equations with their EOS
differentials or by reading row four of the primitive entropy source. Convert
the entropy source with $D_{h,k}=D_p/\rho_k+T_kD_{s,k}$, then apply $J$.
To state the resulting six source rows compactly, set

$$
c=\rho_g\rho_l/D,\quad \eta=c-1,\qquad
\mathcal B=\frac{k\rho_m(\alpha\rho_gu_l+\beta\rho_lu_g)
-\alpha\beta(\rho_l^2-\rho_g^2)(u_g-u_l)}D.
$$

For $k\in\{g,l\}$, the final rows are

$$
D_{m_k}=\sigma_k,\qquad
D_{P_k}=c(F_{v,k}+\sigma_ku_{\rm ex})+\sigma_k\mathcal B,
$$

$$
D_{E_k}=\sigma_k\left[h_{{\rm ex},k}+\tfrac12u_{\rm ex}^2
+u_{\rm ex}u_k\eta+u_k\mathcal B\right]
+F_{v,k}(u_{\rm int}+u_k\eta)+q_k^I
+\varepsilon_k pD_\alpha,
\quad \varepsilon_g=-1,\ \varepsilon_l=+1.
$$

The interfacial pressure-work terms are required by the primitive-to-content
transformation. With `mass_transfer=False`, set $\sigma_k=0$ while retaining
drag, heat exchange and their induced $D_\alpha$. With `closures=None`, all
exchange is explicitly disabled. `source_scale` multiplies exchange sources;
`area_source` independently controls the area contribution.

### Constitutive choices

{class}`~nozzle_hyperbolic.models.two_fluid.configuration.ClosureConfig` selects
interfacial area, drag and heat-transfer laws. Available choices include
number-density or prescribed void-dependent area laws, Schiller–Naumann or
Ishii–Zuber drag policies, and Ranz–Marshall/internal or other retained phase
heat-transfer correlations. One-component phase change uses saturation data,
enthalpy/temperature gates and optionally a pressure cutoff. Two-component
nitrogen–water cases can exchange heat and momentum with mass transfer disabled.
The configuration API lists the precise selectors, parameters and units.

Drag is evaluated with the analytic cancellation of the $1/Re$ factor, giving
a finite Stokes derivative at zero slip. The code does not introduce a physical
slip floor. Correlation bounds, interfacial-area parameters, property-table
domains and closure activation are physical inputs. Two-fluid wall friction
remains unimplemented in this port; the Euler wall laws are not silently added.

## Characteristic analysis and hyperbolicity

The six homogeneous wave speeds are

$$
\lambda=(u_g,u_l,u_m+a_m,u_m-a_m,u_g,u_l).
$$

Define $\widehat\rho_s=\rho_s+k\rho_m$ and

$$
\widetilde a^2=\frac{\widehat\rho_s}{\rho_s}
\frac{\rho_g\rho_l}{D}a_0^2,\quad
a_m^2=\widetilde a^2-
\frac{\alpha\beta\rho_g\rho_l\widehat\rho_g\widehat\rho_l}{D^2}
(u_g-u_l)^2,
$$

$$
u_m=\frac{\alpha\rho_g\widehat\rho_l}{D}u_g
   +\frac{\beta\rho_l\widehat\rho_g}{D}u_l.
$$

The two coefficients in $u_m$ sum to one. Eliminating the coupled
pressure/velocity/void block gives

$$
\det(G_h-\lambda I)
=(u_g-\lambda)^2(u_l-\lambda)^2
[(u_m-\lambda)^2-a_m^2].
$$

The four phase-speed waves carry void/velocity and entropy information, while
the remaining pair is acoustic. For example, the entropy right vectors are
$(0,0,0,0,T_g,0)^T$ and $(0,0,0,0,0,T_l)^T$, with dual rows
$(-1/(\rho_gT_g),0,0,0,1/T_g,0)$ and
$(-1/(\rho_lT_l),0,0,0,0,1/T_l)$. Each acoustic right vector satisfies
$\delta h_g=\delta p/\rho_g$, $\delta h_l=\delta p/\rho_l$;
the coupled four-component block determines its velocities and void variation.
The port supplies closed-form left/right eigenvectors and their biorthogonality
normalizers.

Real acoustics require $a_m^2>0$. The negative slip-squared term shows why
admissible phase EOS values alone do not guarantee mixture hyperbolicity.
The model checks this separately. At zero slip, repeated phase speeds make
the selected individual void eigenvectors degenerate. The compatibility flux
therefore has a separately derived zero-slip branch and blends to the full-slip
formula; it does not divide by vanishing eigenvector normalizers.
This is a numerical treatment of the retained model, not a proof of strict
hyperbolicity at every coincident-speed or phase-limit state.

## Nonconservative discretization

For a smooth state, $F_x=KU_x$ suggests the compatibility action
$XK^{-1}F_x$. The port evaluates this through a linear solve; it never forms
$K^{-1}$ explicitly. For the default `nc_form="cell"`,

$$
N_i=A_iX_iK_i^{-1}(\widehat F_{i+1/2}-\widehat F_{i-1/2}).
$$

`nc_form="staedtke"` uses the area-weighted flux difference. `nc_form="wb"`
subtracts $\Delta A F(U_i)$ from that weighted difference before the action.
The `wb` pairing preserves the tested stationary pressure/void/area balance;
that property is not attributed to the default `cell` pairing.

At literal phase rest, $K$ is singular. A reduced compatible solve uses known
inactive columns and checks the residual range condition. Incompatible right
hand sides are rejected. No arbitrary pseudoinverse or velocity floor is used.
This does not guarantee success at every other singular or sonic state.

The order of operations is important: reconstruct $U$, compute the unfiltered
model FVS, evaluate $N_i$ from that flux, and only then add independently selected
conservative filters. The method is the retained compatibility discretization;
it is not a general path-conservative Riemann solver for arbitrary weak solutions.

## Boundary conditions

Each endpoint has **six primitive trace equations plus two EOS equations**.
{class}`~nozzle_hyperbolic.models.two_fluid.configuration.BoundaryConfig` defines
these laws. Their solution uses the same coupled/segregated machinery as other
models. For forward subsonic phase velocities and one incoming acoustic wave at
the outlet, five of the six primitive characteristics enter at the inlet;
an additional compatibility row completes the trace from interior information.
Other speed signs require a separate physical assessment of the chosen law.

### Stagnation inlet

The selected stagnation law prescribes $H_{g0},H_{l0}$, dominant-phase reservoir
entropy $s_0$, vapor mass-flow quality $x$ and slip ratio $S=u_g/u_l$.
Its six equations are

$$
s_{d,b}=s_0,\qquad u_{g,b}=S u_{l,b},\qquad
u_{2,b}=u_{2,1},
$$

$$
(1-x)m_{g,b}S-xm_{l,b}=0,\qquad
h_{g,b}+\tfrac12u_{g,b}^2=H_{g0},\qquad
h_{l,b}+\tfrac12u_{l,b}^2=H_{l0}.
$$

$u_2$ is the mass-weighted velocity and subscript 1 denotes the adjacent
interior cell. The composition row follows by cross multiplying
$x=m_gu_g/(m_gu_g+m_lu_l)$ with the imposed slip. It avoids division by a
possibly zero trial velocity. The dominant phase is explicitly configured;
it is not selected by an evolving phase-fraction threshold.

Other inlet choices are `static_state` (all six primitives), `transmissive`
(adjacent-cell primitives), `subsonic_inflow` (five specified nonpressure
primitives with adjacent-cell pressure), and `total_state` (extrapolated
pressure/velocities with specified void fraction and phase total enthalpies).
These are explicit alternative boundary policies, not simultaneous constraints.

### Pressure and characteristic outlets

The outlet starts from the adjacent interior primitive state $U_i$. Define
$\delta p=w\,\omega(p_{\rm back}-p_i)$, where $\omega$ is the configured
`pressure_relaxation`. For `outlet="characteristic"`,

$$
U_b=U_i+\delta p\,\frac{r_-}{(r_-)_p},\qquad
w=\tfrac12\left[1+\tanh\frac{1-M_m}{\Delta M}\right],\quad
M_m=|u_m|/a_m.
$$

$r_-$ is the mixture acoustic-minus right eigenvector at the interior state.
This is the retained smooth Mach-weighted pressure policy. It differs from the
sharp pressure/sonic active set in the Euler outlet. The fully outgoing limit
approaches extrapolation; at finite transition width pressure influence decays
smoothly rather than disappearing at an exact switch.

`pressure` and `characteristic_old` instead apply the legacy acoustic target:
$p_b=p_i+\delta p$ and both phase velocities receive
$-\delta p/Z_W$, retaining interior void fraction and phase enthalpies, where

$$
Z_W=\sqrt{\frac{\rho_m}{\alpha/(\rho_ga_g^2)+\beta/(\rho_la_l^2)}}.
$$

The `pressure` law has $w=1$; `characteristic_old` uses the smooth Mach weight.
`transmissive` sets all primitive traces to their adjacent-cell values.
The two EOS trace rows are appended in every case. A time-dependent
backpressure callable is supported by the boundary configuration.

## Stabilization and verification limits

{class}`~nozzle_hyperbolic.models.two_fluid.configuration.StabilizationConfig`
controls optional void/thin-phase filtering and a phase-mass upwind policy.
Defaults leave these additions off. Independently selected phase pins alter
equations: the finite-time momentum relaxation and saturation-energy pin are
restricted to implicit steady solves and are not merely acceleration switches.
Diagnostics retain defects of the original equations when pins are enabled.

The implementation is checked against frozen matrix, source and flux fixtures,
EOS constraints, rest/reversal and open-phase limits. The NICFD comparisons
and demo scripts provide additional case-specific evidence. Full independent
cold-start recovery, grid independence and all closure combinations are not
established by those checks. Inspect all six balance norms, both EOS defects,
boundary defects and the actual nonconservative budget before calling a run
converged. See the [demo page](../demos.md) and
[convergence chapter](../numerics/convergence.md).

## Implementation map

The public entry point is
{class}`~nozzle_hyperbolic.models.two_fluid.model.TwoFluidModel`.
State conversions are in {mod}`nozzle_hyperbolic.models.two_fluid.constrained_state`;
physical policies are in {mod}`nozzle_hyperbolic.models.two_fluid.configuration`.
{mod}`nozzle_hyperbolic.models.two_fluid.matrices`,
{mod}`nozzle_hyperbolic.models.two_fluid.flux` and
{mod}`nozzle_hyperbolic.models.two_fluid.sources` contain the physical matrix,
compatibility-flux and source implementations. Phase property evaluation,
boundary equations, stabilization and closure families have separate owning
modules under {mod}`nozzle_hyperbolic.models.two_fluid`.
