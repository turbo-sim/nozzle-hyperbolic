# General-EOS acoustic backpressure outlets: Euler and a two-fluid extension

[Documentation index](README.md) | [Backpressure comparison](../../development/BC_backpressure/README.md) | [Both outlet formulations](../../development/BC_backpressure/THEORY.md)

The acoustic outlet combines prescribed external pressure with outgoing
acoustic and entropy information from the interior. Its velocity correction
uses a locally frozen acoustic impedance $Z=\rho a$. This note derives that
relation, identifies the approximation, and connects it to the implemented
face-trace equations.

The Euler derivation below describes implemented behavior. The
[two-fluid nozzle extension](#proposed-two-fluid-nozzle-extension) derives a
candidate boundary law from the current two-fluid matrix model. That extension
is a proposal, not an implemented outlet option or a demonstrated performance
improvement.

## Assumptions and orientation

Consider a right outlet, with positive normal velocity $u$ directed out of the
domain. The derivation applies locally to smooth, inviscid, adiabatic Euler
flow with a thermodynamically consistent equilibrium EOS and positive sound
speed. The pressure-controlled branch assumes forward subsonic outflow,
$0<u<a$.

Use the local homogeneous equations to construct the boundary wave. Nozzle
area variation, wall forces and annular centrifugal effects remain in the
model's interior source terms; they are not included in this zero-thickness
boundary-wave construction. This is not an integration of the full nozzle
compatibility equations over a finite characteristic segment.

## Euler equations with a general EOS

Write $D/Dt=\partial_t+u\partial_x$. Mass, momentum and entropy satisfy

$$
\frac{D\rho}{Dt}+\rho u_x=0,
\qquad
\frac{Du}{Dt}+\frac{1}{\rho}p_x=0,
\qquad
\frac{Ds}{Dt}=0.
$$

For an EOS $p=p(\rho,s)$, the equilibrium acoustic speed is

$$
a^2=\left(\frac{\partial p}{\partial\rho}\right)_s.
$$

The chain rule and entropy transport give

$$
\frac{Dp}{Dt}
=a^2\frac{D\rho}{Dt}
+\left(\frac{\partial p}{\partial s}\right)_\rho\frac{Ds}{Dt}
=-\rho a^2u_x.
$$

Thus the pressure and velocity equations are

$$
p_t+u p_x+\rho a^2u_x=0,
\qquad
u_t+u u_x+\frac{1}{\rho}p_x=0.
$$

No identity such as $p=\rho RT$, $a^2=\gamma p/\rho$ or $h=c_pT$ is required.

## Outgoing acoustic information

For the smooth nonlinear homogeneous equations, define
$D_+=\partial_t+(u+a)\partial_x$. Combining the pressure equation with
$\rho a$ times the momentum equation gives the exact differential
compatibility relation

$$
D_+p+\rho a D_+u=0.
$$

This does **not** generally mean $D_+(p+\rho a u)=0$: the impedance varies,
and differentiating that product would introduce an additional term.

To obtain the linear acoustic approximation, linearize around a constant
local base state $(p_0,u_0,\rho_0,a_0)$. For perturbations $p'$ and $u'$,

$$
p'_t+u_0p'_x+\rho_0a_0^2u'_x=0,
\qquad
u'_t+u_0u'_x+\frac{1}{\rho_0}p'_x=0.
$$

Let $Z_0=\rho_0a_0$. Adding the first equation to $Z_0$ times the second gives

$$
\boxed{
\left[\partial_t+(u_0+a_0)\partial_x\right](p'+Z_0u')=0.
}
$$

The combination $p'+Z_0u'$ travels at speed $u_0+a_0$. At a forward subsonic
right outlet, the characteristic speeds have signs

$$
u_0-a_0<0,\qquad u_0>0,\qquad u_0+a_0>0.
$$

One acoustic characteristic enters the domain, while the other acoustic
characteristic and the entropy characteristic leave it. Therefore one scalar
external datum, the backpressure, is supplied. The outgoing acoustic
combination and entropy are estimated from the interior.

Preserving the outgoing combination while changing the boundary pressure
corresponds to an incoming acoustic perturbation:

$$
\delta p+Z_0\delta u=0,\qquad \delta s=0.
$$

The outgoing characteristic supplies the compatibility information; the
pressure correction itself belongs to the incoming wave family.

## Discrete face-trace formulation

Let $i$ and $i-1$ denote the last two physical cells and $b$ the outlet face.
Extrapolate pressure, velocity and entropy separately:

$$
\widetilde q=q_i+\eta(q_i-q_{i-1}),
\qquad
\eta=\frac{x_b-x_i}{x_i-x_{i-1}},
\qquad q\in\{p,u,s\}.
$$

For uniform cells, $\eta=1/2$. In particular, $\widetilde s$ is the
extrapolation of the two EOS entropy values, rather than the entropy evaluated
at the extrapolated primitive state.

The implementation evaluates impedance at the last physical cell,
$Z_i=\rho_i a_i$, and imposes

$$
\boxed{
p_b=p_e,\qquad
p_b-\widetilde p+Z_i(u_b-\widetilde u)=0,\qquad
s(p_b,h_b)=\widetilde s.
}
$$

Consequently,

$$
u_b=\widetilde u-\frac{p_e-\widetilde p}{Z_i}.
$$

Increasing the imposed pressure reduces the boundary velocity. The entropy
equation determines the face enthalpy through the configured EOS. Extrapolation
is a numerical estimate of outgoing information; the characteristic derivation
does not uniquely prescribe this particular spatial extrapolation.

The equations are implemented in
[`EulerOutlet.residual`](../../src/nozzle_hyperbolic/models/euler_outlet.py).
Pressure and acoustic rows are divided by the fixed pressure scale, and the
entropy row by its fixed entropy scale. Scaling changes the numerical
conditioning, not the zero set of these boundary equations. Coupled or
eliminated numerical treatment of the trace is a separate solver choice.

For annular Euler, use meridional velocity in place of $u$ and extrapolate the
advected angular momentum $\ell=r v_\theta$. The acoustic speed is compared
with meridional velocity, not total velocity.

## Meaning of frozen impedance and relation to finite waves

Along an isentropic incoming-family simple-wave curve in state space,

$$
\frac{du}{dp}=-\frac{1}{\rho(p,s)a(p,s)},\qquad ds=0.
$$

For a reference state $(p_r,u_r,s_r)$, the corresponding nonlinear relation is

$$
u_b-u_r
=-\int_{p_r}^{p_b}\frac{dp}{\rho(p,s_r)a(p,s_r)}.
$$

Freezing the integrand at its reference value produces

$$
u_b-u_r
=-\frac{p_b-p_r}{Z_r}+O\!\left((p_b-p_r)^2\right).
$$

The discrete acoustic outlet uses this tangent relation around the
extrapolated face information, with the nearby last-cell impedance $Z_i$.
The finite-wave outlet instead starts at the last cell, integrates the
nonlinear expansion curve, or solves the moving Hugoniot relations for a
compression shock. A finite compression shock generates entropy and cannot
be represented exactly by the isentropic tangent construction above.

"Frozen" refers to the pressure-to-velocity wave correction. It does not mean
that the impedance is held constant over the whole solve or omitted from
automatic differentiation: $Z_i$ is reevaluated from the current interior
state as the boundary residual is evaluated.

## Applicability and limits

- The derivation is general-EOS, provided sound speed and entropy satisfy the
  required thermodynamic identities. Independently interpolated bicubic
  properties can satisfy them only approximately. Computing all properties
  through the selected JAXProp backend does not itself enforce consistency.
- The static pressure-controlled acoustic branch uses these equations even
  when a nonlinear trial state temporarily lies outside subsonic outflow.
  The separate automatic outlet mode adds outgoing/sonic branch selection;
  the subsonic derivation alone does not establish those branches.
- Prescribed static pressure does not guarantee a nonreflecting boundary.
  It constrains the incoming acoustic information and can reflect disturbances.
- Sustained reverse inflow needs additional exterior thermodynamic data.
  Backpressure alone is insufficient.

Measured convergence and accuracy belong to the
[backpressure comparison](../../development/BC_backpressure/README.md). This
derivation does not establish a performance ranking or a general uniqueness
claim.

## Proposed two-fluid nozzle extension

### Model and number of exterior conditions

The same principle extends to the common-pressure two-fluid nozzle: supply
backpressure for the incoming acoustic family and retain outgoing information
from the interior. A single replacement $Z=\rho_m a_m$ is generally insufficient,
because an acoustic wave changes two velocities, void fraction and both phase
enthalpies.

Use the active model's six enthalpy primitives

$$
U=(p,u_g,u_l,\alpha_g,h_g,h_l)^T,\qquad \alpha_\ell=1-\alpha_g.
$$

Locally, omit geometry and interphase-exchange source terms from the
zero-thickness boundary wave and write

$$
U_t+G_h(U)U_x=0.
$$

Virtual mass remains in $G_h$; it is part of the homogeneous acoustic coupling.
Nozzle geometry, drag, heat transfer and phase change still act through the
interior equations. The phase EOSs need not be identical or in equilibrium with
each other. This construction uses the homogeneous model's acoustic speeds,
not an assumed equilibrium-mixture sound speed.

The matrix and eigenstructure are defined in
[`two_fluid/matrices.py`](../../src/nozzle_hyperbolic/models/two_fluid/matrices.py)
and summarized in the [two-fluid theory](../source/theory/two_fluid.md).
Its wave speeds are

$$
\lambda=(u_g,u_l,u_m+a_m,u_m-a_m,u_g,u_l).
$$

Assume an admissible two-phase state, $0<\alpha_g<1$, real nonzero acoustics, and
forward subsonic outflow:

$$
u_g>0,\qquad u_l>0,\qquad
\lambda_-=u_m-a_m<0<\lambda_+=u_m+a_m.
$$

There is one incoming acoustic family and five outgoing families, counting
multiplicity. Thus one exterior scalar, $p_e$, is needed. Prescribing outlet
quality, slip, void fraction or phase temperatures in addition to pressure
would generally constrain outgoing information as well.

### Frozen characteristic derivation

Freeze the matrix at a nearby reference state $U_*$, normally the last physical
cell, and linearize:

$$
\delta U_t+G_*\delta U_x=0,\qquad G_*=G_h(U_*).
$$

For a left eigenvector $\ell_{j,*}$, the linearized characteristic relation is

$$
\left(\partial_t+\lambda_{j,*}\partial_x\right)
\left(\ell_{j,*}^T\delta U\right)=0.
$$

At a regular, diagonalizable reference state, preserving the five outgoing
characteristic amplitudes means that the boundary correction has only an
incoming acoustic component:

$$
\ell_{j,*}^T\delta U=0\quad(j\ne-)
\quad\Longleftrightarrow\quad
\delta U=\delta p\,\frac{r_{-,*}}{(r_{-,*})_p}.
$$

This is the vector counterpart of Euler's
$\delta u=-\delta p/(\rho a)$, rather than a prescription of an incoming wave
speed or a Mach-dependent attenuation of pressure.

The two phase-entropy rows satisfy the Gibbs differentials

$$
\delta s_g=\frac{\delta h_g-\delta p/\rho_g}{T_g},\qquad
\delta s_l=\frac{\delta h_l-\delta p/\rho_l}{T_l}.
$$

Consequently the acoustic direction has
$\delta h_g=\delta p/\rho_g$, $\delta h_l=\delta p/\rho_l$, and
$\delta s_g=\delta s_l=0$. These are local differential statements. For a finite
pressure correction, exact EOS entropy equations provide the same nonlinear
thermodynamic closure used by the Euler acoustic outlet.

### Explicit pressure-to-state coefficients

The following coefficients are obtained by normalizing column 4 of the
production right-eigenvector matrix (`right[:, 3]`). All quantities in this
subsection are evaluated at $U_*$; they are held fixed across the boundary
pressure correction and reevaluated as the interior state changes.

To make the virtual-mass and slip dependence explicit, define

$$
\rho_m=\alpha_g\rho_g+\alpha_\ell\rho_l,\qquad
\rho_s=\alpha_g\rho_l+\alpha_\ell\rho_g,\qquad
D=\rho_g\rho_l+k\rho_m^2,
$$

$$
\widehat\rho_g=\rho_g+k\rho_m,\qquad
\widehat\rho_l=\rho_l+k\rho_m,\qquad
\widehat\rho_s=\rho_s+k\rho_m,\qquad d=u_g-u_l,
$$

$$
a_0^2=\frac{\rho_s}{\alpha_g\rho_l/a_g^2+\alpha_\ell\rho_g/a_l^2},\qquad
\widetilde a^2=\frac{\widehat\rho_s}{\rho_s}
\frac{\rho_g\rho_l}{D}a_0^2,
$$

$$
a_m^2=\widetilde a^2-
\frac{\alpha_g\alpha_\ell\rho_g\rho_l\widehat\rho_g\widehat\rho_l}{D^2}d^2,
\qquad
u_m=\frac{\alpha_g\rho_g\widehat\rho_l u_g+
\alpha_\ell\rho_l\widehat\rho_g u_l}{D}.
$$

Here $k$ is the model's virtual-mass coefficient, and $a_g,a_l$ are phase sound
speeds. The notation $D$ denotes a scalar mixture coefficient in this section,
not a material derivative or the eigenvector-normalization array in the code.

Write the pressure-normalized incoming vector as

$$
\frac{r_-}{(r_-)_p}=
\begin{bmatrix}1&c_g&c_l&c_{\alpha_g}&1/\rho_g&1/\rho_l\end{bmatrix}^T.
$$

Its three kinematic coefficients are

$$
\boxed{
c_g=-\frac{\widehat\rho_l}{D\widetilde a^2}
\left(a_m+\frac{\alpha_\ell\rho_l\widehat\rho_g}{D}d\right),\qquad
c_l=-\frac{\widehat\rho_g}{D\widetilde a^2}
\left(a_m-\frac{\alpha_g\rho_g\widehat\rho_l}{D}d\right),
}
$$

$$
\boxed{
c_{\alpha_g}=-\frac{\alpha_g\alpha_\ell}{\widehat\rho_s}
\left(\frac{\widehat\rho_g}{\rho_g a_g^2}
-\frac{\widehat\rho_l}{\rho_l a_l^2}\right).
}
$$

In general $c_g\ne c_l$ and $c_{\alpha_g}\ne0$. An acoustic pressure correction
therefore need not preserve slip or void fraction. There is no general reason
to replace these three coefficients by a common velocity correction and zero
void correction. Their signs and admissibility must be assessed on the actual
two-fluid branch; Euler's scalar intuition is not a proof for arbitrary slip.

For identical phase states with $\rho_g=\rho_l=\rho$, $a_g=a_l=a$ and
$u_g=u_l=u$, the formulas reduce, for any $k\ge0$, to

$$
a_m=a,\qquad u_m=u,\qquad
c_g=c_l=-\frac1{\rho a},\qquad c_{\alpha_g}=0.
$$

Thus the common velocity satisfies the Euler acoustic correction. This
identical-phase check does not require taking a singular pure-phase limit.

### Proposed face-trace equations

Use the geometry-aware face extrapolation factor

$$
\eta=\frac{x_b-x_i}{x_i-x_{i-1}},\qquad
\widetilde q=q_i+\eta(q_i-q_{i-1}),\qquad
q\in\{p,u_g,u_l,\alpha_g,s_g,s_l\}.
$$

Here $\eta=1/2$ on a uniform mesh. This notation avoids confusing the
extrapolation factor with the gas void fraction. Extrapolate each phase's EOS
entropy independently, as in the Euler outlet; do not substitute the entropy
of an extrapolated enthalpy state.

The proposed fixed-subsonic outlet consists of six equations:

$$
\boxed{
\begin{aligned}
p_b&=p_e,\\
u_{g,b}-\widetilde u_g-c_g(p_b-\widetilde p)&=0,\\
u_{l,b}-\widetilde u_l-c_l(p_b-\widetilde p)&=0,\\
\alpha_{g,b}-\widetilde\alpha_g-c_{\alpha_g}(p_b-\widetilde p)&=0,\\
s_g(p_b,h_{g,b})&=\widetilde s_g,\\
s_l(p_b,h_{l,b})&=\widetilde s_l.
\end{aligned}
}
$$

In the actual eight-variable iteration representation, append the existing
two EOS equations at the trace:

$$
\frac{m_{g,b}}{\alpha_{g,b}}=\rho_g(p_b,h_{g,b}),\qquad
\frac{m_{l,b}}{\alpha_{\ell,b}}=\rho_l(p_b,h_{l,b}).
$$

This gives eight equations for eight trace unknowns. In mixed variables
$Q=(p,u_g,u_l,\alpha_g,s_g,s_l)^T$, the six equations are a correction along
$(1,c_g,c_l,c_{\alpha_g},0,0)^T$. In enthalpy variables they agree with the frozen
acoustic eigenvector to first order about a thermodynamically consistent
reference state. The exact entropy closure intentionally differs at second
order from simply adding $\delta p/\rho_{k,*}$ to each enthalpy. Using last-cell
coefficients with extrapolated face data also introduces a spatial reference
approximation, just as in the Euler formulation.

The entropy equations determine enthalpy at the imposed pressure without
assuming an ideal gas or imposing phase equilibrium. They do not create an
exact finite-amplitude simple wave: velocities and void fraction still use a
frozen tangent. In particular, a finite compression shock need not preserve
phase entropies.

### Replacing the tanh switch and handling choking

The fixed-subsonic equations impose $p_b=p_e$ directly, with no Mach weight or
pressure-target relaxation. They assume the prescribed pressure is reachable
on the forward subsonic branch. As in static Euler, that assumption does not
justify switching equations whenever an intermediate Newton trial crosses
Mach one.

An additional **candidate automatic outlet** can use the Euler active-set idea.
First, if the adjacent interior cell is fully outgoing,

$$
u_{g,i}>0,\qquad u_{l,i}>0,\qquad \lambda_{-,i}\ge0,
$$

extrapolate all six mixed variables $Q_b=\widetilde Q$ and append the EOS rows.
No exterior pressure is imposed on this branch. Otherwise retain the three
kinematic and two entropy rows, replacing the pressure row by

$$
\boxed{
\min\left(\frac{p_b-p_e}{p_*},
          \frac{-\lambda_-(U_b)}{u_*}\right)=0.
}
$$

Here $p_*,u_*$ are fixed positive reference scales and
$\lambda_-(U_b)=u_m(U_b)-a_m(U_b)$ is recomputed from the trace and its phase
EOSs. The condition requires

$$
p_b\ge p_e,\qquad \lambda_-(U_b)\le0,\qquad
(p_b-p_e)\lambda_-(U_b)=0.
$$

It selects either imposed pressure with a subsonic trace, or a sonic trace at
a pressure above the exterior pressure. The relevant sonic condition is the
mixture acoustic eigenvalue $\lambda_-=0$, not $u_g=a_g$ or $u_l=a_l$.
There is no tunable transition width and no half-strength pressure condition
at Mach one.

This branch selection is an extension to investigate, not a consequence of the
subsonic linearization alone. It requires an admissible connected acoustic
branch and a locally determining trace Jacobian. In particular, varying the
trial pressure must cross the sonic condition appropriately; strong slip,
nonconvex thermodynamics or a loss of hyperbolicity can prevent that or produce
multiple roots. The local outgoing test also does not model a finite external
compression that could enter a supersonic interior. A two-fluid finite-wave
outlet would require a separate treatment consistent with its nonconservative
products; Euler's shock Hugoniot cannot simply be reused.

### Relation to the current implementation and checks needed

The current
[`two_fluid/boundaries.py`](../../src/nozzle_hyperbolic/models/two_fluid/boundaries.py)
already uses the incoming acoustic direction for `outlet="characteristic"`:

$$
U_b=U_i+\delta p\frac{r_{-,i}}{(r_{-,i})_p},\qquad
\delta p=w\,\omega(p_e-p_i),\qquad
w=\frac12\left[1+\tanh\frac{1-M_m}{\Delta M}\right].
$$

Thus this proposal generalizes an acoustic mechanism already present. The
changes would be independent choices: extrapolated outgoing face data in place
of adjacent-cell data, exact phase-entropy closure in place of linear enthalpy
increments, full pressure imposition in place of $w\omega$, and, if required,
an explicit sonic/outgoing policy. Setting $w=\omega=1$ alone removes the blend
but does not supply the other changes or a choking model. The `pressure` and
`characteristic_old` policies additionally use a common velocity correction
and retain void fraction and enthalpies, so their difference is larger.

For a future implementation:

- Keep these as model-owned, zero-storage trace residuals, with fixed pressure,
  velocity, void, phase-entropy and EOS row scales. Coupled and segregated
  numerical treatment must call the same residual. Differentiate the
  interior-dependent acoustic coefficients as well as the EOS equations.
- Evaluate the isolated incoming acoustic vector directly. At zero slip, the
  returned void-wave columns coalesce; do not invert the full eigenvector
  matrix or divide by their vanishing normalizers. The three explicit
  coefficients above remain finite there. This avoids a boundary algebra
  singularity but does not establish strong hyperbolicity or well-posedness of
  the repeated-speed interior block.
- Require physical EOS states, $0<\alpha_{g,b}<1$, positive phase masses, real
  nonzero $a_m$, and the intended phase-velocity signs. Reject an inadmissible
  extrapolation or correction; do not silently clip void fraction or phase
  states. Counterflow introduces additional incoming phase families and needs
  more exterior data. At literal rest, phase waves are glancing, so use of the
  limiting algebraic law is not a proof of a strictly outgoing-wave count.
- Check the normalized eigenpair against $G_h$, the identical-phase Euler
  limit, entropy/EOS defects and trace-Jacobian rank. Then compare coupled and
  segregated solutions and audit actual boundary fluxes and conservation.
  A trace law alone does not establish which physical-flux override or
  nonconservative endpoint treatment is appropriate for the two-fluid
  compatibility flux; that assembly decision needs its own verification.
- Compare the boundary changes separately before combining them, including
  subsonic pressure perturbations, startup, sonic passage and unequal phase
  properties. General-EOS entropy/sound-speed consistency remains necessary.
  Better Euler convergence does not establish better two-fluid convergence or
  nonreflection, and no two-fluid performance claim is made here.

The explicit coefficients have been checked against the production matrix and
its normalized incoming acoustic vector, including zero and nonzero slip,
unequal densities/sound speeds and different virtual-mass coefficients. A local
EOS-consistent example also gives a full-rank eight-equation subsonic trace
Jacobian and the expected first-order pressure derivative. These are algebra
checks of the proposal, not integrated nozzle or sonic-transition validation.

## References

[Clawpack's Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html)
describes the characteristic speeds, acoustic eigenvectors, entropy transport
and wave curves for a perfect gas. The derivation above obtains the local
general-EOS relation directly using
$a^2=(\partial p/\partial\rho)_s$, without its perfect-gas specialization.

For related characteristic boundary methodology in a different two-fluid pipe
model, see Section 4.2 of
[van Zwieten et al., *Efficient simulation of one-dimensional two-phase flow
with a new high-order Discontinuous Galerkin method* (2015)](https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Zwi15SHVH.pdf).
That work obtains outgoing information from the interior and supplies incoming
information through external conditions. Its boundary evolution/discretization
and physical model differ from the algebraic nozzle trace proposed here; the
coefficients above are derived from this repository's matrix, not transferred
from that paper.
