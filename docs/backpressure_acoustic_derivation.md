# Derivation of the general-EOS acoustic backpressure outlet

[Documentation index](README.md) | [Backpressure comparison](../development/BC_backpressure/README.md) | [Both outlet formulations](../development/BC_backpressure/THEORY.md)

The acoustic outlet combines prescribed external pressure with outgoing
acoustic and entropy information from the interior. Its velocity correction
uses a locally frozen acoustic impedance $Z=\rho a$. This note derives that
relation, identifies the approximation, and connects it to the implemented
face-trace equations.

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
\widetilde q=q_i+\alpha(q_i-q_{i-1}),
\qquad
\alpha=\frac{x_b-x_i}{x_i-x_{i-1}},
\qquad q\in\{p,u,s\}.
$$

For uniform cells, $\alpha=1/2$. In particular, $\widetilde s$ is the
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
[`EulerOutlet.residual`](../src/nozzle_hyperbolic/models/euler_outlet.py).
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
[backpressure comparison](../development/BC_backpressure/README.md). This
derivation does not establish a performance ranking or a general uniqueness
claim.

## Reference

[Clawpack's Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html)
describes the characteristic speeds, acoustic eigenvectors, entropy transport
and wave curves for a perfect gas. The derivation above obtains the local
general-EOS relation directly using
$a^2=(\partial p/\partial\rho)_s$, without its perfect-gas specialization.
