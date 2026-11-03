# Annular Euler model

The annular model describes axisymmetric, quasi-one-dimensional flow along a
fixed meridional path. It retains swirl and angular-momentum transport. Let $m$
be meridional arc length, $r(m)>0$ the mean radius and $b(m)>0$ the passage
height. The flow area is $A=2\pi r b$. Variables are section averages;
cross-passage profiles, blades and circumferential nonuniformity are not resolved.

## Control-volume derivation

Let $v_m$ and $v_\theta$ denote meridional and tangential velocity. Angular
momentum per unit mass is $\ell=r v_\theta$. Define

$$
H=h+\tfrac12(v_m^2+v_\theta^2),\qquad E=H-p/\rho.
$$

Mass and total energy have the same section-integrated transport form as the
nozzle model. Projecting momentum along the meridional path adds the centrifugal
acceleration $v_\theta^2r'/r$. Taking the moment of tangential momentum about
the axis gives transport of $\ell$; this incorporates the geometric change of
$v_\theta$ as radius changes. With signed wall stress components
$\tau_m,\tau_\theta$, heat flux into the fluid $q_w$ and wall measure per
meridional length $P_w=2A/b$, the equations are

$$
\partial_t(AQ)+\partial_m(AF)=S,
$$

$$
Q=\begin{bmatrix}
\rho\\\rho v_m\\\rho\ell\\\rho H-p
\end{bmatrix},\qquad
F=\begin{bmatrix}
\rho v_m\\\rho v_m^2+p\\\rho v_m\ell\\\rho v_mH
\end{bmatrix},\qquad
S=\begin{bmatrix}
0\\pA'+A\rho v_\theta^2 r'/r-P_w\tau_m\\
-rP_w\tau_\theta\\P_wq_w
\end{bmatrix}.
$$

The geometric pressure force comes from the projected passage walls.
The centrifugal source changes meridional momentum, while angular-momentum
storage accounts for swirl geometry without an additional geometric torque.
A stationary adiabatic wall again has no total-energy sink. Its stresses
decelerate the flow and increase internal energy.

The implemented state is **$W=(p,v_m,\ell,h)$**. The physical and common meshes
share the same face/center coordinates. Geometry preprocessing supplies volumes,
face areas, radii, $r'$ and integrated area derivatives before any JAX solve.
For a cell, the momentum source is evaluated as

$$
(S_i)_m=p_i\int_i A'\,dm
+V_i\rho_i(\ell_i/r_i)^2r_i'/r_i-(2V_i/b_i)\tau_{m,i},
$$

with torque $-r_i(2V_i/b_i)\tau_{\theta,i}$ and heating $(2V_i/b_i)q_{w,i}$.
Only the smooth geometry supplies the metrics; the model does not infer $r'$
or $A'$ from neighboring solution samples.

## Primitive form

In the local primitive frame $w=(p,v_m,v_\theta,h)$, define
$D/Dt=\partial_t+v_m\partial_m$,
$f_m=-P_w\tau_m/A$, $f_\theta=-P_w\tau_\theta/A$ and $q=P_wq_w/A$.
Expansion of the balances gives

$$
\frac{D\rho}{Dt}+\rho(\partial_m v_m+v_mA'/A)=0,
$$

$$
\frac{Dv_m}{Dt}+\frac{p_m}{\rho}
=v_\theta^2r'/r+f_m/\rho,\qquad
\frac{Dv_\theta}{Dt}=-v_mv_\theta r'/r+f_\theta/\rho,
$$

$$
\rho\frac{Dh}{Dt}-\frac{Dp}{Dt}
=q-v_m f_m-v_\theta f_\theta.
$$

The two geometric acceleration terms cancel in the kinetic-energy equation,
leaving the same thermodynamic conversion as nozzle Euler. With
$\mathcal H=q-v_mf_m-v_\theta f_\theta$ and the general-EOS derivatives from
the [nozzle page](euler_nozzle.md),

$$
w_t+G w_m=b,\qquad
G=\begin{bmatrix}
v_m&\rho a^2&0&0\\
1/\rho&v_m&0&0\\
0&0&v_m&0\\
0&a^2&0&v_m
\end{bmatrix},
$$

$$
b=\begin{bmatrix}
-\rho a^2v_mA'/A+\Gamma\mathcal H\\
v_\theta^2r'/r+f_m/\rho\\
-v_mv_\theta r'/r+f_\theta/\rho\\
-a^2v_mA'/A+(1+\Gamma)\mathcal H/\rho
\end{bmatrix}.
$$

The stored $\ell$ representation replaces the tangential equation by
$D\ell/Dt=rf_\theta/\rho$. The radius is fixed, so it adds no unknown or
dynamic storage equation.

## Characteristic analysis

The local homogeneous system has

$$
\det(G-\lambda I)=(v_m-\lambda)^2[(v_m-\lambda)^2-a^2],
\qquad \lambda=v_m+a,\ v_m-a,\ v_m,\ v_m.
$$

Two waves are acoustic; the repeated convective speed carries entropy and
swirl. In the $w$ frame, a dual eigenbasis is

$$
r_+=(\rho a,1,0,a)^T,\quad r_-=(-\rho a,1,0,-a)^T,\quad
r_s=(0,0,0,1)^T,\quad r_\theta=(0,0,1,0)^T,
$$

$$
l_+=(1/(2\rho a),1/2,0,0),\quad
l_-=(-1/(2\rho a),1/2,0,0),\quad
l_s=(-1/\rho,0,0,1),\quad l_\theta=(0,0,1,0).
$$

Consequently, acoustic waves have $dp=\pm\rho a\,dv_m$, the entropy variation
is $dh-dp/\rho$, and swirl is advected with $v_m$. The relevant Mach number for
the boundary characteristic count is **$M_m=v_m/a$**, even when tangential
velocity makes the total speed large.

For numerical flux evaluation, both reconstructed states use the **same face
radius**. The characteristic routines work in the ordinary transverse-Euler
conserved frame $(\rho,\rho v_m,\rho v_\theta,\rho E)$, form projectors with
the storage Jacobian, then convert the third flux component to angular-momentum
flux by multiplying by $r_{\rm face}$. Face area is applied once by the engine.
This ordering avoids interpreting a radius change as a transverse-velocity
jump at one face. Roe groups the entropy and shear projectors into one rank-two
convective eigenspace; the eigenvalue multiplicity remains four.

For smooth, inviscid adiabatic steady flow, $A\rho v_m$, $H$, $s$ and $\ell$
are invariant. Hence $v_\theta=\ell/r$: radial expansion reduces swirl speed.
Across a stationary shock, the four transported fluxes are conserved at the
same local radius; entropy rises. These are useful independent reference
checks, with wall/heat contributions included when present.

## Wall stress and heat transfer

The default `tau_w=0`, `q_w=0` gives inviscid adiabatic flow.
For a scalar stress magnitude $\tau$, the usual directional law is

$$
(\tau_m,\tau_\theta)=\tau\frac{(v_m,v_\theta)}
{\sqrt{v_m^2+v_\theta^2}}.
$$

The code treats zero speed with a finite drag-coefficient evaluation where the
closure supplies one. A directional closure may instead return the two signed
stresses independently. The nozzle Darcy/Haaland closures are reused, while
annular-specific directional and NozzleX adapters live in
{mod}`nozzle_hyperbolic.models.euler_annular.closures`.
Their empirical choices are part of the selected physical model; they are not
an automatically resolved turbulence or heat-transfer system.

## Boundary conditions

### Subsonic inlet

For $0<v_m<a$, three characteristics enter from the left and one leaves.
Specify reservoir $H_0,s_0$ and **one** swirl law: flow angle $\alpha_{\rm in}$
in radians, or angular momentum $\ell_{\rm in}$. The four trace equations are

$$
h_b+\tfrac12[v_{m,b}^2+(\ell_b/r_b)^2]=H_0,\qquad
s(p_b,h_b)=s_0,
$$

$$
(\ell_b/r_b)\cos\alpha_{\rm in}-v_{m,b}\sin\alpha_{\rm in}=0
\quad\text{or}\quad \ell_b=\ell_{\rm in},
$$

$$
p_b-p_{\rm ext}-\rho_1a_1(v_{m,b}-v_{m,\rm ext})=0.
$$

Here `ext` is coordinate extrapolation from the first two cells. Enforcing both
angle and angular momentum would overconstrain the inlet. A selected supersonic
inlet instead prescribes all four primitives through `inlet_data["state"]`.

### Outlet

A forward subsonic outlet has one incoming acoustic wave. The acoustic law is

$$
p_b=p_{\rm back},\qquad
p_b-p_{\rm ext}+\rho_Na_N(v_{m,b}-v_{m,\rm ext})=0,\qquad
s_b=s_{\rm ext},\qquad \ell_b=\ell_{\rm ext}.
$$

Entropy and angular momentum carry the two outgoing convective fields.
`outlet_regime="supersonic"` extrapolates all four variables;
`outlet_regime="auto"` uses the pressure/sonic/outgoing active set described
on the [nozzle page](euler_nozzle.md). Its sonic condition uses $v_m=a$.

`outlet_formulation="finite_wave"` uses the shared Euler PH shock/expansion
wave construction. Tangential velocity is carried unchanged along the sampled
normal acoustic wave at the boundary radius, while normal velocity and
thermodynamic state follow its wave curve. This shares the general-EOS domain
and classical-wave limits of the nozzle outlet. The implementation is in
{mod}`nozzle_hyperbolic.models.euler_annular.boundaries`.

## Implementation map

{class}`~nozzle_hyperbolic.models.euler_annular.model.AnnularModel` binds fixed
geometry, EOS, wall laws and boundaries. State maps are in
{mod}`nozzle_hyperbolic.models.euler_annular.state`, physical fluxes in
{mod}`nozzle_hyperbolic.models.euler_annular.equations`, sources in
{mod}`nozzle_hyperbolic.models.euler_annular.sources` and characteristic
projectors in {mod}`nozzle_hyperbolic.models.euler_annular.flux_jax`.
