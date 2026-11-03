# Euler nozzle model

The nozzle model describes cross-section-averaged, compressible flow along a
fixed duct. Pressure, density, velocity and thermodynamic state are uniform
over each section. Wall friction and heat transfer enter as constitutive
sources; axial viscous diffusion and heat conduction are absent.

## From a control volume to the balances

Let $x$ increase from inlet to outlet, $A(x)>0$ be the flow area and $P(x)$ the
wetted perimeter. Define static internal energy $e$, enthalpy $h=e+p/\rho$,
total energy $E=e+u^2/2$ and total enthalpy $H=h+u^2/2$.

Mass entering and leaving a short control volume gives

$$
\frac{\partial (A\rho)}{\partial t}
+\frac{\partial (A\rho u)}{\partial x}=0.
$$

Axial momentum transport includes the pressure forces on the end sections.
The axial projection of pressure on the sloping wall contributes $pA' dx$.
Writing $\tau$ for signed wall stress opposing the flow gives

$$
\frac{\partial (A\rho u)}{\partial t}
+\frac{\partial [A(\rho u^2+p)]}{\partial x}
=pA'-P\tau.
$$

The pressure work at the end sections combines with internal-energy transport
to give the total-enthalpy flux. For a stationary wall, drag redistributes
kinetic and internal energy without extracting total energy. If $q_w$ is heat
flux into the fluid,

$$
\frac{\partial (A\rho E)}{\partial t}
+\frac{\partial (A\rho uH)}{\partial x}=Pq_w.
$$

Thus the model has the balance-law form

$$
\partial_t(AQ)+\partial_x(AF)=S,\qquad
Q=\begin{bmatrix}\rho\\\rho u\\\rho H-p\end{bmatrix},\quad
F=\begin{bmatrix}\rho u\\\rho u^2+p\\\rho uH\end{bmatrix},\quad
S=\begin{bmatrix}0\\pA'-P\tau\\Pq_w\end{bmatrix}.
$$

The iteration state is $W=(p,u,h)$, rather than $Q$. The
{class}`~nozzle_hyperbolic.models.euler_nozzle.model.NozzleModel` evaluates $Q(W)$
and $F(W)$ using the selected fluid. Cell volumes are $V_i=\int_i A\,dx$;
the geometric momentum source is $p_i\int_i A' dx$. The smooth geometry supplies
this integral directly. A constant-pressure resting state then balances the
pressure flux and geometric source on the supported exact-area meshes.

## Thermodynamic closure

At every state, JAXProp supplies $\rho(p,h)$, sound speed $a$, entropy $s$,
temperature and the required derivatives. With Grüneisen coefficient
$\Gamma=\rho^{-1}(\partial p/\partial e)_\rho$, thermodynamic consistency gives

$$
\rho_p\big|_h=\frac{1+\Gamma}{a^2},\qquad
\rho_h\big|_p=-\frac{\rho\Gamma}{a^2},\qquad
dh=T\,ds+\frac{dp}{\rho}.
$$

These identities follow by combining $de=Tds+p\,d\rho/\rho^2$ with
$h=e+p/\rho$ and $a^2=(\partial p/\partial\rho)_s$. They permit the same
characteristic construction for a general EOS. Constant-$\gamma$ identities
are needed only by the separately selected perfect-gas analytical references.

## Primitive equations and characteristics

Introduce $D/Dt=\partial_t+u\partial_x$, $f=-P\tau/A$ and
$q=Pq_w/A$. Expanding the mass, momentum and total-energy balances yields

$$
\frac{D\rho}{Dt}+\rho\left(u_x+u\frac{A'}A\right)=0,\qquad
\rho\frac{Du}{Dt}+p_x=f,\qquad
\rho\frac{Dh}{Dt}-\frac{Dp}{Dt}=q-uf.
$$

Substituting $D\rho=\rho_pDp+\rho_hDh$ gives

$$
W_t+G W_x=b,\qquad
G=\begin{bmatrix}
u&\rho a^2&0\\
1/\rho&u&0\\
0&a^2&u
\end{bmatrix},
$$

$$
b=\begin{bmatrix}
-\rho a^2uA'/A+\Gamma(q-uf)\\
f/\rho\\
-a^2uA'/A+(1+\Gamma)(q-uf)/\rho
\end{bmatrix}.
$$

The homogeneous characteristic polynomial is
$\det(G-\lambda I)=(u-\lambda)[(u-\lambda)^2-a^2]$.
Its three wave speeds and a convenient right eigenbasis are

$$
\lambda_+=u+a,\quad\lambda_-=u-a,\quad\lambda_0=u,
\qquad
r_+=\begin{bmatrix}\rho a\\1\\a\end{bmatrix},\quad
r_-=\begin{bmatrix}-\rho a\\1\\-a\end{bmatrix},\quad
r_0=\begin{bmatrix}0\\0\\1\end{bmatrix}.
$$

The dual rows are

$$
l_+=\left(\frac1{2\rho a},\frac12,0\right),\quad
l_-=\left(-\frac1{2\rho a},\frac12,0\right),\quad
l_0=\left(-\frac1\rho,0,1\right).
$$

Thus acoustic variations satisfy $dp=\pm\rho a\,du$, while the advected
thermodynamic variation is $dh-dp/\rho=Tds$. Hyperbolicity requires a physical
EOS state with $\rho>0$ and $a^2>0$. Coinciding a wave speed with zero does not
make the storage Jacobian singular: conservative projectors are built as
$P_k=C r_k l_k C^{-1}$ with $C=\partial Q/\partial W$.
This is also why the flux construction remains meaningful at stagnation and
sonic states without inverting a physical-flux Jacobian.

For smooth inviscid adiabatic steady flow, the equations imply constant
$\dot m=A\rho u$, $H$ and $s$. On an isentrope, $dp=a^2d\rho$ and
$u\,du+dp/\rho=0$ give

$$
\frac{dA}{A}=(M^2-1)\frac{du}{u},\qquad M=u/a.
$$

A smooth sonic transition therefore requires an area extremum. Across a
stationary shock the local mass, momentum and total-energy fluxes satisfy
Rankine–Hugoniot conservation, while entropy increases; the isentropic relation
cannot be continued through that discontinuity.

## Wall models

`wall_closure=None` gives an inviscid adiabatic wall. A prescribed closure can
return stress magnitude and heat flux; the model applies the velocity direction
to the drag. {class}`~nozzle_hyperbolic.models.euler_nozzle.closures.wall.HaalandWall`
uses properties from the fluid state, the laminar Darcy factor $64/Re$ and the
turbulent Haaland relation. Darcy stress has magnitude
$f_D\rho |u|^2/8$; it must not be confused with a Fanning factor.
The implemented laminar limit cancels the apparent $1/Re$ singularity before
evaluation, leaving a finite drag derivative at literal rest.

Mixture viscosity/friction correlations are available separately. Choosing such
a closure or an equilibrium property backend does not introduce separate phase
momentum or energy equations.

## Boundary conditions

Endpoint states are independent traces at physical end faces. They have no
storage volume. The following equations define their physical values; the
[boundary-solution chapter](../numerics/boundaries.md) explains how they are solved.

### Reservoir inlet

For forward subsonic flow, two characteristics enter at the left boundary and
one leaves. Reservoir pressure and temperature determine $H_0$ and $s_0$.
The inlet trace $W_b$ satisfies

$$
h_b+\frac12u_b^2=H_0,\qquad s(p_b,h_b)=s_0,\qquad
p_b-p_{\rm ext}-Z_1(u_b-u_{\rm ext})=0,
$$

where $Z_1=\rho_1a_1$ is frozen at the first cell and `ext` denotes linear
coordinate extrapolation from the first two cells to the face. The third row
retains the outgoing $u-a$ acoustic information. These are finite, nonlinear
trace equations with a local acoustic compatibility approximation; they are
not an exact finite-wave reservoir Riemann solution. The reservoir nozzle API
supports this subsonic inlet. `transmissive=True` supplies the separate
zero-gradient endpoints used by shock-tube problems.

### Acoustic pressure outlet

For forward subsonic flow, one characteristic enters from the right. The
default `outlet_formulation="acoustic"`, `outlet_regime="subsonic"` imposes

$$
p_b=p_{\rm back},\qquad
p_b-p_{\rm ext}+Z_N(u_b-u_{\rm ext})=0,\qquad
s(p_b,h_b)=s_{\rm ext}.
$$

Here $Z_N=\rho_Na_N$ and entropy is independently extrapolated from the last
two cell entropies. The plus sign retains the outgoing $u+a$ acoustic field.
The use of entropy, rather than extrapolated enthalpy, closes the advected
thermodynamic mode for a general EOS.

`outlet_regime="supersonic"` extrapolates the complete state, since every
characteristic leaves. `outlet_regime="auto"` selects outgoing extrapolation
when the interior is fully outgoing. Otherwise it solves the acoustic and
entropy rows with the pressure/choking complementarity condition

$$
\min\left(\frac{p_b-p_{\rm back}}{p_*},
           \frac{a_b-u_b}{u_*}\right)=0.
$$

This selects an imposed-pressure subsonic trace or a sonic trace at a pressure
above the unreachable exterior pressure. The fixed positive scales are the
model reference values. Backpressure alone does not supply the entropy and
other exterior data needed for sustained reverse inflow.

### Finite-wave outlet

`outlet_formulation="finite_wave"` instead samples an incoming wave curve at
the boundary. A compressive wave satisfies the moving shock Hugoniot; in the
shock frame its mass flux $j$ obeys

$$
h_b-h_i=\frac12(p_b-p_i)(1/\rho_b+1/\rho_i),\qquad
j^2=\frac{p_b-p_i}{1/\rho_i-1/\rho_b}.
$$

An expansion curve integrates

$$
\frac{dh}{dp}=\frac1\rho,\qquad
\frac{du}{dp}=-\frac1{\rho a}.
$$

Wave head/tail or shock speeds determine whether the trace is exterior-driven,
fully outgoing or sonic inside an expansion fan. The perfect-gas backend has
closed forms; general fluids use PH property evaluations and numerical wave
curves. The formulation assumes classical shocks and monotone fans; arbitrary
nonconvex compound waves are outside this boundary model. Invalid paths reject
the trial state. See {mod}`nozzle_hyperbolic.models.euler_outlet` and
{mod}`nozzle_hyperbolic.models.euler_wave` for the implemented branch controls.

## Implementation map

The state/flux/source/boundary assembly is in
{mod}`nozzle_hyperbolic.models.euler_nozzle.model`; the general-EOS conservative
projectors are in {mod}`nozzle_hyperbolic.models.euler_nozzle.characteristics`.
Independent steady and shock-tube references live under
{mod}`nozzle_hyperbolic.models.euler_nozzle.analytical`.
