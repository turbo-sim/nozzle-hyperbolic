# General-EOS backpressure boundary conditions

This study compares two model-owned outlet laws using the same Euler equations,
physical geometry, fluid properties, interior fluxes and nonlinear stopping rules.
The flow direction is positive toward the right outlet. Boundary traces are
algebraic unknowns at the physical faces; they have no storage volume.

## Euler characteristics and prescribed pressure

For the nozzle, the primitive state is $W=(p,u,h)$. At a locally planar outlet,
the homogeneous Euler characteristic speeds are $u-a$, $u$ and $u+a$.
For forward subsonic outflow, precisely one characteristic enters the domain,
so one external datum, the backpressure $p_e$, is prescribed. Entropy and the
outgoing acoustic information are supplied by the interior. At fully outgoing
supersonic flow, all infinitesimal characteristics leave the domain.

For the annular model, replace $u$ by the meridional velocity $v_m$ and add the
advected angular momentum $\ell=r v_\theta$. The local normal wave calculation
preserves $\ell$ across the wave. Sonic conditions use meridional Mach number
$v_m/a$, rather than the Mach number formed from total velocity. Geometry and
centrifugal terms remain in the model's cell sources; they are not inserted
into a zero-thickness local boundary wave.

These are physical boundary laws. Coupled versus segregated Newton treatment
is a separate numerical choice. This comparison uses coupled implicit Euler
for both laws.

## Acoustic pressure outlet

Let $i$ and $i-1$ be the last two physical cells. The existing geometric
extrapolation factor is

$$
\alpha=\frac{x_b-x_i}{x_i-x_{i-1}},\qquad
\widetilde W=W_i+\alpha(W_i-W_{i-1}),\qquad Z_i=\rho_i a_i.
$$

On a uniform mesh $\alpha=1/2$. Entropy is extrapolated separately:
$\widetilde s=s_i+\alpha(s_i-s_{i-1})$. The subsonic acoustic outlet solves

$$
p_b=p_e,\qquad
p_b-\widetilde p+Z_i(u_b-\widetilde u)=0,\qquad
s(p_b,h_b)=\widetilde s.
$$

Advected components such as annular $\ell$ are extrapolated. Pressure,
velocity and entropy rows use the model's fixed physical scales in Newton's
residual. All density, sound speed and entropy values come from the selected
JAXProp backend.

The velocity equation is a frozen-impedance acoustic compatibility relation.
It approximates a finite wave by its tangent; it does not assume a perfect gas.
It also does not guarantee a nonreflecting boundary: prescribing static
pressure constrains the incoming acoustic information.

The static `subsonic` setting keeps these equations even during an initial
iterate with supersonic interior flow. The package also offers `auto`, which
selects pressure control, sonic choking or outgoing extrapolation. That option
is distinct from the static acoustic law used for the shocked target cases in
this comparison. Automatic acoustic selection uses infinitesimal characteristic
speeds, rather than the propagation speed of a finite compression shock.

## Finite-wave pressure outlet

The finite-wave law is a partial boundary Riemann construction. Starting at the
last interior state $(p_i,u_i,h_i)$, it constructs the incoming-family wave
connected to $p_e$ and samples that wave at the stationary face, $x/t=0$.
It does not specify a complete exterior reservoir state.

### Compression: moving Rankine–Hugoniot conditions

For a classical compressive branch, $p_*=p_e>p_i$. With $v=1/\rho$, solve

$$
h_*-h_i=\frac{p_*-p_i}{2}(v_i+v_*),\qquad
v_*=\frac{1}{\rho(p_*,h_*)}.
$$

The positive shock-frame mass flux, downstream velocity and shock speed are

$$
j=\sqrt{\frac{p_*-p_i}{v_i-v_*}},\qquad
u_*=u_i-\frac{p_*-p_i}{j},\qquad S=u_i-jv_i.
$$

These equations conserve mass, momentum and energy across a moving shock.
They preserve the EOS enthalpy reference; no assumption $h=c_pT$ is made.
If $S<0$, the compression enters the computational domain and the connected
state becomes the face trace. If $S\ge0$, the wave is outgoing and the face
state is extrapolated from the interior.

A finite compression can enter a flow with initial $u_i>a_i$. Consequently,
testing only the initial sign of $u_i-a_i$ cannot determine whether a large
external pressure increase can initiate shock entry.

### Expansion: numerical rarefaction curve

For a classical expansion, $p_e\le p_i$, integrate

$$
\frac{dh}{dp}=\frac1\rho,\qquad
\frac{du}{dp}=-\frac1{\rho a}.
$$

The initial values are $h(p_i)=h_i$ and $u(p_i)=u_i$. For a consistent EOS,
the first equation is Gibbs' identity along constant entropy and the second
is the incoming-family simple-wave relation. The implementation uses
log-pressure as the independent coordinate:

$$
\frac{dh}{d\log p}=\frac p\rho,\qquad
\frac{du}{d\log p}=-\frac p{\rho a}.
$$

Let the head speed be $u_i-a_i$ and the connected-state tail speed be
$u_*-a_*$. If both are negative, impose the connected pressure state. If the
fan straddles zero speed, sample its sonic state $u=a$; the exterior pressure
then lies beyond the pressure reachable at the stationary face. If the wave
is fully outgoing, extrapolate the face state.

The general-EOS integration stops when it samples a sonic crossing. Its saved
positive `tail_speed` in that mode describes the crossing panel's endpoint,
not an evaluated full fan extending to the unreachable exterior pressure.
The perfect-gas closed form can evaluate that full fan directly. Pressure-state
branches reach $p_e$ and report its actual connected-state tail speed.

Calorically perfect-gas backends retain the equivalent closed-form wave
relations. General-EOS backends use numerical integration and a scalar
Hugoniot root through `get_state(HmassP_INPUTS, h, p)`. Trial paths and roots
must remain admissible; an invalid wave is rejected rather than replaced by
clipped properties or a different outlet law.

The numerical defaults allow at most 128 log-pressure integration attempts,
including rejected subdivisions, with panel width at most 0.05. Each trial
uses RK4 step doubling. It accepts the two-half-step state only when the local
defect fits its fraction of the total error budget; otherwise it halves the
panel without advancing. Smooth accepted panels may grow again up to the
maximum width. The accumulated accepted defect estimate, divided by 15 and
normalized with $(a_i^2,a_i)$ for enthalpy and velocity, must not exceed
$2\times10^{-7}$. Exhausting the attempt budget before completing the sampled
wave is an explicit failure.
The sampled panel-end speed $u-a$ must increase along the expansion within
$10^{-10}a_i$. Scalar roots use 48 bracketed bisections and implicit root
derivatives; the normalized Hugoniot or sonic root residual must be at most
$2\times10^{-12}$. These numerical checks do not certify unsampled path points
or resolve arbitrary nonclassical wave structure.

### Boundary flux

For incoming compression, incoming expansion and sonic-fan branches, use the
physical flux of the sampled face trace. Applying another numerical Riemann
split to an already sampled trace could discard the finite incoming wave.
For outgoing extrapolation, retain the chosen interior numerical-flux assembly
at the boundary. HLL and corrected Stadtke therefore share the same physical
outlet construction but still define different spatial discretizations.

## Relationship and limits

For a small pressure increment and a consistent EOS, the finite-wave relation
reduces to

$$
\Delta u=-\frac{\Delta p}{\rho a}+O(\Delta p^2).
$$

An infinitesimal compression is isentropic to leading order, as is a
rarefaction. Thus both formulations have the same local acoustic limit when
evaluated at the same base state. They are not identical discrete boundary
equations: the acoustic law extrapolates the interior information to the
face, whereas the finite-wave curve starts at the last cell. Their treatment
of a finite wave and their boundary flux also differ. Differences between
converged finite-mesh results must be measured, rather than assumed zero.

Pointwise maximum differences are especially sensitive to shock displacement:
a small movement of a steep compression can produce a substantial local
pressure gap while mass flow and smooth-region states remain close. Read the
paired sensitivity metrics alongside shock-position error, volume-weighted
field errors and the plotted profiles. A maximum pressure gap alone does not
measure a uniform error throughout the flow.

The CO2 calculations keep the established 100 by 100 JAXProp bicubic tables,
including their independently interpolated density, entropy and sound speed.
They do not introduce a density-derived sound-speed replacement. Those
interpolants need not satisfy Gibbs and acoustic derivative identities exactly.
In particular, the Hugoniot weak-shock limit follows the density/energy map,
while the rarefaction velocity integral uses tabulated sound speed. Entropy
drift and acoustic-limit disagreement are therefore diagnostics; a small
integration defect does not prove thermodynamic consistency or an exact
Riemann solution for the interpolated system.

The simple shock/fan classification assumes a classical wave branch. An
arbitrary nonconvex EOS can admit compound or nonclassical waves, and the
presence of two phases alone does not establish which structure applies.
Wave-path checks and the selected equilibrium CO2 cases delimit this study;
this is not a general multiphase Riemann-solver certification.

Negative trial velocities are retained during Newton iteration. Neither law
invents an exterior entropy, temperature or swirl for sustained reverse inflow,
and neither substitutes a reflecting wall. A physical backflow application
requires additional exterior data beyond $p_e$.

## Initialization and verification

Each target starts on 50 cells either at uniform reservoir pressure and
enthalpy with zero velocity, or from a separately converged numerical adapted
expansion. The annular stagnant state also has zero angular momentum in the
physical cells; prescribed inlet data still apply at the inlet face. Both
outlet laws receive identical physical adapted seed arrays for a given
geometry, flux and reconstruction. Shocked analytical profiles are assessment
data, never initial guesses.

Successful target states are prolonged through 50, 100 and 200 cells with the
native grid-sequence driver. A new fixed residual normalization is established
at each stage. Any failed transferred-state attempt and any cold restart must
remain distinguishable in the saved evidence. Adapted seed work is recorded
separately from the target shock solve.

Native convergence is checked independently using the original spatial
residuals, boundary equations and global conservation. Physical branch checks,
shock position and analytical field errors are reported separately from
nonlinear residual convergence. A converged adapted flow that fails to admit
the intended shock is not a verified solution of the target case. The 200-cell
comparison assesses initialization and boundary-law sensitivity; it is not a
mesh-convergence study or a repeated performance benchmark.

## Derivation references

- [Euler characteristics, wave curves and Riemann solutions, Clawpack](https://www.clawpack.org/riemann_book/html/Euler.html).
- [Computational hydrodynamics, Euler theory and general-EOS methods, Michael Zingale](https://open-astrophysics-bookshelf.github.io/numerical_exercises/CompHydroTutorial.pdf).
- [Characteristic boundary-condition principles, NASA Wind-US](https://www.grc.nasa.gov/WWW/winddocs/user/bc.html).
