\section{METHODS}

\subsection{Governing equations}

The two-phase flow in the nozzle is described by a single-pressure, six-equation
two-fluid model \cite{ishii_thermo-fluid_2011,stadtke_gasdynamic_2006}, in which
each phase evolves according to its own balance equations for mass, momentum,
and energy, and the phases interact through interfacial exchange terms.
Compared to mixture formulations, the two-fluid description resolves both
mechanical non-equilibrium (velocity slip) and thermal non-equilibrium
(distinct phase temperatures and metastable states), which are the controlling
mechanisms of rapid vaporizing expansions. For unsteady, compressible,
quasi-one-dimensional flow in a duct of variable cross-section $A = A(x)$, the
model reads, for phase $i$ with $i = \mathrm{g}, \mathrm{l}$:

\begin{gather}
    \frac{\partial}{\partial t}(\alpha_i \rho_i)
      + \frac{1}{A} \frac{\partial}{\partial x}(\alpha_i \rho_i u_i A)
      = \sigma_i^{M}
    \label{eq:mass}\\[4pt]
    \frac{\partial}{\partial t}(\alpha_i \rho_i u_i)
      + \frac{1}{A} \frac{\partial}{\partial x}(\alpha_i \rho_i u_i^2 A)
      + \alpha_i \frac{\partial p}{\partial x}
      = F_i^{\mathrm{int}} + \sigma_i^{M} u^{\mathrm{ex}}
    \label{eq:momentum}\\[4pt]
    \frac{\partial}{\partial t}\!\left[ \alpha_i \rho_i
        \left( e_i + \frac{u_i^2}{2} \right) \right]
      + \frac{1}{A} \frac{\partial}{\partial x}\!\left[ \alpha_i \rho_i u_i A
        \left( h_i + \frac{u_i^2}{2} \right) \right]
      + p\, \frac{\partial \alpha_i}{\partial t}
      = \sigma_i^{Q} + F_i^{\mathrm{int}} u^{\mathrm{int}}
      + \sigma_i^{M} \left[ h^{\mathrm{ex}} + \frac{(u^{\mathrm{ex}})^2}{2} \right]
    \label{eq:energy}
\end{gather}

where $\alpha_i$, $\rho_i$, $u_i$, $e_i$, and $h_i$ are the volume fraction,
density, velocity, specific internal energy, and specific enthalpy of phase
$i$, and $p$ is the pressure, shared by the two phases (single-pressure model).
The volume fractions satisfy the saturation constraint
$\alpha_{\mathrm{g}} + \alpha_{\mathrm{l}} = 1$.

The right-hand sides collect the interfacial exchange terms, which must vanish
pairwise when summed over the phases so that mixture mass, momentum, and energy
are conserved:

\begin{itemize}
\item $\sigma_i^{M}$ is the interfacial mass transfer rate
      ($\sigma_{\mathrm{g}}^{M} = -\sigma_{\mathrm{l}}^{M} = \sigma^{M}$, with
      $\sigma^{M} > 0$ for evaporation). It is obtained from the energy balance
      at the interface, which is assumed to be at the local saturation
      temperature $T_{\mathrm{sat}}(p)$:
      %
      $$
      \sigma^{M} = -\frac{\sigma_{\mathrm{g}}^{Q} + \sigma_{\mathrm{l}}^{Q}}
                         {h_{\mathrm{sat,g}} - h_{\mathrm{sat,l}}}
      $$
      %
      The transferred mass leaves the donor phase at its saturation enthalpy
      $h^{\mathrm{ex}}$ and carries the donor-phase velocity $u^{\mathrm{ex}}$
      (donor upwinding), consistent with the convention adopted in two-phase
      system codes \cite{relap5_manual}. Phase change is only activated when
      driven by a metastable state: evaporation requires superheated liquid and
      condensation subcooled vapor; at exact saturation both mechanisms vanish.

\item $\sigma_i^{Q}$ is the sensible heat exchanged between phase $i$ and the
      interface, $\sigma_i^{Q} = \mathrm{htc}_i\, a_{\mathrm{int}}
      (T_{\mathrm{sat}} - T_i)$, where $a_{\mathrm{int}}$ is the interfacial
      area density. The heat transfer coefficient on the continuous-phase side
      follows the Ranz--Marshall correlation
      $\mathrm{Nu} = 2 + 0.6\,\mathrm{Re}^{1/2}\mathrm{Pr}^{1/3}$
      \cite{ranz_evaporation_1952}, while a conduction-limited constant Nusselt
      number is used on the dispersed-phase side.

\item $F_i^{\mathrm{int}}$ is the interfacial force per unit volume, composed
      of an algebraic drag contribution and a differential virtual-mass
      contribution, and $u^{\mathrm{int}}$ is the interface velocity at which
      the associated work is exchanged. The drag force on the dispersed phase
      is
      %
      $$
      F_{\mathrm{v}} = \frac{3}{4}\, \frac{C_{\mathrm{D}}}{d}\,
      \alpha_{\mathrm{d}}\, \rho_{\mathrm{c}}\,
      \left| u_{\mathrm{c}} - u_{\mathrm{d}} \right|
      \left( u_{\mathrm{c}} - u_{\mathrm{d}} \right)
      $$
      %
      with subscripts $\mathrm{c}$ and $\mathrm{d}$ denoting the continuous and
      dispersed phase, and $C_{\mathrm{D}}$ evaluated from the particle
      Reynolds number with the standard drag curve for spheres. The flow
      topology is described by a three-region map in the void fraction: bubbly
      flow for $\alpha_{\mathrm{g}} < 0.3$, droplet flow for
      $\alpha_{\mathrm{g}} > 0.7$, and a smooth transition in between. In the
      dispersed regimes the characteristic particle diameter $d$ follows from a
      prescribed bubble (or droplet) number density, and the interfacial area
      density from $a_{\mathrm{int}} = \pi d^2 N$.

\item The virtual-mass force accounts for the acceleration of the surrounding
      continuous phase induced by the relative acceleration of the dispersed
      phase. It is modeled with a single coefficient $k$ multiplying the
      mixture density, with $k = 0.5$ (added mass of a sphere) throughout this
      work. Being proportional to velocity and pressure \emph{gradients}, the
      virtual-mass terms are differential rather than algebraic; together with
      the interfacial pressure term $\alpha_i\, \partial p/\partial x$ in
      Eq.~\eqref{eq:momentum} they constitute the non-conservative part of the
      system. Their role is not only physical: without virtual-mass coupling
      the single-pressure two-fluid model possesses complex characteristic
      speeds whenever the phase velocities differ, and the initial-value
      problem is ill-posed \cite{stuhmiller_influence_1977}. With the present
      closure the system is hyperbolic over the entire range of conditions
      considered here.
\end{itemize}

Finally, the area variation $A(x)$ contributes, beyond the area-weighted flux
divergences already visible in Eqs.~\eqref{eq:mass}--\eqref{eq:energy}, a
geometric source that represents the pressure force exerted by the lateral wall
and the interfacial pressure work associated with the geometric redistribution
of the void fraction. In the single-phase limit this source reduces to the
familiar $p\,\mathrm{d}A/\mathrm{d}x$ wall-pressure term of the
quasi-one-dimensional Euler equations.

\subsubsection*{Fluid properties}

Each phase is closed by an independent equation of state formulated in the
pressure--enthalpy plane,
$$
\rho_i = \rho_i(p, h_i), \qquad T_i = T_i(p, h_i),
$$
a natural choice for the two-fluid model: the pressure is shared between the
phases, while the phase enthalpies evolve independently and may describe
metastable states (superheated liquid, subcooled vapor) during rapid
expansions. The thermodynamic derivatives required by the characteristic
analysis and the flux linearization follow consistently from the phase speed of
sound and the Gr\"uneisen parameter.

Real-fluid properties are evaluated with \texttt{jaxprop} \cite{jaxprop}, a
Python thermophysical property package designed for compatibility with
automatic differentiation and just-in-time compilation. The underlying
properties derive from multiparameter Helmholtz-energy equations of state
\cite{bell_pure_2014}; for use inside the solver they are pre-tabulated on
structured $(p, h)$ grids and interpolated with $C^1$-continuous bicubic
splines, so that all property calls are smooth, inexpensive, and exactly
differentiable by the automatic-differentiation framework. Separate tables are
built for the vapor and liquid phases, each extended into the metastable region
by evaluating the single-phase Helmholtz surface across the saturation line, so
that the superheated-liquid and subcooled-vapor states produced by finite-rate
interphase transfer remain well defined. The saturation properties required by
the interfacial closures ($T_{\mathrm{sat}}$, $h_{\mathrm{sat,g}}$,
$h_{\mathrm{sat,l}}$) are tabulated separately as functions of pressure.

\subsection{Space discretization}

For the discretization, Eqs.~\eqref{eq:mass}--\eqref{eq:energy} are arranged in
vector form as
$$
\frac{\partial \mathbf{V}}{\partial t}
+ \frac{1}{A}\frac{\partial (A\,\mathbf{F})}{\partial x}
+ \mathbf{H}^{\mathrm{nc}}\,\frac{\partial \mathbf{F}}{\partial x}
= \mathbf{D},
$$
where $\mathbf{V} = [\alpha_i\rho_i,\; \alpha_i\rho_i u_i,\;
\alpha_i\rho_i E_i]^{\mathsf T}$ collects the conserved variables (with
$E_i = e_i + u_i^2/2$), $\mathbf{F}$ the fluxes of
Eqs.~\eqref{eq:mass}--\eqref{eq:energy} augmented by the phasic pressure
contributions $\alpha_i p$ in the momentum components, the operator
$\mathbf{H}^{\mathrm{nc}}$ collects the non-conservative products (interfacial
pressure and virtual mass, recast in terms of the flux gradient), and
$\mathbf{D}$ contains the algebraic interfacial sources and the geometric area
source. The domain is divided into $N$ uniform finite-volume cells, and the
semi-discrete balance for cell $i$ reads
$$
\frac{\mathrm{d} \mathbf{V}_i}{\mathrm{d} t}
= -\frac{1}{A_i\, \Delta x}
  \left( A_{i+\frac{1}{2}} \hat{\mathbf{F}}_{i+\frac{1}{2}}
       - A_{i-\frac{1}{2}} \hat{\mathbf{F}}_{i-\frac{1}{2}} \right)
- \mathbf{H}^{\mathrm{nc}}_i \,
  \frac{\hat{\mathbf{F}}_{i+\frac{1}{2}} - \hat{\mathbf{F}}_{i-\frac{1}{2}}}{\Delta x}
+ \mathbf{D}_i .
$$
The conservative flux difference is weighted by the face areas
$A_{i\pm 1/2}$, which generates all geometric contributions of flux type; the
non-conservative term, being a local product of state gradients, is evaluated
with the plain (un-weighted) flux difference; and $\mathbf{D}_i$ carries only
the geometric leftovers of this splitting — the wall-pressure force
$\alpha_i\, p\, (\mathrm{d}A/\mathrm{d}x)/A$ and the interfacial pressure work.
This partition of the geometry is discretely well balanced: a fluid at rest
with uniform pressure in a duct of arbitrarily varying cross-section is
preserved exactly.

\subsubsection*{Flux vector splitting}

Godunov-type schemes built on approximate Riemann solvers do not carry over
directly to two-fluid models: the non-conservative products preclude a unique
weak form of the equations, so exact jump conditions — and with them exact or
approximate Riemann solutions — are not available. The interface fluxes are
therefore computed with a characteristic-based flux vector splitting in the
spirit of Städtke \cite{stadtke_gasdynamic_2006}, which requires only the
eigenstructure of the quasi-linear system and treats the conservative and
non-conservative parts of the operator consistently.

The idea of the scheme is to decompose the flux into the contributions carried
by the individual waves of the hyperbolic system, and to take each contribution
from its own upwind side. The characteristic analysis of the model yields six
waves: two acoustic waves propagating at $u_{\mathrm{m}} \pm a$, where
$u_{\mathrm{m}}$ is a virtual-mass-weighted mixture velocity and $a$ the
two-phase speed of sound, and four material waves transporting the void
fraction and the phase thermal states at the phase velocities $u_{\mathrm{g}}$
and $u_{\mathrm{l}}$. The two-phase sound speed reduces at vanishing slip to
the frozen (Wood) speed, whose well-known non-monotonic behavior — a deep
minimum at intermediate void fractions, far below the sound speed of either
pure phase — implies that two-phase nozzle flows choke at velocities of a few
tens of meters per second; the scheme captures this mechanism naturally,
because the acoustic upwinding switches at the \emph{mixture} sonic condition.

Given reconstructed states $\mathbf{U}_{\mathrm{L}}$, $\mathbf{U}_{\mathrm{R}}$
at a cell face, the numerical flux is assembled wave by wave as
$$
\hat{\mathbf{F}} = \mathbf{R}^{+}\, \mathbf{F}(\mathbf{U}_{\mathrm{L}})
                 + \mathbf{R}^{-}\, \mathbf{F}(\mathbf{U}_{\mathrm{R}}),
\qquad
\mathbf{R}^{\pm} = \sum_{k=1}^{6}
   \frac{1 \pm \operatorname{sign}(\lambda_k)}{2}\; \tilde{\mathbf{P}}_k ,
$$
where $\tilde{\mathbf{P}}_k$ is the flux-space projector onto wave $k$, built
from the right and left eigenvectors of the quasi-linear coefficient matrix,
evaluated at the face-averaged state. Each wave thus contributes its portion of
the flux from the side determined by the sign of its eigenvalue: the material
waves are upwinded at the phase velocities, the acoustic pair at
$u_{\mathrm{m}} \pm a$.

Three standard modifications, adopted here without further discussion, make
the baseline splitting uniformly robust and accurate over the operating range:
a Harten-type entropy fix applied selectively at expansion sonic points
\cite{harten_high_1983}; a low-Mach-number rescaling of the acoustic
dissipation, which removes the well-known accuracy degradation of compressible
upwind schemes in the slow upstream portion of the nozzle
\cite{rieper_low-mach_2011}; and a regularized evaluation of the projector
basis in the zero-slip limit, where two eigenvector directions coalesce. In
addition, because the upwind dissipation of the material waves vanishes with
the phase velocities, a fourth-order background dissipation of
Jameson--Schmidt--Turkel type \cite{jameson_numerical_1981}, added in
conservative form and confined to the low-velocity region by a smooth velocity
gate, suppresses odd--even void-fraction modes without affecting the smooth
solution.

Second-order spatial accuracy is obtained by MUSCL reconstruction of the
primitive variables with a smooth slope limiter \cite{van_albada_comparative_1982};
the reconstruction reverts locally to first order wherever the reconstructed
states would leave the validity domain of the property tables.

\subsection{Time integration}

The semi-discrete equations are integrated to steady state with the implicit
(backward) Euler method. Implicit integration is dictated by stiffness: the
interfacial drag, heat, and mass transfer introduce relaxation times that are
orders of magnitude shorter than the acoustic time scale, and the acoustic
waves themselves are fast compared with the material transport that sets the
time to steady state.

A key choice is the set of unknowns used by the nonlinear solver. Solving
directly for the conserved variables $\mathbf{V}$ is impractical for a
two-fluid model: recovering the thermodynamic state from $\mathbf{V}$ requires
inverting the equation-of-state closures at every evaluation, an operation
that becomes degenerate when a phase vanishes ($\alpha_i \rho_i \to 0$) and
severely ill-conditioned when the liquid is nearly incompressible, in which
case minute changes of the conserved liquid mass correspond to enormous
pressure changes. Both regimes are unavoidable in vaporizing nozzle flows,
where the void fraction traverses the range from
$\alpha_{\mathrm{g}} \approx 0$ at the subcooled inlet to
$\alpha_{\mathrm{g}} \to 1$ downstream. The solver therefore operates on an
extended per-cell variable vector in which pressure and void fraction appear
explicitly,
$$
\mathbf{W} = \left( p,\; \alpha_{\mathrm{g}},\;
m_{\mathrm{g}},\; u_{\mathrm{g}},\; m_{\mathrm{l}},\; u_{\mathrm{l}},\;
m_{\mathrm{g}} e_{\mathrm{g}},\; m_{\mathrm{l}} e_{\mathrm{l}} \right),
\qquad m_i = \alpha_i \rho_i ,
$$
so that both the conserved variables $\mathbf{V}(\mathbf{W})$ and the primitive
variables needed by the fluxes are purely algebraic in $\mathbf{W}$ — no
equation-of-state inversion appears anywhere in the nonlinear iteration, and
no quantity degenerates as either phase vanishes. The two redundant components
of $\mathbf{W}$ are closed by appending the thermodynamic consistency
conditions as algebraic residual equations,
$$
m_{\mathrm{g}} = \alpha_{\mathrm{g}}\, \rho_{\mathrm{g}}(p, h_{\mathrm{g}}),
\qquad
m_{\mathrm{l}} = \alpha_{\mathrm{l}}\, \rho_{\mathrm{l}}(p, h_{\mathrm{l}}),
$$
which are solved \emph{jointly} with the discrete balance equations. Each cell
thus contributes eight unknowns and eight equations — six discrete balances
and two equation-of-state constraints — and every Newton iterate satisfies the
thermodynamic closure together with the conservation update. This joint
formulation proved essential in practice: enforcing the closures separately
from the conservation update leaves the split between $(p, \alpha_{\mathrm{g}})$
and the partial densities underdetermined, and the resulting drift excites
spurious grid-scale modes in precisely the nearly-incompressible and
vanishing-phase limits described above.

At each time step, the fully implicit residual
$$
\mathbf{R}_i =
V_{c,i} \left( \mathbf{V}_i^{\,n+1} - \mathbf{V}_i^{\,n} \right)
+ \Delta t \left( A_{i+\frac{1}{2}} \hat{\mathbf{F}}^{\,n+1}_{i+\frac{1}{2}}
                - A_{i-\frac{1}{2}} \hat{\mathbf{F}}^{\,n+1}_{i-\frac{1}{2}} \right)
+ \Delta t\, A_i\, \mathbf{H}^{\mathrm{nc},\,n+1}_i \Delta \hat{\mathbf{F}}^{\,n+1}
- \Delta t\, V_{c,i}\, \mathbf{D}_i^{\,n+1} = \mathbf{0},
$$
with $V_{c,i} = A_i \Delta x$ the cell volume, is solved by Newton iteration
with all fluxes, non-conservative terms, and sources evaluated at the new time
level. The complete residual — including the tabulated equations of state, the
eigenstructure of the splitting, and the interfacial closures — is implemented
in the JAX framework \cite{jax2018github}, so the exact Jacobian is obtained by
automatic differentiation rather than by numerical perturbation or hand-coded
linearization.

The time step follows the CFL condition based on the largest characteristic
speed of the two-fluid system,
$$
\Delta t = \mathrm{CFL}\; \frac{\Delta x}
   {\max_i \left( |u_{\mathrm{m}}| + a \right)_i},
$$
where $a$ is the two-phase mixture sound speed — not the much larger
single-phase sound speeds. Because the scheme is implicit, CFL numbers well
above unity are admissible. Convergence to steady state is declared when the
normalized steady-state residual, obtained from the spatial operator alone
scaled by the characteristic acoustic time, falls below a prescribed
tolerance.

\subsection{Boundary conditions}

The number of physical conditions at each boundary follows from the
characteristic structure of the system: at a subsonic inlet five of the six
characteristics enter the domain and one (the upstream-running acoustic wave)
leaves it; at a subsonic outlet a single characteristic enters; at a
supersonic outlet none does.

At the inlet, five stagnation-type quantities are prescribed: the stagnation
enthalpies of both phases $H_{0,\mathrm{g}}$, $H_{0,\mathrm{l}}$, the
stagnation entropy of the gas phase $s_{0,\mathrm{g}}$, the inlet quality
(gas mass fraction) $x_0$, and the slip ratio
$S_0 = u_{\mathrm{g}}/u_{\mathrm{l}}$. The remaining information — the mixture
velocity, which sets the mass flow rate — is extrapolated from the interior
along the outgoing characteristic, so that the mass flow adjusts freely and
the nozzle is allowed to choke. From these data the boundary state is
reconstructed by a small fixed-point iteration: the void fraction follows from
quality and slip,
$$
\alpha_{\mathrm{g}} = \left[\, 1 + \frac{1 - x_0}{x_0}\,
\frac{\rho_{\mathrm{g}}}{\rho_{\mathrm{l}}}\, S_0 \right]^{-1},
$$
the phase velocities from the slip split of the extrapolated mixture velocity,
the static enthalpies from $h_i = H_{0,i} - u_i^2/2$, and the static pressure
from the gas-phase reservoir isentrope
$s_{\mathrm{g}}(p, h_{\mathrm{g}}) = s_{0,\mathrm{g}}$.

At the outlet, a characteristic-based condition switches automatically between
the subsonic and supersonic regimes according to the local mixture Mach
number. In the subsonic regime the static back-pressure $p_{\mathrm{out}}$ is
imposed in non-reflecting form \cite{poinsot_boundary_1992}: pressure and
velocity in the boundary state are displaced jointly along the outgoing
acoustic characteristic,
$$
u^{\mathrm{b}}_i = u_i + \frac{p - p_{\mathrm{out}}}{\rho_{\mathrm{m}}\, a},
$$
so that the imposed pressure does not emit a spurious reflected wave. In the
supersonic regime all variables are extrapolated. The two limits are blended
smoothly across the sonic point, so the boundary operator remains
differentiable for the Newton solver.
