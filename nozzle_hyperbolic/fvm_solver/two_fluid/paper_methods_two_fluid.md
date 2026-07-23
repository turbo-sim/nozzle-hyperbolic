\section{METHODOLOGY}

\subsection{Governing equations}
The two-phase flow in converging–diverging nozzles is described by the single pressure six-equation two-fluid model of Städtke \cite{stadtke_gasdynamic_2006}, in which each phase is governed by its own mass, momentum, and energy conservation equation. Compared to homogeneous mixture formulations, the two-fluid model resolves mechanical non-equilibrium (slip between the phases) and thermal non-equilibrium (distinct phase temperatures), with the interaction between the phases described by interfacial exchange terms for mass, momentum, and energy. For a quasi-one-dimensional, unsteady, compressible flow in a duct of variable cross-section $A = A(x)$, the system can be written as: 
\begin{gather*}
    \frac{\partial}{\partial t}(\alpha_i \varrho_i) + \frac 1 A \frac{\partial}{\partial x}(\alpha_i \varrho_i u_iA) = \sigma_i^M\\\\
    \frac{\partial}{\partial t}(\alpha_i \varrho_i u_i) + \frac 1 A \frac{\partial}{\partial x}(\alpha_i \varrho_i u_i^2 A) + \alpha_i \frac{\partial p}{\partial x} = F_i^{\text{int}} + \sigma_i^M u^{\text{ex}}\\\\
    \frac{\partial}{\partial t} \left[ \alpha_i \varrho_i \left( e_i + \frac{u_i^2}{2} \right) \right] + \frac 1 A \frac{\partial}{\partial x} \left[ \alpha_i \varrho_i u_i A \left( h_i + \frac{u_i^2}{2} \right) \right] + p \frac{\partial \alpha_i}{\partial t} = \sigma_i^Q + F_i^{\text{int}} u^{\text{int}} + \sigma_i^M \left[ h^{\text{ex}} + \frac{(u^{\text{ex}})^2}{2} \right] 
    % \label{two_fluid_equations}
\end{gather*}

with i= l, v

% $$
% \frac{\partial \mathbf{V}}{\partial t}
% + \frac{1}{A}\frac{\partial (A\,\mathbf{F})}{\partial x}
% + \mathbf{H}^{\mathrm{nc}}\,\frac{\partial (A\,\mathbf{F})}{\partial x}
% = \mathbf{D}
% $$

where $\mathbf{V}$ is the vector of conserved variables, $\mathbf{F}$ is the flux vector, $\mathbf{H}^{\mathrm{nc}}$ collects the non-conservative coupling terms, and $\mathbf{D}$ is the source vector. The conserved variables and fluxes are defined per phase as:

$$
\mathbf{V}=
\begin{bmatrix}
\alpha_{\mathrm{g}} \rho_{\mathrm{g}} \\
\alpha_{\mathrm{l}} \rho_{\mathrm{l}} \\
\alpha_{\mathrm{g}} \rho_{\mathrm{g}} u_{\mathrm{g}} \\
\alpha_{\mathrm{l}} \rho_{\mathrm{l}} u_{\mathrm{l}} \\
\alpha_{\mathrm{g}} \rho_{\mathrm{g}} E_{\mathrm{g}} \\
\alpha_{\mathrm{l}} \rho_{\mathrm{l}} E_{\mathrm{l}}
\end{bmatrix},
\qquad
\mathbf{F}=
\begin{bmatrix}
\alpha_{\mathrm{g}} \rho_{\mathrm{g}} u_{\mathrm{g}} \\
\alpha_{\mathrm{l}} \rho_{\mathrm{l}} u_{\mathrm{l}} \\
\alpha_{\mathrm{g}} \left(\rho_{\mathrm{g}} u_{\mathrm{g}}^2 + p\right) \\
\alpha_{\mathrm{l}} \left(\rho_{\mathrm{l}} u_{\mathrm{l}}^2 + p\right) \\
\alpha_{\mathrm{g}} \rho_{\mathrm{g}} u_{\mathrm{g}} H_{\mathrm{g}} \\
\alpha_{\mathrm{l}} \rho_{\mathrm{l}} u_{\mathrm{l}} H_{\mathrm{l}}
\end{bmatrix}
$$

where the subscripts $\mathrm{g}$ and $\mathrm{l}$ denote the gas and liquid phases, $\alpha_i$ is the volume fraction (with the saturation constraint $\alpha_{\mathrm{g}} + \alpha_{\mathrm{l}} = 1$), $\rho_i$, $u_i$ are the phase density and velocity, and $p$ is the pressure, assumed to be shared by both phases (single-pressure model). The specific total energy and total enthalpy of each phase are:

$$
E_i = h_i - \frac{p}{\rho_i} + \frac{u_i^2}{2},
\qquad
H_i = h_i + \frac{u_i^2}{2}
$$

where $h_i$ is the phase specific enthalpy.

Unlike the single-phase Euler equations, the two-fluid system is not fully conservative: the phase momentum and energy equations contain non-conservative products that cannot be written as the divergence of a flux. These arise from (i) the interfacial pressure force $p\, \partial \alpha_{\mathrm{g}}/\partial x$, which exchanges momentum between the phases when the void fraction varies in space, and (ii) the virtual-mass force, which accounts for the acceleration of the surrounding continuous phase induced by the relative acceleration of the dispersed phase. Following Städtke \cite{stadtke_gasdynamic_2006}, the virtual-mass coupling is modeled through a single coefficient $k$ multiplying the mixture density $\rho_{\mathrm{m}} = \alpha_{\mathrm{g}}\rho_{\mathrm{g}} + \alpha_{\mathrm{l}}\rho_{\mathrm{l}}$, with $k = 0.5$ (the added-mass coefficient of a sphere) used throughout this work. The virtual-mass terms are differential (proportional to velocity and pressure gradients) and are therefore part of the non-conservative operator $\mathbf{H}^{\mathrm{nc}}$ rather than of the algebraic source vector $\mathbf{D}$. Their inclusion is not merely a physical refinement: without virtual-mass coupling, the single-pressure two-fluid model has complex characteristic speeds whenever the phase velocities differ, and the initial-value problem is ill-posed \cite{stuhmiller_influence_1977}. The virtual-mass coefficient renders the system hyperbolic for the slip ratios encountered in this work.

The source vector $\mathbf{D}$ collects the algebraic interfacial and geometric contributions:

$$
\mathbf{D} = \mathbf{D}^{\mathrm{int}}\!\left(\sigma_{M},\, F_{\mathrm{v}},\, \sigma_{Q,\mathrm{g}},\, \sigma_{Q,\mathrm{l}}\right) + \mathbf{D}^{A}
$$

where $\sigma_M$ is the interfacial mass transfer rate (evaporation/condensation), $F_{\mathrm{v}}$ is the interfacial drag force per unit volume, $\sigma_{Q,i}$ are the interfacial heat transfer rates, and $\mathbf{D}^{A}$ is the quasi-one-dimensional area-variation source. Because the flux divergence in Eq. (1) is already area-weighted, the area source only contains the terms that the area weighting does not generate, namely the pressure force on the lateral wall and the interfacial pressure work associated with the geometric redistribution of the void fraction:

$$
\mathbf{D}^{A} =
\left[\,0,\;\; 0,\;\; \alpha_{\mathrm{g}}\, p\, G,\;\; \alpha_{\mathrm{l}}\, p\, G,\;\; -p\, D_{\alpha},\;\; +p\, D_{\alpha}\,\right]^{\mathsf T},
\qquad
G = \frac{1}{A}\frac{\mathrm{d}A}{\mathrm{d}x}
$$

with

$$
D_{\alpha} = \alpha_{\mathrm{g}}\alpha_{\mathrm{l}} \frac{a_0^2}{\rho_{\mathrm{s}}}
\left(\frac{\rho_{\mathrm{l}} u_{\mathrm{l}}}{a_{\mathrm{g}}^2} - \frac{\rho_{\mathrm{g}} u_{\mathrm{g}}}{a_{\mathrm{l}}^2}\right) G
$$

where $a_i$ is the single-phase speed of sound of phase $i$, $\rho_{\mathrm{s}} = \alpha_{\mathrm{g}}\rho_{\mathrm{l}} + \alpha_{\mathrm{l}}\rho_{\mathrm{g}}$ is the cross-weighted density, and $a_0$ is the zero-slip (frozen) two-phase sound speed defined below. In the single-phase limit ($\alpha_{\mathrm{g}} \to 1$), $D_\alpha \to 0$ and Eq. (5) reduces to the familiar $p\,\mathrm{d}A/\mathrm{d}x$ wall-pressure source of the quasi-one-dimensional Euler equations.

\subsection{Thermodynamic closure}

Each phase is closed by an independent equation of state expressed in the pressure–enthalpy plane:

$$
\rho_i = \rho_i(p, h_i), \qquad T_i = T_i(p, h_i)
$$

The choice of $(p, h)$ as independent variables is natural for the two-fluid model: pressure is shared between the phases, while the phase enthalpies evolve independently and may correspond to metastable states (superheated liquid or subcooled vapor) during rapid expansions. The partial derivatives of the density required by the quasi-linear analysis and the flux Jacobians follow from the speed of sound $a_i$ and the Grüneisen parameter $\Gamma_i$:

$$
\left(\frac{\partial \rho_i}{\partial p}\right)_{h} = \frac{1 + \Gamma_i}{a_i^2},
\qquad
\left(\frac{\partial \rho_i}{\partial h}\right)_{p} = -\frac{\rho_i\, \Gamma_i}{a_i^2}
$$

which satisfy the thermodynamic identity $1/a_i^2 = (\partial \rho_i/\partial p)_h + (1/\rho_i)(\partial \rho_i/\partial h)_p$ exactly.

The phase properties are evaluated with the Helmholtz energy-based equations of state available in the CoolProp library \cite{bell_pure_2014}. To make the property evaluation compatible with the automatic differentiation and just-in-time compilation framework used by the solver (see Sec. "Time integration"), the equations of state are pre-tabulated on structured $(p, h)$ grids and interpolated with smooth ($C^1$-continuous) bicubic splines. Separate tables are built for the vapor and the liquid phase, each extended into the metastable region by extrapolation of the single-phase Helmholtz energy surface across the saturation line, so that superheated-liquid and subcooled-vapor states arising from finite-rate interphase heat and mass transfer remain well-defined. Saturation properties ($T_{\mathrm{sat}}$, $h_{\mathrm{sat,g}}$, $h_{\mathrm{sat,l}}$) required by the interfacial transfer models are tabulated separately as functions of pressure.

\subsection{Interfacial transfer closures}

The interfacial exchange terms in $\mathbf{D}^{\mathrm{int}}$ require closure laws for the interfacial area density and for the drag, heat, and mass transfer at the phase interface.

**Flow topology and interfacial area.** The flow-regime description follows the standard three-region map based on the void fraction: bubbly flow (gas dispersed in liquid) for $\alpha_{\mathrm{g}} < \alpha_{\mathrm{b}}$, droplet flow (liquid dispersed in gas) for $\alpha_{\mathrm{g}} > \alpha_{\mathrm{d}}$, and a transition regime in between, with $\alpha_{\mathrm{b}} = 0.3$ and $\alpha_{\mathrm{d}} = 0.7$. In the dispersed regimes, the characteristic diameter $d$ of the bubbles or droplets is computed from a prescribed number density, and the interfacial area concentration follows as $a_{\mathrm{int}} = \pi d^2 N$; both quantities are interpolated smoothly across the transition regime.

**Interfacial drag.** The drag force per unit volume exerted on the gas phase is modeled as

$$
F_{\mathrm{v}} = \frac{3}{4}\, \frac{C_{\mathrm{D}}}{d}\, \alpha_{\mathrm{d}}\, \rho_{\mathrm{c}}\, \left| u_{\mathrm{l}} - u_{\mathrm{g}} \right| \left( u_{\mathrm{l}} - u_{\mathrm{g}} \right)
$$

where the subscripts $\mathrm{c}$ and $\mathrm{d}$ denote the continuous and dispersed phases as determined by the flow regime, and the drag coefficient $C_{\mathrm{D}}$ is evaluated from the particle Reynolds number with the standard drag curve for spheres. An equal and opposite force acts on the liquid phase.

**Interfacial heat and mass transfer.** Each phase exchanges sensible heat with the interface, which is assumed to be at the saturation temperature $T_{\mathrm{sat}}(p)$:

$$
\sigma_{Q,\mathrm{g}} = \mathrm{htc}_{\mathrm{g}}\, a_{\mathrm{int}} \left( T_{\mathrm{sat}} - T_{\mathrm{g}} \right),
\qquad
\sigma_{Q,\mathrm{l}} = \mathrm{htc}_{\mathrm{l}}\, a_{\mathrm{int}} \left( T_{\mathrm{sat}} - T_{\mathrm{l}} \right)
$$

The heat transfer coefficient on the continuous-phase side is computed from the Ranz–Marshall correlation \cite{ranz_evaporation_1952}, $\mathrm{Nu} = 2 + 0.6\, \mathrm{Re}^{1/2} \mathrm{Pr}^{1/3}$, while a constant conduction-limited Nusselt number is used on the dispersed-phase side. The net interfacial mass transfer rate follows from the energy balance at the interface:

$$
\sigma_{M} = -\frac{\sigma_{Q,\mathrm{g}} + \sigma_{Q,\mathrm{l}}}{h_{\mathrm{sat,g}} - h_{\mathrm{sat,l}}}
$$

with the convention $\sigma_M > 0$ for evaporation. The transferred mass leaves the donor phase at its saturation enthalpy and carries the velocity of the phase it leaves (donor upwinding), consistent with the convention adopted in system codes \cite{relap5_manual}. Phase change is activated only when the metastable state of a phase drives it: evaporation requires a superheated liquid ($h_{\mathrm{l}} > h_{\mathrm{sat,l}}$ and $T_{\mathrm{l}} > T_{\mathrm{sat}}$) and condensation a subcooled vapor ($h_{\mathrm{g}} < h_{\mathrm{sat,g}}$ and $T_{\mathrm{g}} < T_{\mathrm{sat}}$); at exact saturation both mechanisms vanish. For two-component mixtures (e.g., air–water), the mass transfer term is disabled and the phases exchange only sensible heat, with the two film coefficients combined in series.

% \subsection{Characteristic structure}

% The design of the numerical flux relies on the characteristic analysis of the system. Written in quasi-linear form for the primitive variable vector

% $$
% \mathbf{U} = \left[\, p,\; u_{\mathrm{g}},\; u_{\mathrm{l}},\; \alpha_{\mathrm{g}},\; h_{\mathrm{g}},\; h_{\mathrm{l}} \,\right]^{\mathsf T}
% $$

% the homogeneous part of Eq. (1) becomes $\mathbf{U}_t + \mathbf{G}(\mathbf{U})\,\mathbf{U}_x = \mathbf{0}$, where the coefficient matrix $\mathbf{G}$ includes the interfacial-pressure and virtual-mass coupling. Closed-form expressions for the eigenvalues and the complete set of right and left eigenvectors of $\mathbf{G}$ were derived following Städtke \cite{stadtke_gasdynamic_2006} and verified symbolically. The six eigenvalues are:

% $$
% \lambda_{1,5} = u_{\mathrm{g}}, \qquad
% \lambda_{2,6} = u_{\mathrm{l}}, \qquad
% \lambda_{3,4} = u_{\mathrm{m}} \pm a
% $$

% i.e., two pairs of material waves transporting the void fraction and the phase entropies at the phase velocities, and one pair of acoustic waves propagating at speed $a$ relative to an effective mixture velocity $u_{\mathrm{m}}$. Introducing the virtual-mass-modified densities

% $$
% \hat{\rho}_{\mathrm{g}} = \rho_{\mathrm{g}} + k \rho_{\mathrm{m}}, \qquad
% \hat{\rho}_{\mathrm{l}} = \rho_{\mathrm{l}} + k \rho_{\mathrm{m}}, \qquad
% \hat{\rho}^{\,2} = \rho_{\mathrm{g}} \rho_{\mathrm{l}} + k \rho_{\mathrm{m}}^2
% $$

% the mixture convective velocity is the weighted average

% $$
% u_{\mathrm{m}} = \frac{\alpha_{\mathrm{g}} \rho_{\mathrm{g}} \hat{\rho}_{\mathrm{l}}}{\hat{\rho}^{\,2}}\, u_{\mathrm{g}}
% + \frac{\alpha_{\mathrm{l}} \rho_{\mathrm{l}} \hat{\rho}_{\mathrm{g}}}{\hat{\rho}^{\,2}}\, u_{\mathrm{l}}
% $$

% and the two-phase sound speed is

% $$
% a^2 = \frac{\hat{\rho}_{\mathrm{s}}}{\rho_{\mathrm{s}}} \frac{\rho_{\mathrm{g}} \rho_{\mathrm{l}}}{\hat{\rho}^{\,2}}\, a_0^2
% \;-\; \frac{\alpha_{\mathrm{g}} \alpha_{\mathrm{l}}\, \rho_{\mathrm{g}} \rho_{\mathrm{l}}\, \hat{\rho}_{\mathrm{g}} \hat{\rho}_{\mathrm{l}}}{\hat{\rho}^{\,4}} \left(u_{\mathrm{g}} - u_{\mathrm{l}}\right)^2,
% \qquad
% a_0^2 = \frac{\rho_{\mathrm{s}}}{\dfrac{\alpha_{\mathrm{g}} \rho_{\mathrm{l}}}{a_{\mathrm{g}}^2} + \dfrac{\alpha_{\mathrm{l}} \rho_{\mathrm{g}}}{a_{\mathrm{l}}^2}}
% $$

% where $\hat{\rho}_{\mathrm{s}} = \rho_{\mathrm{s}} + k\rho_{\mathrm{m}}$. In the zero-slip limit, $a$ reduces to the frozen sound speed $a_0$, which exhibits the characteristic non-monotonic behavior of bubbly mixtures (Wood's speed): for an air–water mixture at atmospheric pressure, $a_0$ drops to a minimum of roughly $20\ \mathrm{m/s}$ near $\alpha_{\mathrm{g}} = 0.5$, far below the sound speed of either pure phase. Consequently, two-phase nozzle flows choke at velocities of a few tens of meters per second, and the mixture Mach number — not the single-phase one — governs the flow regime.

\subsection{Space discretization}

The system of Eq. (1) is discretized in space with the finite volume method on a uniform grid of $N$ cells. The semi-discrete form for cell $i$ reads:

$$
\frac{\mathrm{d} \mathbf{V}_i}{\mathrm{d} t}
= -\frac{1}{A_i\, \Delta x} \left( A_{i+\frac{1}{2}} \hat{\mathbf{F}}_{i+\frac{1}{2}} - A_{i-\frac{1}{2}} \hat{\mathbf{F}}_{i-\frac{1}{2}} \right)
- \mathbf{H}^{\mathrm{nc}}_i \, \frac{\hat{\mathbf{F}}_{i+\frac{1}{2}} - \hat{\mathbf{F}}_{i-\frac{1}{2}}}{\Delta x}
+ \mathbf{D}_i
$$

where $\hat{\mathbf{F}}_{i\pm\frac{1}{2}}$ are the numerical fluxes at the cell faces. Second-order spatial accuracy is achieved by MUSCL reconstruction of the primitive variables $\mathbf{U}$ with the van Leer slope limiter; the limiter reverts locally to first order wherever the reconstructed states would leave the validity domain of the property tables.

\subsubsection{Flux vector splitting}

Godunov-type schemes based on approximate Riemann solvers, such as the Roe and HLL schemes commonly used for the Euler equations, do not carry over directly to the two-fluid system: the non-conservative products preclude a well-defined weak (Rankine–Hugoniot) form of the equations, and no exact Riemann solution is available. Instead, the interface fluxes are computed with the characteristic-based flux vector splitting (FVS) scheme proposed by Städtke \cite{stadtke_gasdynamic_2006}, which requires only the eigenstructure of the quasi-linear system. The interface flux between a left state $\mathbf{U}_{\mathrm{L}}$ and a right state $\mathbf{U}_{\mathrm{R}}$ is assembled wave by wave:

$$
\hat{\mathbf{F}} = \mathbf{R}^{+}\, \mathbf{F}(\mathbf{U}_{\mathrm{L}}) + \mathbf{R}^{-}\, \mathbf{F}(\mathbf{U}_{\mathrm{R}}),
\qquad
\mathbf{R}^{\pm} = \sum_{k=1}^{6} \frac{1 \pm s(\lambda_k)}{2}\, \tilde{\mathbf{P}}_k
$$

where the flux-space projector of wave $k$,

$$
\tilde{\mathbf{P}}_k = \frac{\left(\mathbf{K}\, \mathbf{r}_k\right) \left(\boldsymbol{\ell}_k\, \mathbf{K}^{-1}\right)}{\boldsymbol{\ell}_k \cdot \mathbf{r}_k}
$$

is built from the right and left eigenvectors $\mathbf{r}_k$, $\boldsymbol{\ell}_k$ of $\mathbf{G}$ and the flux Jacobian $\mathbf{K} = \partial \mathbf{F} / \partial \mathbf{U}$, all evaluated at the arithmetic average of the reconstructed face states. Each wave thus contributes its portion of the flux from the upwind side determined by the sign of its eigenvalue: material waves ($u_{\mathrm{g}}$, $u_{\mathrm{l}}$) are upwinded at the phase velocities, while the acoustic pair is upwinded at $u_{\mathrm{m}} \pm a$, which naturally captures choking at the mixture sonic condition.

Three modifications make the baseline scheme robust over the full range of operating conditions:

1. **Entropy fix.** The exact sign function in Eq. (15) is discontinuous at $\lambda_k = 0$, which admits unphysical expansion shocks and destroys the differentiability required by the implicit solver. The Harten entropy fix \cite{harten_high_1983} replaces $\mathrm{sign}(\lambda_k)$ with the $C^1$-smoothed function $s(\lambda) = \lambda / |\lambda|_\delta$, where $|\lambda|_\delta = (\lambda^2 + \delta^2)/2\delta$ for $|\lambda| < \delta$. The smoothing is activated selectively: only at faces where an eigenvalue changes sign from negative to positive across the face (an expansion sonic transition, which for accelerating nozzle flow occurs only on the $u_{\mathrm{m}} - a$ wave at the throat), with $\delta = \lambda_{k,\mathrm{R}} - \lambda_{k,\mathrm{L}}$. Elsewhere the exact upwind sign is retained, so no dissipation is added to the slow material waves in smooth flow.

2. **Low-Mach scaling.** In the upstream portion of the nozzle the mixture Mach number is small, and the acoustic dissipation of the standard splitting exceeds the physical pressure–velocity coupling by a factor $\mathcal{O}(1/\mathrm{Ma})$, producing the well-known accuracy degradation of compressible upwind schemes at low Mach number \cite{rieper_low-mach_2011}. The dissipative (jump) part of the two acoustic waves is therefore scaled by $z = \min(\mathrm{Ma}_{\mathrm{m}}, 1)$, where $\mathrm{Ma}_{\mathrm{m}} = |u_{\mathrm{m}}|/a$ is the face mixture Mach number; the scaling deactivates smoothly as the flow accelerates and leaves the transonic and supersonic regions unaffected.

3. **Zero-slip regularization.** At vanishing slip ($u_{\mathrm{g}} \to u_{\mathrm{l}}$), the two void-wave eigenvectors become parallel and the projector decomposition in Eq. (16) is rank-deficient. The combined contribution of the void-wave pair is then recovered from the completeness identity $\sum_k \tilde{\mathbf{P}}_k = \mathbf{I}$ and upwinded at the common material speed, which is exact in the zero-slip limit. A $C^1$ blend between this degenerate form and the full decomposition is applied over a narrow band of slip velocities, so the flux remains smooth across slip sign changes.

Finally, since the eigenvalues of the material waves vanish with the phase velocities, the upwind dissipation of the FVS scheme does not damp odd–even (grid-scale) oscillations of the void fraction in low-velocity regions. A fourth-order background dissipation term of the Jameson–Schmidt–Turkel type \cite{jameson_numerical_1981}, scaled by the local spectral radius and added in conservative flux form, suppresses this mode while perturbing smooth solutions only at $\mathcal{O}(\Delta x^3)$; a smooth velocity gate confines it to the low-velocity region near the inlet.

\subsubsection{Non-conservative terms}

The non-conservative contribution in Eq. (14) is evaluated at the cell centre following the operational form

$$
\mathbf{H}^{\mathrm{nc}}_i\, \Delta \hat{\mathbf{F}} = \mathbf{X}_i\, \mathbf{K}_i^{-1}\, \Delta \hat{\mathbf{F}},
\qquad
\mathbf{X} = \mathbf{J}\,\mathbf{G} - \mathbf{K}
$$

where $\mathbf{J} = \partial \mathbf{V}/\partial \mathbf{U}$ and $\Delta \hat{\mathbf{F}} = \hat{\mathbf{F}}_{i+\frac{1}{2}} - \hat{\mathbf{F}}_{i-\frac{1}{2}}$. The defect matrix $\mathbf{X}$ measures exactly the departure of the system from conservative form: its mass rows vanish identically, its two momentum rows are equal and opposite (so the mixture momentum is conserved discretely to machine precision), and it reduces to zero in the single-phase limit. Closed-form expressions for $\mathbf{X}$ were derived and verified against the analytical structure of the system.

\subsection{Time integration}

The discretized equations are integrated in time with the implicit (backward) Euler method until a steady state is reached. Implicit integration is essential here: the interfacial drag, heat, and mass transfer sources introduce relaxation time scales that are orders of magnitude shorter than the acoustic CFL limit, making the system stiff.

At each time step, the nonlinear residual

$$
\mathbf{R}_i = V_{c,i} \left( \mathbf{V}_i^{n+1} - \mathbf{V}_i^{n} \right)
+ \Delta t \left( A_{i+\frac{1}{2}} \hat{\mathbf{F}}^{n+1}_{i+\frac{1}{2}} - A_{i-\frac{1}{2}} \hat{\mathbf{F}}^{n+1}_{i-\frac{1}{2}} \right)
+ \Delta t\, A_i\, \mathbf{X}_i \mathbf{K}_i^{-1} \Delta \hat{\mathbf{F}}^{n+1}
- \Delta t\, V_{c,i}\, \mathbf{D}_i^{n+1} = \mathbf{0}
$$

is solved by Newton iteration, where $V_{c,i} = A_i \Delta x$ is the cell volume and all fluxes and sources are evaluated at the new time level. The entire residual — including the tabulated equations of state, the eigenstructure, and the interfacial closure laws — is implemented in the JAX framework \cite{jax2018github}, so the exact Jacobian $\partial \mathbf{R} / \partial \mathbf{W}$ is obtained by forward-mode automatic differentiation rather than by numerical perturbation or hand-coded linearization. The banded sparsity of the Jacobian (each cell couples only to its reconstruction stencil) is exploited through graph coloring, reducing the cost of the Jacobian assembly from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ and making it independent of the grid size per cell.

The Newton unknowns are an augmented per-cell variable set $\mathbf{W} = (p, \alpha_{\mathrm{g}}, m_{\mathrm{g}}, u_{\mathrm{g}}, m_{\mathrm{l}}, u_{\mathrm{l}}, m_{\mathrm{g}} e_{\mathrm{g}}, m_{\mathrm{l}} e_{\mathrm{l}})$, with $m_i = \alpha_i \rho_i$ the partial densities and $e_i$ the specific internal energies. This choice makes both the map $\mathbf{W} \mapsto \mathbf{V}$ and the map $\mathbf{W} \mapsto \mathbf{U}$ purely algebraic (no equation-of-state inversion is required inside the Newton loop), at the cost of two redundant unknowns per cell. The resulting rectangular system (six residuals, eight unknowns per cell) is solved in the minimum-norm least-squares sense with row and column equilibration, and the thermodynamic consistency conditions $m_i = \alpha_i\, \rho_i(p, h_i)$ are re-imposed after each Newton update by a local two-variable projection onto the equation-of-state manifold. The projection is a no-op on consistent states and therefore does not affect the converged solution.

The time step is adapted according to the CFL condition based on the largest characteristic speed of the two-fluid eigenstructure:

$$
\Delta t = \mathrm{CFL}\, \frac{\Delta x}{\max_i \left( |u_{\mathrm{m}}| + a \right)_i}
$$

where, importantly, $a$ is the two-phase mixture sound speed of Eq. (13) and not the (much larger) single-phase sound speeds. Since the scheme is implicit, CFL numbers well above unity are admissible once the transient has decayed. The solution is considered converged when the normalized steady-state residual, evaluated from Eq. (18) with $\Delta t$ replaced by the characteristic acoustic time $\Delta x / \lambda_{\max}$, falls below a prescribed tolerance.

\subsection{Boundary conditions}

The number of physical conditions imposed at each boundary follows from the characteristic analysis of Sec. "Characteristic structure". At a subsonic inlet, five of the six characteristics enter the domain and one (the backward acoustic wave $u_{\mathrm{m}} - a$) leaves it; at a subsonic outlet one characteristic enters (again the backward acoustic); at a supersonic outlet all characteristics leave and no physical condition may be imposed.

At the inlet, five physical quantities are prescribed: the stagnation enthalpies of the two phases $H_{0,\mathrm{g}}$ and $H_{0,\mathrm{l}}$, the stagnation entropy of the gas phase $s_{0,\mathrm{g}}$, the inlet quality (gas mass fraction) $x_0$, and the slip ratio $S_0 = u_{\mathrm{g}}/u_{\mathrm{l}}$. The remaining numerical condition — the mixture velocity $u_{\mathrm{m}}$, which sets the mass flow rate — is extrapolated from the interior along the outgoing characteristic, so that the mass flow adjusts freely and the nozzle can choke. Given the extrapolated $u_{\mathrm{m}}$, the ghost state is reconstructed by a small fixed-point iteration: the void fraction follows from the quality and slip ratio,

$$
\alpha_{\mathrm{g}} = \left[ 1 + \frac{1 - x_0}{x_0} \frac{\rho_{\mathrm{g}}}{\rho_{\mathrm{l}}}\, S_0 \right]^{-1}
$$

the phase velocities from the slip split of $u_{\mathrm{m}}$, the static enthalpies from $h_i = H_{0,i} - u_i^2/2$, and the static pressure from the gas-phase reservoir isentrope $s_{\mathrm{g}}(p, h_{\mathrm{g}}) = s_{0,\mathrm{g}}$, evaluated by interpolation of a pre-tabulated isentrope.

At the outlet, a characteristic-based condition automatically switches between the subsonic and supersonic regimes according to the local mixture Mach number, evaluated with the same eigenstructure used by the flux. In the subsonic regime, the static back-pressure $p_{\mathrm{out}}$ is imposed following the non-reflecting (LODI-type) approach \cite{poinsot_boundary_1992}: pressure and velocity in the ghost cell are moved jointly along the outgoing acoustic characteristic,

$$
u^{\mathrm{ghost}}_i = u_i + \frac{p - p_{\mathrm{out}}}{\rho_{\mathrm{m}}\, a}
$$

so that imposing the back-pressure emits no spurious reflected wave. In the supersonic regime all variables are extrapolated (transmissive condition). The two limits are blended smoothly across the sonic point so that the outlet condition remains differentiable for the Newton solver.



