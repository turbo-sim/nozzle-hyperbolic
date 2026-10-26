
Extension of the steady model of Agromayor, Müller & Nord (*Int. J. Turbomach. Propuls. Power* 2019, 4, 31) to the unsteady case, cast in a form suitable for finite-volume discretization.

**Novelty and impact.** Formulating the diffuser equations in time-marching, hyperbolic form would extend the steady-state model int he following ways:

- It reduces exactly to the paper's steady equations and its $\det(\mathbf{A})=0$ choking condition as a special case, so the extension does not depart from the paper's validated results, it generalizes them. Furthermore, it gives a physical explanation for the paper's steady singularity. The paper's own determinant, $\det(\mathbf{A}_{\text{paper}})=\rho^3v_m^2a^2(\mathrm{Ma}_m^2-1)$, vanishes at $\mathrm{Ma}_m=1$, but the paper leaves this as a property of the steady spatial-marching form. The eigenanalysis of Section 3 shows it is exactly the statement that $v_m-a\to0$: an acoustic characteristic stalls relative to the duct, so the steady march loses the ability to sweep information downstream and the governing matrix becomes singular. The unsteady system's eigenvalues, by contrast, are always real regardless of $\mathrm{Ma}_m$, so retaining the time derivative replaces this ill-posed marching problem with a well-posed initial-boundary-value problem that stays hyperbolic everywhere.
- It shows that swirl plays no role in choking or shock formation, and explains why. The tangential velocity $v_\theta$ does not appear in $\mathbf{A}$ at all: it is carried by one of the two linearly degenerate eigenvalues $\lambda=v_m$, alongside entropy, and is simply advected with the meridional flow rather than propagating acoustically. Because this field is linearly degenerate rather than genuinely nonlinear, it cannot steepen into a shock, which is the rigorous version of the paper's own observation that swirl decelerates from supersonic to subsonic without shock waves.
- It enables standard finite-volume and CFD solution strategies, Roe- or HLL-type Riemann solvers and implicit pseudo-time marching, to be applied directly to the diffuser equations, in place of the paper's custom shooting-type ODE integration that must be stopped before the flow chokes.
- It supplies a complete, characteristics-based boundary-condition theory (Section 5) that generalizes the paper's simple initial-value treatment at the diffuser inlet to a properly closed inlet and outlet treatment valid in both subsonic and supersonic regimes.
- It opens the door to transient diffuser analysis, such as part-load operation, surge, and off-design transients, that a strictly steady formulation cannot represent at all. This makes the model useable in coupled, time-resolved low-fidelity turbomachinery simulations.

**Geometry:**

- $A(m) = 2\pi r(m)\,b(m)$ is the duct cross-sectional area, a function of $m$ only since the walls are rigid and non-deforming.
- $r(m)$ is the mean radius of the annular channel.
- $b(m)$ is the channel height, measured normal to the meridional direction.
- $\phi(m)$ is the mean wall cant angle, i.e. the local slope of the meridional channel. Since $m$ is an arc-length coordinate, $dr/dm = \sin\phi$ and $dx/dm=\cos\phi$, equivalently $\tan\phi = dr/dx$ (paper's Eq. 3).

$r(m)$, $b(m)$, $\phi(m)$ follow from the shape of the meridional channel.

**Closures:**

- $p = p(\rho,e)$ is the equation of state, giving pressure from density and specific internal energy.
- $e = E - \tfrac12(v_m^2+v_\theta^2)$ is the specific internal energy, obtained by removing the kinetic energy from the specific total energy $E$.
- $a^2 = (\partial p/\partial \rho)_s$ is the squared speed of sound, the isentropic compressibility of the fluid.
- $H = E + p/\rho$ is the specific stagnation enthalpy, the quantity conserved in adiabatic, no-work flow.
- $\tau_w = C_f\rho v^2/2$, with $v^2=v_m^2+v_\theta^2$, is the wall shear stress from the skin friction coefficient and the local speed.
- $\cos\alpha = v_m/v$ and $\sin\alpha = v_\theta/v$ are the meridional and tangential fractions of the velocity vector, used to resolve the friction force into its meridional and tangential components.

---

## 1. Conservative form

The equations for mass, meridional momentum, tangential momentum, and total energy in conservative, area-weighted form are given by

$$
\begin{gather}
\frac{\partial (A\rho)}{\partial t} + \frac{\partial (A\rho v_m)}{\partial m} = 0 \\
\frac{\partial (A\rho v_m)}{\partial t} + \frac{\partial}{\partial m}\Big[A(\rho v_m^2+p)\Big] = p\,\frac{dA}{dm} + A\rho\,\frac{v_\theta^2}{r}\sin\phi - \frac{2A}{b}\,\tau_w\cos\alpha \\
\frac{\partial (A\rho v_\theta)}{\partial t} + \frac{\partial (A\rho v_m v_\theta)}{\partial m} = -A\rho\,\frac{v_m v_\theta}{r}\sin\phi - \frac{2A}{b}\,\tau_w\sin\alpha \\
\frac{\partial (A\rho E)}{\partial t} + \frac{\partial (A\rho v_m H)}{\partial m} = \frac{2A}{b}\,\dot q_w
\end{gather}
$$

These four equations can be expressed in vector notation as

$$
\begin{equation}
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial m} = \mathbf{S}(\mathbf{U};m)
\end{equation}
$$

$$
\begin{equation}
\mathbf{U}=\begin{pmatrix}A\rho\\ A\rho v_m\\ A\rho v_\theta\\ A\rho E\end{pmatrix},\qquad
\mathbf{F}(\mathbf{U})=\begin{pmatrix}A\rho v_m\\ A(\rho v_m^2+p)\\ A\rho v_m v_\theta\\ A\rho v_m H\end{pmatrix},\qquad
\mathbf{S}(\mathbf{U};m)=\begin{pmatrix}
0\\
p\frac{dA}{dm}+A\rho\frac{v_\theta^2}{r}\sin\phi-\frac{2A}{b}\tau_w\cos\alpha\\
-A\rho\frac{v_mv_\theta}{r}\sin\phi-\frac{2A}{b}\tau_w\sin\alpha\\
\frac{2A}{b}\dot q_w
\end{pmatrix}
\end{equation}
$$

The mass equation has no source term, so mass is exactly conserved. The meridional momentum source has three parts: the term $p\,dA/dm$ is the axial thrust exerted by the pressure acting on the diverging duct walls, the term $A\rho v_\theta^2/r\sin\phi$ is the centrifugal effect of swirl as the mean radius changes along the channel, and the term $-2A\tau_w\cos\alpha/b$ is the meridional component of wall friction. The tangential momentum source mirrors this without the pressure term: $-A\rho v_mv_\theta/r\sin\phi$ is the same curvature coupling acting on the tangential component, and $-2A\tau_w\sin\alpha/b$ is the tangential component of wall friction. The total energy source contains only the wall heat flux $2A\dot q_w/b$, since the duct walls are stationary and therefore do no work on the fluid, even though they exert both a pressure force and a shear stress on it.

---
 
## 2. Quasi-linear form
 
The conservative fluxes can be expanded by the product rule and combined with the mass equation to eliminate the flux-divergence cross-terms, exactly as in the paper's Appendix A. Rather than the primitive variables $(v_m,v_\theta,\rho,p)$ used in the paper, it is more convenient here to work with $(\rho,v_m,v_\theta,s)$, replacing pressure with specific entropy. This is purely a change of basis, since $p=p(\rho,s)$ closes the system just as well as $p=p(\rho,e)$, but it results in a simpler quasi-linear form. With this substitution, the mass, meridional momentum, tangential momentum, and entropy equations are given by
 
$$
\begin{gather}
\frac{\partial \rho}{\partial t} + v_m\frac{\partial \rho}{\partial m} + \rho\frac{\partial v_m}{\partial m} = -\frac{\rho v_m}{A}\frac{dA}{dm} \\
\frac{\partial v_m}{\partial t} + v_m\frac{\partial v_m}{\partial m} + \frac{a^2}{\rho}\frac{\partial \rho}{\partial m} + \frac{1}{\rho}\left(\frac{\partial p}{\partial s}\right)_\rho\frac{\partial s}{\partial m} = \frac{v_\theta^2}{r}\sin\phi - \frac{2\tau_w}{b\rho}\cos\alpha \\
\frac{\partial v_\theta}{\partial t} + v_m\frac{\partial v_\theta}{\partial m} = -\frac{v_mv_\theta}{r}\sin\phi - \frac{2\tau_w}{b\rho}\sin\alpha \\
\frac{\partial s}{\partial t} + v_m\frac{\partial s}{\partial m} = \frac{2(\tau_w v+\dot q_w)}{\rho\, b\, T}
\end{gather}
$$
 
The last of these follows from combining the paper's entropy transport equation, Eq. A57, with its entropy-generation expression, Eq. A58. The wall-temperature terms cancel, leaving the familiar second-law statement that entropy is produced by friction and by heat transfer across the finite temperature difference $T$.
 
The derivative $(\partial p/\partial s)_\rho$ appearing in the momentum equation can be rewritten using the Grüneisen parameter, $\Gamma = (\partial p/\partial e)_\rho/\rho$, a general equation-of-state quantity. From the Gibbs relation $T\,ds=de-(p/\rho^2)\,d\rho$, holding $\rho$ fixed gives $(\partial e/\partial s)_\rho=T$, and the chain rule then gives $(\partial p/\partial s)_\rho=(\partial p/\partial e)_\rho(\partial e/\partial s)_\rho=\rho\,T\,\Gamma$. With this substitution, the momentum equation above can be restated as
 
$$
\begin{gather}
\frac{\partial \rho}{\partial t} + v_m\frac{\partial \rho}{\partial m} + \rho\frac{\partial v_m}{\partial m} = -\frac{\rho v_m}{A}\frac{dA}{dm} \\
\frac{\partial v_m}{\partial t} + v_m\frac{\partial v_m}{\partial m} + \frac{a^2}{\rho}\frac{\partial \rho}{\partial m} + T\Gamma\,\frac{\partial s}{\partial m} = \frac{v_\theta^2}{r}\sin\phi - \frac{2\tau_w}{b\rho}\cos\alpha \\
\frac{\partial v_\theta}{\partial t} + v_m\frac{\partial v_\theta}{\partial m} = -\frac{v_mv_\theta}{r}\sin\phi - \frac{2\tau_w}{b\rho}\sin\alpha \\
\frac{\partial s}{\partial t} + v_m\frac{\partial s}{\partial m} = \frac{2(\tau_w v+\dot q_w)}{\rho\, b\, T}
\end{gather}
$$
 
In vector notation this system reads
 
$$
\begin{equation}
\frac{\partial \mathbf{V}}{\partial t} + \mathbf{A}(\mathbf{V})\,\frac{\partial \mathbf{V}}{\partial m} = \mathbf{S}_s(\mathbf{V};m)
\end{equation}
$$
 
$$
\begin{equation}
\mathbf{V}=\begin{pmatrix}\rho\\ v_m\\ v_\theta\\ s\end{pmatrix},\qquad
\mathbf{A}=\begin{pmatrix}
v_m & \rho & 0 & 0\\
\frac{a^2}{\rho} & v_m & 0 & T\Gamma\\
0 & 0 & v_m & 0\\
0 & 0 & 0 & v_m
\end{pmatrix},\qquad
\mathbf{S}_s(\mathbf{V};m)=\begin{pmatrix}
-\frac{\rho v_m}{A}\frac{dA}{dm}\\
\frac{v_\theta^2}{r}\sin\phi-\frac{2\tau_w}{b\rho}\cos\alpha\\
-\frac{v_mv_\theta}{r}\sin\phi-\frac{2\tau_w}{b\rho}\sin\alpha\\
\frac{2(\tau_w v+\dot q_w)}{\rho\, b\, T}
\end{pmatrix}
\end{equation}
$$
 
Unlike the $(v_m,v_\theta,\rho,p)$ basis, the time-derivative term here is multiplied by the identity matrix, so the system is already in standard quasi-linear form, ready for a direct eigenvalue/eigenvector analysis of $\mathbf{A}$.
 
---
 ## 3. Eigenanalysis
 
The characteristic speeds of the system are the eigenvalues of $\mathbf{A}$, found from
 
$$
\begin{equation}
\det(\mathbf{A}-\lambda\mathbf{I}) = 0
\end{equation}
$$
 
Written out,
 
$$
\begin{equation}
\mathbf{A}-\lambda\mathbf{I}=\begin{pmatrix}
v_m-\lambda & \rho & 0 & 0\\
\frac{a^2}{\rho} & v_m-\lambda & 0 & T\Gamma\\
0 & 0 & v_m-\lambda & 0\\
0 & 0 & 0 & v_m-\lambda
\end{pmatrix}
\end{equation}
$$
 
Rows 3 and 4 have a nonzero entry only on the diagonal, so the determinant can be expanded by cofactors along these rows without touching the off-diagonal terms in the top-left block. This gives
 
$$
\begin{equation}
\det(\mathbf{A}-\lambda\mathbf{I}) = (v_m-\lambda)^2\det\begin{pmatrix}v_m-\lambda & \rho\\ a^2/\rho & v_m-\lambda\end{pmatrix} = (v_m-\lambda)^2\Big[(v_m-\lambda)^2-a^2\Big]
\end{equation}
$$
 
so the full spectrum is
 
$$
\begin{equation}
\lambda_1 = v_m-a,\qquad \lambda_2=\lambda_3=v_m,\qquad \lambda_4=v_m+a
\end{equation}
$$
 
All four eigenvalues are real for any equation of state with $a^2>0$. Reality of the eigenvalues alone is not quite sufficient for hyperbolicity, however: $\mathbf{A}$ must also be diagonalizable, that is, it must possess a complete set of four linearly independent eigenvectors, since a defective matrix with a repeated eigenvalue but too few independent eigenvectors would still fail to decouple into independent scalar wave equations. Since $\lambda=v_m$ here has algebraic multiplicity two, this needs to be checked explicitly rather than assumed, and is confirmed once the eigenvectors are found below. As expected, none of the eigenvalues depend on the swirl velocity $v_\theta$, which does not appear in $\mathbf{A}$ at all.
 
The corresponding eigenvectors follow directly from $(\mathbf{A}-\lambda_k\mathbf{I})\mathbf{r}_k=0$, giving
 
$$
\begin{equation}
\mathbf{r}_1 = \begin{pmatrix}\rho\\ -a\\ 0\\ 0\end{pmatrix},\qquad
\mathbf{r}_2 = \begin{pmatrix}0\\ 0\\ 1\\ 0\end{pmatrix}, \qquad
\mathbf{r}_3 = \begin{pmatrix}-\rho T\Gamma/a^2\\ 0\\ 0\\ 1\end{pmatrix}, \qquad
\mathbf{r}_4 = \begin{pmatrix}\rho\\ a\\ 0\\ 0\end{pmatrix}
\end{equation}
$$
 
$\mathbf{r}_1$ and $\mathbf{r}_4$ carry density and meridional-velocity perturbations only, the left- and right-running acoustic waves. $\mathbf{r}_2$ is a pure swirl perturbation, and $\mathbf{r}_3$ is an entropy perturbation with an associated density change at constant pressure. A characteristic field $(\lambda_k,\mathbf{r}_k)$ is genuinely nonlinear if $\nabla\lambda_k\cdot\mathbf{r}_k\neq0$ everywhere, meaning the wave speed changes along its own eigenvector direction, which is what allows a simple wave in that family to steepen into a shock or spread into a rarefaction. It is linearly degenerate if $\nabla\lambda_k\cdot\mathbf{r}_k=0$ everywhere, meaning the wave speed is unchanged along the eigenvector direction, so a jump in that family propagates undistorted, like a contact discontinuity. For $\lambda_{1,4}=v_m\mp a$, the eigenvalue depends on both $v_m$ and $\rho$ through $a=a(\rho,s)$, and $\mathbf{r}_{1,4}$ perturbs exactly these variables, so $\nabla\lambda_{1,4}\cdot\mathbf{r}_{1,4}\neq0$ in general: these fields are genuinely nonlinear. For $\lambda_{2,3}=v_m$, the eigenvalue depends only on $v_m$, and both $\mathbf{r}_2$ and $\mathbf{r}_3$ have zero component in $v_m$, so $\nabla\lambda_{2,3}\cdot\mathbf{r}_{2,3}=0$ identically: these fields are linearly degenerate.
 
Collecting all four eigenvectors as columns of a matrix gives $\det[\mathbf{r}_1,\mathbf{r}_2,\mathbf{r}_3,\mathbf{r}_4]=2\rho a$, which is nonzero for any physical state, confirming they are linearly independent. Together with the fact that all four eigenvalues are real, this completes the hyperbolicity check: $\mathbf{A}$ is diagonalizable, so the system is unconditionally hyperbolic, consistent with the standard quasi-1D compressible-flow result.
 
Physically, this reproduces and sharpens a statement the paper makes only qualitatively from the steady determinant, that the diffuser chokes on the meridional Mach number alone and that swirl decelerates without shocks. The eigenanalysis shows why directly: $v_\theta$ never enters $\mathbf{A}$, so swirl is carried by a linearly degenerate characteristic field with no acoustic character whatsoever, while compressibility and area-change effects act purely through $v_m$ and $a$.
 
This also connects cleanly to the paper's steady-state singularity condition. Their determinant, from Eq. (16), is
 
$$
\begin{equation}
\det(\mathbf{A}_{\text{paper}}) = \rho^3v_m^2a^2\left(\mathrm{Ma}_m^2-1\right)
\end{equation}
$$
 
which vanishes when $v_m=0$ or $v_m=\pm a$. Setting $\lambda=0$ in the unsteady characteristic polynomial derived above gives exactly $v_m^2(v_m^2-a^2)=0$, the same zero locus up to the always-positive factor $\rho^3$. The paper's singular condition is therefore not an artifact of the steady reduction: it is the statement that one of the full system's genuine characteristic speeds, $v_m-a$ or $v_m+a$, becomes zero, meaning an acoustic wave has stopped moving relative to the duct. In the steady spatial-marching formulation, $m$ plays the role that $t$ plays in the unsteady problem, and a characteristic that cannot advance in $m$ is exactly a characteristic that cannot be swept downstream, so the spatial ODE loses the information it needs to determine $d\mathbf{U}/dm$ and the system becomes singular. This is the familiar choking condition, now grounded in the wave structure of the unsteady equations rather than inferred solely from a vanishing determinant.

---

## 4. Numerical solution

### 4.1 Spatial discretization

A finite-volume scheme integrates the conservative equations over each cell $i$ of extent $\Delta m$ to obtain the semi-discrete form

$$
\begin{equation}
\frac{d\bar{\mathbf{U}}_i}{dt} = -\frac{1}{\Delta m}\left(\hat{\mathbf{F}}_{i+1/2}-\hat{\mathbf{F}}_{i-1/2}\right) + \bar{\mathbf{S}}_i
\end{equation}
$$

where $\bar{\mathbf{U}}_i$ is the cell-averaged state and $\hat{\mathbf{F}}_{i+1/2}$ is a numerical flux built from the reconstructed states on either side of the interface, $\mathbf{U}_L$ and $\mathbf{U}_R$. Since the system is hyperbolic, this interface flux is obtained, exactly or approximately, from the solution of a local Riemann problem, and the eigenstructure of Section 3 is what makes that solution possible.

Two families of schemes use it in different ways.

#### Flux vector splitting

Flux vector splitting methods split the physical flux itself into a part associated with positive eigenvalues and a part associated with negative eigenvalues, $\mathbf{F}(\mathbf{U})=\mathbf{F}^+(\mathbf{U})+\mathbf{F}^-(\mathbf{U})$, built directly from the eigenvalues and eigenvectors of $\mathbf{A}$, and then upwind each part according to its own sign:

$$
\begin{equation}
\hat{\mathbf{F}}_{i+1/2}=\mathbf{F}^+(\mathbf{U}_L)+\mathbf{F}^-(\mathbf{U}_R)
\end{equation}
$$

#### Flux difference splitting

Flux difference splitting methods instead work with the jump $\mathbf{U}_R-\mathbf{U}_L$ across the interface. Roe's scheme evaluates a Roe-averaged Jacobian $\hat{\mathbf{A}}$ with the same eigenstructure derived above, decomposes the jump onto the eigenvector basis, $\mathbf{U}_R-\mathbf{U}_L=\sum_k\alpha_k\mathbf{r}_k$, and forms

$$
\begin{equation}
\hat{\mathbf{F}}_{i+1/2} = \frac{1}{2}\Big[\mathbf{F}(\mathbf{U}_L)+\mathbf{F}(\mathbf{U}_R)\Big] - \frac{1}{2}\sum_k |\hat\lambda_k|\,\alpha_k\,\mathbf{r}_k
\end{equation}
$$

which is exactly where the eigenvalues and eigenvectors of Section 3 enter as the wave speeds and wave strengths of the upwind correction. Simpler solvers such as HLL and HLLC forgo the full eigenvector decomposition and instead bound the Riemann fan with two wave-speed estimates, commonly the extreme eigenvalues $S_L=v_m-a$ and $S_R=v_m+a$ evaluated at a Roe-averaged or arithmetic-averaged interface state, and blend the left and right fluxes within that fan.

#### Practical considerations for flux discretization

The genuinely nonlinear and linearly degenerate character of the fields identified in Section 3 also shapes the discretization choices. The acoustic fields can form shocks and rarefactions and need an entropy fix at Roe's scheme's sonic points to avoid unphysical expansion shocks, while the swirl and entropy fields are simple contact-type discontinuities that reconstruction and limiter choices should aim to capture sharply rather than smear with excess numerical dissipation. Note also that the eigenanalysis above was carried out in the variables $\mathbf{V}=(\rho,v_m,v_\theta,s)$ for convenience, while the flux $\mathbf{F}(\mathbf{U})$ is naturally expressed in the conservative variables $\mathbf{U}$; the eigenvalues are unchanged between the two, since they are properties of the same physical wave speeds, but the eigenvectors must be mapped through the Jacobian $\partial\mathbf{U}/\partial\mathbf{V}$ before being used inside a conservative-variable flux solver such as Roe's.

It is worth checking explicitly whether the area $A(m)$ baked into $\mathbf{U}$ and $\mathbf{F}(\mathbf{U})$ changes any of this. Both $\mathbf{U}$ and $\mathbf{F}(\mathbf{U})$ carry $A$ as a common multiplicative factor on every component, and since $A$ depends only on $m$ and not on the flow state, it factors out of $\mathbf{F}$ as $\mathbf{F}(\mathbf{U})=A\,\mathbf{f}(\mathbf{U}/A)$, where $\mathbf{f}$ is the ordinary unweighted flux function of the physical variables $\mathbf{U}/A$. Differentiating with the chain rule, $\partial\mathbf{F}/\partial\mathbf{U} = A\cdot(\partial\mathbf{f}/\partial\tilde{\mathbf{U}})\cdot(1/A) = \partial\mathbf{f}/\partial\tilde{\mathbf{U}}$, so the two factors of $A$ cancel exactly and the conservative flux Jacobian reduces to the ordinary, area-independent Jacobian evaluated at the local physical state. The eigenvalues and eigenvector directions derived in Section 3 are therefore unaffected by $A(m)$: the local wave speeds $v_m$ and $v_m\pm a$ depend only on the physical state, not on how wide the duct happens to be at that point. The area only enters the discrete equations through the source term, specifically the $dA/dm$ terms already present in $\mathbf{S}(\mathbf{U};m)$ and $\mathbf{S}_s(\mathbf{V};m)$ in Sections 1 and 2, never through the characteristic structure itself.

### 4.2 Time integration to steady state

The quantity of interest for diffuser design is the steady solution, but marching directly in $m$ as a steady ODE is exactly what becomes singular at $\mathrm{Ma}_m=1$, as shown by the paper's determinant and reproduced in Section 3. Retaining the time derivative and marching the unsteady system to steady state avoids this difficulty entirely, since the eigenvalues of $\mathbf{A}$ are always real and the unsteady problem never loses hyperbolicity, even at points where a steady spatial march would fail.

Applying the implicit (backward) Euler method to the semi-discrete finite-volume equations gives, at each cell $i$,

$$
\begin{equation}
\frac{\mathbf{U}_i^{n+1}-\mathbf{U}_i^{n}}{\Delta t} + \frac{1}{\Delta m}\left(\hat{\mathbf{F}}_{i+1/2}^{n+1}-\hat{\mathbf{F}}_{i-1/2}^{n+1}\right) = \mathbf{S}_i^{n+1}
\end{equation}
$$

Because the fluxes and source terms are evaluated at the new time level, this is a nonlinear system in $\mathbf{U}^{n+1}$ at every step, which is typically closed with a Newton iteration. Linearizing the numerical flux with the same flux Jacobian used to build $\hat{\mathbf{F}}$, for instance the Roe-averaged $\hat{\mathbf{A}}$ of Section 4.1, produces a block-tridiagonal linear system for the Newton update at each pseudo-time step, directly reusing the eigenstructure derived above. Implicit Euler is unconditionally stable, so $\Delta t$ is not constrained by the acoustic CFL condition that would otherwise follow from the eigenvalues $v_m\pm a$: this matters in particular near choking, where $v_m-a\to0$ and an explicit scheme's admissible time step would collapse even though the unsteady problem itself remains perfectly well posed there. As the pseudo-time-marching iterations converge, $\mathbf{U}_i^{n+1}\to\mathbf{U}_i^n$, the left-hand side vanishes, and the residual $\left(\hat{\mathbf{F}}_{i+1/2}-\hat{\mathbf{F}}_{i-1/2}\right)/\Delta m - \mathbf{S}_i$ is driven to zero, recovering the paper's steady conservative equations as the converged solution.

---

## 5. Boundary conditions

The eigenanalysis of Section 3 does more than classify the wave types: the sign of each eigenvalue at a boundary determines whether the corresponding characteristic carries information into the domain or out of it. A characteristic entering the domain brings information the interior solution cannot supply on its own, so it must be fixed by an externally prescribed physical boundary condition. A characteristic leaving the domain carries information generated by the interior solution itself, so imposing an external value on it would over-specify the problem and is instead left to be determined by extrapolation, or more precisely by a compatibility relation along that characteristic. The number of physical boundary conditions at a boundary therefore equals the number of incoming characteristics there, with the remaining unknowns supplied by the interior solution. With the flow entering the domain in the direction of increasing $m$, so that $v_m>0$ at both boundaries, this count depends only on whether the meridional flow is locally subsonic or supersonic.

### 5.1 Inlet

**Subsonic inlet ($0<v_m<a$).** The acoustic eigenvalue $\lambda_1=v_m-a$ is negative, so this characteristic propagates upstream and out of the domain through the inlet, while $\lambda_2=\lambda_3=v_m$ and $\lambda_4=v_m+a$ are positive and carry information into the domain. Three characteristics are incoming, so three physical quantities can be prescribed. This matches the usual practice of fixing the inlet stagnation state, $p_0$ and $T_0$, together with the flow angle $\alpha=\arctan(v_\theta/v_m)$. The remaining unknown, equivalent to the inlet static pressure or Mach number, is not prescribed but obtained from a compatibility relation along the single outgoing characteristic $\lambda_1$. For example, the static pressure can be extrapolated from the first interior cell and combined with the prescribed stagnation quantities through the isentropic relations to recover the full inlet state. Because the equation of state is generally nonlinear in $p$ and $\rho$, this last step usually requires solving a nonlinear equation for the inlet velocity or Mach number, typically by a Newton iteration; only for a perfect gas do the isentropic relations reduce to a closed-form algebraic expression.

**Supersonic inlet ($v_m>a$).** Now $\lambda_1=v_m-a$ is also positive, so all four characteristics are incoming and no information reaches the boundary from the interior. All four inlet quantities must therefore be prescribed externally: the stagnation state $p_0,T_0$, the flow angle $\alpha$, and additionally the inlet Mach number, since there is no longer an outgoing characteristic available to determine the remaining degree of freedom from inside the domain.

### 5.2 Outlet

**Subsonic outlet ($0<v_m<a$).** At the downstream boundary, $\lambda_1=v_m-a$ is still negative, but here that means it propagates upstream, into the domain from outside, while $\lambda_2=\lambda_3=v_m$ and $\lambda_4=v_m+a$ leave the domain. Only one characteristic is incoming, so only one physical boundary condition can be prescribed, matching the standard choice of fixing the static pressure $p$ at the outlet. The remaining three quantities, equivalent to entropy, meridional velocity, and tangential velocity, are carried by outgoing characteristics and are obtained by extrapolation from the interior solution rather than imposed.

**Supersonic outlet ($v_m>a$).** Here $\lambda_1=v_m-a$ is also positive, so all four characteristics leave the domain and none enter. No physical boundary condition can be prescribed at a supersonic outlet: the entire outlet state is determined by extrapolation from the interior solution. Imposing any external value here would over-specify the problem and generate spurious reflections propagating back into the domain.