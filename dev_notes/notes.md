
## ChatGPT brainstorm:

- Links with questions and answers
	- [https://chatgpt.com/share/6712148a-2b0c-800c-87cc-9a3acfef7652](https://chatgpt.com/share/6712148a-2b0c-800c-87cc-9a3acfef7652 "https://chatgpt.com/share/6712148a-2b0c-800c-87cc-9a3acfef7652")
	- [https://chatgpt.com/share/67120ca5-c130-800c-aa72-4d33f21ef5fc](https://chatgpt.com/share/67120ca5-c130-800c-aa72-4d33f21ef5fc "https://chatgpt.com/share/67120ca5-c130-800c-aa72-4d33f21ef5fc")
- Extrapolation-Based Boundary Conditions
	- **Method**: For supersonic outflow, where all characteristics are leaving the domain, extrapolation can be used. You assume that flow variables remain constant near the boundary and extrapolate them from the interior to the boundary. This is simple and computationally cheap.
	- **Implementation**: For subsonic outflow, you might extrapolate the velocity and density while imposing a static pressure. For supersonic flows, all variables can be extrapolated from the interior.
	- **Advantage**: Simple to implement and works well in regions with fully developed flow.
	- **Sources**: See works like _Computational Fluid Dynamics_ by John D. Anderson, which describes extrapolation boundary conditions for different flow regimes
- Characteristic-Based Boundary Conditions (Riemann Invariants)
	- **Method**: This approach uses the fact that the 1D Euler equations can be decoupled into characteristic variables, or Riemann invariants, that propagate along specific directions (into or out of the domain).
	- **Implementation**: At the boundary, you impose the characteristics based on the flow direction. For instance, for an outflow, you can allow outgoing characteristics to be extrapolated from the interior, while incoming characteristics can be set to known conditions (e.g., atmospheric pressure or velocity).
	- **Advantage**: This approach is especially robust for subsonic flows, where some information must be imposed (typically pressure) while allowing other variables to "transmit" freely through the boundary.
	- **Sources**: Hirsch's _Numerical Computation of Internal and External Flows_ and Toro’s _Riemann Solvers and Numerical Methods for Fluid Dynamics_ describe this approach in detail.

## Reading Material for Andrea

**Toro's textbook**
- Reading 1.6.2 Flow with Area Variation for different formulations of the Euler equations in a nozzle with area variation.
- Reading "Example 2.39 (Isentropic Gas Dynamics Revisited)" from Toro p. 82 for derivation of the Riemann invariant for perfect gas
- Chapter 3 "Some Properties of the Euler Equations"
	- Equation 3.8 for the Jacobian matrix of the Euler equation system
	- Equations 3.9 and 3.10 for the eigenvalues/eigenvectors of the Jacobian matrix
	- Equations 3.17 and 3.18 for the primitive formulation of the Euler equations
	- Equations 3.24 for the characteristic equations of the conservative form of the Euler equations
	- Equations 3.37 and 3.38 for the Riemann invariants of the Euler equations

**Hirsch's textbook**
- 16.4 CHARACTERISTIC VARIABLES AND EIGENVALUES FOR ONE-DIMENSIONAL FLOWS
	- 16.4.1 Eigenvalues and eigenvectors of Jacobian matrix
	- 16.4.2 Characteristic variables
	- 16.4.4 Physical boundary conditions
- 19.1 ONE-DIMENSIONAL BOUNDARY TREATMENT FOR EULER EQUATIONS
	- 19.1.1 Characteristic boundary conditions
	- 19.1.2 Compatibility relations
	- 19.1.3 Characteristic boundary conditions as a function of conservative and primitive variables
	- 19.1.4 Extrapolation methods
	- 19.1.5 Practical implementation methods for numerical boundary conditions
	- 19.1.6 Non-reflecting boundary conditions




## 1D Euler equations
### Conservative form with area change

Formulation given in Toro
$$
(A \mathbf{U})_t+[A \mathbf{F}(\mathbf{U})]_x=-A \mathbf{S}(\mathbf{U})_x
$$
$$
\mathbf{U}=\left[\begin{array}{c}
\rho \\
\rho u \\
\rho E
\end{array}\right], \mathbf{F}=\left[\begin{array}{c}
\rho u \\
\rho u^2 \\
\rho u H
\end{array}\right], \mathbf{S}=\left[\begin{array}{c}
0 \\
p \\
0
\end{array}\right]
$$

Equivalent formulation with the classic momentum flux formulation
$$
(A \mathbf{U})_t+[A \mathbf{F}(\mathbf{U})]_x=\mathbf{S}(\mathbf{U})
$$
$$
\mathbf{U}=\left[\begin{array}{c}
\rho \\
\rho u \\
\rho E
\end{array}\right], \mathbf{F}=\left[\begin{array}{c}
\rho u \\
\rho u^2 +p \\
\rho u H
\end{array}\right], \mathbf{S}=\left[\begin{array}{c}
0 \\
p A_x \\
0
\end{array}\right]
$$

Formulation removing the area-variation terms from the left hand side
$$
\mathbf{U}_t + \mathbf{F}(\mathbf{U})_x=\mathbf{S}(\mathbf{U})
$$
$$
\mathbf{U}=\left[\begin{array}{c}
\rho \\
\rho u \\
\rho E
\end{array}\right], \mathbf{F}=\left[\begin{array}{c}
\rho u \\
\rho u^2 +p \\
\rho u H
\end{array}\right], \mathbf{S}= -\frac{A_x}{A}\left[\begin{array}{c}
\rho u \\
\rho u^2 \\
\rho u H
\end{array}\right]
$$
### Quasi-linear form with primitive variables
From Toro and Hirsch
$$
\mathbf{W}_t+\mathbf{A}(\mathbf{W}) \mathbf{W}_x=\mathbf{S}(\mathbf{U}),
$$
$$
\mathbf{W}=\left[\begin{array}{c}
\rho \\
u \\
p
\end{array}\right], \quad \mathbf{A}(\mathbf{W})=\left[\begin{array}{ccc}
u & \rho & 0 \\
0 & u & 1 / \rho \\
0 & \rho a^2 & u
\end{array}\right], \quad
\mathbf{S}= -\frac{A_x}{A}\left[\begin{array}{c}
\rho u \\
0 \\
\rho u a^2
\end{array}\right]
$$
Eigenvalues of the problem:
$$
\lambda_1=u-a, \lambda_2=u, \lambda_3=u+a,
$$
with corresponding right eigenvectors
$$
\mathbf{K}^{(1)}=\alpha_1\left[\begin{array}{c}
1 \\
-a / \rho \\
a^2
\end{array}\right], \mathbf{K}^{(2)}=\alpha_2\left[\begin{array}{l}
1 \\
0 \\
0
\end{array}\right], \mathbf{K}^{(3)}=\alpha_3\left[\begin{array}{c}
1 \\
a / \rho \\
a^2
\end{array}\right] .
$$
where $\alpha_1, \alpha_2, \alpha_3$ are scaling factors. The left eigenvectors are
$$
\left.\begin{array}{l}
\mathbf{L}^{(1)}=\beta_1\left(0,1,-\frac{1}{\rho a}\right), \\
\mathbf{L}^{(2)}=\beta_2\left(1,0,-\frac{1}{a^2}\right), \\
\mathbf{L}^{(3)}=\beta_3\left(0,1, \frac{1}{\rho a}\right),
\end{array}\right\}
$$
where $\beta_1, \beta_2, \beta_3$ are scaling factors.

### Characteristic equations and Riemann invariants
The characteristic equation says that in a direction $\mathrm{d} x / \mathrm{d} t=\lambda_i$ the change in primitive variables is zero, $\mathbf{L}^{(i)} \cdot \mathrm{d} \mathbf{W}= 0$.
$$
\mathbf{L}^{(i)} \cdot\left[\begin{array}{l}
\mathrm{d} \rho \\
\mathrm{~d} u \\
\mathrm{~d} p
\end{array}\right]=0
$$

By expanding for $\mathbf{L}^{(1)}, \mathbf{L}^{(2)}, \mathbf{L}^{(3)}$ we obtain the characteristic equations 
$$\begin{align}
\mathrm{d} I_\text{L} = \mathrm{d} p-\rho a \mathrm{~d} u=0 & \quad\text{along}\quad \mathrm{d} x / \mathrm{d} t=\lambda_1=u-a, \\ 
\mathrm{d} I_\text{C} = \mathrm{~d} p-a^2 \mathrm{~d} \rho=0 & \quad\text{along}\quad \mathrm{d} x / \mathrm{d} t=\lambda_2=u, \\
\mathrm{d} I_\text{R} = \mathrm{~d} p+\rho a \mathrm{~d} u=0 & \quad\text{along}\quad \mathrm{d} x / \mathrm{d} t=\lambda_3=u+a
\end{align}$$
For the Generalised Riemann Invariants across 1 and 3 rarefactions are
$$
\left.\begin{array}{rl}
R_{\mathrm{L}}(u, a)=u+\frac{2 a}{\gamma-1} & =\text { constant } \\
s & =\text { constant }
\end{array}\right\} \text { across } \lambda_1=u-a
$$
$$
\left.\begin{array}{rl}
R_{\mathrm{R}}(u, a)=u-\frac{2 a}{\gamma-1} & =\text { constant } \\
s & =\text { constant }
\end{array}\right\} \text { across } \lambda_1=u+a
$$
For the Generalised Riemann Invariants across 2 contact discontinuity is??
$$
\left.\begin{array}{rl}
R_{\mathrm{C}}(u, a)= p/\rho^\gamma & =\text { constant } \\
s & =\text { constant }
\end{array}\right\} \text { across } \lambda_1=u
$$

## Characteristic boundary conditions
The boundary conditions at the inlet and outlet of the nozzle can be specified using information from the direction of travel of information and Riemann invariants of the problem.

**Subsonic inlet**
Boundary conditions from Riemann invariantes and 
$$
\begin{aligned}
p_{0,j=1}^{n+1} &= p_{0,\text{BC}} \\
\rho_{0,j=1}^{n+1} &= \rho_{0,\text{BC}}
\\ 
\left[u + 2 c /(\gamma-1) \right]_{j=1}^{n+1} & = R_\text{R} =  \left[u + 2 c /(\gamma-1) \right]_{j=2}^{n}
\end{aligned}
$$
Solution strategy:
- Guess the Mach number
- Compute inlet pressure as $p/p_0 = \left(1 + \frac{\gamma - 1}{2} \text{Ma} \right)^{-\frac{\gamma}{\gamma-1}}$
- Compute inlet density as $\rho/\rho_0 = \left(1 + \frac{\gamma - 1}{2} \text{Ma} \right)^{-\frac{1}{\gamma-1}}$
- Compute the speed of sound from density and pressure
- Compute velocity from Mach number and speed of sound
- Evaluate the error in the residual equation posed by the Riemann invariant
- Iterate on the guessed Mach number until the residual is zero

Linearized version:
- Not clear how to derive it
- Should we linearize equations of stagnation temperature/pressure
- Relate these expressions with the linearized equation of the other Riemann invariant to solve a system of equations?
- One possible approach could be to use the following equations
$$ \begin{align}
 \Delta u - \Delta p / (\rho a) = 0 \\
 \Delta T + u \Delta u / c_p = \Delta T_0 \\
 \frac{\Delta T}{\Delta T_0} \frac{T_0}{T} = \frac{\Delta p}{\Delta p_0} \frac{p_0}{p}
\end{align}
$$
Where $\Delta T_0$ and $\Delta p_0$ can be determined from the boundary conditions. The equations used are the Riemann invariant, the devinition of stagnation temperature, and the isentropic relation between stagnation and static temperatures and pressures. I am not sure if this approach makes sense.

**Subsonic outlet**
$$
\begin{aligned}
p_{j=N}^{n+1} &= p_{\text{BC}} \\
\left[p/\rho \right]_{j=N}^{n+1} &= R_\text{C}  = \left[p/\rho \right]_{j=N-1}^{n} \\ 
\left[u+ 2 c /(\gamma-1) \right]_{j=N}^{n+1} & = R_\text{L} = \left[u+ 2 c /(\gamma-1) \right]_{j=N-1}^{n} 
\end{aligned}
$$
Solution strategy:
- Get pressure from boundary condition
- Get density from the second Riemann invariant
- Compute speed of sound from pressure and density
- Compute velocity from third Riemann invariant
- Compute the vector of conservative variables

Linearized version:
$$
\begin{aligned}
p_{j=N}^{n+1} &= p_{\text{BC}} \\
\rho_{j=N}^{n+1} &= \rho_{0} + (p_{\text{BC}}-p_0)/a_0^2 \\ 
u_{j=N}^{n+1} &= u_{0} + (p_{\text{BC}}-p_0)/(\rho_0 a_0) \\ 

\end{aligned}
$$
where the subscript $0$ signifies ${j=N-1}$ and $t=n$



**Observations:**
- Would extrapolation boundary conditions work well at the inlet where there are no shock waves? 
- Perhaps use extrapolation for two-phase flows where we do not have the perfect gas form of the Riemann invariants? Or is it still better to use the linearized Riemann equations?
- Should the boundary conditions be applied in a linearized way rather than solving a system of equations?
- Is it very wrong to ignore the source terms when applying characteristic boundary conditions?

**Way forward**
- We implement the linear and nonlinear version of the Riemann boundary conditions for perfect gas
- We compare both approaches to make sure the linear version gives the correct results
- If the linear version is good enough, we can rely on in for two-phase simulations
- How do we linearize the stagnation pressure and temperature boundary conditions?


\[
\begin{array}{|c|c|c|}
\hline
\textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} \\
\hline
Row 1, C1 & Row 1, C2 & Row 1, C3 \\
\hline
Row 2, C1 & Row 2, C2 & Row 2, C3 \\
\hline
Row 3, C1 & Row 3, C2 & Row 3, C3 \\
\hline
\end{array}
\]