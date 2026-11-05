# Nonconservative terms: from Euler finite volumes to state-space integration

This note starts from conservative finite volumes and explains the treatment
of nonconservative terms in the two-fluid nozzle solver. The development is
organized around the numerical problem that motivated the change:

1. **Old approach:** use the physical-flux Jacobian to reconstruct a state
   change from a numerical-flux difference.
2. **Problem:** understand why that inverse-based representation can become
   poorly conditioned and obstruct nonlinear convergence.
3. **Alternative:** integrate the coupling directly along state changes, then
   work through its path definition, Gauss quadrature and finite-volume assembly.

Sections 1-2 establish the Euler background and notation. Sections 3-4 explain
the old method and its failure mechanism. Sections 5-9 develop the replacement
and its mathematics and numerics. The remaining sections compare the resulting
solutions, explain the qualifications and alternatives, and summarize the
existing validation evidence.

The relevant implementation is split between the native package and the
[v2 development study](../../development/two_fluid_test_v2/README.md). This note
explains the prototype; its numerical changes have not been promoted into
`src/`.

## 1. What conservative Euler discretization gives us

For Euler, write the conserved state as $U$ and its flux as $F$:

$$
\frac{\partial U}{\partial t}+\frac{\partial F(U)}{\partial x}=0.
$$

Integrating over a cell gives

$$
\frac{d}{dt}\int_{x_{i-1/2}}^{x_{i+1/2}}U\,dx
+F_{i+1/2}-F_{i-1/2}=0.
$$

A finite-volume method replaces each physical boundary flux by a numerical
flux $\widehat F(U_L,U_R)$. The same flux is used by both neighbors, with
opposite signs. Internal face contributions therefore cancel when cells are
summed. This is discrete conservation, independently of the particular
upwind flux chosen.

Conservation also gives a precise meaning to a discontinuity. A shock moving
with speed $\sigma$ satisfies

$$
-\sigma[U]+[F]=0,
\qquad [f]=f_R-f_L.
$$

This Rankine–Hugoniot condition depends only on the states on either side.
Entropy admissibility is still needed to select physical shocks, but the
conservation jump itself has no ambiguity about the unresolved transition.

## 2. General vector notation

Use $U$ for the conservative transport variables and $W$ for the working
variables in which the equations are expressed and solved. They need not be
the same variables: write $U=U(W)$. First consider a square transport system
with $m$ independent working coordinates, for example primitives:

$$
\boxed{
\frac{\partial U(W)}{\partial t}+\frac{\partial F(W)}{\partial x}+B(W)\,\frac{\partial W}{\partial x}=Q(W).
}
$$

Here $Q$ denotes ordinary local sources, such as interphase exchange. The
nonconservative term depends on a spatial derivative and needs separate
treatment.

Writing $F(W)$ means evaluating the physical flux from the state represented
by $W$; when it is written naturally as a function of conservative variables,
this is the composition $F_{\mathrm{cons}}(U(W))$.
From this section onward, $F_L=F(W_L)$, $F_R=F(W_R)$ and the numerical
flux takes working-coordinate traces as its inputs. Its output still has
conservative transport-flux ordering and units.
Cell subscripts indicate evaluation at the cell state, for example
$B_i=B(W_i)$ and $K_i=K(W_i)$. Overbars denote frozen face-state evaluations;
the prototype uses the arithmetic mean $\overline W=(W_L+W_R)/2$.

| Symbol | Meaning |
|---|---|
| $W\in\mathbb R^m$ | Working variables used to express and solve the equations, for example primitives |
| $U(W)\in\mathbb R^m$ | Conservative transport variables expressed as a function of $W$ |
| $F(W)\in\mathbb R^m$ | Physical transport flux, in conservative transport-equation ordering |
| $J(W)=\frac{\partial U}{\partial W}$ | Storage Jacobian, converting working-variable changes into conservative-variable changes |
| $K(W)=\frac{\partial F}{\partial W}$ | Physical-flux Jacobian |
| $B(W)\in\mathbb R^{m\times m}$ | Coupling matrix with transport-equation rows and working-coordinate columns |
| $A(W)=J^{-1}(K+B)$ | Full primitive-space propagation matrix, when $J$ is invertible |

For smooth solutions, the chain rule gives

$$
\begin{aligned}
J(W)\frac{\partial W}{\partial t}
+[K(W)+B(W)]\frac{\partial W}{\partial x}&=Q(W),\\
\frac{\partial W}{\partial t}
+A(W)\frac{\partial W}{\partial x}&=J^{-1}Q(W).
\end{aligned}
$$

The eigenvalues of $A$ describe the propagation of the **complete system**.
$K$ alone omits the nonconservative coupling. Hyperbolicity of the complete
system does not require $K$ to be invertible. A zero wave speed is also allowed;
it is not by itself an invalid physical state.

The $J$ above is a physical storage Jacobian, not the large Jacobian of the
nonlinear residual assembled by Newton's method.

At the nozzle code's analytic matrix-policy level, the correspondence is

$$
W=(p,u_g,u_\ell,\alpha_g,h_g,h_\ell)^T,
\qquad \alpha_\ell=1-\alpha_g,
\qquad B=X,\qquad A_{\mathrm{policy}}=G_h.
$$

In these coordinates the conservative variables are

$$
U(W)=
\begin{pmatrix}
\alpha_g\rho_g\\
\alpha_\ell\rho_\ell\\
\alpha_g\rho_g u_g\\
\alpha_\ell\rho_\ell u_\ell\\
\alpha_g\rho_g E_g\\
\alpha_\ell\rho_\ell E_\ell
\end{pmatrix},
\qquad E_k=e_k+\frac12u_k^2,
\qquad h_k=e_k+\frac{p}{\rho_k}.
$$

Here $k\in\{g,\ell\}$ labels the phases, $e_k$ is specific internal energy,
and $E_k$ is specific total energy. Calling these the conservative variables
describes the stored quantities; it does not assert that every individual
balance is free of nonconservative terms or sources.

The six-component $W$ above is the working coordinate vector used by the
flux and path kernels. The actual Newton iterate in the package is an
eight-component constrained representation, denoted $W_{\mathrm{code}}$ here,
with six transport balances and two EOS constraints. The square Jacobian
identities in this note refer to the six independent coordinates on the
EOS-consistent state manifold; they are not an inverse of a six-by-eight
Jacobian with respect to the full constrained iterate.

The matrix module calls the six primitive coordinates `U_h` and the six
transport quantities `V`; this note consistently calls them $W$ and $U$,
respectively. The implementation's variable names are unchanged. Also, the
analytic `J_matrix` and `K_matrix` use a thermodynamic derivative policy;
they obey $J_{\mathrm{policy}}G_h=K_{\mathrm{policy}}+B$.
Unsubscripted $J$, $K$ and $A$ in the general derivations denote the exact
theoretical Jacobians and propagation matrix. The policy matrices need not
equal the derivatives of the actual interpolated EOS; Section 11 makes that
distinction explicit.

**A change of variables is not, by itself, a genuinely nonconservative model.**
Euler written in primitives has a quasilinear matrix, but its conservative
storage and flux still define the shock jumps. The additional difficulty is
a coupling $B(W)\frac{\partial W}{\partial x}$ that cannot generally be absorbed into a flux derivative.

## 3. Old approach: reconstructing state changes from flux differences

The old approach starts from the smooth chain rule. If the physical-flux
Jacobian is invertible,

$$
\begin{aligned}
\frac{\partial F}{\partial x}
&=K(W)\frac{\partial W}{\partial x},\\
\frac{\partial W}{\partial x}
&=K(W)^{-1}\frac{\partial F}{\partial x},\\
B(W)\frac{\partial W}{\partial x}
&=B(W)K(W)^{-1}\frac{\partial F}{\partial x}.
\end{aligned}
$$

This motivates recovering a primitive-space change from a flux-space change.
Write $\mathcal A_i$ for the cell's cross-sectional area, distinct from the
propagation matrix $A$. In schematic constant-area notation, the native cell
operation is

$$
N_i^{\mathrm{old}}=B_i\,y_i,
\qquad
K_i y_i=\delta f_i,
\qquad
\delta f_i=\mathcal A_i
  (\widehat F_{i+1/2}-\widehat F_{i-1/2}).
$$

An important implementation detail is that these are differences of the
**transported numerical compatibility flux before optional filter fluxes**.
They are not simply endpoint differences of the physical function $F(W)$.
Numerical dissipation therefore also enters this construction.

The geometry policies in
[`integrated_nonconservative`](../../src/nozzle_hyperbolic/models/two_fluid/model.py)
are:

| Policy | Integrated right-hand side $\delta f_i$ |
|---|---|
| `cell` | $\mathcal A_i(\widehat F_{i+1/2}-\widehat F_{i-1/2})$ |
| `staedtke` | $\mathcal A_{i+1/2}\widehat F_{i+1/2}-\mathcal A_{i-1/2}\widehat F_{i-1/2}$ |
| `wb` | The preceding expression minus $F(W_i)(\mathcal A_{i+1/2}-\mathcal A_{i-1/2})$ |

The last subtraction is motivated by the smooth identity

$$
\frac{\partial(\mathcal A F)}{\partial x}
-F\frac{\partial\mathcal A}{\partial x}
=\mathcal A\frac{\partial F}{\partial x}.
$$

The code does not explicitly form a dense inverse. It calls
[`compatible_flux_solve`](../../src/nozzle_hyperbolic/models/two_fluid/flux.py),
with row/column equilibration and special range-compatible handling of exactly
resting phases, and then multiplies by $X=B$. Writing $K^{-1}$ here describes
the mathematical action where the inverse exists, not a naive implementation.

The same state-from-flux reconstruction also occurs in the native face flux.
For a full characteristic branch, its schematic form is

$$
\widehat F^{\mathrm{old}}
=\frac{F_L+F_R}{2}
-\frac12\overline K\,S\,\overline K^{-1}(F_R-F_L),
$$

Here $S$ contains signed characteristic upwind weights. Acoustic rescaling and
the small-slip branch have additional details, but the important dependency
is already visible: both the cell correction and the face dissipation use an
action of $K^{-1}$. The next section explains why that can be a fragile way to
evaluate an otherwise meaningful physical coupling.

## 4. Problem: why the inverse-based representation became fragile

The smooth chain-rule argument does not guarantee that inverting $K$ is a
robust numerical representation at every state visited by an implicit solve.

Suppose a finite flux change can be written as

$$
\Delta F=\overline K\Delta W+e.
$$

$e$ can contain nonlinear finite-jump effects, interpolation inconsistency
and arithmetic error. For the actual cell correction, the numerical-flux
construction adds further differences from the simple physical-flux identity.
Recovering the change through $\overline K$ gives

$$
\overline K^{-1}\Delta F
=\Delta W+\overline K^{-1}e.
$$

A small $e$ can become large near a singular $K$. For example,

$$
K_\varepsilon=
\begin{bmatrix}\varepsilon&0\\0&1\end{bmatrix},
\qquad
\Delta F=K_\varepsilon\Delta W+
\begin{bmatrix}\eta\\0\end{bmatrix}
$$

produces an extra state change $(\eta/\varepsilon,0)^T$. The physical
nonconservative action $B\Delta W$ need not have any such singularity.

There are two different conditioning issues. Mixed physical units can make a
raw condition number large, and scaling can help. However, the saved failure
diagnosis also found very large condition numbers after equilibration, and
high-precision evaluation retained the large inverse actions. That is evidence
of a badly behaved mathematical action, not merely a low-quality linear solve.
The earlier local study record
`development/multigrid/TWO_FLUID_ROBUSTNESS.md` contains those probes and
their limitations. That record is not included in this documentation commit;
the versioned v2 results and report are linked in Section 13.

Implicit differentiation illustrates the consequence. If $K(W)y(W)=b(W)$,
then

$$
dy=K^{-1}(db-(dK)y).
$$

Large inverse actions can create large residual derivatives and a very narrow
region where a Newton update decreases the residual. Repeatedly reducing the
pseudo-time step can then produce negligible spatial progress. An accurately
solved Newton system does not remove the pole in the residual it is solving.

These problems occurred in admissible interior states; they were not explained
by terminal ghost/EOS defects or a ghost phase fraction reaching zero in the
inspected failures. Also, the singular surfaces of $K$ are not simply the
mixture Mach-one surface. Allowing a real acoustic pair, allowing sonic flow
and avoiding an inverse pole are distinct issues. A real acoustic pair alone
does not establish strong hyperbolicity of the entire system.

As Section 3 shows, this representation appears in two places. Changing only
the cell nonconservative term would leave the inverse-dependent face flux in
place. The replacement therefore addresses both operations.

## 5. Alternative: use state changes directly

The failure mechanism suggests a different starting point: use the state
change already available from the neighboring states, rather than recover it
from a flux change through $K^{-1}$.

For the cell coupling, start again from the intended physical integral
$N_i=\int_{C_i}B(W)\frac{\partial W}{\partial x}\,dx$. The substitution $dW=\frac{\partial W}{\partial x}\,dx$ motivates
integrating $B(W)\,dW$ directly. This removes the problematic inverse, but
leaves two questions: which intermediate states connect the endpoints, and
how should that integral be evaluated numerically? Sections 6-9 answer these
questions in order.

The face dissipation also changes. Instead of the flux-space reconstruction
in Section 3, the prototype applies a viscosity to the primitive-state jump:

$$
\boxed{
\widehat F^{\mathrm{new}}
=\frac{F_L+F_R}{2}
-\frac12\overline J_{\mathrm{policy}}\,\mathcal D\,(W_R-W_L).
}
$$

$\mathcal D$ is a numerical viscosity with speed-like magnitudes, and
$\overline J_{\mathrm{policy}}$ converts the primitive-space action into
transport-flux units. Acoustic rescaling and the grouped small-slip treatment
require their additional implementation qualifications; the formula is not
a claim that every branch is a simple diagonalization.

Both $K$-inverse actions have now been removed from this spatial package.
The physical flux, stored quantities, EOS configuration, closures and boundary
equations are reused. The implicit Newton solver still solves linear systems;
“inverse-free” refers specifically to the physical-flux $K$ representation.

This removes the requirement that $K$ be invertible in order to evaluate the
nonconservative path or the new dissipation. It does not remove physical EOS
singularities, invalid states or every possible characteristic degeneracy.
The grouped small-slip viscosity is a bounded approximation, not a claim of
the exact matrix absolute value at a material Jordan block.

A separate change replaces a hard sonic-activation switch by a continuous
Harten magnitude. It improves the continuity of that numerical weighting,
but is distinct from integrating $X\,dW$. The v2 experiment retains the old
sonic law in one inverse-free arm specifically to separate these effects at
the package level.

We now develop the cell integral itself: first its meaning across a jump,
then the path mathematics, the quadrature and the assembly into cell balances.

## 6. Why direct integration needs a path

Before choosing quadrature points, we must decide what transition is being
integrated. Consider the scalar contribution

$$
p\,\frac{\partial\alpha_g}{\partial x}.
$$

It appears pressure-like, but it is not simply $\frac{\partial(p\alpha_g)}{\partial x}$:

$$
\frac{\partial(p\alpha_g)}{\partial x}
=p\,\frac{\partial\alpha_g}{\partial x}+\alpha_g\,\frac{\partial p}{\partial x}.
$$

Replacing one by the other without retaining the extra term changes the
equation. More generally, a row of $B(W)\,dW$ need not be the differential of
any scalar flux function.

The two-fluid matrix contains, among other couplings, opposite phase-momentum
contributions $-p\,\frac{\partial\alpha_g}{\partial x}$ and
$+p\,\frac{\partial\alpha_g}{\partial x}$. They redistribute momentum between the phases and
cancel in their sum. The following $+p\,d\alpha_g$ example illustrates the
integration issue; it is not the complete two-fluid matrix.

Suppose a transition connects $(p_L,\alpha_{g,L})$ to
$(p_R,\alpha_{g,R})$. Three different transitions give

| Transition in state space | $\int p\,d\alpha_g$ |
|---|---|
| Change pressure first, then phase fraction | $p_R(\alpha_{g,R}-\alpha_{g,L})$ |
| Change phase fraction first, then pressure | $p_L(\alpha_{g,R}-\alpha_{g,L})$ |
| Change both linearly together | $\tfrac12(p_L+p_R)(\alpha_{g,R}-\alpha_{g,L})$ |

For $p_L=1$ MPa, $p_R=2$ MPa, $\alpha_{g,L}=0.2$ and
$\alpha_{g,R}=0.8$, these are respectively $1.2$, $0.6$ and $0.9$ MPa.
The endpoints are identical in all three calculations. The integral differs
because the sequence of intermediate states differs.

In a smooth resolved flow, the actual profile supplies those intermediate
states. Across a mathematical jump, $\frac{\partial W}{\partial x}$ contains a singular contribution
while $B(W)$ may itself jump. Ordinary multiplication of distributions does
not define their product uniquely. A prescription for the transition is
needed. The path-based weak-solution framework supplies one such prescription;
see [Muñoz-Ruiz and Parés (2007)](https://www.numdam.org/articles/10.1051/m2an:2007011/).

## 7. Defining the path integral and its shock condition

The preceding example shows why endpoint states alone are insufficient.
To specify the transition, choose a family of admissible curves

$$
\Phi:[0,1]\times\Omega\times\Omega\longrightarrow\Omega,
\qquad
\Phi(0;W_L,W_R)=W_L,
\quad \Phi(1;W_L,W_R)=W_R.
$$

$\Omega\subset\mathbb R^m$ is the admissible domain of working states $W$.
Define the vector contribution

$$
\boxed{
\mathcal P_B(W_L,W_R)
=\int_0^1 B(\Phi(s;W_L,W_R))\,
             \frac{\partial\Phi(s;W_L,W_R)}{\partial s}\,ds.
}
$$

This specifies how the nonconservative product contributes across a jump. For
a constant-area problem with bounded local sources, the resulting generalized
shock condition is

$$
-\sigma[U]+[F]+\mathcal P_B(W_L,W_R)=0.
$$

The jumps here are $[U]=U(W_R)-U(W_L)$ and $[F]=F(W_R)-F(W_L)$.
The path is parameterized in $W$, but the storage jump remains a jump of
the conservative vector $U$. For nonlinear storage, it is not generally
equal to $J(\overline W)(W_R-W_L)$ at a single frozen state.

It reduces to the conservative jump condition when $B=0$. If $B=\frac{\partial G}{\partial W}$
for some vector function $G$, its path integral is simply $[G]$ and is independent
of the curve. The difficulty concerns couplings without such a flux potential.

The curve is part of the interpretation of a shock, not just an integration
convenience. A physical viscosity or relaxation model can provide information
about the internal shock profile and hence motivate a family of paths.
An entropy condition or other physical admissibility information may still
be required; a chosen path alone is not a complete uniqueness theorem.

## 8. Numerical integration: state space and Gauss quadrature

With the path defined, we can make its numerical evaluation explicit. There
are two related integration spaces and an ordinary quadrature parameter.

### Physical space: the nozzle coordinate $x$

For now, divide out any constant cross-sectional area, so contributions are
written per unit area. The cell contribution, at a fixed time, is

$$
N_i=\int_{x_{i-1/2}}^{x_{i+1/2}}B(W(x))\,\frac{\partial W(x)}{\partial x}\,dx.
$$

It describes changes along the spatial profile in a cell. On a smooth part
of that profile, the substitution $dW=\frac{\partial W}{\partial x}\,dx$ gives

$$
\int B(W(x))\frac{\partial W(x)}{\partial x}\,dx=\int_{\Gamma}B(W)\,dW,
$$

where $\Gamma$ is the curve traced by $W(x)$ through state space.

### State space: pressure, velocities, phase fraction and enthalpies

In the prototype, the interface curve is the straight primitive path

$$
\Phi(s)=W_L+s\Delta W,
\qquad \Delta W=W_R-W_L,
\qquad 0\le s\le1.
$$

For the two-fluid model, this is a curve in six-dimensional state space.
At each value of $s$ there is a complete intermediate state: one pressure,
two velocities, one independent phase fraction and two enthalpies.

All components advance together with the same $s$. This is a **one-dimensional
line integral through a six-dimensional space**, not a six-dimensional volume
integral and not six separate integrations holding other variables fixed.

The parameter $s$ is dimensionless. It is neither physical time nor another
mesh coordinate. The path derivative is the vector

$$
\frac{\partial\Phi}{\partial s}=\Delta W.
$$

Thus

$$
\boxed{
\mathcal P_B(W_L,W_R)
=\left[\int_0^1B(W_L+s\Delta W)\,ds\right]\Delta W.
}
$$

The EOS is evaluated at these intermediate states. There is no need to solve
an ODE for this straight path: its states are obtained by interpolation.
Densities, temperatures and sound speeds are then derived from each
intermediate pressure/enthalpy state; they are not independently interpolated
between their endpoint values.

### Quadrature space: the interval $s\in[0,1]$

The remaining numerical task is an ordinary one-dimensional integral over $s$.
The prototype uses two-point Gauss–Legendre quadrature on $[0,1]$:

$$
s_1=\frac12-\frac{1}{2\sqrt3}\simeq0.2113248654,
\qquad
s_2=\frac12+\frac{1}{2\sqrt3}\simeq0.7886751346,
\qquad w_1=w_2=\frac12.
$$

Consequently,

$$
\boxed{
\mathcal P_B(W_L,W_R)
\approx\frac12 B(W_L+s_1\Delta W)\Delta W
       +\frac12 B(W_L+s_2\Delta W)\Delta W.
}
$$

The weights already include the transformation from the usual Gauss interval
$[-1,1]$ to $[0,1]$. Their sum is one. There is no additional factor of two.

For each path, the actual calculation is:

```python
def nc_path(W_L, W_R, coupling_matrix):
    jump = W_R - W_L
    contribution = zeros(number_of_transport_equations)
    for s, weight in gauss_rule_on_zero_one:
        W_q = W_L + s * jump
        B_q = coupling_matrix(W_q)  # includes EOS evaluation at W_q
        contribution += weight * (B_q @ jump)
    return contribution
```

This is pseudocode for the square transport system. The accumulator has
transport-equation ordering and the result has flux units; it is not a
working-variable increment, even though both vectors have $m$ components.
The executable
[`path_nc`](../../development/two_fluid_test_v2/scripts/numerics.py) evaluates
both phase EOS objects, constructs the imported $X$ matrix and performs the
matrix-vector action at each of the two points. It uses fixed-size JAX mapping,
so the operation remains compatible with JIT and automatic differentiation.
It never solves a system with $K$.

Two-point Gauss quadrature integrates polynomials in $s$ through degree three
exactly. The two-fluid integrand contains nonlinear thermodynamics and is
generally not such a polynomial, so the result is an approximation. Two Gauss
points also do not, by themselves, establish second-order accuracy of the
complete PDE scheme: reconstruction, geometry, dissipation and shocks matter.

The integral has the units of a flux contribution: the derivative and spatial
integration have already combined into $dW$. A face jump therefore receives
**no extra factor of $\Delta x$**. Division by cell length or volume belongs to
the subsequent finite-volume residual assembly.

## 9. Assembling the integrals: interfaces, cell interiors and nozzle area

The quadrature gives one vector contribution for a pair of states. We now
place those contributions into the finite-volume cell balance.

Let $W_i^-$ be cell $i$'s reconstructed trace at its left face, and $W_i^+$ its
trace at its right face. At the right interface, the pair is
$(W_i^+,W_{i+1}^-)$.

For a constant-area cell, again dividing out its common area, the prototype
uses the following contribution per unit area:

$$
\boxed{
N_i=
\frac12\mathcal P_B(W_{i-1}^+,W_i^-)
+\mathcal P_B(W_i^-,W_i^+)
+\frac12\mathcal P_B(W_i^+,W_{i+1}^-).
}
$$

Let $U_i$ denote the discrete conservative cell state represented by $W_i$,
with $U_i=U(W_i)$ in the independent-coordinate description. The semidiscrete
balance per unit area is then

$$
\Delta x_i\frac{dU_i}{dt}
+\widehat F_{i+1/2}-\widehat F_{i-1/2}+N_i
=\Delta x_i Q_i.
$$

### What happens with constant reconstruction?

Both traces of a cell equal its cell state: $W_i^-=W_i^+=W_i$.
The internal path has zero length and contributes zero. Each interface path
is evaluated once, and half its vector contribution is assigned to each
neighboring cell.

These halves have the **same orientation** in the two cell balances: both use
the path from the left state to the right state. They do not cancel like the
conservative face flux. Their sum represents the nonconservative contribution
across that interface.

One way to see the bookkeeping is to define left/right interface fluctuations

$$
D^- =\widehat F-F_L+\tfrac12\mathcal P_B,
\qquad
D^+ =F_R-\widehat F+\tfrac12\mathcal P_B.
$$

Then

$$
D^-+D^+=[F]+\mathcal P_B.
$$

This is the interface balance that a path-consistent method must represent.
The left cell receives $D^-$ from its right face and the right cell receives
$D^+$ from its left face. These are contributions to the spatial residual,
added to the storage time derivative, rather than direct state increments.
With constant reconstruction these reproduce the
flux-difference-plus-$N_i$ formula above. The framework generalizes conservative
approximate Riemann solvers; see [Parés (2006)](https://epubs.siam.org/doi/10.1137/050628052).

### Why is there also an internal term with linear reconstruction?

A reconstructed cell varies from $W_i^-$ to $W_i^+$. That variation has its
own nonconservative contribution. For a profile linear in the primitive
coordinates, its physical-space integral maps directly to the straight
state-space path between those traces.

In smooth flow, accurate reconstruction makes neighboring face states close
to each other. Much of the resolved variation is then **inside the cells**,
not in their interface jumps. Integrating only the jumps would omit that
variation and fail to approximate the intended smooth nonconservative term.
The internal path is therefore essential, not an optional extra smoothing.

### What changes in a nozzle?

Use $\mathcal A(x)$ for cross-sectional area, reserving $A$ for the propagation
matrix. Restoring area and replacing the boundary fluxes in the integrated
balance by numerical fluxes gives

$$
\frac{d}{dt}\int_{C_i}\mathcal A U\,dx
+[\mathcal A\widehat F]_{i-1/2}^{i+1/2}
+\int_{C_i}\mathcal A B(W)\frac{\partial W}{\partial x}\,dx
=\int_{C_i}\mathcal A Q\,dx+Q_{\mathcal A,i}.
$$

$Q_{\mathcal A,i}$ denotes the separately implemented geometric source.
The prototype approximates the nonconservative integral by

$$
N_i=
\frac{\mathcal A_{i-1/2}}2\mathcal P_B(W_{i-1}^+,W_i^-)
+\mathcal A_i\mathcal P_B(W_i^-,W_i^+)
+\frac{\mathcal A_{i+1/2}}2\mathcal P_B(W_i^+,W_{i+1}^-).
$$

The face factors use the actual face areas. The internal term freezes area
at its cell-center value, a midpoint approximation. More elaborate volume
quadrature could evaluate both $\mathcal A(x)$ and the reconstructed state at
physical-space quadrature points instead. The current code does not do that.

The geometric source retains its existing exact face-area difference. A
consistent path integral does not automatically make every flowing nozzle
equilibrium exactly well-balanced; flux, geometry and source discretizations
must also agree.

### Practical safeguards

Each quadrature state must remain admissible for the EOS and physical model.
Straight interpolation keeps $\alpha_g$ between admissible endpoint fractions,
but a general admissible thermodynamic/hyperbolic domain need not be convex.
Admissible endpoints do not prove every intermediate state is admissible.
The prototype does not silently clip path states.

For the straight path and symmetric Gauss rule,
$\mathcal P_B(W,W)=0$ and
$\mathcal P_B(W_R,W_L)=-\mathcal P_B(W_L,W_R)$.
These are useful kernel checks. They do not imply that all nonconservative
contributions cancel globally.

## 10. Why the physical profiles can remain essentially the same

There is a useful exact calculation behind this expectation. Consider a
constant-coefficient linear system, after dividing out its constant
cross-sectional area, with

$$
U=JW,\qquad F=KW,\qquad JA=K+B,
$$

and assume $J,K$ are invertible and $A$ is diagonalizable with real speeds.
Ignore sources, filters, special slip grouping and sonic modifications for
this calculation. Put $S=\operatorname{sign}(A)$ and
$|A|=A\operatorname{sign}(A)$.

With constant reconstruction, the old numerical flux becomes

$$
\widehat F^{\mathrm{old}}_{i+1/2}
=\frac12K(W_i+W_{i+1})
-\frac12KS(W_{i+1}-W_i).
$$

Because the old cell correction acts on the same flux difference, its full
spatial residual density is

$$
\begin{aligned}
R_i^{\mathrm{old}}
&=(I+BK^{-1})
  \frac{\widehat F^{\mathrm{old}}_{i+1/2}
       -\widehat F^{\mathrm{old}}_{i-1/2}}{\Delta x}\\
&=JA\frac{W_{i+1}-W_{i-1}}{2\Delta x}
-J|A|\frac{W_{i+1}-2W_i+W_{i-1}}{2\Delta x}.
\end{aligned}
$$

For the new flux, use $J|A|$ directly as the dissipation matrix. Since $B$ is
constant, each path integral is exactly $B\Delta W$, so the new cell term is

$$
N_i^{\mathrm{new}}=\frac12B(W_{i+1}-W_{i-1}).
$$

Adding this to the new flux difference gives

$$
R_i^{\mathrm{new}}
=(K+B)\frac{W_{i+1}-W_{i-1}}{2\Delta x}
-J|A|\frac{W_{i+1}-2W_i+W_{i-1}}{2\Delta x}
=R_i^{\mathrm{old}}.
$$

Thus, in this restricted linear case, the complete discretizations agree
exactly even though the old one passes through a $K$-solve. The new expression
computes the combined action without that potentially dangerous intermediate
representation. This is an explanation of the mechanism, not a proof of
nonlinear equivalence of the implemented methods.

For nonlinear, thermodynamically consistent smooth problems, both constructions
are intended to approximate the same $\frac{\partial F}{\partial x}+B \frac{\partial W}{\partial x}$ balance. Differences in
dissipation and quadrature can then be truncation effects. Keeping the same
physical data and boundary equations can produce close resolved profiles.
At finite jumps or on coarse meshes, the paths and viscosity can instead
produce measurable changes. Merely observing agreement of a few curves does
not prove identical shock selection or identical refinement limits.

## 11. Two qualifications that prevent an overly strong equivalence claim

### The interpolated EOS and analytic derivative policy need not agree exactly

The implementation distinguishes the actual interpolated physical flux from
analytic matrices built using sound speed and Grüneisen data. Distinguish
the actual storage and flux derivatives

$$
J_{\mathrm{AD}}=\frac{\partial U_{\mathrm{actual}}}{\partial W},
\qquad
K_{\mathrm{AD}}=\frac{\partial F_{\mathrm{actual}}}{\partial W}
$$

from $J_{\mathrm{policy}}$ and $K_{\mathrm{policy}}$. The exact quasilinear
matrix of the written smooth balance is
$A=J_{\mathrm{AD}}^{-1}(K_{\mathrm{AD}}+B)$, where the storage derivative is
invertible. It need not coincide with the code's analytic $G_h$ for fixed,
independently interpolated property tables.

Where the inverse exists, the literal native smooth NC limit is

$$
B K_{\mathrm{policy}}^{-1}K_{\mathrm{AD}}\frac{\partial W}{\partial x},
$$

which need not equal $B \frac{\partial W}{\partial x}$ exactly. The new direct integral targets the
intended $B \frac{\partial W}{\partial x}$ term. This can remove amplification of a small thermodynamic
derivative mismatch, as well as amplification of finite-jump errors.

Consequently the change is not just row scaling or a mathematically identical
conditioning trick. It retains the intended physical model and the configured
property data while changing the numerical construction, including this
fixed-table compatibility artifact. The analytic $J_{\mathrm{policy}}$ still
appears in the prototype's artificial dissipation; the physical flux itself
is evaluated from the actual EOS interface.

### Formal path consistency is not a complete shock-convergence guarantee

Changing the curve changes the nonconservative shock prescription in general.
Even with a declared curve, numerical viscosity can affect the resolved shock
profile and the limiting jump. Smooth consistency and small nonlinear residuals
are therefore insufficient validation on their own. This limitation is
examined by [Castro, LeFloch, Muñoz-Ruiz and Parés (2008)](https://arxiv.org/abs/0808.2065).

Shock profiles, location, conservation identities and refinement must be
checked independently. It would be incorrect to justify “the same solution”
solely by saying that the original differential equations look the same.

## 12. Are straight paths and two Gauss points the only usual approach?

No. Straight paths with a fixed Gauss rule are a common practical starting
point: they are simple, inexpensive and easy to differentiate. The method has
several separate choices, with different consequences.

| Choice | Alternatives | What changes? |
|---|---|---|
| State-space curve | Straight primitives, straight stored variables, piecewise curves, characteristic/equilibrium paths, paths motivated by viscous or relaxation profiles | Generally changes the jump prescription; requires physical justification and shock validation |
| Quadrature on a fixed curve | Midpoint, trapezoidal, two-/three-point Gauss, higher fixed rules, adaptive quadrature, analytic integration | Changes integration accuracy and cost; does not by itself select a different physical curve |
| Interface distribution/upwinding | Symmetric half-contributions, characteristic fluctuation splitting, Roe/Osher/Rusanov-type path-conservative solvers | Changes numerical dissipation and propagation; must preserve the required total interface balance |
| Cell-interior integration | Reconstruction paths, direct physical-space volume quadrature, higher-order finite-volume or discontinuous-Galerkin constructions | Controls how resolved within-cell variation is represented |
| Geometry/source balance | Simple midpoint treatment, equilibrium reconstruction, augmented stationary geometry variables | Controls preservation of selected steady balances |

For example, midpoint integration on the same straight path gives

$$
\mathcal P_B\approx B\!\left(\frac{W_L+W_R}{2}\right)\Delta W,
$$

whereas the trapezoidal rule gives

$$
\mathcal P_B\approx\frac12[B(W_L)+B(W_R)]\Delta W.
$$

Both are exact for constant $B$; neither is generally identical to the
two-point Gauss result for nonlinear $B$. Three-point Gauss adds another
state/EOS evaluation and integrates polynomials through degree five.
Adaptive integration can resolve difficult variation, but introduces variable
work and more complicated JIT/AD behavior. No quadrature rule repairs a badly
chosen or inadmissible curve.

An analytically integrated or appropriately defined Roe matrix may also
represent the action as $\overline B_\Phi\Delta W$. The essential requirement
is that this matrix-vector action represents the chosen path integral; an
arbitrary average does not automatically satisfy that condition. Computing
it need not involve a physical-flux inverse.

**Changing coordinates and changing the curve are different operations.**
For a smooth, invertible coordinate change $Z=\Psi(W)$, the same geometric
curve can be described in $Z$ coordinates with the transformed coupling
$B_Z(Z)=B(W(Z))\,\frac{\partial W}{\partial Z}$. Its exact line integral is
unchanged. In particular, one may choose $Z=U(W)$ where that storage map is
invertible. But choosing a new straight line in $Z$ usually maps
back to a curved path in $W$, so “straight in primitives” and “straight in
conserved variables” are generally different prescriptions.

Likewise, reparameterizing the same oriented curve leaves the exact integral
unchanged, although a finite quadrature rule can approximate it differently.
This is why the physical/path choice should be settled separately from the
quadrature accuracy study.

## 13. What conservation and the existing evidence actually establish

For any constant projection $c^T$ with $c^TB(W)=0$ along the path,

$$
c^T\mathcal P_B=0.
$$

The two-fluid $X$ has zero phase-mass rows and opposite momentum rows.
Direct quadrature preserves these identities: each phase mass still uses a
common conservative numerical flux, and the NC momentum terms cancel in
the mixture sum. External geometry and closure sources retain their own
physical contributions. The phase-energy NC rows do not generally sum to
zero at finite slip; they must be checked as part of the complete retained
energy balance, not discarded to manufacture a different conservation law.

The [v2 results](../../development/two_fluid_test_v2/RESULTS.md) and
[standalone report](../../development/two_fluid_test_v2/REPORT_two_fluid_inverse_free.html)
compare independently seeded single-grid implicit solves of water–nitrogen
subsonic, shock and adapted flows, using 50–400 cells and both reconstructions.
Their convergence counts were:

| Spatial package | Converged configurations |
|---|---:|
| Native compatibility operator | 15/24 |
| Inverse-free flux/NC, retaining the old sonic law | 24/24 |
| New sonic law with native flux/NC construction | 18/24 |
| Inverse-free flux/NC and new sonic law | 24/24 |

This supports the usefulness of removing the inverse-based **flux/NC package**
for these tested initializations. It does not isolate the face-flux change
from the NC-path change, nor establish universal convergence from cold states.
Boundary equations and the native boundary dissipation setting were held fixed.

An example makes “essentially the same” more precise. Comparing the native and
combined **400-cell linear shock** saved profiles, both of which pass the
declared reference checks, gives a pairwise pressure RMS difference of about
$0.141\%$ of the campaign pressure scale and a gas-temperature RMS difference
of about $0.174\%$ of its temperature scale. The mean mass flows differ by
about $0.0727\%$ relative to the native value. These are comparisons of saved
numerical profiles, not new simulations. Locally, the maximum gas-temperature
difference is about $4.82$ K, so the states are plainly not identical.
The RMS normalizations are the native record's declared reference scales;
they are not pointwise relative errors.

Finally, convergence is not the same as accuracy. The combined method passes
all profile limits in 11/12 linear configurations but only 3/12 constant ones.
Even its 400-cell constant shock case has a gas-temperature error of about
$1.08\%$ against a $1\%$ limit. These misses are retained in the study.
The evidence supports robust recovery and close resolved profiles in the
validated cases, with separate finite-grid accuracy and shock-path limitations.

## Implementation and background references

- [Native coupling, storage and flux matrices](../../src/nozzle_hyperbolic/models/two_fluid/matrices.py): `X_matrix`, `J_matrix`, `K_matrix`, `G_h_matrix` and `phase_props`.
- [Native inverse action](../../src/nozzle_hyperbolic/models/two_fluid/flux.py): `compatible_flux_solve` and `nonconservative_correction`.
- [Native cell assembly](../../src/nozzle_hyperbolic/models/two_fluid/model.py): `integrated_nonconservative`.
- [Prototype quadrature and assembly](../../development/two_fluid_test_v2/scripts/numerics.py): `path_nc` and `AblationModel.integrated_nonconservative`.
- Earlier local formulation, `development/two_fluid_test_v1/formulation.md`: sonic regularization, small-slip grouping and additional validation qualifications. This local study record is not included in this documentation commit; the versioned [v2 study guide](../../development/two_fluid_test_v2/README.md) documents the retained formulation and qualifications.
- Dal Maso, LeFloch and Murat (1995), [*Definition and weak stability of nonconservative products*](https://iris.sissa.it/handle/20.500.11767/16373), J. Math. Pures Appl. 74, 483–548. Foundational reference for path-defined products.
- Parés (2006), [*Numerical methods for nonconservative hyperbolic systems: a theoretical framework*](https://epubs.siam.org/doi/10.1137/050628052), SIAM J. Numer. Anal. 44, 300–321.
- Muñoz-Ruiz and Parés (2007), [*Godunov method for nonconservative hyperbolic systems*](https://www.numdam.org/articles/10.1051/m2an:2007011/), ESAIM: M2AN 41, 169–185.
- Castro, LeFloch, Muñoz-Ruiz and Parés (2008), [*Why many theories of shock waves are necessary: convergence error in formally path-consistent schemes*](https://arxiv.org/abs/0808.2065), J. Comput. Phys. 227, 8107–8129.
