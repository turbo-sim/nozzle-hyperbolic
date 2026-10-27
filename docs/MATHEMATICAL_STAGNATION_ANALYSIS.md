# Mathematical analysis of the shocked-nozzle stagnation

This analysis describes the historical flux, retained as
`_jax_F_fvs_stadtke_legacy`. The subsequent implementation and controlled
LM/HYBR/reconstruction tests are documented in
[SOLVER_REGULARITY_IMPROVEMENTS.md](SOLVER_REGULARITY_IMPROVEMENTS.md).

Newton can solve discretized flows containing shocks. These failures concern
the particular residual, initialization, and globalization used here. The
strongest sampled obstruction is the rapid change in the MUSCL reconstruction
along a large Newton direction. There are also independently verified
regularity and consistency defects in the implemented FVS flux.

This investigation evaluated saved states and diagnostic models before the
subsequent production corrections. Numerical evidence
below refers to the original 201-point cases and first saved repeat in
`development_new/newton_solve/results_comparison/summary.json`.

## 1. An accurate Newton solve can give an unusable finite step

Let $R(z)$ be the scaled steady residual and $J=R'(z)$. Exact Newton solves

$$
Jd=-R.
$$

For the squared-residual merit $Phi(z)=\frac12\|R(z)\|_2^2$, at a differentiable
point,

$$
\nabla\Phi(z)^T d=R^TJd=-\|R\|_2^2<0.
$$

Thus a sufficiently small step must decrease the merit at a smooth non-root
state with nonsingular $J$. A non-root stationary point of this smooth merit
would require singular $J$: $J^TR=0$ and invertible $J$ imply $R=0$.

This is only a local statement. The predicted residual is

$$
R(z+\alpha d)=(1-\alpha)R(z)+O(\alpha^2\|d\|^2).
$$

Large directions and large curvature can overwhelm the predicted decrease.
If the merit gradient has a local Lipschitz constant $L_\Phi$, a sufficient
Armijo bound is

$$
\alpha\leq
\frac{2(1-c)\|R\|_2^2}{L_\Phi\|d\|_2^2},
$$

where $c$ is the Armijo parameter. This bound can become extremely small.
It is not a uniform guarantee across nonsmooth or discontinuous branches.

At the saved `shock70` Newton state, residual RMS is $8.906\times10^{-4}$,
while the Newton direction has scaled RMS 4.374. The direction is large on
the natural pressure/velocity/enthalpy scales. The global Jacobian's condition
number is about $1.4\times10^5$. Its linear solve is accurate, but that does not
make the finite nonlinear update accurate. A full step would also make
pressure negative. Backtracking can protect positivity while yielding almost
no useful progress. The previous 200-trial Newton runs accepted only 15 to 17
updates; rejected trials consumed most of their budgets.

Agreement between compressed and dense AD checks the selected formula's
derivative. Both AD calculations follow the same branch and therefore cannot
certify continuity across a branch boundary.

## 2. The Van Leer limiter has unbounded near-flat curvature

The primitive MUSCL reconstruction uses adjacent differences $a,b$ and

$$
\sigma(a,b)=
\begin{cases}
2ab/(a+b), & ab>0,\\
0, & ab\leq0.
\end{cases}
$$

On the active branch,

$$
\nabla^2\sigma
=-\frac{4}{(a+b)^3}
\begin{bmatrix}
b^2&-ab\\
-ab&a^2
\end{bmatrix},
$$

so along differences changing in direction $(\dot a,\dot b)$,

$$
\frac{d^2\sigma}{d\alpha^2}
=-\frac{4(b\dot a-a\dot b)^2}{(a+b)^3}.
$$

When $a,b$ shrink together toward a flat region, curvature grows like the
inverse of their size, except for proportional changes. This occurs with
same-sign differences; it does not require cancellation between opposite
slopes. The limiter is continuous and locally Lipschitz but has derivative
kinks on $ab=0$. At $(0,0)$ it has no Frechet derivative: axis directional
derivatives are zero, while the $(1,1)$ directional derivative is one. The
forward-mode JAX calculation selects the inactive branch's zero derivative
there, which cannot predict every direction through that point.

The exact expression with its $10^{-300}$ denominator guard has an additional
AD robustness issue in the installed JAX 0.6.2: at $(a,b)=(0,0)$, `jacfwd`
returns `[0, 0]`, while `jacrev` returns `[NaN, NaN]`. The reverse-mode result
is recorded in the diagnostic JSON. This is a separate compatibility defect;
the sampled failed runs had finite forward/colored Jacobians, so it is not
their established cause. JAX documents the general hazard of nonfinite
adjoints in a `where` branch in its
[autodiff FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where).

Near such states, a Newton perturbation can change reconstruction behavior
after a very small fraction of its direction. Local AD agreement is entirely
compatible with this difficulty.

### Controlled evidence at the worst saved Newton state

The diagnostic independently reconstructs the unchanged spatial residual,
matching production to about $2.3\times10^{-16}$ relative error for this state.
It then constructs a **hypothetical diagnostic residual** in which only the
limiter slopes are held to their tangent at the starting field. Boundaries,
EOS, source, and the face-flux formula remain nonlinear. The diagnostic and
original agree in value and first directional derivative at that field
(relative errors approximately $2.3\times10^{-16}$ and $1.9\times10^{-10}$).

Along the identical Newton direction for `shock70`:

| Fraction $\alpha$ | Linear prediction for norm ratio | Original residual norm ratio | Tangent-limiter diagnostic norm ratio |
|---:|---:|---:|---:|
| $10^{-7}$ | 0.9999999 | 0.9999996831 | 0.9999999000 |
| $10^{-6}$ | 0.999999 | 1.0003169183 | 0.9999990043 |
| $10^{-5}$ | 0.99999 | 1.0333399372 | 0.9999904299 |
| $10^{-4}$ | 0.9999 | 1.9090255918 | 0.9999468528 |

The $10^{-6}$ test is the cleanest attribution: the predicted decrease is
overwhelmed by the original reconstruction's higher-order change. Four
cell/variable limiter masks differ at this trial. The located entropy and
average-sonic floor crossings occur later, around $3.87\times10^{-5}$ and
$4.35\times10^{-5}$ along the original path. At $10^{-4}$, changing the
reconstruction also changes downstream flux branches, so the larger contrast
includes those interactions.

This isolates an obstruction for **this state and direction**. It does not
establish that every failed iterate or `shock35` has the same dominant cause.
For example, linearizing the limiter did not improve all `shock35` trials.
The diagnostic is not a proposed replacement residual or a validated solver;
away from the starting field it is a different function.

![Limiter probe at the saved shock70 Newton state](../development_new/newton_solve/results_stagnation_mechanism/shock70_newton_limiter_probe.png)

## 3. The entropy-fix activation creates a finite flux jump

In [flux_jax.py](../nozzle_hyperbolic/models/euler_nozzle/flux_jax.py), the acoustic
projector is multiplied by $c=|\bar\lambda|/\lambda_{\rm safe}$. The strict
expansion condition is $\lambda_L<0<\lambda_R$. When active,

$$
\lambda_{\rm safe}
=\frac{\bar\lambda^2+(\lambda_R-\lambda_L)^2}
{2(\lambda_R-\lambda_L)}.
$$

Otherwise the denominator is $\max(|\bar\lambda|,\delta_{\rm global})$.
These two formulas do not match at the activation boundary.

Take equal pressure and enthalpy, hence equal sound speed $a$, and choose
$u_L\to a$ and $u_R=a+b$ with $b>0$ well above the tiny global floor. For the
slow acoustic wave,

$$
\lambda_L\to0^-,\qquad \lambda_R=b,\qquad \bar\lambda\to b/2.
$$

On the active side,

$$
\lambda_{\rm safe}\to5b/8,\qquad c\to4/5.
$$

At $\lambda_L=0$, the strict test becomes false and $c=1$. The coefficient
jumps by 20%. The corresponding projected flux is generally nonzero, so the
**flux itself has a finite jump**. This is stronger than a continuous limiter
kink.

Direct evaluation with air at 100 kPa and 300 K, $u_R=1.1a$, and
$u_L=a(1\pm\epsilon)$ confirms the flux difference tends to a nonzero limit
as $\epsilon\to0$. Scaled by $\rho a[1,a,h]$, that limiting jump is

$$
\Delta\widehat F_{\rm scaled}
\approx(0.00361733,\;0.000180867,\;0.00289262),
\qquad \|\Delta\widehat F_{\rm scaled}\|_2\approx0.00463520.
$$

This example has average Mach 1.05, so the jump is distinct from a singular
sonic matrix inverse. An independent nonsingular projector representation
reproduces it to roundoff. Full-field probing also finds finite residual jumps
at located entropy boundaries on several saved-state Newton paths. However,
the small-step `shock70` obstruction above occurs earlier; this flux defect
does not alone explain all stagnation.

## 4. The projector formula introduces an avoidable singular inverse

Let $U=(p,u,h)$, $Q$ be conservative variables, and

$$
C=\frac{\partial Q}{\partial U},\qquad
K=\frac{\partial F}{\partial U},\qquad B=C^{-1}K.
$$

The code evaluates characteristic projectors through $KP_iK^{-1}$. But

$$
\det C=\frac{\rho^2}{a^2},\qquad
\det K=\frac{\rho^2}{a^2}\,u(u^2-a^2).
$$

Thus its intermediate inverse becomes singular at rest and at sonic flow.
Away from those points, because $BP_i=\lambda_iP_i$,

$$
KP_iK^{-1}=CBP_iB^{-1}C^{-1}=CP_iC^{-1}.
$$

The expression on the right remains defined at sonic flow. The physical
Euler eigenvalues there are $2a,0,a$, still distinct; a zero eigenvalue does
not make the eigenvectors coincide. The inverse singularity is therefore
removable from this projector representation.

The numerical probe produces NaNs at uniform rest using production, while
the independent $C$ representation returns the finite physical flux. A nominal
Mach-one sample happened to remain finite through roundoff; the determinant
identity establishes the exact singularity. This defect is a robustness risk,
but we have not established it as the dominant floating-point error at the
saved failed states.

## 5. The tiny global floor changes consistency as well as smoothness

The source calls its global floor $C^1$, but
$\max(|\lambda|,\delta)$ has derivative kinks. With
$\Pi_i=CP_iC^{-1}$ and $c_i=\min(1,|\lambda_i|/\delta)$, equal states give

$$
\widehat F(U,U)-F(U)=\sum_i(c_i-1)\Pi_iF(U).
$$

This is generally nonzero for near-sonic nonzero eigenvalues: the weighted
projectors no longer sum to identity. The equal-state probe confirms the
defect. For homogeneous perfect-gas Euler flux, $\Pi_iF=\lambda_i\Pi_iQ$, so
this particular defect tends to zero at an exactly zero eigenvalue; it is
different from the inverse singularity.

In a frozen linear system $F=AQ$, the characteristic dissipation implied by
the implementation is $\lambda_i^2/\lambda_{\rm safe}$, while its central
transport is modified too. Where $\lambda_{\rm safe}\geq|\lambda_i|$, this
reduces dissipation. It is not equivalent to keeping the central flux and
increasing the absolute-eigenvalue dissipation in a conventional Roe entropy
fix. This algebra is a statement about that linear setting, not a proof of
entropy violation for every nonlinear Euler state. See the primary
[Steger-Warming flux-splitting report](https://ntrs.nasa.gov/citations/19790020779)
for the homogeneous-flux basis, and
[SU2's entropy-fix documentation](https://su2code.github.io/docs_v7/Convective-Schemes/)
for the conventional minimum-dissipation interpretation.

## 6. Choking makes the transition difficult, but does not prohibit Newton

For isentropic perfect-gas nozzle flow, the area-Mach function has

$$
\frac{A}{A^*}-1\approx\frac{2}{\gamma+1}(M-1)^2.
$$

The two Mach branches meet at $M=1$ and the inverse sensitivity grows there.
The differential relation

$$
\frac{dA}{A}=(M^2-1)\frac{du}{u}
$$

requires throat compatibility for a regular transonic solution. A subsonic
initial field must establish this transition and the downstream shock. This
helps explain large global corrections and sensitivity, but does not prove a
singular assembled discrete Jacobian. The throat compatibility, numerical
viscosity, and boundary coupling matter. Nor have these tests established a
weak shock-location mode as the dominant cause. The continuum relations are
given by [NASA's nozzle equations](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/nozzle-design/)
and [choked mass-flow derivation](https://www.grc.nasa.gov/WWW/k-12/airplane/mflchk.html).

## 7. SLSQP shares the direction; implicit marching changes it

For a square, full-rank equality-feasibility problem with zero objective,
SLSQP's linearized constraints determine

$$
R+Jd=0.
$$

There are no remaining degrees of freedom for its Hessian approximation to
choose a different feasible direction. SLSQP changes merit/globalization,
but inherits the same underlying local constraint model. Similar failures
of the two solvers are therefore mathematically expected in this setup.

Scalar damping merely replaces $d$ by $\alpha d$. In contrast, the first
correction of an implicit pseudo-time step satisfies

$$
\left(\frac{\mathcal M}{\Delta t}+J_R\right)d=-R,
$$

where $\mathcal M$ contains cell volumes and the derivative of conservative
storage with respect to primitive variables. This changes the direction and
relative characteristic responses. For small $\Delta t$ it follows the
transient dynamics; for large $\Delta t$ it approaches steady Newton. The
transient path can temporarily increase the steady residual norm, unlike a
monotone Newton line search. This is the mathematical reason implicit progress
can succeed when repeated shortening of one Newton direction does not.
Our reference performs multiple inner corrections per backward-Euler step.
The broader rationale is established by
[Kelley and Keyes' pseudo-transient analysis](https://doi.org/10.1137/S0036142996304796)
and described in [PETSc's pseudo-transient documentation](https://petsc.org/release/manualpages/TS/TSPSEUDO/).

The existing warm-start results fit this picture: switching very late succeeds
once the choked/shocked field is established. It does not show that a smaller
initial scalar step solves the earlier regularity and reconstruction problems.

## Reproduction and implications

Run from the repository root:

```powershell
poetry run python -m development_new.newton_solve.verify_flux_regularity
poetry run python -m development_new.newton_solve.analyze_stagnation_mechanism
# Regenerate only the saved diagnostic plots.
poetry run python -m development_new.newton_solve.analyze_stagnation_mechanism --plot-only
```

The first command is independent of previous nozzle runs. The second requires
the original comparison JSON and matching baseline equations/settings. Its
JSON records the full residual replication check, directional derivative
checks, limiter/entropy/floor events, counterfactual probes, and finite jump
tests. Generated results are ignored by Git.

The appropriate next work is to restore flux consistency and continuity,
remove the unnecessary singular inverse, and then assess globalization and
reconstruction treatment with controlled comparisons. Pseudo-transient or
parameter continuation provides a principled transition toward steady Newton.
Changes to limiter treatment require final evaluation against the intended
high-order residual; the tangent-limiter diagnostic alone is not a solver fix.
That diagnostic phase did not implement production changes. The subsequent
corrections and their measured limitations are recorded in the implementation
report linked at the top of this document.
