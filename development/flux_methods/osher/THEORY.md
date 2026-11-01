# Osher-type conservative path viscosity

The sole production selector is `flux="osher"`, with flat Scheme parameters
`quadrature_points=5` and `integration="gauss"` by default. Integration may be
`"gauss"` or `"sonic_partitioned"`. The study configuration IDs `gauss_3`,
`gauss_5`, `gauss_9` and `sonic_partitioned_5` are metadata identifiers, not
additional flux methods. The point count is **quadrature count, not spatial
order**; constant or Van Leer MUSCL reconstruction supplies the face states.
For partitioned integration it is the count per nonempty panel.

All measured statements belong to the recorded source/case/client epoch. The
original source guard stopped on an external Roe-source edit; any continuation
must preserve the frozen epoch and document isolated execution and origins.
The current working checkout is not implicitly certified by those results.

## Flux and model-declared path

Let $W_L,W_R$ be reconstructed states, $U(W)$ conservative storage, $F(W)$ the
physical face flux and $(P_k(W),\lambda_k(W))$ coincident-state characteristic
data in the model's common face frame. The regularized absolute speeds are
$\phi_k(W)=\phi_{\delta_k(W)}(\lambda_k(W))$, with

$$
\phi_\delta(\lambda)=
\begin{cases}
(\lambda^2+\delta^2)/(2\delta),&\delta>0,\ |\lambda|<\delta,\\
|\lambda|,&\text{otherwise}.
\end{cases}
$$

Define

$$
B_\phi(W)=\sum_k\phi_k(W)P_k(W),\qquad
\boxed{\widehat F=\frac{F_L+F_R}{2}
-\frac12\int_0^1 B_\phi(W(s))\frac{dU(W(s))}{ds}\,ds.}
$$

The default is $W(s)=W_L+s(W_R-W_L)$ in the model's reconstruction variables,
**not an undocumented straight segment in conservative variables**. An
optional model path callback may supply another path and tangent. Conservative
storage and its directional derivative are model-owned. Annular frame/radius
conversion and face area remain in the established model/operator assembly.

$$
U'(s)=D_WU(W(s))\,W'(s).
$$

This is the established Osher-type strategy of integrating characteristic
viscosity along a state path; see [Dumbser and Toro (2011)](https://www.global-sci.com/cicp/article/view/5859)
and [Castro, Gallardo and Marquina (2014)](https://ww3.math.ucla.edu/camreport/cam14-96.pdf).
The implemented path, regularized speeds and integration policy are explicit
numerical choices; the general name does not transfer every entropy or
positivity theorem to this implementation and its coupled implicit update.
The 2014 reference also develops polynomial/rational approximations to the
absolute Jacobian; those approximations are not the spectral implementation
tested here. The publisher record and the December 5, 2014 preprint were
checked directly for this documentation.

## Fixed Gauss rules

For nodes/weights $(\xi_q,w_q)$ on $[0,1]$,

$$
D_N=\sum_{q=1}^{N}w_q B_\phi(W(\xi_q))U'(\xi_q),\qquad
\widehat F_N=\tfrac12(F_L+F_R-D_N),\qquad N=3,5,9.
$$

The three `gauss` configurations therefore differ in flux-integration accuracy
and cost. They do
not imply third-, fifth- or ninth-order finite-volume solutions. Reconstruction,
limiters, source discretization and regularity determine spatial convergence.

The signed chain rule, for correct physical Jacobian callbacks, is

$$
\int_0^1 A(U(s))U'(s)\,ds=F_R-F_L.
$$

Agreement with this signed identity does not bound error in the absolute
viscosity integral. A transonic Burgers pair $(-1,1)$ is a useful counterexample:
unsplit fixed Gauss rules can violate the scalar E-flux bound despite a tiny
signed defect. They remain separate, selectable numerical experiments.

## Sonic-partitioned production policy

`integration="sonic_partitioned"` samples eight path intervals and captures
bracketed roots of

$$
\lambda_k(W(s))=0,\qquad
\lambda_k(W(s))=\pm\delta_k(W(s)).
$$

The captured roots partition the path into panels $[a_m,a_{m+1}]$. The configured $Q=\texttt{quadrature\_points}$ Gauss rule is applied on every
nonempty panel (the study uses $Q=5$):

$$
D_{\rm sonic}=\sum_m(a_{m+1}-a_m)
\sum_{q=1}^{Q}w_q B_\phi(W(s_{mq}))U'(s_{mq}),\qquad
s_{mq}=a_m+(a_{m+1}-a_m)\xi_q.
$$

The compact execution algorithm takes up to eight safeguarded Illinois/secant
steps, with relative root residual $2^{-24}$; an unresolved captured bracket
receives 24 bisections. It packs at most $4K$ total candidate brackets per face,
where $K$ is the number of characteristic families. If capacity is exceeded,
the complete full-capacity evaluation is used without dropping captured roots.
That fallback does not discover crossings missed by the original sampling.

Root-location differentiation uses the supported implicit derivative at
captured simple roots. Exact endpoints are fixed cuts. Stable family labels,
admissible hyperbolic paths and differentiable supported conservative storage
are assumptions. Several unseen crossings in one interval, tangential/multiple
roots and invalid thermodynamic paths are not certified by finite sampling.
Sampled invalid quadrature nodes are rejected by production validity checks;
they are not repaired by clipping or a hidden HLL substitution.

## Per-face cost and explicit timestep guard

Each integration node needs a model state, characteristic data and the
conservative-storage directional derivative. Sonic partitioning adds crossing
searches and panel construction. Separate sampled-path admissibility checks
repeat some path/physical work; the explicit-speed guard can repeat it again.
Fused callbacks and compiled reuse can reduce cost, but neither the point
count nor a nonlinear iteration reduction supplies a universal timing ratio.
This per-flux work is separate from accepted/rejected pseudo-time steps and
Newton corrections over a complete solve.

For explicit Euler/SSPRK3, production bounds the step with the largest sampled
regularized characteristic speed at the endpoints and actual quadrature
nodes. The sonic method also includes its crossing-search grid. Physical
endpoint bounds, when available, and existing cell/model restrictions are
retained. For the resulting $s_f$,

$$
\Delta t\le\mathrm{CFL}\min_i
\frac{\Delta x_i}{\max(s_{i-1/2},s_{i+1/2})}.
$$

This is a sampled viscosity guard, not a certified maximum between samples,
nor a nonlinear CFL/positivity theorem. It cannot certify unseen path-speed
extrema. The requested steady campaign uses backward Euler and retains its
existing implicit CFL controller.

## Study interpretation

The requested population is four nozzle/annular shock cases, grids
50/100/200/400, constant and Van Leer MUSCL reconstruction and four parameter configurations
of the single Osher flux:
128 recorded independent cold solves. The metadata migration itself performs
no new flow solves: derived records preserve numeric payloads and bind their
archived origin hashes, while rebuilt current Operators check the explicit
parameter configurations. Geometry, EOS backend, source/boundary equations,
Newton/linear solver, stopping tolerances and adaptive CFL policy are controlled
by the shared verification runner. No flow kernel is copied into this study.

The report compares actual residual/CFL trajectories, accepted/rejected work,
all Newton corrections and analytical errors against conservative cell
projections. Compilation-inclusive elapsed values are diagnostics, not matched
performance timings. Path integration and derivatives can cost more per face
even when fewer nonlinear steps are needed; iteration reductions alone are
not speedups.

No universal entropy stability, positivity or absence of every local extremum
is claimed. The full backward-Euler/Newton, source and boundary assembly needs
its own analysis. A pressure improvement may accompany a different total-
enthalpy defect or shock thickness. Failed and incomplete states remain visible,
and the report does not promote one quadrature merely because its point count is larger.

The [Stadtke study](../stadtke/THEORY.md) instead uses the existing mean-state
projectors plus two explicitly custom algebraic corrections. The
[Roe study](../roe/roe_theory.md) uses a model-owned finite-jump matrix. These
are distinct constructions, even when some tested profiles look similar.
