# Steady Newton and implicit Euler for shocked nozzle flow

This study asks whether the current finite-volume equations can reach a
prescribed internal-shock solution directly, and how that depends on the
initial field, numerical flux, reconstruction and outlet law. Direct Newton
and implicit Euler solve the same spatial balances and the same algebraic
boundary equations for each configuration. They reach those equations by
different nonlinear trajectories.

The physical cases are the packaged perfect-gas Air nozzle cases
[`shock_35`](../../src/nozzle_hyperbolic/models/euler_nozzle/cases/shock_35.yaml)
and [`shock_70`](../../src/nozzle_hyperbolic/models/euler_nozzle/cases/shock_70.yaml).
Both have reservoir pressure $p_0=200\,000\ \mathrm{Pa}$, temperature
$T_0=300\ \mathrm{K}$ and length $L=0.2\ \mathrm{m}$. The throat is at $L/2$.
The analytical normal shock lies at 35% or 70% of the divergent length,
respectively $x_s=0.135$ or $0.170\ \mathrm{m}$. The independent continuous
reference selects the backpressure; its shocked profile is assessment data.
The target mesh has 200 physical cells, plus two boundary traces. Fluid states
are evaluated through the configured JAXProp perfect-gas backend.

The executable settings are in [settings.json](settings.json). The
[backpressure study theory](../BC_backpressure/THEORY.md) describes the outlet
laws for general equations of state, including their real-gas limitations.
This document specializes the wave formulas to the perfect gas used here.

## The coupled spatial system

The model's primitive state is $W=(p,u,h)$. With
$h_t=h+u^2/2$, its conservative density and physical flux are

$$
U(W)=
\begin{pmatrix}
\rho\\ \rho u\\ \rho h_t-p
\end{pmatrix},
\qquad
F(W)=
\begin{pmatrix}
\rho u\\ \rho u^2+p\\ \rho u h_t
\end{pmatrix}.
$$

For physical cell $i$, let $V_i=\int_i A(x)\,dx$. The signed, integrated
steady residual is

$$
R_i(W)=
A_{i+1/2}\widehat F_{i+1/2}(W)
-A_{i-1/2}\widehat F_{i-1/2}(W)-S_i(W),
\qquad
S_i(W)=
\begin{pmatrix}0\\p_i\int_i A'(x)\,dx\\0\end{pmatrix}.
$$

The case has no wall friction or heat transfer. The source uses the model's
analytical geometry integrals. Rusanov, HLL and corrected Stadtke define
$\widehat F$ from the selected constant or Van Leer MUSCL face states. Both
Stadtke corrections are enabled: conservative-jump characteristic viscosity
and the endpoint characteristic-speed envelope. The characteristic width
factor is 0.1, and the additional fourth-difference filter is disabled.

There are $N+2$ independent primitive states: $N$ physical cells and one trace
at each physical endpoint. Their six boundary equations are ordinary rows of
the square coupled system. Traces have zero storage volume; their equations
are neither multiplied by the timestep nor advanced as physical cells.

The inlet imposes reservoir total enthalpy and entropy, together with an
outgoing acoustic compatibility relation. If $\widetilde W_l$ denotes the
extrapolation of the first two cells to the inlet and $Z_1=\rho_1a_1$, these
equations are

$$
h_l+\frac{u_l^2}{2}=h_0,\qquad
s(p_l,h_l)=s_0,\qquad
p_l-\widetilde p_l-Z_1(u_l-\widetilde u_l)=0.
$$

The outlet contributes the three equations described below. The complete
steady residual implemented by `Operator.steady_residual` is

$$
\mathcal R(W)=
\operatorname{pack}\left[
B_l(W),\;
\left\{\frac{t_{\rm ref}R_i(W)}{V_iq_{\rm ref}}\right\}_{i=1}^N,
\;B_r(W)\right]=0.
$$

Division by a vector means componentwise division. The boundary rows $B_l$
and $B_r$ already contain their model-defined normalization. For this nozzle,

$$
D_W=\operatorname{diag}(p_0,a_0,a_0^2),\qquad
q_{\rm ref}=\rho_0(1,a_0,a_0^2),\qquad
t_{\rm ref}=\frac{L}{a_0},\qquad
f_{\rm ref}=A_{\rm inlet}a_0q_{\rm ref}.
$$

These scales are fixed from reservoir data and geometry. An enthalpy-reference
offset is retained in all state and flux evaluations; scaling enthalpy by
$a_0^2$ does not replace it by $c_pT$ with an assumed zero reference.

## Two paths to the same steady equations

### Implicit Euler

Each accepted outer step solves

$$
V_i\big[U(W_i^{n+1})-U(W_i^n)\big]
+\Delta t\,R_i(W^{n+1})=0,
\qquad B_l(W^{n+1})=B_r(W^{n+1})=0.
$$

The implementation divides each cell equation by
$\max(1,\Delta t/t_{\rm ref})V_iq_{\rm ref}$. This uniform factor avoids
amplifying roundoff at large timesteps and leaves the timestep equation's
root unchanged. Boundary rows keep their original normalization.

The storage term regularizes each intermediate nonlinear solve. The CFL
controller starts at 0.2, grows by 1.2 after accepted steps up to $10^6$, and
halves after a failed attempt. Each inner Newton solve permits at most ten
corrections, with backtracking and up to twenty reductions. It requires both
the configured scaled update target $10^{-10}$ and timestep-equation RMS
target $10^{-12}$. The outer steady audit remains necessary: satisfying a
timestep equation does not by itself imply $R_i=0$.

The implicit implementation uses colored automatic differentiation and a
banded linear solve. Its accepted outer updates, inner corrections and
rejected attempts are different work counters.

### Direct Newton

The direct adapter calls the installed custom PySolverView backend with
`library="pysolver"` and `method="newton_raphson"`. It solves the unshifted
system $\mathcal R(W)=0$ immediately, including both endpoint traces.
There is no storage term, implicit-Euler prefix or pseudo-transient
continuation within this strategy.

For a supplied full initial field $W^{(0)}$, the dimensionless coordinates are
centered at that field:

$$
W(z)=W^{(0)}+D_Wz,\qquad z^{(0)}=0.
$$

The scale matrix is repeated for every cell and trace. Centering guarantees
that the first physical residual evaluation contains exactly the supplied
floating-point initial field; dividing and multiplying an arbitrary warm
start by scales need not preserve every bit. Both the requested and actual
initial-state hashes are retained.

The Newton direction is computed from

$$
J(z_k)\,\delta z_k=-\mathcal R(W(z_k)),\qquad
J=\frac{\partial\mathcal R}{\partial W}D_W,
\qquad
z_{k+1}=z_k+\alpha_k\delta z_k.
$$

The adapter supplies an exact dense forward-AD Jacobian of the current
residual graph. It applies no second row scaling. PySolverView's numerical
column equilibration is enabled; full-rank Newton directions are unchanged
by an invertible column scaling in exact arithmetic. The dense backend uses
pivoted QR and can use SVD to assess an ill-conditioned model. A numerically
rank-deficient square Jacobian terminates strict Newton instead of silently
selecting a truncated minimum-norm least-squares direction.

The initial fraction cap is 0.25. After each accepted move it grows by 1.5,
up to one:

$$
\alpha_{\rm cap}^{(k)}=\min(1,0.25\,1.5^k).
$$

Armijo backtracking may reduce the actual fraction below this cap. The merit
function is $\phi=\|\mathcal R\|_2^2/2$, with sufficient-decrease coefficient
$10^{-4}$ and contraction factor $1/2$. Rejected trials do not advance the
fraction schedule. After an accepted backtracked move, the next cap grows
from its preceding cap, rather than resetting to the small fraction that
was actually accepted. Once the cap reaches one, each fresh direction can
therefore incur several reductions again. The separate absolute-displacement
envelope is disabled with `backtracking.step_limits=null`.

The main comparison permits at most **400 accepted Newton updates**. The
initial field is record zero and does not count as an update. At update 400,
the common original-equation checks run first: a root reached on that update
is accepted; otherwise the solve stops as failed with reason
`newton_step_limit_400`. There is no continuation beyond this limit.

Secondary backend guards remain in place. The limit of
10,000 counts trial attempts, including rejected trials; it does not mean
10,000 accepted Newton updates. The residual-evaluation limit is explicitly
20,000, rather than the backend's separate default evaluation budget.
Exhausting either limit establishes that a run
did not finish within the work budget, rather than proving that no convergent
Newton trajectory exists.
Supplemental fraction variants, when explicitly selected by the driver, are
stored separately from the baseline 0.25 campaign.

Every trial must satisfy the model's admissibility checks for physical cells,
boundary traces and reconstructed faces. Invalid trials are rejected through
the backend's domain-error mechanism. Velocities may be zero or signed during
iteration. No velocity clipping, pressure floor, state projection or alternate
linear direction is introduced by the adapter.

## Backpressure boundary conditions

For forward subsonic Euler outflow, the characteristic speeds are $u-a$, $u$
and $u+a$. One characteristic enters the domain, so a prescribed backpressure
provides one exterior datum. How that datum connects to interior information
distinguishes the two model-owned outlet laws.

### Acoustic compatibility

For the last two cells, define

$$
\eta=\frac{x_b-x_N}{x_N-x_{N-1}},\qquad
\widetilde W=W_N+\eta(W_N-W_{N-1}),\qquad
\widetilde s=s_N+\eta(s_N-s_{N-1}),\qquad Z_N=\rho_Na_N.
$$

The target study uses the static `subsonic` acoustic law:

$$
p_b=p_e,\qquad
p_b-\widetilde p+Z_N(u_b-\widetilde u)=0,\qquad
s(p_b,h_b)=\widetilde s.
$$

The acoustic velocity relation freezes the interior impedance and linearizes
the finite pressure change. Entropy, density and sound speed come from the
configured EOS; the law does not require a perfect-gas expression for them.
The pressure and compatibility rows are divided by $p_0$, and the entropy row
by $a_0^2/T_0$. The selected numerical flux remains active at the outlet.

The static law continues to impose these equations when an intermediate
iterate has supersonic outflow. This study does not select the package's
separate automatic acoustic pressure/sonic/outgoing active-set option.

### Finite incoming-family wave

The finite-wave law starts at the last cell's state $(p_i,u_i,h_i)$ and
constructs the incoming-family wave connected to $p_*=p_e$. It samples that
wave at the stationary boundary, $x/t=0$. Its algebraic boundary residual is
the primitive difference between the independent trace and the sampled
target, divided by $(p_0,a_0,a_0^2)$.

For a perfect gas, set $r=p_e/p_i$ and $\beta=(\gamma-1)/(\gamma+1)$. A
compression ($r>1$) has connected state and propagation speed

$$
\rho_* = \rho_i\frac{r+\beta}{\beta r+1},\qquad
u_* = u_i-(p_e-p_i)
\sqrt{\frac{2}{(\gamma+1)\rho_i(p_e+\beta p_i)}},
$$

$$
h_*=h_i+\frac{\gamma}{\gamma-1}
\left(\frac{p_e}{\rho_*}-\frac{p_i}{\rho_i}\right),\qquad
S=u_i-a_i\sqrt{\frac{\gamma+1}{2\gamma}r+
\frac{\gamma-1}{2\gamma}}.
$$

These are the moving Rankine–Hugoniot shock relations, with the EOS enthalpy
reference preserved by the increment. If $S<0$, the wave enters the domain
and the connected state is imposed at the face. Otherwise the face state is
extrapolated from the last two cells. A sufficiently strong finite compression
can enter even when the initial infinitesimal speed $u_i-a_i$ is positive.

An expansion ($r\le1$) follows the isentropic simple-wave relations

$$
a_*=a_i r^{(\gamma-1)/(2\gamma)},\qquad
u_*=u_i+\frac{2(a_i-a_*)}{\gamma-1},\qquad
h_*=h_i+\frac{a_*^2-a_i^2}{\gamma-1}.
$$

The head and tail speeds are $u_i-a_i$ and $u_*-a_*$. If the head is
nonnegative the wave is outgoing. If both speeds are nonpositive, the
pressure-connected state reaches the boundary. If the head is negative and
the tail positive, the face samples a sonic state within the fan:

$$
a_b=u_b=\frac{(\gamma-1)u_i+2a_i}{\gamma+1},\qquad
p_b=p_i\left(\frac{a_b}{a_i}\right)^{2\gamma/(\gamma-1)},\qquad
h_b=h_i+\frac{a_b^2-a_i^2}{\gamma-1}.
$$

The external pressure then lies beyond the part of the expansion that can
reach the stationary face; the face pressure need not equal $p_e$.

For incoming compression, incoming expansion and sonic-fan modes, the model
uses the physical flux $F(W_b)$ of the sampled trace at the outlet. This avoids
applying a second numerical Riemann split to an already sampled boundary
wave. The outgoing mode retains the selected numerical flux. The boundary
wave speed also enters the timestep restriction used by implicit Euler.

Both laws share the infinitesimal relation
$\Delta u=-\Delta p/(\rho a)+O(\Delta p^2)$ at the same base state. They are
different finite-mesh boundary discretizations: the acoustic law extrapolates
interior pressure and entropy to the face, the finite-wave law builds its
curve from the last cell, and their incoming boundary flux treatment differs.
Agreement of their converged results must therefore be assessed rather than
assumed. Prescribed pressure is not in general a nonreflecting condition.
Neither law supplies the extra exterior thermodynamic information needed for
sustained physical reverse inflow.

## A common steady stopping rule

Stopping uses the original signed spatial balances, independently of a small
Newton update or a backend exit flag. Define

$$
r_{ij}=\frac{t_{\rm ref}R_{ij}}{V_iq_{{\rm ref},j}},\qquad
w_i=\frac{V_i}{\sum_kV_k},\qquad
E_j=\sqrt{\sum_iw_i r_{ij}^2}.
$$

Every equation must satisfy either $E_j\le10^{-9}$ or
$E_j/E_j^{(0)}\le10^{-6}$. The baseline $E_j^{(0)}$ is fixed from the actual
initial field for that target solve. If it is at or below $10^{-14}$, only
the absolute criterion applies. Each equation is assessed separately; one
large equation cannot be hidden by averaging across equation components.

The additional requirements are

$$
\|B(W)\|_\infty\le10^{-9},\qquad
G_j=\left|\frac{\sum_iR_{ij}}{f_{{\rm ref},j}}\right|\le10^{-9}
\quad\text{for every }j,
$$

together with finite values and admissibility. Global cancellation does not
replace the local equation norms. In momentum, the global balance includes
the integrated geometry source; it does not require constant axial momentum
flux through a variable-area nozzle.

Direct Newton's accepted-point callback obtains these reductions from the
backend's existing fixed-scaled steady residual. When they first pass, it
also checks the standalone original spatial and boundary graphs before
requesting a stop. A fresh post-solve audit recomputes the signed spatial rows,
boundary equations and initial baseline without relying on stored optimizer
norms. Implicit Euler is audited against the same policy.

PySolverView labels a callback-requested stop with native status 5. This is
expected when the common target is reached before its separate Euclidean
residual target. The native flag is retained unchanged and is reported
separately from independent root convergence. Status 5 alone does not prove
convergence: a callback can also stop for a wall-time budget, so the recorded
reason and original-equation audit decide which situation occurred.

The native Euclidean target is a conservative fallback. With
$c_{ij}=V_iq_{{\rm ref},j}/(t_{\rm ref}f_{{\rm ref},j})$, it is

$$
\varepsilon_{\rm native}=
\min\left(
\frac{\varepsilon_{\rm abs}}{\sqrt{\max_iw_i}},\;
\varepsilon_{\rm BC},\;
\frac{\varepsilon_{\rm global}}{\max_j\|c_{\cdot j}\|_2}
\right).
$$

Cauchy–Schwarz makes this sufficient for the absolute equation, boundary and
global bounds. It does not replace the earlier shared relative-or-absolute
callback criterion or the independent final audit.

An accepted-update-limit stop also has native status 5. It is distinct from
`common_target_reached`: the recorded reason, effective 400-update cap and
independent residual checks must be read together. The secondary trial,
evaluation and wall-time budgets may terminate an attempt earlier.

Root convergence and the intended physical solution remain separate checks.
The report also examines forward flow, a resolved supersonic region before
the compression, subsonic downstream flow, shock location and analytical
field errors. The packaged two-cell shock-location tolerance is checked with
only a floating-point coordinate-roundoff allowance. Continuous analytical
primitive point values are used for field diagnostics; those errors are not
claimed to compare exact conservative cell averages.

## Initial fields and scope of the comparison

A stagnant start is genuinely uniform: $p=p_0$, $h=h_0$ and $u=0$ in all
cells and both traces. The changed outlet pressure initially leaves a boundary
residual. An adapted start is a separately converged numerical adapted
expansion, prepared with the same flux and reconstruction using 50, 100 and
200 cells. Both strategies and outlet laws receive the identical saved
200-cell seed for each flux/reconstruction pair. No shocked analytical
profile is used as a seed.

Each shocked target begins directly on 200 cells. It receives no coarse-grid
shocked continuation, outlet-layer adjustment or implicit warm-up before
direct Newton. Adapted-seed preparation is disclosed separately and shared
across target configurations. This isolates direct convergence from the
specified initial field while preserving the requested coarse-to-fine
acceleration for seed preparation.

The execution budget is 600 seconds per attempt. Direct Newton checks it at
accepted points, so a running Jacobian or line-search sequence completes
before that callback can stop it. The implicit driver also records its
declared wall-time and collapsed-CFL plateau stops. These are failures to
finish within the experiment controls, not weaker convergence criteria.
Final accepted states and failed/rejected work are retained.

The methods use different linear algebra: banded colored AD inside implicit
Euler and dense exact AD with the custom PySolverView backend for direct
Newton. Compilation, solve work and adapted preparation are recorded
separately where available. Single trajectories, potentially run alongside
another worker, do not establish a repeatable timing advantage.

## Differentiability and adapter verification

Van Leer MUSCL is piecewise smooth. At a uniform field, both neighboring
slopes vanish and the limiter's active set changes under arbitrarily small
perturbations. JAX differentiates its selected branch; a symmetric directional
finite difference can instead sample different active sets on its two sides.
The two need not agree at that nonsmooth point. Finite-wave branch changes
also make the complete residual only piecewise smooth. Full-rank Jacobians
and exact differentiation of the selected graph do not guarantee global
Newton convergence through such changes.

The executable [adapter audit](scripts/audit_direct.py) checks a known linear
root through the actual installed backend, the prescribed fraction schedule,
callback status handling and the bit-identical initial field. It also checks
the scaling chain with dense AD, JVPs and centered differences on admissible
smooth fields away from the limiter switches.

In the eight-cell audit, all twelve combinations of the three fluxes, two
reconstructions and two outlet laws have full-rank stagnant Jacobians
($30/30$), with condition numbers about $10^3$ to $2.3\times10^4$ in the stated
fixed coordinates. Smooth-state AD/JVP relative differences are below
$3.1\times10^{-16}$, and the best centered finite-difference relative defects
are below $6.4\times10^{-10}$. The approximately 0.25 directional defect for
MUSCL at rest is recorded as a nonsmooth-switch diagnostic. These are adapter
and small-grid linearization checks, not convergence results for the
200-cell shocked targets.

## Implementation references

- [Nozzle storage, flux, sources, boundary rows and physical audits](../../src/nozzle_hyperbolic/models/euler_nozzle/model.py).
- [Shared coupled spatial and implicit operator](../../src/nozzle_hyperbolic/solvers/fvm/operator.py).
- [Model-owned acoustic and finite-wave outlet laws](../../src/nozzle_hyperbolic/models/euler_outlet.py).
- [Residual definitions and stopping semantics](../../docs/notes/residual_monitoring.md).
- [General-EOS backpressure derivation and scope](../BC_backpressure/THEORY.md).
- [Direct adapter](scripts/direct.py), [initial-field construction](scripts/problem.py) and [independent audit](scripts/audit.py).
