# Why Newton stalls while implicit marching and PTC can progress

The difference is the path through flow fields. Ordinary Newton tries to cancel
all steady imbalances with one local linearization. Implicit marching and
physical-storage pseudo-transient continuation (PTC) allow those imbalances to
change stored mass, momentum and energy. They change the correction direction,
not merely its length. This interpretation fits the measured nozzle cases;
it is not an unconditional convergence theorem.

The [experiment overview](../../../README.md) records the tested methods, outcomes and
retained comparison commands.

## An imbalance is also a rate of change

Let $q(z)$ contain cell-integrated conservative variables and let $R$ be the
outward-flux-minus-source spatial residual. The semi-discrete evolution is

$$
\frac{d q(z)}{dt}+R(z)=0,\qquad
M(z)\dot z=-R(z),\qquad M(z)=\frac{\partial q}{\partial z}.
$$

Each Euler cell stores $V(\rho,\rho u,\rho E)$ with
$E=h+u^2/2-p/\rho$. A momentum imbalance can accelerate fluid rather than needing
to vanish immediately. Mass and energy imbalances similarly change density and
thermodynamic state. This supplies a physically meaningful direction for the
next update, without analytically finding the shock or using a 1D nozzle formula.

In this experiment's scaled primitive coordinates,

$$
M=D_R^{-1}\operatorname{blockdiag}
\left(V_i\frac{\partial Q_i}{\partial w_i}\right)D_w,
\qquad w_i=(p_i,u_i,h_i).
$$

$D_R$ and $D_w$ are fixed equation and variable scales. The matrix has units of
time and is generally nonsymmetric. An arbitrary identity shift in primitive
coordinates would not represent the same conservative storage. The PTC controller
is generic; the current storage adapter implements the Euler state mapping.

## Why an accurate Newton direction can still make little progress

Ordinary Newton solves

$$
J_R(z)d_N=-R(z).
$$

Its local prediction is $R(z+\alpha d_N)\approx(1-\alpha)R(z)$. That can be
excellent near a regular root. Far away, it describes only the current local
flow structure. Establishing a supersonic section and a captured shock requires
coordinated changes across cells; characteristic speeds, reconstruction and
boundary responses can change within a proposed update.

The stalled fields had a small supersonic patch near the throat, rather than
the developed supersonic region of the final solution. Some full Newton steps
would make throat pressure negative. Backtracking prevented invalid states but
could leave fractions near $10^{-7}$ and negligible progress. Positivity alone
did not ensure residual decrease.

Accurate dense/colored AD agreement and small linear-solve defects were observed
at sampled stalled fields. This verifies the local derivative and linear solve,
not their predictive quality over a finite path. Conditioning worsened near
choking and finite-difference agreement required very small displacements.
Flux singularities and discontinuous activation made the original algebra worse;
repairing them did not eliminate cold stalls. Constant reconstruction also
failed, so MUSCL cannot be the sole explanation.

Local Newton convergence requires an adequate initial neighborhood and suitable
regularity/nonsingularity. Shocked flow does not invalidate Newton everywhere:
the four custom direct methods did recover the measured roots from sufficiently
advanced implicit fields and perturbed solutions. A narrow basin is consistent
with those local successes and cold failures.

## Why LM, HYBR and SLSQP need not supply the missing path

LM regularizes least-squares motion, schematically through
$(J^TJ+\mu D^TD)d=-J^TR$. Powell HYBR combines Newton-like and gradient-like
steps in a trust region. These are useful safeguards but do not introduce the
conservative evolution operator $M$.

For a square nonsingular system, $J^TR=0$ implies $R=0$. Thus full-rank terminal
LM fields are not proof of exact non-root stationary minima. Weak singular
modes can nevertheless make $J^TR$ small while $R$ remains appreciable. Native
progress-based success can then describe stagnation rather than a root, as in
some historical LM runs. Independent original-residual audits are necessary.

For zero-objective SLSQP with a square full-rank equality Jacobian, the constraint
linearization already fixes $Jd=-R$. The direction can coincide with Newton.
Shorter fractions, different trust regions and positive coordinates can improve
safety without creating the missing conservative evolution direction.

## Backward Euler adds storage to the correction equation

An implicit timestep solves

$$
G(z;z_n)=\frac{q(z)-q(z_n)}{\Delta t}+R(z)=0.
$$

At its initial guess $z_n$, the storage difference is zero but its derivative
is not. The first inner Newton direction satisfies

$$
\left[J_R(z_n)+\frac{M(z_n)}{\Delta t}\right]d=-R(z_n).
$$

For small pseudo-time, $d\approx-\Delta t M^{-1}R$: imbalance produces a small
evolution step. For large pseudo-time, where $J_R$ is invertible, the direction
approaches ordinary steady Newton. The intermediate direction is generally not
a scalar multiple of $d_N$. Scalar line-search damping cannot reproduce this
mode-dependent change of direction.

Physically, the algorithm can accelerate and redistribute the flow in response
to present imbalances while the structure develops. Adaptive timestep and
increment controls limit abrupt changes. The production method continues the
inner nonlinear solve of each timestep and uses compiled banded linear algebra.
Its adaptive steady-state process is not a time-accurate transient simulation.

## A local mode calculation makes the connection explicit

Freeze an invertible $M$ near a root. Writing the linearized dynamics as
$\dot e=-Ae$ with $A=M^{-1}J_R$, a full shifted correction gives

$$
e_{k+1}=(I+\Delta t A)^{-1}e_k.
$$

An eigenmode with $\operatorname{Re}\lambda>0$ has factor
$1/(1+\Delta t\lambda)$ of magnitude below one. This explains damping of
locally stable evolution modes by the storage shift. In conservative coordinates
it is the familiar time-mass shift, even though primitives make it look
nonsymmetric. It differs from damping the normal equations in LM.

This frozen calculation is not a nonlinear convergence proof. Nonnormality can
produce transient amplification; stability-changing modes, varying matrices,
domain restrictions and branch switches need additional control. A shift does
not guarantee every matrix is well conditioned. In a genuinely linear problem,
exact unshifted Newton would solve the root immediately: the practical benefit
here concerns finite steps through a nonlinear flow problem.

## PTC takes one evolution-based correction per trial

The implemented loop uses

$$
[J_R(z_k)+\sigma_k M(z_k)]d_k=-R(z_k),\qquad
\sigma_k=1/\Delta t_k,\qquad z_{\rm trial}=z_k+\alpha_kd_k.
$$

One custom PySolverView Newton correction solves this frozen affine model.
The outer loop evaluates the original nonlinear residual, checks the domain,
relative changes and model error, and accepts or rejects the physical candidate.
Affine backend success does not establish a PDE root. Unlike implicit marching,
it does not fully solve the nonlinear backward-Euler timestep.

The relation to the first backward-Euler Newton linearization is described in
[PETSc's PTC documentation](https://petsc.org/release/manualpages/TS/TSPSEUDO/).
Our implementation adds its own acceptance, nonmonotonicity and shift controls;
it does not claim to reproduce PETSc's timestep controller exactly.

The CFL/SER schedule can increase pseudo-time when progress permits and decrease
it after failures. It need not steadily approach infinity. Near the root, a
sufficiently well-modeled trajectory permits zero-shift Newton trials. Failed
zero-shift trials return to finite regularization. Successful measured runs
finished with ordinary Newton corrections inside this same solver loop.

## Why a temporarily growing residual can be acceptable

Residual norm is instantaneous imbalance, not a thermodynamic potential and not
distance to the desired solution. Evolving waves and a developing shock can
redistribute imbalance before steady balance is reached. This is a possible
physical interpretation of nonmonotonicity, not a diagnosis of every recorded
spike.

The successful cold PTC paths included 542 and 610 accepted residual increases.
The controller allowed bounded growth over a recent window, subject to domain,
step and nonlinear-model limits. That can admit a useful evolution path that a
strictly monotone residual line search repeatedly rejects. The experiment tested
the combination of direction regularization and nonmonotone controls; neither
was established as a sufficient remedy in isolation.

## Why neither method has a physical-branch guarantee

Hyperbolicity describes wave propagation; it does not guarantee that an arbitrary
boundary-value problem has a unique attracting steady state. Implicit stability
for a linear evolution problem does not establish global nonlinear convergence,
positivity of every computed trial, or correct branch selection.

The implemented outlet switches toward pressure extrapolation for supersonic
exit flow. A previously converged fully supersonic field was also a numerical
root at the high nominal backpressures because the floating-point pressure
weight vanished. The early stalled fields had subsonic outlets, so this finding
is not a demonstrated cause of their stalls. It does show why $R=0$ alone cannot
certify the desired shocked branch. An implicit method supplied with that
alternative zero-residual field would also accept it immediately.

Check final Mach profiles, actual numerical mass flux, and analytic branch
agreement in addition to the original residual. Numerical full rank and local
root recovery do not prove global uniqueness or continuum well-posedness.

The defensible conclusion is that physical storage supplied a useful globalizing
trajectory for the two original cold fields. Implicit marching and PTC share
that mechanism, while ordinary root solving lacks it. The PTC implementation
successfully connects evolution-like corrections to Newton near these roots,
but it remained much slower than implicit marching and supplies no unconditional
guarantee for other seeds, meshes, fluids or boundary conditions.
