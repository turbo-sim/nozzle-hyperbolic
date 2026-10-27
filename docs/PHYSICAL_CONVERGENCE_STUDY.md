# Physical initialization, local convergence, and steady nozzle branches

Follow-up: the single-loop Newton/PTC experiment now demonstrates cold
convergence for both shocked cases using the same spatial equations and no
implicit warm start. See [NEWTON_PTC_EXPERIMENT.md](NEWTON_PTC_EXPERIMENT.md).
The unshifted and continuation results below describe the preceding study.

The four custom nonlinear solvers can converge to the shocked discrete
solutions. They recover those solutions from partial implicit fields and from
small perturbations of converged fields. They fail from both tested cold
initializations. The implemented outlet also admits a different, fully
supersonic numerical branch at the same high backpressures. Consequently,
local solver convergence, global uniqueness, and selection of the intended
physical branch are separate questions.

The runs below use 201 points, 597 scaled interior unknowns, the perfect-gas
backend, the corrected characteristic flux, and `muscl_van_leer`
reconstruction with its safe inactive arithmetic. This is not the opt-in
`muscl_van_leer_smooth` discretization. The common root target is scaled
spatial residual RMS $10^{-8}$. All four direct methods assert
`library="pysolver"`; there is no native SciPy fallback.

## What the physical initialization supplies

The uniform spouting initialization uses only reservoir data and the requested
backpressure:

$$
h_b=h(p_b,s_0),\qquad
u_{\rm sp}=\sqrt{2(h_0-h_b)},\qquad
(p,u,h)=(p_b,u_{\rm sp},h_b).
$$

It starts with $H=h+u^2/2=h_0$ and $s=s_0$ everywhere. For a perfect gas,
$T_b=T_0(p_b/p_0)^{(\gamma-1)/\gamma}$. These are the usual isentropic
spouting relations; the implementation evaluates the actual perfect-gas
backend rather than assuming $\gamma=1.4$ or an entropy reference zero.
[NASA's ideal nozzle velocity derivation](https://www.grc.nasa.gov/WWW/wind/papers/NASA-TM-2009-215507-Yoder.pdf)
gives the corresponding total-enthalpy and pressure-temperature relations.

The measured spouting Mach numbers are 0.41454 for `shock35` and 0.71854 for
`shock70`. Initial relative total-enthalpy error is zero, and absolute entropy
error is at most $4.55\times10^{-13}$ J/(kg K).

This is admissible initial data, not a steady variable-area solution. Uniform
$\rho u$ gives spatially varying $\rho uA$. For a consistent flux, the
uniform field's integrated spatial residual is

$$
R_i=(A_{i+1/2}-A_{i-1/2})
\begin{bmatrix}\rho u\\\rho u^2\\\rho uH\end{bmatrix},
$$

because the pressure-area source cancels the pressure part of momentum flux.
Here the nodal mass-flow span is 50% of its maximum. Equal inlet/outlet mass
flow in the symmetric geometry does not eliminate this interior imbalance.

## Why implicit marching can reach a root that a cold direct solve misses

Let $R(w)$ denote integrated spatial balances in primitive unknowns
$w=(p,u,h)$, and let $M(w)=V\,\partial Q/\partial w$ be the conservative
storage Jacobian. Direct Newton computes

$$
J_R(w_n)d=-R(w_n).
$$

Backward Euler instead solves successive problems

$$
\frac{V[Q(w)-Q(w_n)]}{\Delta t}+R(w)=0.
$$

Its first inner Newton direction satisfies

$$
\left[\frac{M(w_n)}{\Delta t}+J_R(w_n)\right]d=-R(w_n).
$$

The storage term changes direction and conditioning. It is not equivalent to
choosing a smaller scalar fraction of the steady Newton direction. Small
pseudo-timesteps give a regularized evolution; larger steps approach a direct
steady solve. This distinction is explicit in
[PETSc's pseudo-transient formulation](https://petsc.org/release/manualpages/TS/TSPSEUDO/).

The present adaptive CFL/SER process is a steady-state acceleration method,
not an accurate physical transient. Hyperbolicity and backward Euler do not
guarantee convergence to a steady state, or to the desired branch. Such
claims require assumptions on attraction, stability, admissibility, boundary
conditions, and numerical solves. Those assumptions appear explicitly in
[Kelley and Keyes' convergence analysis](https://doi.org/10.1137/S0036142996304796).
Entropy selection also depends on a suitable spatial/time discretization;
the corrected flux does not come with a general nonlinear Euler entropy
proof. See [Tadmor's entropy-stability analysis](https://www.math.umd.edu/~tadmor/pub/TV%2Bentropy/Tadmor.MathComp1987.pdf).

### Local convergence is verified

The tight implicit references use RMS $10^{-10}$ and take 81 and 196 accepted
updates. Their scaled Jacobians have numerical rank 597/597:

| Case | Jacobian condition number | Partial implicit updates | Partial-field RMS | Direct methods reaching the root |
|---|---:|---:|---:|---|
| `shock35` | $4.36\times10^4$ | 68 | $1.80\times10^{-6}$ | Newton, LM, HYBR, SLSQP |
| `shock70` | $2.79\times10^4$ | 181 | $9.78\times10^{-6}$ | Newton, LM, HYBR, SLSQP |

All four methods also recover both roots after a $10^{-4}$ scaled sinusoidal
perturbation of the tight reference. These are intentionally local tests;
the partial implicit fields already contain most of the required flow
structure, and their cost is recorded separately from the direct phase.
They do not demonstrate a cold-start acceleration.

| Custom method | `shock35` accepted updates / final RMS | `shock70` accepted updates / final RMS |
|---|---:|---:|
| Newton | 43 / $1.64\times10^{-12}$ | 6 / $2.96\times10^{-13}$ |
| LM | 51 / $1.71\times10^{-9}$ | 4 / $8.88\times10^{-9}$ |
| HYBR | 51 / $5.32\times10^{-9}$ | 4 / $5.39\times10^{-9}$ |
| SLSQP | 33 / $3.03\times10^{-12}$ | 6 / $2.96\times10^{-13}$ |

These are accepted updates from the partial implicit field, not backend
trial counts. The perturbed-field tests require five accepted Newton/SLSQP
updates and one LM/HYBR update in each case. A smaller residual need not mean
a closer field: the `shock35` partial implicit field has RMS residual
$1.80\times10^{-6}$ but scaled distance $2.10\times10^{-3}$ from the root;
the perturbed field has larger residual $5.93\times10^{-6}$ yet is about
48 times closer in state space.

On a differentiable branch, a nonsingular Jacobian supports local isolation
and local parameter dependence. Numerical full rank at the measured roots
does not establish global uniqueness, global Newton convergence, or Hadamard
well-posedness of the underlying continuum boundary-value problem.

Recorded evidence: [warm and perturbed-field results](../development_new/newton_solve/results_custom_warm_proof/summary.json).

### Both cold initializations still fail

The simple initialization and the physical spouting initialization were each
given to all four methods. The budget is 300 trials for Newton/LM/HYBR and
300 major iterations for SLSQP; these counts do not measure equal work.
None of the 16 cold runs reached RMS $10^{-8}$.

| Case | Seed | Newton RMS | LM RMS | HYBR RMS | SLSQP RMS |
|---|---|---:|---:|---:|---:|
| `shock35` | Simple | $6.00\times10^{-4}$ | $6.24\times10^{-5}$ | $1.93\times10^{-4}$ | $4.99\times10^{-4}$ |
| `shock35` | Spouting | $8.12\times10^{-4}$ | $6.24\times10^{-5}$ | $6.78\times10^{-4}$ | $8.15\times10^{-4}$ |
| `shock70` | Simple | $8.91\times10^{-4}$ | $2.93\times10^{-4}$ | $4.26\times10^{-4}$ | $8.69\times10^{-4}$ |
| `shock70` | Spouting | $2.40\times10^{-3}$ | $2.92\times10^{-4}$ | $2.39\times10^{-3}$ | $2.23\times10^{-3}$ |

Thermodynamically coherent initialization therefore does not by itself supply
a large basin of convergence. These outcomes apply to the recorded budgets,
controls, and fields; they do not establish that every cold solver strategy
must fail. [Cold initialization results](../development_new/newton_solve/results_custom_initialization/summary.json)
retain returned fields, residual histories, statuses, and initial-state hashes.

The spouting seed is closer in scaled state distance than the simple seed:
0.237 versus 0.241 for `shock35`, and 0.263 versus 0.345 for `shock70`.
It still lacks the choked throat, supersonic region, shock position, and
downstream entropy rise. Satisfying reservoir enthalpy and entropy alone
does not supply this flow structure.

Stagnation is not proof of a nonzero-residual local minimum. For a square
differentiable system with nonsingular $J$, $J^TR=0$ implies $R=0$.
An almost singular $J$ can nevertheless make the least-squares gradient
small while the residual remains appreciable. Newton, LM, HYBR, and SLSQP
have different safeguards but none promises global root convergence from
arbitrary admissible initial fields.

## Ordinary backpressure continuation also meets a narrow barrier

Starting from $p_b/p_0=0.98$, ordinary pressure continuation with Newton and
HYBR solves intermediate steady systems and reuses each accepted solution.
Both requested shock targets stop near $p_b/p_0=0.934197$:

| Requested target | Last accepted $p_b/p_0$ range | Throat Mach range | RMS evaluated at the requested target |
|---|---:|---:|---:|
| `shock35`, 0.888338 | 0.9341966–0.9341971 | 0.99935–0.99949 | $7.05\times10^{-4}$ |
| `shock70`, 0.708722 | 0.9341974–0.9341982 | 0.99712–0.99848 | $3.47\times10^{-3}$ |

The last accepted intermediate systems satisfy the stage RMS target
$10^{-8}$. That is not convergence at the requested backpressure. Failed
stages are rolled back and remain in the report.

The common stopping pressure and near-sonic throat are consistent with a
choking-related continuation barrier. They do **not** prove a fold or identify
a unique mechanism. Natural parameter continuation can fail when its
state Jacobian becomes singular or when local correction/globalization cannot
follow the branch. Pseudo-arclength continuation addresses simple folds by
augmenting the system, but its success still requires suitable regularity and
transversality. [Dickson et al.'s condition analysis](https://ipsen.math.ncsu.edu/ps/sinum65438.pdf)
states those distinctions. [Recorded ordinary pressure runs](../development_new/newton_solve/results_pressure_continuation/summary.json)
separate stage and target residuals.

A second test starts from the already converged `shock35` field and lowers
pressure toward `shock70`, so it does not need to create a shock from scratch.
Newton and HYBR still stop after only about 0.532% of the pressure path:
from 177,667.63 Pa to approximately 177,476.53 Pa. The pressure drop is only
$0.000955\,p_0$. Last accepted stage residuals are $2.93\times10^{-9}$ and
$1.72\times10^{-9}$, while target residuals remain about $2.67\times10^{-3}$.
This demonstrates a further limitation of the tested natural continuation.
Branch turning, shock positioning on the mesh, and correction strategy remain
possible explanations, not established diagnoses.
[Known-shock tracking results](../development_new/newton_solve/results_pressure_shock_tracking/summary.json)
record the anchor provenance and unsuccessful stages.

## Verified outlet-branch nonuniqueness

The outlet implements

$$
p_g=w(M_{\rm out})p_b+[1-w(M_{\rm out})]p_{\rm extrap},\qquad
w=\tfrac12\left[1+\tanh\left(\frac{1-M_{\rm out}}{0.004}\right)\right].
$$

For the fully supersonic numerical field, $M_{\rm out}=2.187117$ and the
floating-point weight is exactly zero. With fixed limiter scales, the entire
spatial residual therefore loses its backpressure dependence in this regime.

One custom Newton iteration polishes the existing no-shock numerical CSV
field to RMS $1.32\times10^{-14}$. The **same interior vector** then gives
that residual at backpressures 18,747.77, 177,667.63, and 141,744.48 Pa.
The maximum residual change between pressures is exactly zero. Its ghost
pressure remains 18,718.80 Pa, with zero inlet total-enthalpy error.

Distinct existing shocked numerical fields also satisfy the same corrected
Van Leer equations at the high pressures:

| Imposed backpressure [Pa] | Supersonic-field RMS | Shocked-field RMS | Shocked outlet Mach |
|---|---:|---:|---:|
| 177,667.63 | $1.32\times10^{-14}$ | $5.95\times10^{-9}$ | 0.32556 |
| 141,744.48 | $1.32\times10^{-14}$ | $7.48\times10^{-9}$ | 0.40590 |

This is numerical evidence of multiple converged branches of the implemented
problem, not a uniqueness proof for a physical nozzle connected to a specified
ambient reservoir. An implicit integrator supplied with the alternative
zero-residual field would also accept it immediately. Time marching cannot
provide an unconditional physical-branch guarantee for this outlet model.

Supersonic characteristic extrapolation is standard when all characteristics
leave the local boundary; [NASA Wind-US documents the same subsonic-pressure /
supersonic-extrapolation distinction](https://www.grc.nasa.gov/www/winddocs/user/keywords/downpressure.html).
It does not by itself describe every finite-amplitude interaction with an
exterior high-pressure reservoir. Thus this result does not make local
characteristic extrapolation inherently wrong; it identifies a difference
between the implemented branch-dependent boundary model and a global ambient
reservoir specification. In exact real arithmetic the tanh weight is
exponentially small rather than zero, so the exact same-vector invariance here
is specifically a floating-point result.

The original stalled Newton/SLSQP fields had subsonic outlet Mach numbers
around 0.35–0.44, where this pressure weight remains active. Outlet
nonuniqueness is therefore a separate global finding, not a demonstrated
cause of those early near-throat stalls.

Reproduce with
[verify_outlet_branches.py](../development_new/newton_solve/verify_outlet_branches.py).
The [outlet-branch record](../development_new/newton_solve/results_outlet_branches/summary.json)
uses only numerical CSV columns and physical inputs, and also records the
large ghost-pressure mismatch on the unintended supersonic branch.

## Residual homotopy and pseudo-arclength results

Residual anchoring starts from the exact seed root at $\lambda=0$ and solves

$$
H(z,\lambda)=R(z)-(1-\lambda)R(z_{\rm seed})=0.
$$

Only $\lambda=1$ solves the original problem. Starting from the uniform
spouting field, custom Newton and HYBR both stall before that endpoint for
both shock cases. The retained states have original RMS approximately
$9.33\times10^{-4}$ (`shock35`) and $3.08\times10^{-3}$ (`shock70`).
This algebraic path is easy to construct but has no guarantee of remaining
on an admissible, regular path to the desired root.
[Residual-homotopy records](../development_new/newton_solve/results_residual_homotopy/summary.json)
include rejected stages and both homotopy and original residuals.

The pseudo-arclength experiment treats $\beta=p_b/p_0$ as an unknown and
corrects a tangent predictor using

$$
\begin{bmatrix}
R(z,p_0\beta)\\
\sqrt{n}\,t^T([z,\beta]-y_{\rm predictor})
\end{bmatrix}=0.
$$

It uses exact JAX state/pressure derivatives and allows pressure to reverse.
The arc equation is scaled so that the augmented stopping criterion controls
both physical RMS and absolute arc error at $10^{-8}$; both are also audited
independently. The driver passes a separate cubic test that crosses two
known simple folds using the actual custom Newton backend.

| Path | Custom correctors | Last accepted $p_b/p_0$ | Desired target reached? |
|---|---|---:|---|
| Uniform seed at 0.98 toward `shock35` | Newton / HYBR | 0.93419693 / 0.93417487 | No |
| Existing `shock35` root toward `shock70` | Newton / HYBR | 0.88780086 / 0.88780117 | No |

These runs stop at the configured minimum arclength step, rather than
reporting an intermediate pressure root as target convergence.
The cold Newton path contains a small pressure reversal near choking;
this is consistent with turning but does not establish a smooth simple fold.
Pseudo-arclength is therefore a tested negative result here, not a demonstrated
remedy for this discretization and these controls.
[Cold weighted-arclength records](../development_new/newton_solve/results_pressure_arclength_weighted/summary.json)
and [shock-branch arclength records](../development_new/newton_solve/results_shock_arclength/summary.json)
retain all failed corrections. The earlier
[unweighted-arc run](../development_new/newton_solve/results_pressure_arclength/summary.json)
used a stricter augmented tolerance and is retained separately.

### What the stalled continuation derivatives show

The independent [pressure-path diagnostic](../development_new/newton_solve/diagnose_pressure_paths.py)
reconstructs each saved state at its actual stage pressure. Its
[derivative and mode record](../development_new/newton_solve/results_pressure_path_diagnostics/summary.json)
checks square, wide, and bordered Jacobians and directional differences.
All recorded operator hashes match the source runs.

At the final cold Newton stage, the smallest singular value of $J$ is
$6.07\times10^{-7}$, but that of $[J,R_\beta]$ is
$9.64\times10^{-4}$. At the shocked Newton stage the corresponding values
are $2.54\times10^{-4}$ and $2.41\times10^{-3}$. Thus the sampled curve
equations retain full row rank: there is no numerical evidence here that
the entire curve has a branchpoint. The cold weak mode has 99.92% of its
squared scaled-state norm within three cells of the throat. Shock-tracking
weak modes instead concentrate around the shock, near $x=0.134$-$0.139$ m.

The finite differences agree with AD within sufficiently small neighborhoods.
Those neighborhoods can be much smaller than the continuation step floor.
For example, along the cold Newton null tangent, the central derivative
defect is about 0.26 at displacement $10^{-5}$ but falls to
$2.75\times10^{-8}$ at $10^{-7}$. On the shocked path the defect changes
from approximately 195 at $10^{-4}$ to $1.38\times10^{-10}$ at $10^{-5}$.
These numbers are absolute norms in the fixed scaled equations, not relative
errors against a zero tangent derivative. They demonstrate a very narrow
useful linearization neighborhood; they do not establish an AD error or
identify which flux/limiter transition is responsible.

Some failed correctors turn nearly perpendicular to the previous tangent:
the incoming-plane bordered condition number reaches approximately
$10^8$-$10^9$ even though the wide curve Jacobian remains healthy. On the
shocked path a HYBR corrector can move 0.688 in scaled augmented state from
a predictor with arclength step $10^{-4}$. The tangent-plane equation alone
does not constrain distance from that predictor. Cold failures also occur
with zero invalid-domain trials, so positivity recovery is not the common
explanation.

The measured difficulty is therefore a combination of a weak fixed-pressure
direction, rapid nonlinear changes, and insufficient local control of some
curve corrections. A bounded follow-up reduces the shock-path Newton minimum
step from $10^{-4}$ to $10^{-6}$, retaining the same 180-attempt budget.
It still stops at $p_b/p_0=0.88780136$, with target RMS
$2.67\times10^{-3}$. Thus reducing that floor by 100 alone does not fix the
tested path. [Small-step follow-up](../development_new/newton_solve/results_shock_arclength_small_steps/summary.json)
records this result. Predictor-distance/trust-radius control remains a more
targeted next experiment; a full-curve singularity or impossibility of
continuation is not established.

## Practical implications

The earlier demonstrated strategy was implicit pseudo-time evolution followed
by a custom direct solve once the shock structure is established. The current
warm-ups consume 68/81 and 181/196 implicit updates respectively, so they are
evidence of local convergence rather than an efficient early handoff. No
robust speed advantage is established: most warm-plus-direct totals exceed
even the tighter implicit reference time. The marginal `shock70` HYBR
difference is below 0.1% in these single runs. References stop at $10^{-10}$
while direct solves stop at $10^{-8}$, so matched-tolerance, repeated,
end-to-end measurements are needed before claiming a hybrid speedup.

The subsequent Newton/PTC study also reaches the shocked roots directly from
cold fields, with regularization inside one steady-solver loop. This resolves
the earlier lack of a demonstrated cold-start path; its current performance
is documented separately.

For broader robustness, the next controlled investigations should be:

1. Continue the pseudo-time regularization itself, using the storage-shifted
   direction and increasing the timestep as the flow becomes established.
   Switch based on correction quality and residual progress, with a fallback
   to pseudo-time evolution if the unshifted solve stalls.
2. Use the measured curve-derivative changes to control continuation locally:
   adapt the predictor step to tangent rotation and limit corrector distance
   or trust radius relative to that step. Separately isolate the flux and
   limiter contributions to the narrow nonlinear layers before altering the
   target discretization.
3. Test dissipation or mesh continuation: establish a shock on a more
   dissipative/coarse discretization and continue to the desired one. This
   changes intermediate equations and requires a final audit of the original
   target discretization; it has not been shown to work in this study.
4. Make physical branch acceptance explicit: a shocked target needs the
   intended subsonic outlet and pressure coupling, alongside residual,
   conservation, and shock/admissibility checks. Reconsider the downstream
   reservoir model separately if it must select a unique ambient-pressure
   solution from any initial field.

The experiment scripts, full histories, settings, backend identities, and
CSV/PNG/PDF outputs are under `development_new/newton_solve/`. The production
finite-volume equations and boundary conditions were not modified in this
physical-path study. The project `AGENTS.md` records the custom-only policy
for future investigations.

Validation: the 23 focused custom-backend, double-fold/rollback/arclength,
and physical-initialization tests pass. Backend identities, finite saved
data, seed provenance, stage acceptance, and final-root audits were also
checked independently across the seven main result reports.
