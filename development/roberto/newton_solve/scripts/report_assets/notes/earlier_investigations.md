# Earlier Newton/PTC investigations

## Earlier Newton and PTC investigations

Physical-storage pseudo-transient continuation (PTC) reached both shocked steady
solutions from the original cold initialization, without an implicit warm start.
Unshifted Newton and the tested alternatives did not reliably reach those roots
from that seed. The existing implicit solver remained much faster. This is a
successful globalization experiment, not an established replacement for it.

The companion [physical and mathematical explanation](convergence_mechanism.md)
connects the observed failures to the evolution-based directions that worked.
The measurements below summarize the earlier investigations; they are not new
performance measurements made during cleanup.

The [PTC performance investigation](ptc_performance.md) separates linear-algebra
cost from the controller-induced iteration gap, with measured inner-solve counts
and controlled one-correction replays.

## Five-case Newton/PTC comparison

The default [comparison script](scripts/investigate_ptc.py) now runs all five
baseline cases: fully subsonic, sonic-throat/subsonic-exit limit, shocks at
35% and 70%, and adapted supersonic exit. Each case has three independent runs:
implicit integration, Newton/PTC, and **controlled custom PySolverView Newton**.
The last method is unshifted Newton with initial fraction 0.25, growth 1.5,
and backtracking; it is not vanilla SciPy or unconditional full-step Newton.

Open [outputs/REPORT.md](../../../outputs/REPORT.md) after running the script. Its two
main figures are [convergence histories](../../../outputs/convergence_histories.png)
and [final pressure profiles](../../../outputs/pressure_profiles.png), also saved as
vector PDFs. Both use one subplot per case and the same three method colors
(plasma between 0.2 and 0.8). The pressure panels include analytic references.
A failed solver's final returned field is explicitly labeled unconverged.
Root convergence and agreement with the analytic solution remain separate.

## Shared problem and acceptance criterion

The physical setup comes from the [steady EulerNozzle example](../euler_nozzle/README.md):
perfect-gas air, reservoir conditions 200 kPa and 300 K, a 0.2 m symmetric nozzle
with area ratio two, and 201 points including ghost points. The two shocked backpressure ratios
$p_b/p_0=0.888338$ and $0.708722$ place shocks at 35% and 70% of the divergent
section. The remaining backpressures select the subsonic, limiting sonic, and
adapted supersonic analytic branches. The scaled interior primitive unknowns $(p,u,h)$ total 597.

All retained methods solve the same original spatial equations and use a
fixed-scale residual RMS target of $10^{-8}$. This differs from the production
driver's state-dependent SSR. Backend success and small steps do not establish
a physical root. Analytic accuracy and agreement with the intended implicit
branch are audited separately; the scaled state RMS agreement threshold is
$10^{-4}$. The analytic solution is never supplied as the initial field.

## Approaches that did not resolve the cold-start problem

| Approach | What happened | Interpretation and limits |
|---|---|---|
| Ordinary Newton and zero-objective, equality-constrained SLSQP | Smooth cases could converge, but both shocked cold cases stalled around RMS $5\times10^{-4}$ to $9\times10^{-4}$ in the first legacy-flux study. | For a square full-rank equality problem, SLSQP's feasibility step can coincide with Newton. A different interface need not provide a different direction. |
| Initial step-size sweeps | Fractions 1, 0.25, 0.1, 0.03, 0.01 with growth 1.5; also fraction 0.1 with growth 1.0 and 1.1. None of 28 Newton/SLSQP runs met the target within the tested budgets. | Smaller fractions of the same direction did not cure the later bottleneck. An initial cap also regrows unless persistent damping is specified. This was a finite sweep, not a proof of impossibility. |
| Short implicit prefixes | Early switches to a direct method still stalled; sufficiently advanced fields restored convergence. | A useful starting field must already contain much of the required flow structure. Prefix generation costs count toward hybrid runtime; measured hybrids did not show a robust speedup. |
| Historical native SciPy LM and Powell HYBR | Cold runs missed the physical target, including some native LM progress-success statuses. | A least-squares progress flag can stop at a non-root. These historical native adapters were removed; retained Newton calls use custom PySolverView. |
| Constant, ordinary Van Leer, and smooth MUSCL reconstruction | Direct cold failures persisted with constant reconstruction and after smoothing MUSCL. | MUSCL regularity matters but cannot be the sole cause. Removing it did not give a reliable global path. |
| Flux and limiter repairs | Fixed singular projector algebra, discontinuous entropy activation, and unsafe reverse-AD arithmetic; cold direct failures persisted. | Genuine defects were repaired, but the repairs were not a complete globalization remedy. The [separate numerical-fix study](../numerical_fixes/README.md) owns their code, tests, and comparisons. |
| Positive/log coordinates | Tested LM/HYBR cold runs and a short implicit-prefix variant still missed tolerance. | Maintaining positivity and changing trust-region geometry did not by themselves resolve the trajectory problem. |
| Residual-anchoring homotopy | Newton/HYBR stopped before the original-equation endpoint; target RMS remained about $9.33\times10^{-4}$ and $3.08\times10^{-3}$. | Successful intermediate modified equations do not imply an admissible regular path to the original root. |
| Natural backpressure continuation | Cold paths stopped near $p_b/p_0=0.934197$, close to choking. Tracking an already shocked solution toward the downstream-shock target also stopped early. | Consistent with a difficult conditioning/branch region, without proving a simple fold. Intermediate stage roots were not target-pressure solutions. |
| Pseudo-arclength continuation | Cold weighted paths stopped near pressure ratios 0.934197/0.934175; shock-anchored paths stopped near 0.887801 instead of 0.708722. A smaller arclength floor did not fix the latter path. | Allowing pressure reversal and passing a separate cubic-fold test was insufficient here. The obstruction was not conclusively isolated. |
| A larger ordinary Newton budget | With a 3000-trial cap, the two original-seed runs still exhausted backtracking after 944 and 268 trials, at RMS about $5.82\times10^{-4}$ and $8.91\times10^{-4}$. | The original failure was not merely the initial 300-trial cap. |

Early derivative diagnostics found accurate dense/colored Jacobian agreement
and small linear-solve defects at sampled stalled fields. Conditioning worsened
near choking and the useful linearization neighborhood became very narrow.
Some full directions would make throat pressure negative; tiny admissible steps
could still increase residual. This supports a globalization problem, not an
established exact non-root minimum or a universally incorrect Jacobian.

## What the warm-start tests established

All four custom PySolverView methods (Newton, LM, HYBR, SLSQP) converged from
sufficiently advanced implicit fields and from small perturbations of tight
implicit roots. The partial implicit fields required 68 and 181 updates; their
original RMS values were $1.80\times10^{-6}$ and $9.78\times10^{-6}$.
Measured root Jacobians had full numerical rank, with condition numbers about
$4.36\times10^4$ and $2.79\times10^4$.

This establishes useful local basins for the measured discrete roots. It does
not prove global uniqueness, continuum well-posedness, or cold-start convergence.
Generating most of the flow with implicit integration is evidence of local
solvability, not an efficient direct cold solve.

## What worked: one-loop physical-storage Newton/PTC

The retained solver computes

$$
[J_R(z_k)+M(z_k)/\Delta t_k]d_k=-R(z_k),\qquad
z_{\rm trial}=z_k+\alpha_kd_k.
$$

The physical storage Jacobian $M$ is expressed in the same scaled primitive
coordinates as $J_R$; it is generally nonsymmetric, rather than an identity.
Each trial uses one custom PySolverView Newton correction of a frozen affine
model. The outer controller evaluates the original nonlinear residual,
checks domain validity, relative changes, nonlinear model error and bounded
nonmonotone growth, and can reject the candidate and strengthen the shift.

Pseudo-time adapts through CFL/SER control. The regularization does not decrease
monotonically: difficult regions can require recovery to smaller pseudo-times.
Near a root, zero-shift Newton is tried and can fall back to regularization.
No shock-position preprocessing, analytic seed, implicit prefix, or nested
nonlinear timestep solve is used.

Historical results for corrected FVS, safe ordinary Van Leer, original simple
cold initialization, and initial PTC CFL one:

| Case | Accepted PTC updates | Rejected trials | Final original RMS | Same shocked root? | PTC time [s] | Implicit time [s] |
|---|---:|---:|---:|---|---:|---:|
| shock35 | 1514 | 58 | $5.59\times10^{-14}$ | Yes | 76.97 | 0.554 |
| shock70 | 1190 | 41 | $5.84\times10^{-10}$ | Yes | 60.75 | 1.521 |

The matched implicit references required 77 and 193 accepted outer updates.
Both successful PTC fields passed the analytic checks and finished with two
accepted zero-shift Newton corrections. Scaled distances from the implicit
roots were $1.72\times10^{-7}$ and $8.41\times10^{-7}$. Mass imbalance was below
$5\times10^{-15}$. Successful paths included 542 and 610 accepted residual
increases, showing that strictly monotone residual descent was unnecessary.

PTC was much slower: it made more corrections and used dense custom-backend QR
while implicit marching used compiled banded solves. These are single serial
implementation measurements, not repeated intrinsic-algorithm benchmarks.
Success on these two cases does not establish global convergence or robustness
for arbitrary seeds, grids, thermodynamic models, or boundary conditions.

## Why this connects to implicit integration

At the start of a backward-Euler timestep the storage difference is zero, but
its derivative is $M/\Delta t$. Its first Newton linearization therefore gives
the same shifted direction as an undamped PTC trial. Implicit marching continues
solving the nonlinear timestep; this PTC method instead controls one linearized
correction using the original residual. Both allow imbalances to drive an
evolution-based redistribution before approaching steady Newton near the root.
See the [companion explanation](convergence_mechanism.md) for the derivation and
physical interpretation, including why neither method guarantees branch choice.
