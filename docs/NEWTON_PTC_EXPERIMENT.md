# Single-loop Newton/PTC experiment

This experiment keeps the original steady finite-volume residual $R(z)$ and
its exact AD Jacobian $J(z)$. It changes the correction equation through a
physical storage shift:

$$
[J(z_k)+\sigma_k M(z_k)]\Delta z=-R(z_k),\qquad
z_{\rm trial}=z_k+\alpha_k\Delta z.
$$

Here $z$ contains scaled interior primitive variables. For
$w=(p,u,h)$ and $Q=(\rho,\rho u,\rho(h+u^2/2)-p)$,

$$
M=D_R^{-1}\operatorname{blockdiag}
\left(V_i\frac{\partial Q_i}{\partial w_i}\right)D_w.
$$

$D_R$ and $D_w$ are the existing residual and variable scales. The matrix is
computed by local EOS AD, includes interior storage only, and has units of
time. It is generally nonsymmetric in primitive coordinates. Thus
$\sigma=1/\Delta t$ gives the appropriate scaled physical shift, rather than
an arbitrary identity addition.

This is pseudo-transient continuation: with an exact Jacobian, an undamped
correction has the form of the first Newton linearization of backward Euler
at the previous state. The experiment does **not** solve a nested nonlinear
backward-Euler timestep. The distinction is described in
[PETSc's PTC formulation](https://petsc.org/release/manualpages/TS/TSPSEUDO/).
These pseudo-times control the steady solver; this is not a time-accurate
transient simulation.
The PTC core has no shock-position or nozzle-profile knowledge; the present
storage adapter implements the EulerNozzle primitive mapping.

## One correction per outer trial

Each trial calls the public custom PySolverView Newton backend with
`library="pysolver"`, `max_iterations=1`, and no inner backtracking. It solves
one correction of the frozen affine model

$$
F_k(y)=R(z_k)+[J(z_k)+\sigma_kM(z_k)](y-z_k).
$$

The affine Jacobian is exact for that model. The original nonlinear residual
is independently evaluated at every proposed physical state. The direction
fraction starts at 0.25 and grows by 1.5 after accepted updates, capped at one.
There is no analytic initial flow, shock fitting, shock-position preprocessing,
or implicit warm-start phase. The driver gives ordinary Newton and PTC the
same copied cold seed; its implicit reference is computed afterwards.

Affine backend success concerns only $F_k$. It does not establish a steady
physical root, and a partial affine correction may legitimately exhaust its
single-update budget. Only the original residual RMS $\leq10^{-8}$ establishes
outer success. Physical calls and affine calls are counted separately.

## Acceptance and shift control

The characteristic time uses the existing interior wave-speed CFL kernel.
The controller sets $\Delta t_k=\mathrm{CFL}_k\tau_k$ and
$\sigma_k=1/\Delta t_k$. Jacobian and storage matrices are refreshed after
acceptance and reused while retrying an unchanged anchor.

The controller checks domain validity, affine linear-solve accuracy, the scaled
increment bound $\max_j|d_j|/(1+|z_j|)\leq0.15$, and nonlinear model error

$$
E=\frac{\|R(z+d)-[R(z)+J(z)d]\|}
{\max(\|R(z)\|,\|R(z)+J(z)d\|)}.
$$

Defaults require $E\leq1$ and permit residual RMS up to twice the maximum
among the last ten accepted states. Controlled nonmonotone growth avoids
requiring a physical evolution direction to decrease the residual at every
step. Rejected finite-shift trials halve CFL; every rejected trial is retained.

Accepted steps update CFL using a bounded SER factor
$(r_{\rm old}/r_{\rm new})^{0.7}$. A small growth probe is allowed on a
well-modeled plateau. After three well-modeled steps and a thousandfold
residual reduction, a zero-shift Newton trial is permitted; an unsuccessful
zero-shift trial returns to finite regularization. These are experimental
controls, not an unconditional convergence theorem. PTC convergence analyses
require stability and regularity assumptions;
[Kelley and Keyes](https://doi.org/10.1137/S0036142996304796) discuss them.

The target roots are unchanged, including any alternative branches admitted
by the current boundary model. Small steps alone are never reported as roots.

## Reproduction and recorded outputs

Run from the repository root:

```powershell
poetry run python -m development_new.newton_solve.investigate_ptc --phases simple --initial-cfl 1 --no-show --output development_new/newton_solve/results_ptc_simple
```

Defaults cover both shock cases, 201 points, and at most 3000 accepted PTC
updates. Solve timing includes backend construction, rejected trials, and
monitoring; JAX first calls, reference generation, and final audits are separate.
No speed claim follows from accepted-update counts alone.

- [PTC core](../development_new/newton_solve/ptc.py)
- [Physical storage adapter](../development_new/newton_solve/ptc_storage.py)
- [Experiment driver](../development_new/newton_solve/investigate_ptc.py)
- [Recorded output directory](../development_new/newton_solve/results_ptc_simple/)

Outputs include `summary.json`, `results.csv`, accepted histories, all-trial
histories, and per-case `simple/residual_convergence.png` and
`simple/ptc_controls.png`. Native affine status, outer root status, failed
trials, CFL, shift, fraction, and model-error records remain distinguishable.
Six storage tests and seven core tests passed before PDE comparisons.

## Measured results

All rows use the same 201-point mesh, corrected FVS flux, ordinary Van Leer
MUSCL reconstruction, perfect-gas backend, and true residual target $10^{-8}$.
Initial CFL is one unless stated otherwise. No low-order spatial Jacobian,
shock initialization, or implicit warm start is used.

| Case | Cold seed | Accepted PTC updates | Rejected trials | Final true RMS | Same shocked root as implicit? | PTC time [s] | Implicit time [s] |
|---|---|---:|---:|---:|---|---:|---:|
| `shock35` | Original simple | 1514 | 58 | $5.59\times10^{-14}$ | Yes | 76.97 | 0.554 |
| `shock70` | Original simple | 1190 | 41 | $5.84\times10^{-10}$ | Yes | 60.75 | 1.521 |
| `shock35` | Uniform spouting | 1984 | 68 | $2.16\times10^{-13}$ | Yes | 99.77 | 0.640 |
| `shock70` | Uniform spouting | 3000 | 186 | $6.85\times10^{-3}$ | No: iteration budget exhausted | 135.41 | 2.389 |

The three successful fields pass the analytic accuracy checks. Their scaled
RMS differences from matched-seed implicit roots are respectively
$1.72\times10^{-7}$, $8.41\times10^{-7}$, and $9.19\times10^{-7}$;
inlet/outlet numerical mass-flow imbalance is below $5\times10^{-15}$.
Each ends with two accepted zero-shift Newton corrections. Thus the
regularization eventually disappears without a separate solver handoff.
The implicit references require 77, 193, 71, and 391 accepted outer updates
respectively and use the same original-residual stopping target.

The successful simple-seed paths contain 542 and 610 accepted residual
increases. The shifts are strongly nonmonotone during globalization; their
early reduction sometimes has to be reversed near narrow nonlinear layers.
This is evidence for a useful evolution-based trajectory, not a claim that
the steady residual decreases throughout the solve.

The failed spouting `shock70` field remains nonsteady: mass imbalance is
13.7%, the analytic Mach error check fails, and residual/CFL cycling continues.
Its outlet remains subsonic, so this is not escape to the pressure-independent
supersonic outlet branch. All affine corrections still succeed numerically.
Most accepted updates are small-CFL recovery probes, and no zero-shift
Newton trial qualifies. Saved scalar histories establish residual/control
cycling; they do not establish a physical limit cycle.

A bounded follow-up sets initial CFL to 20, matching the existing implicit
solver's starting value, with everything else unchanged. It also exhausts
3000 accepted updates, at RMS $7.02\times10^{-3}$.
[Follow-up record](../development_new/newton_solve/results_ptc_spouting_cfl20/summary.json)
retains this negative result. Changing that initial shift alone does not fix
the remaining case.

### Ordinary Newton and performance

Ordinary custom Newton fails from all four case/seed combinations. To check
that this observation is not solely due to its original 300-trial budget,
an additional run raises the maximum to 3000 trials. All four still terminate
with exhausted backtracking, after 944, 268, 385, and 354 trials respectively.
Their final RMS values remain approximately $5.82\times10^{-4}$,
$8.12\times10^{-4}$, $8.91\times10^{-4}$, and $2.40\times10^{-3}$.
[Extended unshifted Newton records](../development_new/newton_solve/results_ptc_newton_budget/summary.json)
preserve statuses and identical cold-seed hashes.

The new implementation is much slower than the existing implicit solver.
It makes many more outer corrections, and custom PySolverView 0.7.4 factors
a dense pivoted QR matrix for every correction. Its public Newton API has
no banded/sparse/LU or user-supplied linear-solver option. The existing
implicit method uses compiled banded solves. This comparison therefore
includes substantial implementation differences; the remaining runtime has
not been profiled enough to attribute it entirely to QR. These are single
serial measurements, not repeated performance benchmarks.

The result demonstrates cold convergence of one adaptive Newton/PTC loop
for both original baseline fields. It does not yet demonstrate uniformly
robust convergence across the tested initializations, or a speed advantage.
The next optimization targets are shift recovery/control and linear-solve
cost. The present data do not justify relaxing the physical root criterion.

## Validation and provenance

All 38 focused tests pass, including 15 tests for the new PTC/storage code.
They cover the independently computed shifted correction, nonsymmetric
physical storage, EOS AD/finite differences, matrix locality, cached retries,
custom backend identity, actual domain failures, and separation of affine
success from physical convergence. They also verify that the solver does not
invoke an implicit timestepper or reference initialization.

Every measured PTC trial performs exactly one custom Newton correction.
Seed hashes, accepted/trial histories, call counts, final residuals, and
same-root distances were independently audited. Full results are in
[simple-seed records](../development_new/newton_solve/results_ptc_simple/summary.json)
and [spouting-seed records](../development_new/newton_solve/results_ptc_spouting/summary.json).

The measured controller and driver are retained in
[measurement sources](../development_new/newton_solve/results_ptc_sources/).
After measurements, small fixes enforce a caller's CFL growth cap below
1.05, report unrepresentable affine-step stagnation without making the shift
larger, reject nonfinite growth settings, and correct baseline CSV trial
counts. The solver branches affected by those fixes were inactive in the
recorded PDE runs: all use growth cap 1.5 and every affine trial makes one
update. Original source hashes and the archived copies remain in each report;
postprocessing hashes distinguish the final code. No production spatial
equations or boundary conditions were changed.
