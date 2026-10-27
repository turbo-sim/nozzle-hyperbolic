# Matched implicit-boundary benchmark

2 of 2 measured solves reached the
original steady and boundary targets on the 13-point mesh.
There are exactly two methods across five backpressures. Each method/case has
**one excluded full compilation run, then 1 measured runs**.
The 2 excluded runs do not enter any timing statistic.

Both methods now use the **same source-native JAX Newton loop algorithm,
colored block Jacobian assembly, and block-banded Thomas solver code**.
The old PySolverView dense inner solver has been removed. The inspected source
path does not use Optimistix; introducing it would add a different solver rather
than matching the current source implementation.

The current source timestep kernel is called directly for internal BCs. The
experimental kernel reuses the source residual/assembly/linear-solver code with
six additional ghost unknowns and algebraic boundary rows. Both use the same
relative-update stopping formula, maximum of eight inner Newton iterations,
CFL/SER and increment/retry controls. Ghost rows have zero storage. Exact
reservoir equations replace the finite three-update inlet inversion; no nested
inlet root solve is called by the coupled residual or shared monitor.

## Timing: minimum, maximum and mean

![Ten-run warm full-solve timing distributions](../../../poetry_setup_audit/boundary_commit_smoke/timings.png)

[Vector PDF](../../../poetry_setup_audit/boundary_commit_smoke/timings.pdf).

Markers show means, whiskers span minimum to maximum, and faint points show
every measured run. All starts are cold physical starts; warm timing means
compiled kernels are reused, not that a solution is warm-started. Both methods
use the identical initial full field, reset controller state on every run,
and alternate execution order between repeats. No method is given a previous
solution. First-run histories/statuses are retained separately.

| Case | Method | Measured runs | Minimum [s] | Maximum [s] | Mean [s] | All roots verified? |
|---|---|---:|---:|---:|---:|---|
| Shock at 35% of divergence | Source: internal BCs | 1 | 0.0117 | 0.0117 | 0.0117 | Yes |
| Shock at 35% of divergence | Experimental: coupled BC equations | 1 | 0.0101 | 0.0101 | 0.0101 | Yes |

| Case | Coupled/source mean-time ratio |
|---|---:|
| Shock at 35% of divergence | 0.865 |

Timing includes synchronized timestep work, both methods' identical monitor,
controller and rejected trials. Geometry/reference construction, kernel setup,
final audits, file writing and plotting are outside solve time. The first
complete solve of each method/case is discarded even if some kernels were
already compiled by prior initialization. Kernels are built once per case and
are not recreated between measured runs. Device results are materialized before
stopping the timer. All measured solves are sequential, with one BLAS thread.
These are local-machine observations, not portable performance guarantees.

## Convergence: five backpressures

![Five-panel convergence history comparison](../../../poetry_setup_audit/boundary_commit_smoke/convergence_histories.png)

[Vector PDF](../../../poetry_setup_audit/boundary_commit_smoke/convergence_histories.pdf).

One actual median-duration measured run per method/case supplies each plotted
trajectory; residual histories are not averaged or selected by best speed.
Each subplot fixes backpressure and contains one line per boundary treatment.
The common residual is the maximum of the fixed-scale interior spatial RMS
and dimensionless boundary RMS; the dotted target is $10^{-8}$. Both methods
use the same JIT monitor on their actual ghost-inclusive state. No internal
inlet inversion is evaluated merely to monitor the coupled solution.

## Final pressure: five backpressures

![Five-panel final pressure comparison](../../../poetry_setup_audit/boundary_commit_smoke/pressure_distributions.png)

[Vector PDF](../../../poetry_setup_audit/boundary_commit_smoke/pressure_distributions.pdf).

The same representative measured runs supply actual pressure fields, excluding
ghosts. Dotted analytic curves are used for verification only, never as seeds.
The sonic-limit discrepancy is preserved in the analytic checks.

| Case | Method | Accepted updates | Rejected trials | Final RMS | Analytic check |
|---|---|---:|---:|---:|---|
| Shock at 35% of divergence | Source: internal BCs | 18 | 0 | $4.2e-09$ | Fail |
| Shock at 35% of divergence | Experimental: coupled BC equations | 18 | 0 | $4.2e-09$ | Fail |

| Case | Maximum pressure difference [Pa] | Scaled field RMS difference |
|---|---:|---:|
| Shock at 35% of divergence | $8.25e-07$ | $2.53e-12$ |

## Interpretation and limits

The former order-of-magnitude gap mixed compiled banded source solves with a
Python/custom dense solver, so it could not establish a boundary-formulation
cost. This benchmark removes that mismatch. Remaining differences include the
six additional algebraic variables, two extra cell rows/superblock grouping,
boundary equations/EOS evaluations, and any resulting Newton/retry work.
They are consequences of the boundary treatment within a matched implementation.

The relative-update stopping formula is applied to all unknowns: interior
variables in source, and interior plus ghosts in the coupled system. Iterates
and factorization roundoff therefore need not be bitwise identical. Both use
the source colored JVP and block-Thomas algorithms, including within-superblock
pivoting rather than a different globally pivoted dense method. Independent
dense AD/solve tests verify the enlarged stencil and linear corrections.

Eight inner updates are a cap, not a proof that every transient nonlinear
system is solved exactly. Native source behavior is retained rather than adding
an experimental least-squares/root tolerance. Final convergence is independently
audited against original spatial equations, boundary constraints, mass flow
and analytic profiles. The comparison uses a shared fixed-scale steady stop,
so standalone driver SSR/plateau counts can differ (56 versus 58 for subsonic
flow in the baseline study).

The coupled inlet solves exact reservoir relations instead of three finite
pressure updates and does not replicate inlet enthalpy clipping outside the
tested admissible branch. This limitation is unchanged. No general EOS/grid/
initialization robustness theorem follows from these five cases.

## Data and source integrity

All `src` Python hashes match before and after the benchmark. No source file
was edited or monkeypatched. The read-only adapter rebinds private source
closures in new experimental function objects. Only residual/boundary equations
and cell-count bindings change; assembly and linear-solver code are shared.
The current source snapshot and tests guard against silently stale assumptions.

[All 2 timing samples](../../../poetry_setup_audit/boundary_commit_smoke/timings.csv), [full measured and excluded histories, fields and source hashes](../../../poetry_setup_audit/boundary_commit_smoke/summary.json). See [README.md](README.md) and [BC_IMPLEMENTATION.md](BC_IMPLEMENTATION.md).
