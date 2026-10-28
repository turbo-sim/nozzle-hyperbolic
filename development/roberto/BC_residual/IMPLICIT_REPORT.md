# Matched implicit-boundary benchmark

100 of 100 measured solves reached the
original steady and boundary targets on the 201-point mesh.
There are exactly two methods across five backpressures. Each method/case has
**one excluded full compilation run, then 10 measured runs**.
The 10 excluded runs do not enter any timing statistic.

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

![Ten-run warm full-solve timing distributions](outputs/steady/timings.png)

[Vector PDF](outputs/steady/timings.pdf).

Markers show means, whiskers span minimum to maximum, and faint points show
every measured run. All starts are cold physical starts; warm timing means
compiled kernels are reused, not that a solution is warm-started. Both methods
use the identical initial full field, reset controller state on every run,
and alternate execution order between repeats. No method is given a previous
solution. First-run histories/statuses are retained separately.

| Case | Method | Measured runs | Minimum [s] | Maximum [s] | Mean [s] | All roots verified? |
|---|---|---:|---:|---:|---:|---|
| No shock (supersonic exit) | Source: internal BCs | 10 | 0.0521 | 0.0567 | 0.0541 | Yes |
| No shock (supersonic exit) | Experimental: coupled BC equations | 10 | 0.0553 | 0.0596 | 0.0576 | Yes |
| Shock at 35% of divergence | Source: internal BCs | 10 | 0.3394 | 0.3675 | 0.3553 | Yes |
| Shock at 35% of divergence | Experimental: coupled BC equations | 10 | 0.3658 | 0.4021 | 0.3766 | Yes |
| Shock at 70% of divergence | Source: internal BCs | 10 | 0.8507 | 0.9536 | 0.8891 | Yes |
| Shock at 70% of divergence | Experimental: coupled BC equations | 10 | 0.9091 | 0.9985 | 0.9469 | Yes |
| Sonic throat, subsonic exit | Source: internal BCs | 10 | 0.2120 | 0.2677 | 0.2272 | Yes |
| Sonic throat, subsonic exit | Experimental: coupled BC equations | 10 | 0.2305 | 0.2668 | 0.2373 | Yes |
| Subsonic throughout | Source: internal BCs | 10 | 0.1551 | 0.2036 | 0.1710 | Yes |
| Subsonic throughout | Experimental: coupled BC equations | 10 | 0.1632 | 0.2072 | 0.1817 | Yes |

| Case | Coupled/source mean-time ratio |
|---|---:|
| No shock (supersonic exit) | 1.066 |
| Shock at 35% of divergence | 1.060 |
| Shock at 70% of divergence | 1.065 |
| Sonic throat, subsonic exit | 1.045 |
| Subsonic throughout | 1.062 |

Timing includes synchronized timestep work, both methods' identical monitor,
controller and rejected trials. Geometry/reference construction, kernel setup,
final audits, file writing and plotting are outside solve time. The first
complete solve of each method/case is discarded even if some kernels were
already compiled by prior initialization. Kernels are built once per case and
are not recreated between measured runs. Device results are materialized before
stopping the timer. All measured solves are sequential, with one BLAS thread.
These are local-machine observations, not portable performance guarantees.

## Convergence: five backpressures

![Five-panel convergence history comparison](outputs/steady/convergence_histories.png)

[Vector PDF](outputs/steady/convergence_histories.pdf).

One actual median-duration measured run per method/case supplies each plotted
trajectory; residual histories are not averaged or selected by best speed.
Each subplot fixes backpressure and contains one line per boundary treatment.
The common residual is the maximum of the fixed-scale interior spatial RMS
and dimensionless boundary RMS; the dotted target is $10^{-8}$. Both methods
use the same JIT monitor on their actual ghost-inclusive state. No internal
inlet inversion is evaluated merely to monitor the coupled solution.

## Final pressure: five backpressures

![Five-panel final pressure comparison](outputs/steady/pressure_distributions.png)

[Vector PDF](outputs/steady/pressure_distributions.pdf).

The same representative measured runs supply actual pressure fields, excluding
ghosts. Dotted analytic curves are used for verification only, never as seeds.
The sonic-limit discrepancy is preserved in the analytic checks.

| Case | Method | Accepted updates | Rejected trials | Final RMS | Analytic check |
|---|---|---:|---:|---:|---|
| No shock (supersonic exit) | Source: internal BCs | 19 | 0 | $8.19e-09$ | Pass |
| No shock (supersonic exit) | Experimental: coupled BC equations | 19 | 0 | $8.19e-09$ | Pass |
| Shock at 35% of divergence | Source: internal BCs | 77 | 14 | $5.95e-09$ | Pass |
| Shock at 35% of divergence | Experimental: coupled BC equations | 77 | 14 | $5.95e-09$ | Pass |
| Shock at 70% of divergence | Source: internal BCs | 193 | 45 | $7.49e-09$ | Pass |
| Shock at 70% of divergence | Experimental: coupled BC equations | 193 | 45 | $7.49e-09$ | Pass |
| Sonic throat, subsonic exit | Source: internal BCs | 80 | 0 | $9.97e-09$ | Fail |
| Sonic throat, subsonic exit | Experimental: coupled BC equations | 80 | 0 | $9.97e-09$ | Fail |
| Subsonic throughout | Source: internal BCs | 56 | 0 | $9.14e-09$ | Pass |
| Subsonic throughout | Experimental: coupled BC equations | 56 | 0 | $9.14e-09$ | Pass |

| Case | Maximum pressure difference [Pa] | Scaled field RMS difference |
|---|---:|---:|
| No shock (supersonic exit) | $4.22e-09$ | $3.13e-15$ |
| Shock at 35% of divergence | $4.86e-07$ | $1.8e-13$ |
| Shock at 70% of divergence | $5.38e-08$ | $1.62e-14$ |
| Sonic throat, subsonic exit | $4.95e-10$ | $1.27e-15$ |
| Subsonic throughout | $5.24e-10$ | $9.72e-16$ |

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

[All 100 timing samples](outputs/steady/timings.csv), [full measured and excluded histories, fields and source hashes](outputs/steady/summary.json). See [README.md](README.md) and [BC_IMPLEMENTATION.md](BC_IMPLEMENTATION.md).
