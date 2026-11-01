# Perfect-gas nozzle results

The main campaign completed all 96 requested target solves on 200 physical
cells. Direct Newton had a strict limit of 400 accepted updates. Failed runs
were not extended, retried with another nonlinear method or replaced by an
implicit-Euler result. Their actual convergence histories and final accepted
flow profiles remain selectable in [REPORT.html](REPORT.html).

| Strategy | Independently verified discrete roots | Roots also passing canonical shock branch and position checks |
|---|---:|---:|
| Implicit Euler | 48 / 48 | 44 / 48 |
| Direct PySolverView Newton | 2 / 48 | 0 / 48 |

Here a verified root satisfies the original spatial equations, algebraic
boundary equations, global conservation, admissibility and selected boundary
regimes. Shock accuracy is assessed separately. Native callback status 5 is
expected for a successful common-target stop; its native success flag does not
override the independent equation checks.

## Dependence on discretization and initialization

| Flux | Implicit-Euler roots | Direct-Newton roots |
|---|---:|---:|
| Rusanov | 16 / 16 | 2 / 16 |
| HLL | 16 / 16 | 0 / 16 |
| Corrected Stadtke | 16 / 16 | 0 / 16 |

| Controlled subset | Implicit-Euler roots | Direct-Newton roots |
|---|---:|---:|
| Constant reconstruction | 24 / 24 | 2 / 24 |
| Van Leer MUSCL reconstruction | 24 / 24 | 0 / 24 |
| Stagnant initial field | 24 / 24 | 2 / 24 |
| Numerical adapted initial field | 24 / 24 | 0 / 24 |
| Acoustic outlet | 24 / 24 | 1 / 24 |
| Finite-wave outlet | 24 / 24 | 1 / 24 |

All six numerical adapted seeds were verified, including all eighteen
50 → 100 → 200 preparation stages. Their use did not make direct Newton
converge in this campaign. Both outlet formulations gave the same number of
Newton successes. This does not establish identical convergence paths or
equivalence of their discrete boundary fluxes.

The two direct successes use Rusanov, constant reconstruction, a stagnant start
and the 35% shock target: one with each outlet formulation. All direct attempts
for the 70% target failed within the specified controls.

## A converged root can still miss the analytical position

The four implicit-Euler accuracy misses are the Rusanov/constant 35% shock
cases, covering both initial fields and both outlets. They resolve the intended
shock branch but place the largest pressure jump at $x=0.131$ m, compared with
the analytical $x_s=0.135$ m. The four-cell offset exceeds the canonical
two-cell tolerance; that tolerance was not relaxed.

The two converged direct solutions have the same position discrepancy. For
the acoustic/stagnant pair, direct Newton converged in 222 accepted updates.
Its agreement with implicit Euler is

$$
\max_i |p_i^{\mathrm{Newton}}-p_i^{\mathrm{IE}}|/p_0
=1.47\times10^{-9},\qquad
\max_i |M_i^{\mathrm{Newton}}-M_i^{\mathrm{IE}}|
=2.71\times10^{-9}.
$$

Thus Newton finds the same finite-volume root in that comparison. The
continuous-reference position error remains a separate spatial-discretization
issue. The report permits root-agreement comparisons even when the analytical
position check fails, and displays the accuracy outcome alongside them.

## What stopped the unsuccessful Newton attempts

Of the 46 unsuccessful direct attempts:

- 31 reached 400 accepted updates without satisfying the common checks.
- 15 exhausted Armijo backtracking before reaching 400 updates.

There were no terminal execution exceptions, rank-failure exits, trial-budget
exits or wall-time stops in the main campaign. Rejected property-domain trials,
where present, remain in the saved work counts; they are not accepted updates.

A representative Rusanov/MUSCL/stagnant 35% shock case with the acoustic outlet
stopped after 147 accepted updates. Its last accepted fraction was $2^{-30}$;
the next direction failed sufficient decrease at all 31 tested fractions.
Those final trials were admissible. The numerical Jacobian had rank $606/606$
and the linear Newton residual was $2.16\times10^{-12}$, while the spatial
equation RMS values remained about $0.0024$–$0.0030$ and the boundary defect
about $0.069$. This establishes line-search stagnation away from a root.
Limiter active-set changes are a plausible contributor, but these saved
diagnostics do not prove a unique cause.

The successful implicit results show that each tested flux, reconstruction,
outlet and initial field can reach a verified root through implicit Euler.
They do not establish global convergence of unshifted Newton. The current
initial-step policy and corrected Stadtke flux are insufficient to make direct
Newton robust for these cases within 400 updates. These failures occur with
the perfect-gas backend, without CO₂ bicubic property tables.

## Evidence and interpretation

The saved-evidence audit passed all 96 records and all 48 matched initial-state
pairs. A separate fresh-process audit re-evaluated every final state, including
the 46 failed Newton attempts, without taking any nonlinear or time steps.
Every root/target flag and individual check reproduced exactly. The maximum
difference in all saved spatial norms, boundary/global diagnostics and direct
original-residual arrays was zero. Source, numerical-client, settings and raw
record hashes remained unchanged.

The machine-readable proofs are
[the campaign audit](outputs/audit_campaign.json) and
[the fresh-state audit](outputs/verify_saved.json).

[settings.json](settings.json) records the executable controls;
[THEORY.md](THEORY.md) defines the equations and stopping policy. Full initial
and final states, scalar histories, native trial vectors and source provenance
are retained under `outputs/`. The HTML embeds all plotted fields and scalar
histories, including failed solutions; raw native event vectors remain in the
SHA-identified compressed run records on disk.

Timings describe single trajectories with different linear algebra and
compilation/cache conditions. They are not repeated speed benchmarks.
