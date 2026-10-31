# Choosing eliminated or coupled boundary equations



## Highlights


My recommendations:

- **Coupling is more general:** an inlet may have no solution with the interior state temporarily frozen, while the combined system remains solvable. Coupling lets both adjust. It cannot fix genuinely inconsistent boundary conditions.
- **Keep elimination as Euler’s default.** For a difficult implicit two-fluid inlet, I would start with coupled equations as the reference implementation and validate robustness before choosing a production default.
- **Coding simplicity depends on the boundary:** elimination fits today’s solver more easily; coupled equations often express complicated physical laws more cleanly, without separate inversion algorithms.
- **Supporting both is reasonable:** define the model-specific boundary equations once, then select `"eliminated"` or `"coupled"` in one assembly factory. Share the timestepper and diagnostics.
- **Avoid duplicated physics and automatic switching.** Those would create more complexity than the explicit two-option design.

The note includes an algebraic example showing how a failed inner boundary inversion can coexist with a regular, solvable coupled system.


## Recommendation

Use **one physical boundary residual as the canonical definition**, and allow
two numerical realizations of that definition. Keep eliminated boundaries as
the Euler default because the current tests show reliable behavior and a small
speed advantage. For a new, strongly coupled implicit two-fluid inlet, I would
start with the coupled residual formulation as the reference implementation,
then add elimination where a reliable local map has been demonstrated.

The coupled formulation is more general: it does not require the boundary
state to exist as a single-valued function of every trial interior state.
It is not universally more robust. A well-bracketed scalar or explicit
boundary map can be simpler, safer and faster than a large coupled solve.
The right default can differ by physical boundary model.

These are design recommendations, not new two-fluid results. The completed
[Euler benchmark](IMPLICIT_REPORT.md) found all 100 measured solves converged,
with essentially identical histories/fields and coupled mean costs about
4.5-6.6% higher. That modest Euler overhead is not a performance prediction for
a two-fluid system or a general EOS. No source implementation changes are
proposed as part of writing this note.

## 1. What does it mean for the inner boundary solve to have no solution?

Let $x$ be interior unknowns and $b$ the boundary/ghost unknowns. Write the
implicit interior equations as $F(x,b)=0$ and physical boundary equations as
$C(x,b)=0$.

There are several different failure mechanisms:

| Situation | Meaning | Would coupling help? |
|---|---|---|
| A root exists, but the local algorithm misses it | Poor initialization, step, bracket, tolerance, or thermodynamic inversion | Possibly; improving the inner algorithm may be sufficient |
| No admissible boundary root for the current frozen $x$ | The local map cannot be evaluated at that trial, but another interior state may permit a root | This is a strong reason to consider coupling |
| Several admissible boundary roots | A branch must be selected using physics, continuation, or operating history | Neither formulation selects the correct branch automatically |
| Boundary map is locally ill-conditioned or turns | $C_b$ is nearly singular; $b=B(x)$ is a poor local parameterization | Coupling may remain regular, but this must be checked |
| No admissible simultaneous solution at all | Boundary data, physical closure, EOS domain, or equation count are inconsistent | Neither formulation fixes the model |
| EOS evaluation itself fails at a trial | $C$ or $F$ cannot be evaluated there | Coupling still needs domain handling/globalization |

An inner solve failing is not, by itself, evidence that the physical inlet
problem has no solution. Conversely, a clipped finite value is not evidence
that the intended boundary equations have been satisfied.

### A small example of a genuine advantage for coupling

Consider the abstract equations

$$
F(x,b)=b-1=0,\qquad C(x,b)=b^2-x=0.
$$

The joint solution is $(x,b)=(1,1)$. But if the interior trial has $x<0$,
there is no real solution of the inner problem $b^2=x$. An eliminated residual
using $b=\sqrt{x}$ is not even defined there. The joint residual remains
well-defined and can move $x$ and $b$ together toward a feasible state.

Its Jacobian is

$$
\begin{bmatrix}0&1\\-1&2b\end{bmatrix},
$$

whose determinant is one, including at $b=0$, where $C_b=2b$ is singular.
This illustrates why a conditional boundary inversion can fail while the full
system is regular. It is an algebraic example, not proof that a particular
inlet's EOS remains evaluable on every coupled trial.

If instead the boundary condition were $C=b^2+1=0$ for real $b$, there would
be no joint solution either. Adding that equation to a larger residual would
not make it solvable.

## 2. Mathematical strength and its limits

Elimination assumes a boundary map $b=B(x)$ and solves $F(x,B(x))=0$.
If $C_b$ is invertible on a smooth branch, then

$$
B_x=-C_b^{-1}C_x,\qquad
J_{\rm reduced}=F_x-F_bC_b^{-1}C_x.
$$

The full Newton matrix is

$$
J_{\rm coupled}=\begin{bmatrix}F_x&F_b\\C_x&C_b\end{bmatrix}.
$$

Eliminating its boundary correction gives the reduced matrix when that block
elimination is legitimate. Thus exact elimination is not intrinsically an
inferior Newton method, and augmentation does not create a new convergence
mechanism when both are regular and solved accurately.

The distinction is the assumption required to construct the reduced residual:
if $C_b$ fails, the chosen boundary map can cease to exist or become extremely
sensitive, while the complete matrix may still be invertible through the
interior/boundary coupling. The full matrix can also be singular or badly
conditioned; augmentation does not guarantee otherwise. Enlarging a system
can make scaling and globalization harder.

Keeping the full nonlinear boundary formulation does **not** require always
factoring a large dense matrix. Local/sparse assembly or block linear algebra
can exploit its structure. Linear block elimination inside a coupled Newton
solve is different from solving the nonlinear boundary equations to completion
before every residual call. Schur block methods are standard numerical tools;
[PETSc's field-split documentation](https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/)
describes that linear-system structure. This does not recommend adding PETSc
to this project.

For implicit time integration, boundary constraints remain algebraic, with no
invented ghost storage. Coupled time-dependent differential/algebraic systems
also require consistent initialization and regularity assumptions;
[SUNDIALS IDA's mathematical documentation](https://sundials.readthedocs.io/en/latest/ida/Mathematics_link.html)
explains those distinctions. This is a formulation analogy, not a proposal to
change our timestep library. If boundary physics genuinely has its own dynamics
(for example an external reservoir model), its physical storage is a separate
modeling choice.

## 3. Which is stronger in practice?

### When elimination is an excellent choice

Elimination is attractive for explicit prescribed states, simple extrapolation,
and unique scalar inversions with reliable brackets. It keeps every trial on
the boundary manifold and reduces unknown count. That can protect boundary
admissibility, simplify the cell solver and avoid extra algebraic scaling.

Its main hidden cost is the need for a valid and differentiable boundary map
throughout the outer solver's useful search region. A nested solve also needs
failure reporting and sufficiently accurate sensitivities. A fixed small
iteration count can be useful computationally, but then the algorithm defines
an approximate map; its returned state must be checked against $C=0$.

### When I would favor coupling

I would favor coupling when the inlet has several mutually dependent unknowns,
local solves have narrow feasibility regions, branch changes obstruct the map,
or a local fixed-point/root iteration is fragile. One outer Newton process
then handles all interactions and one globalization policy can control the
joint proposed move. It exposes constraint defects directly rather than hiding
inner failures behind clipped ghost values.

The trade-off is that Newton trials can violate boundary constraints and leave
the admissible domain. Coupling still needs physical state checks, sensible
variable/equation scales, damping or timestep reduction as appropriate, and
correct branch handling. Allowing temporary constraint residuals does not mean
allowing arbitrary EOS-invalid states. Initial guesses should be as consistent
as practical; coupling is not permission to ignore consistency requirements.

**My preference for difficult implicit boundary physics is the coupled
formulation as the reference, with elimination as a validated specialization.**
For simple boundaries, the preference reverses: eliminate the obvious algebra
and avoid unnecessary unknowns. This is a per-boundary-physics judgment, not a
claim that one representation should replace the other everywhere.

## 4. Why the two-fluid concern is plausible

The current [two-fluid solver](../../../../src/nozzle_hyperbolic/models/two_fluid/solver.py)
already treats some stagnation boundaries with a precomputed dominant-phase
isentrope lookup and fixed-point updates linking density, phase velocity,
void fraction and static enthalpy. The source comments identify fragile reverse
thermodynamic calls and table-range limitations. This code observation supports
your concern about conditional local solvability; it does not establish a
failure of the current two-fluid implementation or prove coupling would cure it.

The lookup is a finite-resolution model of the isentrope and clamps at its
range limits. Those safeguards can prevent invalid numerical evaluations,
but the original thermodynamic boundary residual must still be audited to
distinguish a satisfied constraint from an endpoint clamp.

Before choosing a production default, I would specifically check:

- Which variables are prescribed and which are unknown? Count independent
  physical incoming conditions and outgoing compatibility relations for the
  actual model/regime; a square algebraic system alone does not ensure a
  well-posed hyperbolic boundary problem.
- Are shared pressure, phase total enthalpies, entropy conditions, quality,
  slip and extrapolated velocities mutually compatible? Do not append both
  phase entropy conditions to all current prescriptions without checking
  independent degrees of freedom. The existing stagnation closure selects a
  dominant-phase isentrope; changing that would be a physical-model change.
- What happens near a disappearing phase or saturation/table limit? Rank loss,
  undefined phase properties and multiple thermodynamic branches can affect
  either formulation. Coupling does not repair an undefined EOS.
- Does a failed frozen-interior root become feasible when the adjacent flow
  moves, or do the requested reservoir/closure data remain inconsistent?

A useful next robustness test would retain identical physical BC equations and
compare admissible near-limit cases, deliberately difficult initial fields,
constraint defects, timestep rejections and failures in both representations.
A modest Euler timing penalty alone should not decide the two-fluid design.

## 5. Coding simplicity and a clean code structure

There are two different meanings of "simpler":

| Coding concern | Eliminated boundary | Coupled boundary |
|---|---|---|
| Fit into today's interior-only FVM solver | Usually simpler | Requires algebraic unknown/row support |
| State the physical boundary laws once | Requires a map or an inner algorithm | Directly express $C(x,b)=0$ |
| Handle a difficult implicit closure | Separate root algorithm, tolerances, feasibility and derivative logic | One common nonlinear solve, but larger state and scaling |
| Ensure boundary constraints at every trial | Enforced through the map if the map succeeds | Must tolerate and monitor off-manifold trials |
| JAX AD | AD through finite iterations, or a regular implicit derivative at a converged root | Direct AD of the BC residual |
| Generic future multiphysics boundaries | Can accumulate specialized inversion machinery | Often cleaner physical composition |

Once the common solver accepts differential and algebraic rows, coupled BC
laws can be cleaner to implement and differentiate than a collection of nested
inversion procedures. But imposing that capability on every explicit boundary
would add avoidable bookkeeping. For the existing Euler code, retaining its
current eliminated default is the simpler immediate choice.

Keep **physical boundary definitions beside the physical model**, consistent
with the repository's model-specific organization. The reusable FVM layer
should know how to assemble/eliminate algebraic boundary blocks, not know
reservoir entropy, phase-slip laws or two-fluid thermodynamics.

A proposed interface, not code implemented by this note, is:

```python
boundary.residual(interior_trace, boundary_state, data)
boundary.scales(data)
boundary.initial_guess(interior_trace, data)
boundary.admissible(interior_trace, boundary_state, data)
# Optional specialization, only when supported robustly:
boundary.eliminate(interior_trace, data)
```

Expose local stencil/dependency information as part of assembly metadata.
Colored/banded Jacobians are only valid for the actual dependency pattern;
a future globally coupled reservoir cannot silently reuse today's local band.
Use structured domain/status results for inner failures rather than returning
NaNs or silently accepting a clipped non-root. Both formulations should audit
the same original $C$, with the same physical data and tolerances.

## 6. Should both approaches be selectable?

**Yes, provided they are two strategies around one boundary model.** A single
factory-level option is sufficient:

```python
boundary_formulation = "eliminated"  # or "coupled"
```

This must be distinct from the physical boundary kind, such as reservoir,
prescribed pressure or transmissive. Construct the appropriate residual layout
and kernels once, before JIT compilation. The different array sizes mean
separate compiled functions; do not scatter mode branches through numerical
kernels or attempt runtime switching between differently sized JIT states.

For a clean first implementation:

1. Define physical laws, scales, admissibility and guesses once in the model's
   boundary module.
2. Centralize unknown packing and row assembly in one strategy factory.
3. Permit elimination only when that boundary object provides a tested map;
   reject unsupported combinations clearly.
4. Keep one timestep/controller implementation, one convergence policy and one
   diagnostics path for both. Match kernel/linear algebra choices when timing.
5. Verify boundary residuals, AD derivatives, reduced/coupled timestep agreement
   where elimination is regular, and deliberately invalid/no-local-root cases.

The original benchmark used a read-only closure adapter to study the question
without editing `src`. The current adapter has since migrated to public
physical-residual and solver-metadata interfaces, as described in the theory
tab. This compatibility migration did not rerun the 100-sample benchmark.
A production formulation selector around a canonical physical boundary residual
remains a recommendation; it is not implemented by this reporting change.

I would not initially add per-boundary mixed strategies, automatic mode changes
halfway through a solve, or a broad `auto` selector. They multiply packing,
initialization, compilation, testing and provenance cases. An automatic fallback
would require a deliberate restart with a new state layout and recorded extra
cost; it should not be a hidden exception handler.

Supporting two named strategies is reasonable. Maintaining two independent
copies of the boundary physics, with different clipping, closure equations or
EOS assumptions, is what would make the structure messy and the comparison
untrustworthy. Our current exact-constraint versus three-update Euler
experiment illustrates why that distinction must be explicit.

## Decision for this repository

- **Current EulerNozzle:** keep eliminated BCs as the production default;
  retain the coupled experimental/reference path.
- **New difficult two-fluid inlet:** formulate and validate the physical BC
  residual first; use coupled assembly as the initial reference if local
  inversion is unreliable. Do not declare it superior before robustness tests.
- **Shared infrastructure:** support algebraic BC rows and optional elimination
  through one strategy interface; do not put model-specific BC laws in a generic
  shared folder.
- **Production options later:** one explicit formulation selector is useful;
  introduce it only with common equations/diagnostics and tests, not duplicate
  solver implementations or silent fallbacks.
