# Can multigrid accelerate the annular Euler solver?

Design assessment and implementation plan for **EulerAnnular**, 2026-10-28.
This remains a proposal for that model. The separate
[EulerNozzle grid-sequencing demo](README.md) tests coarse-to-fine initialization;
its measurements do not establish FAS performance or an annular speedup.

## Assessment

**Yes, this is worth testing, particularly for cold starts with a shock far
from its final position.** A coarse mesh can establish the large-scale pressure,
mass-flow and shock-location pattern cheaply; the fine mesh can then recover
the final shock profile. The best first experiment is coarse-to-fine
initialization. Repeated nonlinear coarse corrections are the next step if
initialization alone leaves a substantial convergence plateau.

The distinction matters here: every current implicit update already uses a
global, colored-Jacobian, banded direct Newton solve. Multigrid is therefore
unlikely to help primarily by making an individual linear solve cheaper at
400 cells. The target is the thousands of nonlinear pseudo-time updates,
retries and line-search evaluations needed to settle the flow. Linear algebraic
multigrid as a replacement for the banded solver is not the first priority.

### What the existing results support

The [CFL study](../historical/euler_annular/cfl_investigation/REPORT.md) provides the
following evidence for shock35 at 400 cells:

- With initial CFL 0.2, accepted-step growth 1.2, failure shrink 0.5 and a cap
  of eight Newton corrections, constant reconstruction takes 3,488 accepted
  updates and MUSCL takes 4,933. Both meet the original steady targets.
- In the separately labeled five-correction, residual-verified MUSCL run,
  the compression-gradient proxy moves from approximately 0.100 m initially
  to 0.078 m at update 1,000, 0.053 m at update 3,000 and finally 0.035 m.
  This is an observed trajectory for that run, not a trajectory inferred for
  every solver variant. The early proxy need not identify a unique shock.
- Some converged trajectories retain different local shock-cell profiles.
  A smaller residual and the same approximate shock station do not by
  themselves establish agreement with the same discrete root.

These observations support the hypothesis that coarse meshes could reduce
the cost of moving the compression front. They do not establish how much of
the plateau multigrid can remove or a numerical speedup factor.

### Why a coarse mesh could help

The current timestep is

$$
\Delta\tau=C\min_i\frac{\Delta m_i}{|v_{m,i}|+a_i}.
$$

At comparable CFL and wave speeds, doubling cell width approximately doubles
the dimensional pseudo-timestep. There are also fewer cells to evaluate and
differentiate. A compression front can cross the domain in fewer cell-scale
updates. Coarse corrections can communicate a global adjustment without
requiring the fine-grid iteration to transport it entirely through small steps.

This is a heuristic for the current pseudo-time controller, not a propagation
limit for implicit integration: a global implicit solve can couple distant
cells, and permitted CFL, nonlinear damping and shock strength can differ
between levels. A shock-location error is also nonlinear and nonsmooth; the
usual smooth-error argument for elliptic multigrid is not a proof for this flow.

Finite-volume Euler multigrid with agglomerated control volumes has precedent
in [NASA's unstructured Euler multigrid work](https://ntrs.nasa.gov/citations/19940017011).
The [NASA FAS study by Nishikawa, Diskin and Thomas](https://fun3d.larc.nasa.gov/papers/AIAA-2010-4731.pdf)
describes recursively transferring states and defects, solving coarse problems
and prolonging corrections. These support the algorithmic direction; their
performance results do not predict a speedup for this 1D solver.

## Three approaches and the recommended order

| Approach | Operation | Role in this repository |
|---|---|---|
| Coarse-to-fine initialization, also called grid sequencing | Solve on a small mesh, interpolate to the next mesh and continue; no return to a coarser mesh | Implement first; smallest change and a direct test of the shock-location hypothesis |
| Nonlinear FAS multigrid | Repeatedly restrict the current state and residual, solve a defect-forced coarse problem, prolong its correction and smooth | Implement second; retains the fine-grid target through a consistency correction |
| Linear multigrid inside Newton | Approximately solve each linearized correction on a hierarchy | Defer; the present narrow-band direct solver already avoids a large iterative linear bottleneck |

Grid sequencing is not itself a recurring multigrid cycle. It may capture most
of the benefit if the difficulty is predominantly the initial shock transit.
FAS becomes attractive if global errors reappear during fine-grid iteration.
Neither approach refines the mesh adaptively or improves the accuracy of the
chosen fine-grid discretization.

## Equations and fine-grid target

Keep the current variables and general-EOS physics:

$$
W=(p,v_m,\ell,h),\qquad \ell=r v_\theta,
\qquad q=(\rho,\rho v_m,\rho\ell,\rho E),
$$

$$
E=h-\frac{p}{\rho}
 +\frac12\left(v_m^2+v_\theta^2\right).
$$

The physical-cell integrated spatial residual and storage are

$$
R_i(W)=\widehat F_{i+1/2}(W)-\widehat F_{i-1/2}(W)-S_i(W),
\qquad Q_i(W)=V_i q_i(W).
$$

Fluxes already include face area and $S_i$ is the integrated geometric source.
The complete steady problem includes the two four-variable boundary traces:

$$
\mathcal A_h(W_h)=
\begin{bmatrix}
B_{h,\mathrm{in}}(W_h)\\
R_h(W_h)\\
B_{h,\mathrm{out}}(W_h)
\end{bmatrix}=0.
$$

For transfer algebra, use **raw integrated physical rows**, with boundary rows
in their existing dimensionless normalization. This mixed block representation
is intentional: its restrictions are defined separately below. Apply each
level's established row scales when assembling Newton's equations and merit
functions. Do not sum the existing volume-normalized spatial rows as though
they were integrated balances.

All flux, source, boundary, storage and thermodynamic functions remain the
existing ones. Numerical methods must use EOS sound speed and Gruneisen as
needed, with no $c_p/c_v$ closure. Perfect-gas analytical solutions remain
verification references only. Every final solution must satisfy the original
**unforced fine-grid** equations with its requested constant or MUSCL
reconstruction and selected inlet/outlet regimes.

## Mesh hierarchy from the same NURBSpy geometry

Start with $400\rightarrow200\rightarrow100\rightarrow50$ physical cells
for restriction, traversing the reverse order during initialization. All levels
cover the same meridional arclength and use the same NURBSpy geometry object,
physical endpoints, fluid object and physical boundary data.

Build coarse faces by selecting every second fine face and retaining the last
endpoint. For arbitrary cell counts, pair adjacent cells and permit an unpaired
final cell. Record explicit parent/child maps instead of assuming equal volumes.
Keep important geometric breaks or throat stations as shared faces when these
are already represented in the fine mesh; stop coarsening across an essential
feature if its representation becomes inadequate.

Use `make_annular_mesh(geometry, m_faces=...)` for centers, face metrics and
source geometry. Verify that each coarse volume equals the sum of its children
to the geometry quadrature tolerance. If exact agglomerated volumes are needed,
construct the experimental mesh adapter with those sums and retain the
independently evaluated geometry metrics. Do not average radii, fit a new
spline, average primitive states to create geometry, or replace the integrated
volumes by midpoint area times width.

FAS allows rediscretized coarse sources to differ from summed fine sources;
the defect correction accounts for this difference. The pressure-source
construction using face-area differences must remain intact on every mesh.
Stop at 50 cells initially; test 25 only after the shock and geometry are still
adequately represented. No analytical shock station may be used to place faces.

## State and defect transfer

### First prototype: transfer primitive states explicitly

For coarse-to-fine initialization, use limited interpolation in physical
arclength of $(p,v_m,\ell,h)$, with piecewise-constant injection as the fallback
near a shock. Transfer $\ell$, deriving $v_\theta=\ell/r$ at the destination
radius. Copy the two physical boundary trace guesses at the unchanged endpoints.
These are initial guesses, not prescribed traces; the next coupled Newton solve
updates them along with the cells.

For the first FAS prototype, restrict physical primitive states with
volume weights:

$$
(I_h^H W_h)_I=\frac{\sum_{i\in I}V_i W_i}{V_I}.
$$

Inject each boundary trace without volume weighting. Prolong primitive
**corrections** by constant injection to children; consider limited linear
prolongation only after this robust version works. Explicit primitive transfer
avoids requiring an EOS inverse that the current solver does not otherwise need.
It is not a conservative state remap. Record the change in integrated contents
caused by transfer, and rely on the original residual audit for final steady
conservation. This compromise is acceptable for a first steady convergence
experiment, but is not an accuracy-preserving physical-time remap.

### Conservative transfer: a later controlled variant

A conservative alternative restricts $q$ by volume average and prolongs $q$
with child corrections whose volume-weighted sum matches the parent correction.
Density, meridional momentum, angular momentum and total energy must all use
the same transfer weights. Transfer $\rho\ell$, not $\rho v_\theta$, because
the latter is not the transported angular-momentum density in this model.

This needs recovery of $W$ at each destination radius. The existing
`conservative_to_primitive` supports density/internal-energy EOS recovery and
a checked host HP inversion fallback. That fallback is **not JIT-compatible**.
Do not assume that every JAXProp object supports a traceable density/energy pair,
or silently substitute a perfect-gas inversion. Either keep this variant on a
clearly measured host transfer path, require an explicitly supported traceable
EOS pair for that variant, or design a separate coupled HP transfer system.
Primitive transfer remains the general HP-backend prototype.

Neither primitive nor conservative averaging guarantees admissibility for
every real-gas domain. Check destination states and reconstructed face states;
fall back to an admissible child primitive state for restriction if necessary,
and damp or reject invalid prolongations. In FAS the actual fallback restriction
must be used consistently when forming the coarse forcing. Never clip pressure,
enthalpy or density into a valid range and call the result converged.

### Restrict residuals differently from states

For raw integrated physical residuals,

$$
(\mathcal R_h^H R_h)_I=\sum_{i\in I}R_{h,i}.
$$

Inject the two boundary residual blocks directly. This preserves telescoping
of internal face fluxes within a parent. Averaging an integrated defect would
introduce a factor-of-two error under pairwise coarsening. If code receives
scaled physical residuals, first undo their level-specific scales.

## Nonlinear FAS formulation

On a level, write the current forced equation as
$\mathcal A_h(W_h)=f_h$, with $f_h=0$ at the finest level. After pre-smoothing,
compute

$$
W_H^0=I_h^H W_h,\qquad
f_H=\mathcal A_H(W_H^0)
 +\mathcal R_h^H\left(f_h-\mathcal A_h(W_h)\right).
$$

Approximately solve $\mathcal A_H(W_H)=f_H$, then update

$$
W_h^{\mathrm{trial}}=W_h+alpha P_H^h(W_H-W_H^0),
\qquad 0<\alpha\le1.
$$

Post-smooth on the fine level. Restriction of a solved fine state makes
$W_H^0$ solve the forced coarse equation, so the correction should be zero
to numerical tolerance. This fixed-point property is essential; it also permits
constant reconstruction on coarse levels while retaining MUSCL on the target
fine mesh. Solving the unforced coarse problem on every cycle would generally
pull the fine solution toward a different discrete problem.

This formulation follows the nonlinear correction principle documented in
[PETSc's FAS reference](https://petsc.org/release/manualpages/SNESFAS/SNESFAS/).
PETSc is a reference here, not a proposed new dependency.

### Coupled boundary equations on all levels

Every level retains its independent inlet and outlet traces, four equations
per boundary, zero boundary storage, and the same physical boundary data and
selected regimes. There are no nested nonlinear boundary solves.

The coarse FAS forcing includes its boundary components. Coarse boundary
stencils depend on their adjacent coarse cells, so injecting the trace alone
does not in general make the coarse boundary residual equal to the fine one.
Artificial boundary forcing is a consistency correction inside the cycle;
the finest level always has zero forcing. Do not freeze coarse boundary traces,
overwrite them after prolongation, or omit boundary defects from restriction.

### Reuse implicit Newton as the smoother

For each smoothing pseudo-step, solve

$$
Q_H(W_H^{k+1})-Q_H(W_H^k)
 +\Delta\tau_H\left[R_H(W_H^{k+1})-f_{H,R}\right]=0,
$$

$$
B_H(W_H^{k+1})-f_{H,B}=0.
$$

The FAS forcing is fixed during that level visit and is not differentiated as
a function of the trial state. Its physical part receives the same timestep
factor as $R_H$; its boundary part receives no timestep or storage factor.
Use existing content scales for physical implicit rows and existing boundary
scales for algebraic rows. A nonzero forcing changes the residual, but not
the stencil width or its Jacobian sparsity.

The current `solver.step(state, dt)` only solves an unforced problem. It cannot
be used unchanged for coarse FAS smoothing. Build an experimental wrapper around
the public `storage`, `spatial_residual` and `boundary_residual` callbacks and
reuse `make_coupled_residual`, `make_block_operators` and `make_newton_solver`.
No flux, boundary law, source or Newton algorithm needs to be copied. Ordinary
grid sequencing can use the existing unforced timestep directly.

## Cycle, controls and shock safeguards

Start with two levels, 400 and 200 cells, and a V(1,1) cycle: one accepted
implicit pseudo-step before and one after coarse correction. Each pseudo-step
has a maximum of eight Newton corrections, not one correction; retain both
inner tolerances at $10^{-10}$. A coarse visit initially permits four accepted
pseudo-steps, then returns its approximate correction without claiming a
converged coarse root. Record exhausted or failed steps separately.

For a multilevel cycle, recursively apply the same V-cycle down to 50 cells.
Begin with independent level CFL controllers: initial CFL 0.2, growth 1.2
after each accepted smoothing step, shrink 0.5 on failure, and cap $10^6$.
Compute dimensional timesteps from each level's own widths and EOS wave speeds.
These are provisional settings borrowed from the single-grid study, not
validated multigrid tuning. Do not inherit the fine mesh's dimensional timestep
on every level or count a rejected attempt as smoothing work completed.

Use constant reconstruction on coarse levels first; retain the requested
constant or MUSCL reconstruction on the fine level. Limit coarse work rather
than converging each intermediate level to $10^{-8}$ during every visit.
Coarse stages may be stopped by a defect reduction or work budget; only the
unforced fine-grid root determines final success.

Safeguard the prolongated correction with trials
$\alpha=1,1/2,1/4,\ldots$ (initially at most eight trials). Check EOS validity,
physical and reconstructed-face admissibility and the intended boundary
regimes. Initially require decrease in the level's fixed-scale forced residual
merit at the correction stage. If no correction passes, retain the pre-smoothed
state and perform ordinary smoothing; record the rejected coarse visit.
This conservative guard may reject useful shock translations and is itself
an experimental choice. If it blocks progress, test a separately labeled
bounded nonmonotone guard, judged after post-smoothing, instead of silently
weakening the acceptance rule.

The whole residual history need not become monotone: ordinary pseudo-time
smoothing can still increase the steady defect. Plot compression location,
boundary defects and conservation as well as residual RMS. A shock can be
broader and at a different cell position on a coarse mesh; unchecked linear
interpolation may produce oscillations or inadmissible states on prolongation.
Repeatedly rejected corrections are a reason to reduce hierarchy depth or
change transfer, not evidence that more cycles will eventually help.

## Implementation stages and reuse map

| Stage | Proposed work in this study | Completion gate |
|---|---|---|
| 1: hierarchy and initialization | `scripts/hierarchy.py`, `scripts/transfers.py`, `scripts/run_grid_sequence.py`; 50 to 100 to 200 to 400 cells | Same cold-start physical setup, valid transferred states, original fine-grid root reached; include all coarse work |
| 2: two-level FAS | `scripts/forced_level.py`, `scripts/fas.py`; 400/200 first, constant then MUSCL target | Zero-defect invariance, correct boundary forcing and no state clipping; matched fine-grid root |
| 3: multilevel | Recurse 400/200/100/50; compare V(1,1) and V(2,2) only if useful | Improvement survives total-work accounting and repeated timing |
| 4: extension | Shock70, smooth subsonic, transonic and supported real-gas cases; deeper hierarchy or conservative transfer as separate variants | No loss of fine-grid conservation, branch correctness or general-EOS support |

Existing modules to import:

- [Annular FVM factory](../../../src/nozzle_hyperbolic/models/euler_annular/fvm.py):
  `make_annular_solver`, `AnnularOptions`, original operators, timestep,
  admissibility and metrics.
- [NURBSpy geometry and mesh](../../../src/nozzle_hyperbolic/geometry/annular.py):
  `make_annular_mesh` and the existing geometry object.
- [Coupled residual assembly](../../../src/nozzle_hyperbolic/solvers/finite_volume.py),
  [Newton solver](../../../src/nozzle_hyperbolic/solvers/time_marching/implicit.py)
  and [colored block operators](../../../src/nozzle_hyperbolic/numerics/jacobians.py):
  the forced coarse smoother uses these unchanged building blocks.
- [State maps](../../../src/nozzle_hyperbolic/models/euler_annular/state.py):
  EOS evaluation, $q$ construction and the optional conservative recovery path.
- [Verification case factory](../historical/euler_annular/scripts/verify_annular.py):
  reuse `make_case` and the same boundary data, reference and cold-start rule.
  Analytical profiles must never initialize the numerical hierarchy.

All proposed code stays in this study initially. Promote reusable hierarchy,
transfer or forcing interfaces into `src/` only after evidence supports the
method and a separate production change is requested. Do not create a second
annular solver or new thermodynamic implementation.

## Verification and performance experiment

First add executable study audits in `scripts/audit_multigrid.py`, rather than
development-specific pytest tests in the production test suite:

1. Verify nested faces, positive volumes, volume sums, physical endpoints and
   the sum restriction of integrated defects. Check internal flux cancellation
   on a nonuniform hierarchy, not just equal-width cells.
2. Verify transfer of a constant primitive state and the identity
   $I_h^H P_H^h e_H=e_H$ for constant-injection corrections. For a conservative
   variant, separately audit all four integrated contents and EOS recovery.
3. At an arbitrary admissible state, construct synthetic right-hand side
   $f_h=\mathcal A_h(W_h)$. A complete forced cycle must leave that state
   unchanged to inner tolerance, including both boundary traces. This tests
   the zero-defect property without needing an analytical solution.
4. Verify the forced Jacobian against AD/directional checks and its expected
   band structure. Include nonzero boundary forcing and geometrically varying
   source terms. Re-audit a saved converged unforced fine-grid state after a
   cycle; an already solved root must not drift significantly.
5. Exercise strong-shock correction rejection and an out-of-domain EOS trial.
   Preserve the previous valid state and record failure without clipping or
   silently switching EOS. Test one HP-compatible real-gas backend within its
   declared domain before claiming general-EOS transfer support.

Begin with shock35, 400 cells, both reconstructions, comparing:

- Single-grid cold solve with the CFL study's eight-correction settings.
- Grid sequencing from the same physical cold guess on 50 cells.
- Two-level FAS starting cold on the fine mesh, without grid sequencing.
- Grid sequencing plus FAS, only after their separate effects are understood.

Use constant reconstruction on all levels for the first constant-target
experiment; for the MUSCL target, first use constant coarse levels and MUSCL
on the target mesh. Declare these differences. Compare coarse-stage stopping
budgets rather than requiring unnecessarily tight coarse initialization roots.
Then repeat the useful variants for shock70 and a smooth subsonic control.

Final success requires all original fine-grid spatial RMS, boundary maximum
and integrated balance defects below $10^{-8}$, admissibility and intended
boundary regimes. Check signed flow and final pressure, Mach, $C_p$, mass flux,
angular-momentum flux and stagnation enthalpy against the single-grid solution.
Initially retain the CFL study's fixed-scale primitive agreement threshold
$10^{-5}$ and shock-location tolerance of two fine cells; flag other roots
separately and inspect shock cells. Analytical verification remains a separate
accuracy assessment, with analytical, constant and MUSCL curves and numerical
markers, rather than being replaced by agreement with another numerical solve.

Record every level visit, accepted/rejected pseudo-step, Newton correction,
backtrack, correction damping factor, forcing norm, actual CFL, dimensional
pseudo-time and compression proxy. Never combine pseudo-time accumulated on
different meshes into a purported physical evolution. Plot fine-grid residual
against both elapsed time and cumulative work; coarse steps must count too.

For performance claims, follow the repository protocol: one excluded complete
solve and ten measured solves per configuration, identical cold-start rules,
sequential alternating method order, JAX synchronization and compilation reuse.
Include hierarchy transfers, EOS calls, coarse solves, rejected corrections,
monitoring and final fine-grid polishing within the solve timing. Report setup
and compilation separately; multiple mesh shapes entail additional compiled
kernels. Save source fingerprints and every failed sample. Compare elapsed
time, not only cycle counts: fewer fine updates can still cost more overall.

## Decision after the first experiment

If grid sequencing removes most of the plateau, keep that simpler approach
unless repeated FAS corrections deliver a further measured advantage. If FAS
frequently rejects shock corrections, first examine transfer and coarse-grid
shock representation. If neither approach reduces total time to the same
fine-grid root, retain the single-grid banded solver and report the negative
result. The immediate proposal is therefore **grid sequencing first, nonlinear
FAS second**, with shock movement as the motivating hypothesis and original
fine-grid convergence as the non-negotiable final test.
