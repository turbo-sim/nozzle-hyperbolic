# Research backlog

These are unresolved investigations, not a release plan or supported API.
They preserve useful questions from the retired roadmap scratchpad and dated
source audits. Each needs a defined physical problem, numerical scope and
validation before implementation.

## Nonlinear steady-state control

Assess [conditional CFL adaptation and bounded Newton extensions](adaptive_cfl_and_newton.md)
with fixed physical equations, initialization and final convergence criteria.
Separate cap exhaustion from invalid trials and record rejected work. This is
distinct from accepting unconverged backward-Euler steps.

## HEM field inversion and optimization

- Define a homogeneous-equilibrium-model field-inversion problem and assess a
  simultaneous state/parameter formulation with SLSQP.
- Compare it with a reduced or adjoint-based formulation on the same objective,
  constraints, discretization and physical branch.
- Define the intended sensitivity first: a residual JVP, an accepted timestep
  derivative and a converged-solution derivative have different requirements.
  Establish derivative correctness before drawing optimization conclusions.

## Nonlinear solver interfaces and sensitivities

Investigate initial-step/radius control, infeasible-trial handling and the
derivative contract of any proposed Optimistix/custom-solver interface.
A safeguarded solve followed by a local differentiable solve is a candidate to
test, not an established strategy. Distinguish implicit differentiation of a
converged root from differentiating through an iteration history, especially
with dynamic loops, branch switches and clipped properties.

## Deferred two-fluid review

Reassess the original [repository](../archive/alberto_repository_audit.md) and
[JAX](../archive/jax_implementation_audit.md) findings against the intended port:

- Physical balances versus nonlinear unknowns, nonconservative coupling,
  hyperbolicity and boundary well-posedness.
- EOS recovery, invalid-state reporting and consistency of derivatives through
  phase bounds, clipping and property tables.
- Global limiter/filter dependencies, stopped gradients and declared Jacobian
  stencils; compare directional finite differences where relevant.
- Mass-transfer metastability and closure-domain assumptions. Preserve existing
  expected-failure evidence until a physical-model decision is made.
- Physical-time stage consistency, conservation budgets and explicit stiffness
  restrictions, independently of steady pseudo-time convergence.

The original audits are evidence for specific historical source, not proof that
all of these defects apply to the current Euler engine. A future two-fluid port
must establish its own capability and verification record.
