# Numerical solver

A solve follows a small number of steps. The physical model first defines
storage, fluxes, sources, algebraic constraints and boundary equations on a fixed
mesh. `Scheme` selects spatial and temporal methods. `Operator` validates the
combination and binds JAX kernels. A driver then advances the state, accepts or
rejects trials and checks the requested outcome.

1. Construct geometry, fluid properties and a physical initial field.
2. Complete or couple the endpoint traces, depending on the integrator.
3. Reconstruct left/right states at each face and evaluate the numerical flux.
4. Assemble flux divergence, nonconservative work and integrated sources.
5. Advance conserved contents explicitly or solve an implicit step by Newton.
6. Check admissibility, inner convergence and the original physical residuals.
7. Adjust CFL for steady pseudo-time, or advance physical time to the next output.
8. Optionally transfer a successful steady solution to a finer grid or new
   explicitly specified physical parameter stage.

These chapters follow that dependency order. The engine is independent of the
number and identities of equations. Two-fluid constraints and model-specific
nonconservative work enter through declared callbacks.

```{toctree}
:maxdepth: 2

spatial_operator
reconstruction
fluxes
boundaries
time_integration
jacobians
newton
convergence
acceleration
```
