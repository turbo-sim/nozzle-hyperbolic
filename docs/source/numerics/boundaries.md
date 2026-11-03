# Algebraic boundary solution

The physical models define one canonical boundary residual $B(W,t)=0$.
Reservoir data, acoustic compatibility, pressure conditions and phase relations
belong to those equations. `Scheme.boundary_treatment` chooses how they are
solved. Changing the numerical treatment does not define a new physical law.

| Integrator | `auto` resolves to | `coupled` | `segregated` |
|---|---|---|---|
| Backward Euler | Coupled | Supported | Supported |
| Forward Euler | Segregated | Rejected | Required |
| SSPRK3 | Segregated | Rejected | Required |
| Classical RK4 | Segregated | Rejected | Required |

## Coupled backward Euler

Endpoint traces and cells are unknowns in one Newton vector. Each endpoint
contributes $V$ scaled physical rows; each cell contributes $T$ time-discrete
transport rows and $C$ local constraints. Boundary rows have zero storage and
are never multiplied by the timestep or assigned a cell volume.

All stencil dependencies, including the first/last two-cell extrapolation,
enter the augmented Jacobian. Its block-major layout is

$$
z=(W_L,W_1,\ldots,W_N,W_R).
$$

Trial states are evaluated directly. The physical boundary callback does not
overwrite a trace or run an internal projection during residual evaluation.

## Segregated traces and implicit elimination

For a given cell state $c$, a nonlinear completion solve finds the endpoint
values $b=b(c,t)$ satisfying $B(b,c,t)=0$. Failed or inadmissible completions
make the outer trial fail. This is also how explicit stages obtain their
physical boundary states.

In segregated backward Euler the converged roots are eliminated, leaving a
cell-only residual $H(c,b(c))$. Implicit differentiation supplies

$$
\frac{db}{dc}=-B_b^{-1}B_c,\qquad
\frac{dH}{dc}=H_c-H_bB_b^{-1}B_c.
$$

The eliminated Jacobian therefore includes boundary sensitivity. Treating
completed traces as constant would miss this term and change the Newton
linearization. A locally invertible boundary Jacobian is required; root
completion and the original full boundary residual are checked separately.

## Explicit stages

An explicit method advances conserved cell contents. It then recovers the
iteration state, including any EOS constraints, and completes traces at the
appropriate stage time before the next flux evaluation. Traces are algebraic
values, so averaging them with Runge–Kutta weights would generally violate
their nonlinear equations. The final state is completed again as needed.

Boundary conditions can affect physical-time flux budgets and stability.
The operator includes model-supplied endpoint speed bounds in timestep
selection. Some Euler wave/sonic branches also supply a physical endpoint-flux
override so a second numerical Riemann split does not change the chosen trace law.

See {mod}`nozzle_hyperbolic.solvers.boundaries` and
{mod}`nozzle_hyperbolic.solvers.fvm.elimination` for completion and elimination,
and the three [physical-model pages](../theory/index.md) for the equations.
