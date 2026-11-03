# Newton steps and trial acceptance

For an implicit timestep, Newton solves the scaled square residual $H(z)=0$.
The previous accepted state provides the initial guess. The residual includes
the backward-Euler transport balances, local constraints and either coupled
boundary equations or their converged nonlinear elimination.

## One correction

At iteration $k$,

$$
J_k=\frac{\partial H}{\partial z}(z_k),\qquad
J_k\Delta z_k=-H(z_k),\qquad
z_{k+1}=z_k+\alpha_k\Delta z_k.
$$

The Jacobian is recomputed at the current point. Numerical fluxes and closures
are differentiated as selected; the generic engine has no Euler-specific
variable identities or EOS branches.

`newton_backtracking=False` is the default: try the full correction
$\alpha_k=1$. It still must give a finite, physically admissible candidate.
A failed full trial rejects the timestep; it does not clip pressure, enthalpy,
phase fractions or reconstructed states into a valid range.

With `newton_backtracking=True`, the initial full trial is followed by up to
`max_backtracks` halvings. Let $\nu(z)=\sqrt{\operatorname{mean}(H(z)^2)}$.
A finite admissible candidate must satisfy

$$
\nu(z_k+\alpha\Delta z_k)\leq(1-10^{-4}\alpha)\nu(z_k),
$$

or already meet the configured residual tolerance. The implementation uses
this Armijo condition on the residual RMS. If no trial passes, it retains the
preceding admissible iterate, reports failure and lets the outer controller
reduce the timestep.

## Inner convergence

The correction measure uses fixed model variable scales:

$$
\eta_k=\max_j\frac{|\alpha_k\Delta z_{k,j}|}{z_{*,j}}.
$$

By default both criteria are active:

| Setting | Default | Meaning |
|---|---:|---|
| `newton_iterations` | 10 | Maximum corrections per inner solve |
| `newton_update_tolerance` | $10^{-10}$ | Maximum scaled correction |
| `newton_residual_tolerance` | $10^{-12}$ | Scaled residual RMS |
| `max_backtracks` | 20 | Additional trial halvings when backtracking is enabled |

Every active criterion must pass. Setting one tolerance to zero disables that
criterion; at least one must remain enabled. An initial residual already below
the residual threshold can return without a correction. Reaching the iteration
cap with unfinished criteria is a failure, not a successful approximate root.

`NewtonInfo` records iteration count, update size, residual norm, convergence,
admissibility, rejected trials and status. A rejected Newton line-search trial
is distinct from a rejected timestep and from an accepted pseudo-time update.

## Outer acceptance and failure

After the inner solve, the driver checks its convergence/admissibility status
and evaluates the resulting physical state. Failure retains the last accepted
outer state and reduces CFL before retrying. Acceptance advances the accepted
step count and stores diagnostics. It need not make the steady residual
monotonically smaller: steady convergence is assessed independently using the
original spatial equations.

Backward Euler at a large pseudo-time step approaches a Newton-like steady
correction, while its finite storage term can regularize difficult early
iterations. This motivates the [CFL strategy](convergence.md). It does not justify
relaxing inner criteria or equating a small update with a solved physical balance.

See {mod}`nozzle_hyperbolic.solvers.time_marching.implicit` for the shared
Newton algorithm and {class}`~nozzle_hyperbolic.solvers.fvm.config.Scheme`
for validated public settings.
