# Time integration

Every integrator advances **conserved cell contents**
$\mathcal Q_i=V_iQ(W_i)$ on a fixed mesh. Iteration variables may be primitive
or constrained, so advancing $W$ directly with the same Runge–Kutta arithmetic
would generally solve a different discrete problem. Write
$L(\mathcal Q,t)=-R(W(\mathcal Q,t))$, with physical boundary completion
included in each stage evaluation.

## Forward Euler

`time_integrator="euler-explicit"` computes

$$
\mathcal Q^{n+1}=\mathcal Q^n-\Delta t R(W^n).
$$

The solver recovers cell states from these target contents and completes the
new boundary traces. It is first order in physical time. Source, filter and
wave restrictions all enter its timestep selection.

## SSPRK3

`time_integrator="runge-kutta-3"` uses the third-order SSP construction

$$
\mathcal Q^{(1)}=\mathcal Q^n+\Delta tL(\mathcal Q^n,t_n),
$$

$$
\mathcal Q^{(2)}=\tfrac34\mathcal Q^n+
\tfrac14[\mathcal Q^{(1)}+\Delta tL(\mathcal Q^{(1)},t_n+\Delta t)],
$$

$$
\mathcal Q^{n+1}=\tfrac13\mathcal Q^n+
\tfrac23[\mathcal Q^{(2)}+\Delta tL(\mathcal Q^{(2)},t_n+\tfrac12\Delta t)].
$$

Each stage requires admissible content recovery and boundary completion.
The SSP property inherits an appropriate forward-Euler stability property
under its assumptions; it does not establish that property for arbitrary
coupled closures or invalid reconstructed states.

## Classical RK4

`time_integrator="runge-kutta-4"` evaluates

$$
k_1=L(\mathcal Q^n,t_n),\quad
k_2=L(\mathcal Q^n+\tfrac12\Delta t k_1,t_n+\tfrac12\Delta t),
$$

$$
k_3=L(\mathcal Q^n+\tfrac12\Delta t k_2,t_n+\tfrac12\Delta t),\quad
k_4=L(\mathcal Q^n+\Delta t k_3,t_n+\Delta t),
$$

$$
\mathcal Q^{n+1}=\mathcal Q^n+
\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
$$

It is fourth order in time for sufficiently smooth problems with sufficiently
accurate algebraic solves. It is not an SSP method in the same sense as SSPRK3.
The implementation also accumulates boundary/source transfers with the matching
stage weights for the physical-time budget.

## Recovery for nonlinear or constrained storage

At each explicit stage, the generic cell solve finds $W_i$ satisfying

$$
V_iQ(W_i)=\mathcal Q_{i,\mathrm{target}},\qquad g(W_i)=0.
$$

This is a square local system even when the storage map alone is rectangular.
For two-fluid flow there are six target contents and two EOS rows for eight
unknowns. Constraints are solved at every stage, not allowed to drift and
projected only at output times. A failed recovery rejects the attempted step.

## Backward Euler

`time_integrator="euler-implicit"` solves

$$
H_i(W^{n+1})=
\mathcal Q_i(W_i^{n+1})-\mathcal Q_i(W_i^n)
+\Delta tR_i(W^{n+1})=0,
$$

together with $g(W_i^{n+1})=0$ and $B(W^{n+1},t_{n+1})=0$.
It is first order in physical time. Newton solves the nonlinear step with
either coupled or eliminated traces. Implicit integration removes the explicit
wave-step restriction as an algorithmic requirement, but a large timestep can
still fail nonlinear convergence or lose physical transient accuracy.

Transport rows are divided by fixed content scales and by
$\max(1,\Delta t/t_*)$ to limit roundoff amplification at large pseudo-time
steps. This uniform transport-row scaling preserves their zero set. Cell
constraints and boundary equations keep their independent fixed scaling and
zero storage.

## Physical time versus steady pseudo-time

`solve_transient(model, output_times, scheme, ...)` starts at time zero,
shortens steps to hit each requested output, retains rejected attempts without
advancing time and returns results at the attempted outputs. Time accuracy and
the integrated conservation budget matter; a small steady spatial residual is
not a transient stopping condition.

`solve_single_grid` instead uses repeated time steps to find $R(W)=0$.
Its pseudo-time and CFL growth accelerate convergence and are not a physical
trajectory; the reported `Solution.time` remains zero. Boundary callbacks are
evaluated at steady time zero. Inner Newton convergence proves only that the
current discrete step was solved, so the driver separately checks the original
steady spatial residual.

The production code implements global pseudo-time stepping. Local cell time
steps, dual-time physical integration and FAS multigrid are not selectable
methods in the current solver. The implemented acceleration is described in
[grid sequencing and reuse](acceleration.md).

Implementations: {mod}`nozzle_hyperbolic.solvers.time_marching.explicit`,
{mod}`nozzle_hyperbolic.solvers.time_marching.implicit` and
{mod}`nozzle_hyperbolic.solvers.fvm.drivers`.
