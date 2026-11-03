# Pseudo-time and dual-time stepping

Physical transient integration advances every cell through the same physical
time. The global timestep respects the most restrictive cell and any source
stability limits. Runge–Kutta stages must recompute properties, sources,
constrained-cell recovery and physical boundary traces at their stage states
and times. The shared solver provides this route; classical RK4 is selected by
`Scheme(time_integrator="runge-kutta-4")` and is not SSP.

Local pseudo-time stepping assigns each cell a different artificial increment,
usually proportional to its cell width divided by its local wave speed. This
can accelerate the solution of a steady residual $R(W)=0$ when a small region
otherwise limits the global step. It removes any interpretation of the iterates
as a common physical transient. Stiff sources, strong cell coupling, shocks and
boundary changes can still limit convergence; larger local steps are not a
guarantee of lower total cost. The final convergence test must use the original
spatial residual with fixed physical scales, including cell and boundary
algebraic equations. Pseudo-time weighting must not change the target root or
provide storage for algebraic constraints.

Dual-time stepping introduces an inner pseudo-time solve for each common
physical timestep. For backward Euler, the inner target is

$$
\frac{V_i[Q(W_i^{n+1})-Q(W_i^n)]}{\Delta t}+R_i(W^{n+1})=0,
\qquad g_i(W_i^{n+1})=0,\quad B(W^{n+1},t^{n+1})=0.
$$

Local pseudo-time increments can accelerate this inner nonlinear solve while
the converged outer solution retains the physical timestep. Benefits include
reuse of steady solvers and potential robustness for stiff flow evolution.
Costs include an additional convergence loop and control parameters. An
insufficiently converged inner solve introduces physical-time errors and can
spoil temporal order and conservation. Inner work, rejected attempts and
physical-time error must be reported separately.

Neither local pseudo-time nor dual-time stepping is implemented by this port.
Andrea's per-cell stepping and diagonal update advertised as implicit were not
adopted. The existing nonlinear backward-Euler solver remains the implicit
method. This note records possible future methods, not available settings.
