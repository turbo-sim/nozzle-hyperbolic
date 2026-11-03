# CFL strategy and convergence

The steady driver has two separate decisions: whether a proposed pseudo-time
step is acceptable and whether the original spatial equations have converged.
Increasing CFL changes the former solve; it must not change the definition of
the latter residual.

## Wave and explicit timestep limits

The base wave time is a global minimum

$$
\Delta t_{\rm wave}=\min_i\frac{\Delta x_i}{s_i},
$$

where $s_i$ bounds the model's physical wave speeds. Endpoint speed callbacks
also bound the end cells. For explicit Roe, corrected Stadtke and Osher, the
operator additionally accounts for their selected interface dissipation speeds,
which can exceed cell-centered physical speeds.

Implicit stepping uses $\Delta t=\mathrm{CFL}\,\Delta t_{\rm wave}$.
For explicit integration, source/filter decay rates combine with wave transport:

$$
\frac1{\Delta t}=
\frac1{\mathrm{CFL}\,\Delta t_{\rm wave}}
+\frac1{\Delta t_{\rm source}}+\frac1{\Delta t_{\rm filter}}.
$$

An inactive restriction is infinity and contributes zero rate. Merely taking
the minimum of separate stable bounds can miss their combined decay rate.
For the generic fourth-difference filter,

$$
\Delta t_{\rm filter}=\min_i\frac{\Delta x_i}
{8\epsilon_f\max(s_{i-1/2}^{\rm filt},s_{i+1/2}^{\rm filt})}.
$$

Model-specific closures may provide an additional source limit. For constrained
explicit states, content recovery and trace completion occur before evaluating
these limits. The history distinguishes requested CFL from actual wave CFL
$\Delta t/\Delta t_{\rm wave}$.

## Accepted-step CFL controller

{class}`~nozzle_hyperbolic.solvers.fvm.config.SteadySettings` defaults are:

| Setting | Default | Role |
|---|---:|---|
| `cfl` | 0.2 | Initial request |
| `cfl_growth` | 1.2 | Multiplier after an accepted step |
| `cfl_shrink` | 0.5 | Multiplier after a rejected step |
| `cfl_max` | $10^6$ | Maximum implicit request |
| `max_steps` | 4000 | Maximum accepted steps |
| `max_retries` | 14 | Additional attempts per step |
| `stagnation_steps` | 12 | Consecutive bitwise-unchanged accepted states before stagnation |

After acceptance the next request is
$\min(\mathrm{CFL}_{\max},1.2\,\mathrm{CFL})$; after rejection it is halved.
Explicit steady integration additionally caps the request at 0.45.
The accepted-state controller is not a residual-based or Newton-iteration-count
adaptation. A failed trial leaves the previous accepted state unchanged.
`stagnation_steps=0` disables that diagnostic.

`solve_transient` has its own requested CFL (default 0.45), output-time clipping
and retry behavior. Steady accepted-step growth does not describe its physical
time accuracy policy.

## Fixed-scale spatial residuals

Let $Q_{*,k}$ be a fixed conserved-density reference and $t_*$ a fixed reference
time. For each transported equation,

$$
\widetilde R_{ik}=\frac{t_*R_{ik}}{V_iQ_{*,k}},\qquad
E_k=\left(\sum_iw_i\widetilde R_{ik}^2\right)^{1/2},\qquad
w_i=\frac{V_i}{\sum_jV_j}.
$$

This is the default volume-weighted L2 norm. Cell weighting and L1 are also
available. Each equation is checked separately, preventing a large-scale
equation from hiding a smaller one. Relative norms compare with that solve's
initial spatial residual, with an explicit policy for the initial floor;
absolute norms retain the fixed physical scaling. Neither uses the current CFL
or the implicit step residual as a normalization.

The default {class}`~nozzle_hyperbolic.solvers.residuals.ResidualPolicy` accepts
each equation when its relative norm is at most $10^{-6}$ **or** its absolute
norm is at most $10^{-9}$. `criterion` can select `both`, `relative` or `absolute`
instead of the default `either`.

Boundary equations, cell constraints and global balance defects have independent
default absolute tolerances of $10^{-9}$. EOS constraints retain fixed intrinsic
density scales; no local void fraction downweights them.

## Global and physical checks

The global defect uses actual endpoint fluxes, integrated sources and
nonconservative contributions, with fixed model flux scales. A local residual
norm can be small while a domain budget is still unacceptable, and cancelling
cell residuals can make a global defect small while local equations remain
unfinished. Both are checked.

Optional engineering monitors use their configured windows and criteria.
A final model audit can also reject the solution, for example when a requested
physical branch has not been recovered. Two-fluid equation pins retain
unmodified-balance diagnostics separately from the selected equations.

Inspect `solution.success`, `reason`, `metrics` and the accepted/rejected
`history`. Useful metric groups include `equation_norm`, `constraint_norm`,
`boundary_equation_max` and `global_defect`. A native optimizer success flag,
small correction or smooth convergence plot alone is not a physical balance
audit. Finite-volume convergence, analytical accuracy and physical validation
are different checks.

See {mod}`nozzle_hyperbolic.solvers.residuals`,
{mod}`nozzle_hyperbolic.solvers.monitoring` and
{mod}`nozzle_hyperbolic.solvers.fvm.drivers`.
