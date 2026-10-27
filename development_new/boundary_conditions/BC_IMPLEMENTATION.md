# Internal versus simultaneous boundary equations

Both formulations solve implicit timesteps of the same discretized Euler
nozzle. The difference is whether ghost variables are eliminated before each
inner residual evaluation or solved as part of that residual.

## Current source boundaries

The source factory uses interior primitive unknowns $x=(p,u,h)$ and reconstructs
the inlet/outlet ghost states $g=B(x)$ internally.

At the inlet, with first physical indices 1 and 2:

$$
u_g=2u_1-u_2,\qquad
h_g=\operatorname{clip}(h_0-u_g^2/2),\qquad
s(h_g,p_g)=s_0.
$$

Pressure starts from the clipped first-cell pressure. Three unrolled Newton
updates apply

$$
p_g\leftarrow\operatorname{clip}\left[p_g+\rho_gT_g(s_g-s_0)\right].
$$

This is a fixed-work approximate nonlinear inversion, rather than a nested
solver with a separate convergence tolerance. AD differentiates the actual
three updates and active clips. It includes boundary sensitivities in the
interior Jacobian; BC equations are not mathematically missing.

At the outlet, define interior EOS densities and sound speed. The current
code forms area-weighted extrapolations

$$
\widetilde\rho_g=\operatorname{clip}
\frac{2\rho_{N-2}A_{N-2}-\rho_{N-3}A_{N-3}}{A_g},\qquad
u_g=\frac{2\rho_{N-2}u_{N-2}A_{N-2}-\rho_{N-3}u_{N-3}A_{N-3}}
{A_g\widetilde\rho_g}.
$$

The density proxy supplies the velocity extrapolation; the actual ghost EOS
density follows its final pressure and enthalpy and need not equal this proxy.
Enthalpy is linearly extrapolated and clipped. Pressure blends backpressure
and extrapolation using the last physical-cell Mach number:

$$
w=\tfrac12[1+\tanh((1-|u_{N-2}|/a_{N-2})/0.004)],\qquad
p_g=wp_b+(1-w)p_{\mathrm{extrap}}.
$$

The outlet implementation is explicit algebra: it does not run the commented
historical bisection. Subsonic flow activates backpressure; supersonic flow
extrapolates pressure. The present study keeps those same outlet equations.

## Two implicit timestep systems

Let $S(x)$ be the scaled integrated conservative content of the interior cells
and $F(x,g)$ their scaled spatial balances. The current timestep Newton solve
has only interior unknowns:

$$
\frac{S(x_{n+1})-S(x_n)}{\Delta\tau}
+F(x_{n+1},B(x_{n+1}))=0.
$$

The tentative implementation instead adds six ghost primitive unknowns and
six dimensionless algebraic rows $C(x,g)=0$:

$$
\begin{bmatrix}
[S(x_{n+1})-S(x_n)]/\Delta\tau+F(x_{n+1},g_{n+1})\\
C(x_{n+1},g_{n+1})
\end{bmatrix}=0.
$$

There is **no storage term in the ghost rows**. Newton solves this entire
system inside each implicit timestep, with repeated inner corrections.
This is not the earlier direct steady root solve or single-correction PTC.
For 201 points there are 597 interior unknowns versus 603 coupled unknowns.
Adding equations without adding the ghost unknowns would overdetermine the
system.

The coupled inlet rows impose

$$
s(h_g,p_g)-s_0=0,\qquad
u_g-(2u_1-u_2)=0,\qquad
h_g+u_g^2/2-h_0=0.
$$

They are scaled respectively by $h_0/T_0$, $a_0$ and $h_0$. The outlet rows
are differences between the independent ghost state and the existing explicit
outlet map, scaled by $(p_0,a_0,h_0)$. **No three-update inlet pressure solve
is called by the coupled residual or convergence monitor.** A consistent
initial ghost guess is prepared once; final independent source-equation
audits may evaluate the source map outside the timed marching loop.

Exact inlet thermodynamic constraints reproduce the intended un-clipped source
boundary relations, not every out-of-range clip or finite-iteration error of
the existing algorithm. This equivalence is tested on the admissible branch;
it is not a claim of global equivalence through all clipped states.

## Why similar histories are possible

At a boundary-consistent state and where derivatives and inverses exist,
the augmented timestep Jacobian has blocks

$$
\begin{bmatrix}M/\Delta\tau+F_x&F_g\\C_x&C_g\end{bmatrix}.
$$

Eliminating the ghost correction gives
$M/\Delta\tau+F_x-F_gC_g^{-1}C_x$, precisely the reduced implicit Jacobian
for exact boundary elimination. Thus the two formulations can generate nearly
identical updates when the source inlet inversion is already accurate.
Explicit BC rows do not add a new physical mechanism or guarantee faster
convergence. Differences can arise from finite inner solves, off-manifold
trial states, clipping, scaling and linear-algebra costs.

## Matched experimental implementation

The baseline calls the source timestep kernel directly. Source currently uses
its own **JAX-compiled Newton loop**, not Optimistix. The tentative implementation
now uses the very same Newton correction, colored JVP block assembly and
block-Thomas solver code. The former PySolverView dense timestep backend was
removed for this matched benchmark; its timing gap was not a fair measure of
boundary-condition organization alone.

The source Newton cap (eight) and relative-update stopping formula
$\max_j|\Delta U_j|/U_{\mathrm{ref},j}$ are retained. Reference scales use local
pressure and global velocity/enthalpy magnitudes, following the exact source
helper. Source applies this rule to interior unknowns; the enlarged system
naturally includes ghosts. Some inner timestep systems can still reach the
cap before exact convergence, just as in source. No new line search,
least-squares tolerance, transient BC relaxation, or artificial ghost storage
is introduced. The outer controller and monitor are shared between methods.

Unknowns and residual rows are arranged in physical cell order: inlet ghost,
interior cells, outlet ghost, with three variables per cell. Both new boundary
rows depend on at most their ghost and the nearest two physical cells. Thus
the source width-two coloring and block-banded algorithms remain valid. Only
cell-count/padding bindings change. Extra ghost cells alter superblock grouping
and can change floating-point roundoff; identical trajectories are not assumed.

Interior rows retain the exact source form
$V[Q(U)-Q(U_n)]+\Delta\tau F(U)$. Boundary rows are dimensionless constraints
scaled to primitive units $(p_0,a_0,h_0)$, with zero storage. The independent-ghost
adapter must supply **current ghost unknowns** to spatial flux evaluation,
while retaining **old interior values** for the storage term. It must not lag
the ghosts in time. A dedicated dependency test and a full source-timestep
parity test guard this distinction.

The read-only adapter creates separate function objects from existing factory
closures, substituting the coupled residual and enlarged row count. Original
functions, closure cells and source files are not mutated. Dense AD and dense
linear solves are used only as independent verification, never as the measured
coupled solver. Eight focused checks span odd/even sizes and verify that the
colored stencil and block solve remain valid.

The common steady monitor uses each actual full state, identical spatial/EOS
code and the same fixed equation scales. It does not call the internal inlet
inversion to monitor the coupled field. Its target is spatial and algebraic
boundary RMS $10^{-8}$. The source native SSR is retained for the shared SER
controller. Final original-equation and analytic audits are separate from
measured solve time. Neither a small increment nor an inner native flag alone
establishes final root convergence.

## Repeated timing protocol

Each case/method has one excluded complete solve, followed by ten measured
complete solves. The same initial full state and fresh timestep-controller
state are restored every time; only compiled kernels are reused. Solves run
sequentially with one BLAS thread, alternating which method runs first.
A device synchronization is required before stopping the timer. Setup, final
audits, writing and plotting are excluded equally; monitors and rejected
attempts are included equally. Every warmup and measured outcome is saved.

The reported minimum, maximum and mean use all ten measured values, not the
excluded first solve. Representative flow/residual plots use an actual median-
duration run. No timing samples are dropped for being inconvenient, and no
speed ratio is presented as a successful-root result if a method fails its
independent root audit.

This now measures boundary handling within matched numerical machinery.
Remaining overhead includes six additional equations/unknowns, two additional
block rows and changed boundary EOS work, plus any changed inner/retry work.
It does not establish performance on another machine or general EOS, mesh or
initialization robustness. The exact thermodynamic inlet versus finite
three-update/clipped source relation remains the earlier stated limitation.
