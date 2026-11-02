# Equations, numerical method and retained evidence

## Physical problem

The five operating points use the same symmetric converging-diverging Euler
nozzle. The flow is inviscid and adiabatic, with no wall friction. Reservoir
stagnation pressure is 200 kPa and stagnation temperature is 300 K. The nozzle
length is 0.2 m, its inlet and exit areas are $3.0\times10^{-4}$ m$^2$, and
its throat area is $1.5\times10^{-4}$ m$^2$ at $x=0.1$ m:

$$
A(x)=A_{\mathrm{in}}-4(A_{\mathrm{in}}-A_t)
\frac{x}{L}\left(1-\frac{x}{L}\right).
$$

The saved default mesh has 201 points including two boundary/ghost slots;
199 interior states are the physical unknowns. A perfect-gas Air object supplies
properties. The recorded backend is JAXProp 0.5.7, with JAX 0.6.2 and 64-bit
arithmetic. The embedded metadata retains the exact recorded versions.

Backpressure selects fully subsonic flow, the sonic-throat/subsonic-exit limit,
shocks at 35% and 70% of the divergent section, and an adapted supersonic exit.
Analytical nozzle relations supply reference pressure curves and the physical
backpressures. Numerical starts use the same simple cold-state rule; no
analytical flow or specified shock station initializes a solve.

## Conservative equations and wave speeds

Use pressure, meridional velocity and static enthalpy as primitive variables,
$W=(p,v_m,h)$. The source API's velocity slot is retained without modification;
the report uses $v_m$ for readability. Conservative storage and inviscid flux are

$$
q=\begin{bmatrix}\rho\\\rho v_m\\\rho E\end{bmatrix},\qquad
F=\begin{bmatrix}\rho v_m\\\rho v_m^2+p\\\rho v_m(h+v_m^2/2)\end{bmatrix},
\qquad E=h-\frac{p}{\rho}+\frac{v_m^2}{2}.
$$

The quasi-one-dimensional balance is

$$
\frac{\partial(Aq)}{\partial\tau}
+\frac{\partial(AF)}{\partial x}
=\begin{bmatrix}0\\p\,dA/dx\\0\end{bmatrix}.
$$

The local characteristic speeds are $v_m-a$, $v_m$ and $v_m+a$, where $a$ is
the thermodynamic sound speed. The middle field convects entropy/contact
disturbances; the other fields carry acoustic information. The direction of
incoming information changes across sonic conditions, which also affects the
outlet boundary branch. In this study, pseudo-time $\tau$ is a device for
finding a steady root, not a physical startup clock.

## Spatial discretization and boundary handling

The retained solver uses Städtke flux-vector splitting with the corrected,
regularized characteristic flux, entropy regularization parameter
$\delta=10^{-6}$, and safe ordinary Van Leer MUSCL reconstruction of primitive
variables. All comparisons hold the spatial method fixed. No constant-
reconstruction results are present in this dataset, so this report does not
invent an order-of-accuracy comparison.

For interior cell $i$, the integrated original spatial residual is

$$
R_i(W)=A_{i+1/2}\widehat F_{i+1/2}
-A_{i-1/2}\widehat F_{i-1/2}-V_iS_i(W).
$$

The existing geometry/source construction uses $V_i=A_i\Delta x$, adjacent-node
face areas and a face-consistent pressure-area momentum source. The original
factory's reservoir inlet and subsonic/supersonic outlet ghost maps are retained.
This sensitivity study does not substitute the separately studied coupled
boundary formulation.

## Implicit pseudo-time integration

Backward Euler gives a square interior nonlinear system

$$
\mathcal F_i(W^{n+1})=
V_i\left[q_i(W^{n+1})-q_i(W^n)\right]
+\Delta\tau R_i(W^{n+1})=0.
$$

Newton corrections solve the linearization

$$
J(W^k)\Delta W^k=-\mathcal F(W^k),\qquad
W^{k+1}=W^k+\Delta W^k.
$$

The saved runs use JAX differentiation, colored block Jacobian assembly and a
banded direct solve. They use the production implicit algorithm rather than
a separate SciPy or PySolverView direct root method. Instrumentation adds
diagnostic returns for the actual correction count and last relative update.

The cap sweep uses maxima 1, 3, 5, 10 and 100. The inner tolerance is a
relative-update criterion of $10^{-8}$, with the production scaling. A cap of
100 permits early termination; it does not force 100 corrections. A low cap
may reach a steady root without accurately solving every backward-Euler
timestep. This is legitimate as a steady iteration but does not establish
physical-time accuracy. The saved stopping test does not include a separate
norm requirement for each transient nonlinear equation.

After a candidate update, existing finite-state/domain and increment checks
can reject the timestep. Relative pressure and enthalpy limits are 0.2 and 0.1;
the sound-speed-scaled velocity increment limit is 0.2. A rejected timestep is
halved, with up to eight halvings. Rejected work remains included in the total
Newton/linear-solve counts and recorded solve time.

## CFL and SER controller

The nominal timestep is controlled by the fastest acoustic/convective speed:

$$
\Delta\tau_0=C_0\frac{\Delta x}
{\max_i(|v_{m,i}|+a_i)}.
$$

The anchored switched-evolution relaxation (SER) target is

$$
C_{\mathrm{target}}=C_0\operatorname{clip}
\left[\left(\frac{\mathrm{SSR}_1}{\mathrm{SSR}}\right)^\beta,
1,\frac{C_{\max}}{C_0}\right].
$$

Here $\mathrm{SSR}_1$ is established after the first accepted timestep. A
separate growth bound limits the next timestep relative to the preceding
accepted timestep. Wave speeds change with the state, and safety retries can
reduce the timestep further. Thus a configured maximum CFL is not the actual
accepted CFL; the report plots the saved accepted values and can overlay
rejected attempts.

| Variant | Starting CFL | Maximum CFL | SER exponent | Timestep growth bound |
|---|---:|---:|---:|---:|
| Baseline | 20 | 500 | 1 | 1.2 |
| Start CFL 1 | 1 | 500 | 1 | 1.2 |
| Start CFL 100 | 100 | 500 | 1 | 1.2 |
| Max CFL 100 | 20 | 100 | 1 | 1.2 |
| Max CFL 100000 | 20 | 100000 | 1 | 1.2 |
| No SER ramp | 20 | 500 | 0 | 1.2 |
| SER exponent 2 | 20 | 500 | 2 | 1.2 |
| Timestep growth 1.05 | 20 | 500 | 1 | 1.05 |
| Timestep growth 2 | 20 | 500 | 1 | 2 |

Each non-baseline variant changes one controller parameter. The same nine
variants are compared at caps 100 and 1. This is a controlled sensitivity,
not a factorial optimization or a claim about the best combination of settings.
The two baseline cap configurations are shared with the Newton sweep: the three
pages contain 115 entries but only 105 unique solves.

## Original residual, root agreement and analytical accuracy

The independent outer stopping quantity is a fixed-scale spatial RMS:

$$
\varepsilon_R=\sqrt{\frac{1}{N_R}\sum_j
\left(\frac{R_j}{R_{\mathrm{ref},j}}\right)^2},\qquad
\varepsilon_R\le10^{-8}.
$$

The retained problem defines fixed primitive scales from reservoir pressure,
sound speed and enthalpy. Spatial scales are
$V_i q_{\mathrm{ref}}/(\Delta x/a_0)$. The native state-normalized SSR used by
SER differs from this fixed-scale convergence norm. A small Newton update or
small physical timestep does not by itself establish a spatial root.

Agreement with each case's cap-100 baseline is the RMS distance between the
saved dimensionless interior primitive vectors, with threshold $10^{-4}$.
This is the historical comparison definition, not a uniqueness proof.
Analytical acceptance independently requires normalized mean absolute errors
in pressure, density and Mach below 0.03. These full-domain errors retain shock
contributions. The sonic-limit discrepancy remains a failure in 21 runs despite
small equation residuals; it is not corrected or relabeled by this report.

The new report audit checks saved histories, count arithmetic, inner-cap
compliance, stopping flags, common initial-state hashes, field dimensions and
baseline distances. It does not evaluate today's changed solver against the
old saved states. The original source hashes and historical path strings are
preserved as provenance.

## Timing interpretation and current-source limitations

Each configuration has **one serial timed solve** after kernel warmup.
Compilation/first-call costs are retained separately. Solve times include
controller monitoring and rejected attempts; reference construction, final
diagnostics, file output and plots are excluded. There are no repeated samples
from which to derive a timing distribution or a reliable speedup estimate.
Use accepted outer updates and total Newton corrections for stronger work
comparisons, and treat wall times as descriptive observations on that machine.

The source tree and adapter interfaces have evolved since this dataset was
generated under the historical `development_new` layout. Rebuilding this report
does not import a numerical solver, rerun a case, rewrite the original source
fingerprints or assert that the current counter adapter can reproduce the old
campaign without a compatibility review. A new numerical campaign would be a
separate task.
