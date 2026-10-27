# EulerNozzle physical verification cases

These two examples compare numerical Euler solutions with independent
analytical or semi-analytical references. They verify the implementation
against known solutions, rather than validating it against experiments.
Each script puts the physical conditions, mesh, method, and tolerances at
the top, so refactors can be checked against the same problems.

Both use `jaxprop.FluidPerfectGas(name="Air")`. The references use the same
heat-capacity ratio and gas constant as the backend; Air is not forced to
have exactly $\gamma=1.4$.

## Running and outputs

From the repository root, after `poetry install`:

```powershell
poetry run python -m development_new.euler_nozzle.verify_euler_nozzle_steady --no-show
poetry run python -m development_new.euler_nozzle.verify_euler_nozzle_sod --no-show
```

The files can also be run directly by path. Omit `--no-show` to display figures.
Each saves PNG/PDF comparisons, numerical/reference CSV files, and a
`summary.json` in a neighboring `results_euler_nozzle_steady/` or
`results_euler_nozzle_sod/` directory. Generated outputs are ignored by Git.
Existing local results were moved here with the scripts; they are not
committed reference fixtures. Regenerate them when changing solver versions.

Plots use `plasma` between 0.2 and 0.8. A color identifies a back pressure or
a physical time. Solid lines show the reference; hollow circles show the
numerical solution. A failed check saves the comparisons and returns a
nonzero exit status.

## 1. Steady converging-diverging nozzle

The quasi-1D flow is inviscid and adiabatic, without wall friction. The Euler
equations include the pressure-area momentum source. The symmetric area is

$$
A(x)=A_{\rm in}-4(A_{\rm in}-A_t)\frac{x}{L}\left(1-\frac{x}{L}\right).
$$

| Condition | Value |
|---|---|
| Length $L$ | 0.2 m |
| Inlet and exit area | $3.0\times10^{-4}$ m$^2$ |
| Throat area $A_t$ | $1.5\times10^{-4}$ m$^2$ |
| Throat position | 0.1 m |
| Reservoir stagnation pressure $p_0$ | 200,000 Pa |
| Reservoir stagnation temperature $T_0$ | 300 K |
| Uniform grid | 201 points including two boundary/ghost points; 1 mm spacing |

The inlet uses reservoir enthalpy and entropy. The model outlet imposes static
pressure for subsonic outflow and extrapolates for supersonic outflow.

| Reference regime | Approximate $p_b/p_0$ | Physical behavior |
|---|---:|---|
| Supersonic exit, no internal shock | 0.093739 | Choked throat and supersonic expansion |
| Shock at 35% of divergence | 0.888338 | Supersonic acceleration, normal shock at $x=0.135$ m, then subsonic flow |
| Shock at 70% of divergence | 0.708722 | A stronger normal shock farther downstream, at $x=0.170$ m |
| Sonic throat, subsonic exit | 0.937112 | Limiting isentropic branch with $M_t=1$ and subsonic flow on both sides |
| Entirely subsonic | 0.956415 | Unchoked branch with reference throat Mach $M_t=0.6$ |

The references combine isentropic nozzle relations and, where needed,
normal-shock jump conditions. Subsonic cases use the subsonic area-Mach root
on both sides. The unchoked case has a virtual critical area below the
geometric throat area and a mass flow below the choked value.

Shock positions and throat Mach numbers specify the reference branches and
their back pressures. The numerical solver receives the geometry, reservoir
conditions, and back pressure. It receives neither an imposed shock position
nor the exact flow as initialization; it uses its ordinary simple startup.

The numerical method is Städtke FVS with primitive Van Leer MUSCL and implicit
backward-Euler/Newton pseudo-time marching with one Newton correction per
timestep. The starting CFL is 10, maximum SER CFL is $10^6$, SER exponent is
one, and the timestep growth bound is 1.5 per accepted update. Adaptive
increment checks can halve a trial timestep. The budget is 5000 updates and
the residual target is $10^{-8}$.
Pseudo-time is a convergence device, not a time-accurate nozzle startup.

The combined CFL 10 / growth 1.5 / maximum $10^6$ setting was run on all five
cases. Accepted-update counts were 17 (supersonic), 114 (shock 35%), 199
(shock 70%), 43 (sonic limit), and 41 (subsonic). All met the residual target;
the same four analytic checks passed, with the known sonic-limit discrepancy
remaining. This is a verified usable setting, not a claim of minimum work for
every case. `dt_growth_factor` is a production-driver parameter with fallback
1.2 for callers that do not specify it.


**Why it is useful:** smooth branches check the geometric source and
isentropic acceleration; choking checks the sonic region and operating
branch; shock cases check discontinuity capture, downstream compression,
and shock position. All branches expose mass-flow imbalance and the
difference between residual convergence and physical accuracy. These are
standard nozzle-verification objectives; [NASA's CDV example](https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html)
uses different geometry and conditions from ours.

The script compares pressure, density, and Mach number. Acceptance requires
convergence, boundary mass-flow imbalance below $10^{-4}$, and full-domain
normalized mean absolute errors below 0.03. Pressure and density use reservoir
scales; Mach uses one. Additional smooth-region errors exclude three grid
spacings around a reference shock, but acceptance includes the shock region.
Shock-position and throat-Mach errors are reported separately.

## 2. Unsteady Sod shock tube

A diaphragm at $x=0.5$ m separates two initially stationary states in a
constant-area, one-metre domain. Removing it at $t=0$ creates a Riemann
problem, with no area source, wall friction, or heat transfer.

| Condition | Left | Right |
|---|---:|---:|
| Density | 1.0 kg/m$^3$ | 0.125 kg/m$^3$ |
| Pressure | 100,000 Pa | 10,000 Pa |
| Velocity | 0 m/s | 0 m/s |

The mesh has 400 physical cells and two transmissive ghost cells, with the
diaphragm on a face. The method is the existing JAX HLL flux, piecewise-constant
reconstruction, forward Euler, and CFL 0.45.

One simulation advances through $\tau=t a_L/L=0.1,0.2,0.3$, approximately
0.267, 0.534, and 0.801 ms for this backend. All cells share a physical timestep,
clipped before the update to reach each requested time. All waves remain
away from the ends, so the infinite-domain Riemann reference is appropriate.

The exact solution contains a left-moving rarefaction, a contact discontinuity,
and a right-moving shock. Pressure and velocity are continuous across the
contact while density jumps. A scalar star-pressure solve determines states
and wave speeds; sampling uses $(x-x_d)/t$. See the
[Clawpack Euler chapter](https://www.clawpack.org/riemann_book/html/Euler.html)
for the wave structure and conservation laws.

**Why it is useful:** comparing all three waves checks propagation speeds,
the physical clock, shock/contact diffusion, and the rarefaction profile.
It complements the steady case, whose residual cannot reveal an incorrect
transient clock. The first-order method intentionally diffuses discontinuities.

Normalized mean errors in density, velocity, and pressure must be below 0.04.
Independent mass, momentum, and energy budgets must agree with the numerical
boundary flux integrals to $10^{-10}$ in normalized units:

$$
\int Q(x,t)\,dx-\int Q(x,0)\,dx
=\int_0^t\left(F_{\rm left}-F_{\rm right}\right)\,dt.
$$

Momentum need not be constant: unequal end pressures exert a net force even
before a wave reaches an end. Accounting for boundary fluxes is essential.

## Verification status and scope

Both examples were rerun against an isolated copy of the committed solver,
without pending FVS/limiter/root-finding modifications. Sod passes all three
snapshots. All five steady cases converge; four pass the accuracy checks.
The sonic-limit case fails the unchanged 3% threshold, with numerical throat
Mach about 0.846 instead of 1. This failure is retained: the steady script
currently exits unsuccessfully after saving every comparison.

The scripts use the installed solver implementation. Pending working-tree
changes can affect a rerun; see the [numerical changes review](../../docs/NUMERICAL_CHANGES_REVIEW.md).
This commit adds examples and documentation without approving those changes.

These are fixed-mesh verification baselines, not mesh/order studies or
certification of real-fluid and TwoFluid physics. Sod checks HLL physical-time
integration, not the nozzle driver's FVS pseudo-time path.
