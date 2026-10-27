# EulerNozzle baseline checks

The new baseline work lives in `development_new/`. The older research scripts
remain in `development/`. The direct steady-solve experiment and residual/time
comparisons are documented in [newton_solve/README.md](newton_solve/README.md).

These two scripts establish readable starting points for checking the ongoing
refactor. Each has its physical problem, mesh, numerical settings, and acceptance
thresholds at the top. They run one numerical configuration and compare it with
the reference functions in `models/euler_nozzle/analytical/`. The steady script
also evaluates the perfect-gas subsonic area-Mach relation directly for the
unchoked and sonic-limit cases.

```powershell
poetry run python development_new/verify_euler_nozzle_steady.py
poetry run python development_new/verify_euler_nozzle_sod.py
```

Add `--no-show` to save the figures without opening a window. Both scripts save
`comparison.png`, `comparison.pdf`, per-case CSV data, and a `summary.json` in
neighboring `results_euler_nozzle_*` directories. Those generated results are
ignored by Git. A failed convergence/error check produces a nonzero exit status.

| Script | Problem and numerical configuration | Exact comparison |
|---|---|---|
| `verify_euler_nozzle_steady.py` | Smooth symmetric nozzle, one mesh, Städtke FVS, primitive Van Leer MUSCL, implicit backward Euler/Newton; perfect-gas Air | Five back pressures: smooth supersonic exit, two internal normal shocks, sonic throat with subsonic exit, and fully subsonic flow |
| `verify_euler_nozzle_sod.py` | Standard Sod initial states with zero velocity, one mesh, existing JAX HLL flux, piecewise-constant reconstruction, forward Euler | Three physical times from a single run; the exact left rarefaction, contact discontinuity, and right shock |

Both use `plasma` sampled uniformly from 0.2 to 0.8. Each back pressure or time
has one color: the exact reference is a solid line, and numerical values are
hollow circles in the same color. The reference and numerical perfect gas use
the same heat-capacity ratio from `jaxprop.FluidPerfectGas`.

The steady script gives the solver the back pressure derived from each reference
case; it does not provide the exact shock location or initialize from the exact
flow. It reports the residual, full-domain and smooth-region normalized errors,
shock-position error, throat Mach number, and boundary mass-flux imbalance. Its
pseudo-time is only a steady convergence device.

The sonic/subsonic reference takes the subsonic area-Mach root upstream and
downstream, with exactly $M_t=1$ at the throat. The fully subsonic reference uses
$M_t=0.6$. Its virtual critical area is smaller than the geometric throat, so
its mass flow is below the choked value. Each reference determines its own back
pressure; the numerical solver receives the same reservoir conditions and only
that back pressure. The reference throat Mach number is not imposed on the
numerical flow. A finite mesh need not attain exactly $M=1$, so the summaries
record both reference and numerical throat Mach numbers.

The Sod script uses a single global CFL timestep and clips the timestep before
the update to hit each output time. All plotted times precede any interaction
with the domain boundaries. It also checks integrated mass, momentum, and energy
against the numerical boundary fluxes. The first-order scheme deliberately
diffuses discontinuities; the comparison tolerances describe this fixed mesh,
not an accuracy order or a mesh-convergence study.

The branch review found `dev_scripts/shock_tube_fvs.py` on Alberto's branch and
its relocated copy at `development/shock_tube_fvs.py`. That older example adds
a small background velocity to avoid a stationary FVS singularity and caps the
time counter after taking a full step. The new Sod baseline uses the existing
HLL kernel to retain the canonical initially stationary data. The production
driver's local time stepping is a steady pseudo-time path, so the physical-time
loop is kept explicit in this development script. Existing solver functions
are preserved.

For the mathematical references, see the
[NASA nozzle verification example](https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html)
and the [Clawpack Riemann book's Euler/Sod discussion](https://faculty.washington.edu/rjl/riemann_book/Euler_equations.html).
The nozzle reference uses isentropic branches and normal-shock jump conditions;
the Sod reference solves the scalar star-pressure equation and samples the
self-similar solution. Their scalar root solves are evaluated to numerical
tolerance, independently of the finite-volume mesh.

## Baseline results, including the two subsonic cases

These results were obtained on 2026-10-26 with the settings shipped here.
Normalized L1 is the mean absolute error divided by the scales used in each
script; Mach error uses a scale of one. The original three steady cases and all
Sod snapshots pass. The extended steady check flags the sonic-limit case.

| Steady case | $p_b/p_0$ | Final residual | Largest normalized L1 | Result |
|---|---:|---:|---:|---|
| No internal shock, supersonic exit | 0.093739 | 8.04e-9 | 7.09e-4 | Pass |
| Shock at 35% of divergence | 0.888338 | 5.75e-9 | 5.61e-3 | Pass |
| Shock at 70% of divergence | 0.708722 | 6.31e-9 | 1.09e-3 | Pass |
| Sonic throat, subsonic exit | 0.937112 | 9.74e-9 | 3.12e-2 | Fails existing 3% error limit |
| Fully subsonic, $M_t=0.6$ | 0.956415 | 9.72e-9 | 1.60e-2 | Pass |

On the 201-point mesh, numerical throat Mach is 0.846401 for the sonic-limit
reference and 0.573769 for the fully subsonic reference. All five cases satisfy
the residual criterion, but convergence alone does not establish agreement with
the exact solution. The 3% acceptance threshold is unchanged: the steady script
saves all five comparisons and exits with a failure status for the sonic case.
This exposes an accuracy issue for later solver/discretization investigation;
no production solver functions were changed for this extension.

All steady boundary mass imbalances were below 3.6e-6. The three Sod output
times were 0.267100, 0.534200, and 0.801300 ms. Their largest normalized L1 error
was 8.93e-3; the largest normalized conservation-budget error was 1.90e-16.
The JSON reports retain the unrounded values and individual field errors.

Future method and mesh comparisons can extend these scripts after this single
configuration is established. Keep the reference problem and error definitions
fixed when comparing different numerical options.
