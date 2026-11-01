# Analytical references and verification assumptions

The reusable references belong beside their physical models under
`src/nozzle_hyperbolic/models/`. They are opt-in verification tools; the shared
FVM machinery does not use them to construct its numerical solution.

| Model | Canonical source |
|---|---|
| Perfect-gas nozzle | [euler_nozzle/analytical/perfect_gas.py](../src/nozzle_hyperbolic/models/euler_nozzle/analytical/perfect_gas.py) |
| General-EOS nozzle | [euler_nozzle/analytical/general_eos.py](../src/nozzle_hyperbolic/models/euler_nozzle/analytical/general_eos.py) |
| Steady perfect-gas annular | [euler_annular/analytical/perfect_gas.py](../src/nozzle_hyperbolic/models/euler_annular/analytical/perfect_gas.py) |
| Annular special transients | [euler_annular/analytical/transient.py](../src/nozzle_hyperbolic/models/euler_annular/analytical/transient.py) |
| Sod | [euler_nozzle/analytical/shock_tube.py](../src/nozzle_hyperbolic/models/euler_nozzle/analytical/shock_tube.py) |
| Scalar translation and Burgers | [scalar/analytical.py](../src/nozzle_hyperbolic/models/scalar/analytical.py) |

The [installed verification API](../src/nozzle_hyperbolic/verification/README.md)
loads versioned physical YAML resources and exposes point fields and conservative
cell projections. Benchmark settings and numerical solver choices belong to
study profiles. References must not become unreported analytical initial guesses.

## Ownership and independence

An analytical solution is a property of a physical problem under stated
assumptions. It is independent of constant/MUSCL reconstruction, numerical flux,
time integrator, Newton solver, or coupled/segregated boundary treatment.
Keeping it beside the corresponding model makes the assumptions and comparison
API discoverable without putting physics into the generic numerical engine.

There are two distinct forms of independence:

- **Independent of the numerical method:** the reference must not call the FVM
  residual, flux discretization, reconstruction, boundary-completion solver, or
  time integrator to manufacture its answer. These modules do not do so.
- **Independent verification of the reference:** tests must check conservation,
  limiting cases, jump conditions, known values, and alternative derivations.
  Importing the same reference twice is not an independent accuracy check.

The references may share the physical geometry and EOS with the numerical
problem. Their tests check the mathematical identities separately.

Perfect-gas algebra in these explicitly named analytical verification modules
does **not** change the general-EOS production equations. The earlier restriction
on perfect-gas relations applies to the numerical physical model and solver:
their only intended exception remains the optional finite-wave outlet.
The analytical modules are opt-in tools with explicit EOS restrictions.

## Canonical modules and mathematical scope

### Nozzle: calorically perfect gas

`perfect_gas.py` supplies `area_mach`, `mach_from_area_ratio`, `pressure_ratio`,
`normal_shock`, `subsonic_reference`, and `PerfectGasReference`.

For a steady, inviscid, adiabatic, quasi-one-dimensional perfect gas,

$$
\frac{A}{A_*}=\frac{1}{M}
\left[\frac{2}{\gamma+1}
\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{(\gamma+1)/(2(\gamma-1))}.
$$

The algebraic relation has separate subsonic and supersonic roots. For an
unchoked throat, $A_*=A_t/f(M_t)$ is a virtual critical area, not the geometric
throat. `PerfectGasReference` assembles these relations for the package's
symmetric nozzle and imposed backpressure, optionally locating a normal shock.
External under/overexpansion is classified; the exterior jet is **not** solved.

The standard formulas are attributed in the source to
[NACA Report 1135](https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf)
and NASA Glenn's [area–Mach](https://www.grc.nasa.gov/www/k-12/BGP/astar.html)
and [normal-shock](https://www.grc.nasa.gov/WWW/k-12/airplane/normal.html) pages.
The operating-map assembly and one-sided shock sampling are project choices.

### Nozzle: general EOS

`general_eos.py::nozzle_exact_real_gas` is **semi-analytical**: it numerically
solves scalar thermodynamic/root equations, rather than discretizing the PDE.
The governing invariants are

$$
s=s_0,\qquad h+\frac{u^2}{2}=h_0,\qquad A\rho u=\dot m.
$$

Sonic mass flux selects a choked solution. Across a prescribed stationary shock,
mass, momentum, and stagnation enthalpy are continuous, and entropy increases.
The reduction to scalar isentrope and compression-ratio roots is derived in
this project from those conservation laws; the source says so explicitly.

This is not a universal nonclassical-EOS Riemann solver. It assumes accessible
single-valued property branches and conventional subsonic/supersonic roots,
finite EOS search bounds, and an admissible compressive shock within its search
range. Read the function's bounds and limitations before using a new fluid.
Sharing the same EOS is necessary for comparisons, but does not prove the
reference construction is valid in a nonclassical regime.

### Annular: steady swirl and shocks

`PerfectGasReference.from_inlet(...)` constructs steady invariants

$$
\dot m=\rho v_m A,\qquad \ell=r v_\theta,\qquad
h_0=h+\frac{v_m^2+v_\theta^2}{2},\qquad K=\frac{p}{\rho^\gamma}.
$$

At each station, `perfect_gas.py` solves the scalar density equation

$$
\frac{\gamma K}{\gamma-1}\rho^{\gamma-1}
+\frac{1}{2}\left(\frac{\dot m}{A\rho}\right)^2
+\frac{1}{2}\left(\frac{\ell}{r}\right)^2
=h_0-h_{\rm shift}.
$$

Its density-root reduction, branch brackets, and sonic-endpoint quadrature
mapping are project derivations. Standard perfect-gas normal-shock relations
are applied to the **meridional** Mach number; tangential velocity is continuous
across the shock. Total Mach and meridional Mach are not interchangeable.

`cell_average` integrates conserved quantities with physical volume measure,
splitting at shocks and treating sonic endpoint behavior. The returned `W` is
an equivalent primitive state recovered from conservative averages; it is not
the direct average of each primitive component. `fields_from_state` provides
the shared diagnostic conversion for comparisons.

### Annular: transient contact and prescribed wall sources

`transient.py` documents two project-derived special solutions:

- A compact density/entropy contact translates at constant meridional velocity
  in a uniform annulus with fixed pressure and swirl. The compact bump is a
  chosen verification profile. Its pressure-preserving conservative projection
  is restricted to the supported geometry/EOS assumptions, which are checked.
- At fixed density in a uniform annulus, constant prescribed wall shear slows
  the velocity vector without rotating it, and constant wall heat flux changes
  total specific energy linearly:

$$
|v|(t)=|v|(0)-\frac{2\tau_w}{\rho b}t,\qquad
E(t)=E(0)+\frac{2q_w}{\rho b}t.
$$

The wall solution is valid only up to stopping and uses the fluid's enthalpy
datum consistently. It verifies prescribed sources; it is not an exact solution
with an arbitrary friction/heat-transfer closure such as Haaland.

### Sod and scalar problems

The Sod sampler is a calorically perfect-gas, left-rarefaction/right-shock
solution. The name does not imply a solver for every Euler Riemann pattern.
Supplying custom sound speeds does not make its gamma-based formulas exact
for a general EOS. Both samplers now share their wave construction and handle
time zero explicitly.

The scalar module contains characteristic translation and the entropy solution
of the Burgers Riemann problem, with shock speed $(q_L+q_R)/2$ or a centered
rarefaction. These relations are described in the primary
[Riemann Problems and Jupyter Solutions](https://www.clawpack.org/riemann_book/html/Burgers.html)
reference; the trigonometric initial profile is a project fixture.

## Limits and assessment

Perfect-gas references establish accuracy only for their declared caloric EOS.
Using PH property calls in the numerical model does not make the corresponding
perfect-gas analytical curve a real-gas reference. General-EOS comparisons need
their own matching reference and physical branch checks.

Prescribed wall-source special solutions do not validate arbitrary Haaland or
heat-transfer closures. Dominant-phase/Bernoulli helpers in legacy two-fluid
code are not exact solutions of the complete coupled two-fluid equations.
Case-specific thermodynamic trajectories are not PDE solutions either.

Historical copies in [legacy/pre_generic_fvm](../legacy/pre_generic_fvm/README.md)
are frozen comparison evidence. Development reference wrappers and the former
`generic_interface_v3` study are not current APIs. New clients should import the
canonical modules or the installed verification API directly.

Preserve source fingerprints when reusing saved reference arrays. Reference
construction belongs outside solver timing. Report original-equation convergence,
conservation, intended branch and analytical accuracy separately. An equivalent
state recovered from conservative cell averages is not a point sample or a
primitive-variable average.

Focused reference checks can be run from the repository root:

```powershell
poetry run pytest -q -p no:cacheprovider tests/test_nozzle_analytical.py tests/test_annular_analytical.py tests/test_scalar_analytical.py
```

These tests check reference identities and limiting cases; they do not certify
every EOS, flow branch or full numerical benchmark. The historical consolidation
inventory and its migration-time test counts remain available in Git history;
this guide preserves the mathematical assumptions and attribution separately
from that completed restructuring work.
