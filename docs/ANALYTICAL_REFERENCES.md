# Analytical references: inventory and consolidation

Assessed on **31 October 2026**. The reusable Euler references now belong to
their physical models in `src/nozzle_hyperbolic/models/`. They are optional
verification tools; the shared FVM machinery does not import them.

## Finding

The references were **not all in `development/roberto/verification_references/`**.
That directory held the main steady nozzle, steady annular, and Sod references,
but further implementations and useful derivations were scattered among studies
and the pre-port archive.

| Before consolidation | What it contained | Current owner |
|---|---|---|
| `verification_references/nozzle.py` | Perfect-gas symmetric-nozzle operating map, choking, internal shocks, external-adjustment classification | [`euler_nozzle/analytical/perfect_gas.py`](../src/nozzle_hyperbolic/models/euler_nozzle/analytical/perfect_gas.py) |
| `backpressure_sensitivity/scripts/reference.py` | A second implementation of the same operating map; five of eight methods were identical | Same canonical nozzle module; obsolete wrapper deleted |
| `euler_nozzle/verify_euler_nozzle_steady.py::subsonic_reference` | Unchoked area–Mach construction and repeated perfect-gas formulas | Callers invoke canonical `subsonic_reference` directly with their physical parameters |
| `euler_annular/scripts/case_config.py::build_case` | Local area–Mach inversion to select a sonic-exit fixture | Calls canonical `mach_from_area_ratio`; case selection remains in the study |
| `verification_references/annular.py` | Steady swirl invariants, branch roots, normal shock, conservative cell projection | [`euler_annular/analytical/perfect_gas.py`](../src/nozzle_hyperbolic/models/euler_annular/analytical/perfect_gas.py) |
| `verification_references/shock_tube.py` | Two Sod samplers with substantially duplicated logic | [`euler_nozzle/analytical/shock_tube.py`](../src/nozzle_hyperbolic/models/euler_nozzle/analytical/shock_tube.py) |
| `euler_annular/scripts/verify_transient.py` | Translated compact entropy contact and conservative volume projection | [`euler_annular/analytical/transient.py`](../src/nozzle_hyperbolic/models/euler_annular/analytical/transient.py) |
| `euler_annular/scripts/verify_wall_sources.py::exact_homogeneous` | Homogeneous prescribed-shear/heating solution before the flow stops | Same annular transient module |
| Archived nozzle `analytical/nozzle_exact_real_gas.py` | General-EOS isentropic branches and prescribed normal-shock reference; four nonarchive callers still imported its missing production path | [`euler_nozzle/analytical/general_eos.py`](../src/nozzle_hyperbolic/models/euler_nozzle/analytical/general_eos.py); callers updated |
| Scalar models and v3 fixtures | Repeated translating trigonometric profile and Burgers shock/rarefaction | [`scalar/analytical.py`](../src/nozzle_hyperbolic/models/scalar/analytical.py) |

Paths in the first column are beneath `development/roberto/`, unless marked
otherwise. The earlier studies have since moved beneath
`development/roberto/historical/`. The entire `historical/verification_references/`
compatibility package and `historical/backpressure_sensitivity/scripts/reference.py`
have been deleted. Callers import the canonical source modules directly;
provenance no longer reads a deleted wrapper file.

## Why this ownership works

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

## Intentional exceptions and remaining work

- **Historical archive:** `legacy_before_generic_fvm/` remains unchanged,
  including its old analytical modules. Its purpose is reproducible old-code
  comparisons, not a second current implementation.
- **Older `nozzle_exact2`:** archived `exact_solution.py` and
  `nozzle_exact_solution.py` duplicate a hardcoded-`R=287` implementation.
  They are not promoted. `development/alberto/euler_nozzle/main.py` still
  targets this retired API and needs a separate legacy-client migration.
- **Two-fluid limits:** archived `models/two_fluid/analytical.py` contains
  dominant-phase isentropic, shock, Bernoulli, and initial-profile helpers.
  They are not exact solutions of the full coupled two-fluid model. Four
  legacy/application callers still target the missing old module. Porting
  these requires a separately checked API and assumptions; this consolidation
  does not claim to restore the deferred two-fluid solver.
- **Independent test oracles:** analytical tests use independent area–Mach,
  conservation, and quadrature identities to cross-check the references. These
  test equations are not benchmark reference generators.
- **Thermodynamic trajectories:** the case-specific metastable-isentrope and
  thermodynamic plotting helpers under `cases/` are not analytical solutions
  of the nozzle/annular PDE. They remain case tools.
- **Legacy drivers:** several historical studies still import retired FVM
  factories. Updating their reference imports does not migrate their numerical
  driver API. Use `src/nozzle_hyperbolic/solvers/fvm` for the supported engine.
- **Unused case wrapper:** `cases/nicfd/nicfd_figures/isentropic_ic.py` only
  re-exported source names and had no remaining callers. It has been deleted;
  the legacy NICFD scripts already import the model's source API directly.

Saved reports/results retain their original values and source fingerprints.
New comparisons should evaluate the canonical reference, or validate its source
fingerprint before reusing cached arrays. Reference generation belongs outside
benchmark timing.

## Reference corrections made during consolidation

This includes a structural move and the following isolated reference fixes;
they are not changes to the numerical equations:

- Coordinate arrays are converted to floating point, preventing integer-input
  truncation in the nozzle/Sod samplers.
- Sod time-zero sampling now returns the initial discontinuity. Unsupported
  wave patterns raise an error instead of silently using the Sod sampler.
- The homogeneous wall-source solution retains the backend's enthalpy datum.
  Incompatible EOS/geometry assumptions in special references are checked.
- The general-EOS nozzle reference checks entropy-root convergence, falls back
  to a bracketed solve after a failed warm start, and rejects out-of-domain
  shock trials. It exposes pressure/enthalpy/compression search bounds. The
  old heuristic enthalpy fallback remains documented; specify `h_min` for an
  arbitrary enthalpy datum. This is deliberately described as a semi-analytical
  construction with limited branch coverage.
- Comparison clients must use the canonical source rather than treating
  unversioned saved reference arrays as the current formula authority.

The FVM fluxes, physical boundary equations, Newton policy, integration methods,
and controller settings were not changed. Scalar profiles were extracted into
shared model-owned functions with the same mathematical definitions.

## Verification

The consolidation adds direct production tests in
[`test_nozzle_analytical.py`](../tests/test_nozzle_analytical.py),
[`test_annular_analytical.py`](../tests/test_annular_analytical.py), and
[`test_scalar_analytical.py`](../tests/test_scalar_analytical.py).
They check conservation invariants, shock jumps, branch behavior, EOS/datum
assumptions, projection, and transient limiting cases. This work does not rerun
full numerical benchmark campaigns or alter their retained timings.
The original focused analytical/model/FVM/boundary verification passed **122
tests**, including **41 analytical tests**, with no failures. The 41 analytical
tests are rerun for the compatibility-wrapper removal.

The original migration-parity audit compared **20 cases/APIs** with retained
code: nine nozzle operating points, nine annular cases, and the two positive-time
Sod APIs. All pass. Annular profiles and conservative projections are bit-for-bit
equal; the largest scaled nozzle difference is $2.1\times10^{-16}$, and the Sod
difference is $5.6\times10^{-17}$. The deliberate reference fixes above are checked
against mathematical expectations, rather than forced to match the old defects.
The separate annular invariant/jump/quadrature audit passed **nine checks**.
These are migration-time observations; they do not claim that the removed v3
development directory or its generated artifacts remain in the checkout.

Run from the repository root:

```powershell
poetry run pytest -q -p no:cacheprovider tests/test_nozzle_analytical.py tests/test_annular_analytical.py tests/test_scalar_analytical.py
```

The production tests import source modules directly and do not require historical
wrappers, development verification drivers, or the immutable pre-port archive.
