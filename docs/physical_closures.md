# Physical closure laws

The canonical library is [`models/closures`](../src/nozzle_hyperbolic/models/closures/__init__.py).
Closures specify constitutive data; each physical model applies its geometry,
momentum direction, angular momentum convention and total-energy balance. The
FVM machinery does not select a friction or heat-transfer law.

Both Euler models accept a wall policy with the signature
`wall(physical, radius, fluid) -> (shear_magnitude, heat_flux)`. Nozzle primitives
are $(p,u,h)$; annular primitives are $(p,v_m,\ell,h)$, with
$v_\theta=\ell/r$. Returned arrays have the physical cells' shape.

```python
from nozzle_hyperbolic.models.closures import HaalandWall, PrescribedWall

adiabatic = PrescribedWall()
prescribed = PrescribedWall(tau_w=2.0, q_w=100.0)
friction = HaalandWall(hydraulic_diameter=0.01, roughness=1e-5)
```

`PrescribedWall` retains the zero-default annular placeholders: shear is a
nonnegative magnitude in Pa; heat flux is in W/m² and positive into the fluid.
Scalar or fixed cell profiles are supported. A nonzero prescribed shear law has
no unique direction at zero velocity; the model applies zero force exactly
there. A viscous law provides the smooth behavior needed for linearization at
zero speed.

`HaalandWall` obtains density and dynamic viscosity directly from
`fluid.get_state(HmassP_INPUTS, h, p)`. It does not compute thermodynamic
properties from an ideal-gas identity. The fluid must provide positive density
and viscosity in its admissible domain. The hydraulic diameter and roughness
are fixed scalar or cell profiles. The speed includes both velocity components
for annular flow.

The friction convention is **Darcy**:

$$
\tau_w = \frac{f_D}{8}\rho |\boldsymbol{v}|^2,
\qquad f_D=\frac{64}{\mathrm{Re}}\quad(\mathrm{Re}<2300).
$$

The turbulent branch retains the existing Haaland correlation, including its
coefficient $-1.81$. Its switch at Reynolds number 2300 is sharp; no smoothed transition correlation is supplied. `Haaland` preserves the historical
turbulent-only public function and its zero value at nonpositive Reynolds
numbers. `darcy_friction_factor` is the laminar/turbulent policy and returns
infinity at zero Reynolds number, reflecting the singular factor. Negative
Reynolds numbers return NaN. Compute stresses with `HaalandWall` or
`haaland_wall_shear`, rather than multiplying that infinite factor by zero.

The signed laminar shear is evaluated analytically:

$$
\tau_w = \frac{8\mu}{D_h}u.
$$

There is no Reynolds floor. Its derivative at zero speed is $8\mu/D_h$.
`HaalandWall.drag_coefficient` exposes the finite coefficient
$\tau_w/|\boldsymbol{v}|$ so that the model applies it to velocity components
directly, without dividing by a zero norm. Wall shear removes momentum and
annular angular momentum. For a stationary adiabatic wall it does **not** remove
total energy; dissipation increases internal energy as kinetic energy falls.
Prescribed wall heat alone contributes a wall energy source.

The policy's optional `explicit_timestep(physical, radius, fluid,
wall_measure_per_volume)` returns a per-cell drag relaxation bound. It uses the
larger radial/tangential shear derivative while holding the current EOS
transport properties fixed, and returns infinity when wall area is zero. The
model returns the physical bound to the shared numerical operator, which
combines wave, source and filter restrictions through inverse time limits.
This is a local estimate: it does not
establish stability across the sharp transition, arbitrary thermoviscous
feedback, or imposed heating. Stage admissibility and accepted-step residual
checks remain necessary. Custom stiff sources should provide their own model
`explicit_timestep` restriction.

The retained mixture functions can be used by user-defined physical models:

| Function | Preserved meaning / domain |
|---|---|
| `viscosity_Beattie` | Beattie mixture viscosity; vapor mass fraction $0\le x\le1$; positive phase densities and viscosities |
| `viscosity_MacAdams` | Harmonic mixture viscosity; historical spelling retained |
| `LM_gronnerud` | Grønnerud two-phase multiplier; nonzero liquid-reference mass flux, positive geometry/properties |
| `LM_condensation` | Historical condensation multipliers and reference phase shear, with Darcy factors; $0<x<1$, positive reference Reynolds number and speed |
| `f_HEM_purdue` | **Fanning** factor: laminar $16/\mathrm{Re}$; multiply by four for Darcy |
| `richardson` | Richardson multiplier using mixture void fraction; singular pure-vapor limit |

The empirical formulas are retained in their existing valid domains; this does
not validate their applicability to a new fluid or two-fluid model. Singular
endpoints remain explicit. The condensation correlation returns NaN outside
its stated domain; no quality, velocity or Reynolds clipping substitutes a
different law. These functions support JIT, vectorization and differentiation
inside their valid domains.

Run the production closure checks with:

```powershell
poetry run python -B -m pytest tests/test_physical_closures.py -q
```

## Directional annular loads and optional NozzleX import

An annular wall policy can supply
`directional_stress(physical, radius, fluid) -> (tau_m, tau_theta, q_w)`.
The two stresses are signed components in Pa, subtracted from meridional
momentum and radius-weighted angular momentum respectively. Heat is in W/m²,
positive into the fluid. This takes precedence over the scalar callable
interface; existing scalar and Haaland policies keep their behavior.

`models.euler_annular.closures.NozzleXWall` is an optional adapter to the
separately installed NozzleX package. It imports the existing friction and
heat factories and the shared `evaluate_wall_sources` function; it does not
copy correlations or make NozzleX a required dependency of the base package.
Bind it to the same native mesh as the model, and supply NozzleX's
`model_options` mapping and `alpha_in_deg`. The bound geometry and inlet
context stay fixed. The EOS must support HP and HS evaluations with density,
entropy, temperature, viscosity, conductivity and heat capacity.

The adapter supports zero friction, constant Fanning friction, Aungier wall,
diffusion and curvature losses, adiabatic walls and Reynolds-analogy heating.
Diffusion and curvature losses act only meridionally. Curvature uses the
nonnegative magnitude; Reynolds analogy uses local stagnation temperature
and wall friction alone. Existing NozzleX Reynolds floors, entrance correction
and hard-coded transition behavior remain explicit imported approximations.
The empirical policies assume forward meridional flow. No universal explicit
source-timestep bound is supplied by this adapter; the verification uses
implicit pseudo-time marching.

See [the annular verification study](../cases/verification_annular/README.md)
for installation, exact configurations, shared geometry, reference-equation
corrections and measured coverage. Optional adapter tests skip when NozzleX
is not installed; they do not make it a production installation requirement.

## Two-fluid closures from the frozen Alberto reference

`models.two_fluid.ClosureConfig` configures the port of
`origin/alberto-branch` at `faafff704e54fd866abff95fcb88fc0a5217405c`.
The implementation is in `_alberto_closures.py` and `_alberto_sources.py`;
the older retained two-fluid files are not its reference or import path.
See [the migration guide](two_fluid_migration.md) for state, boundary and
discretization details. Pass `closures=None` to explicitly disable interphase
exchange, or `ClosureConfig(...)` to enable it. No implemented upstream
two-fluid wall-friction law was available.

| Selector | Implemented choices | Limits and approximations |
|---|---|---|
| `area_law` | `original`, `polynomial`, `parabola`, `segments` | Interfacial area in m$^{-1}$ and diameter in m; `original` uses bubble/droplet counts per m$^3$. Other laws prescribe area from void fraction. |
| `drag_bubbly`, `drag_droplet` | `schiller_naumann`, `ishii_zuber` | Independent selectors; Ishii–Zuber includes its viscous regime only, not distorted/Newton regimes. |
| `htc_liquid_bubbly` | `ranz_marshall`, `jakob`, `max`, `liao` | Jakob/Liao choices require the mass-transfer/saturation formulation. |
| `htc_liquid_droplet` | `internal`, `jakob`, `max` | Used when `regime_htc=True`; internal Nu is the declared upstream approximation. |
| `regime_blend` | `linear`, `smoothstep` | Linear uses frozen end-regime drag values across the transition; smoothstep blends actual-state correlations with a $C^1$ weight. |
| `mass_transfer` | true/false | One-component heat/mass exchange or two-component heat/drag. Positive mass source means liquid-to-gas evaporation. |

`area_params` are passed by name and now override the base correlation defaults,
including `d_min_b`; upstream's call-site keyword could shadow this override.
No normalization by evolving phase mass or volume is used in convergence tests.

The following numerical/empirical regularizations are explicitly retained;
they are not replacements for EOS domain checks:

| Policy | Frozen value/behavior |
|---|---|
| Drag at zero slip | The port cancels the $1/Re$ factor analytically against $|\Delta u|\Delta u$. It preserves nonzero-slip correlations and restores the finite Stokes derivative at exact rest; upstream's $Re\ge10^{-12}$ and viscosity denominator floors are removed from drag. Accepted states require finite positive viscosity. |
| Ranz–Marshall Reynolds input | $\max(Re,10^{-12})$ before its square root; finite zero-slip AD with a tiny conduction-limit offset. |
| Liao Nusselt square root | Argument bounded below by $10^{-12}$; declaration retained from the frozen closure. |
| Ishii–Zuber dispersed fraction | Capped at 0.9 inside the mixture-viscosity correlation to keep the inactive regime finite. This is not a cell void-fraction clip. |
| Dispersed diameter | Law-specific physical diameter floors; original source call uses $d_{min,b}=5\times10^{-6}$ m unless `area_params` overrides it. |
| Effective slip | `smooth_slip=False` by default; opt-in $\sqrt{(u_g-u_l)^2+u_0^2}$ changes HTC and uses `smooth_slip_u0=2` m/s by default. |
| Evaporation gate | Binary liquid enthalpy/temperature thresholds, OR vapor fraction above 0.05; optional pressure cutoff. The latter must remain inside a valid saturation domain. |

The phase property calls do not clip pressure or enthalpy. Accepted states must
remain in both PH table domains, have positive density, temperature, sound
speed and transport data, and remain in the tabulated saturation-pressure
interval when the selected physics reads saturation. This prevents JAXProp's
endpoint saturation interpolation from silently supplying out-of-range physics.

`PhasePinConfig` selects independent, off-by-default equation changes: the
binary vapor saturation-energy pin and the finite-time vapor momentum pin.
The momentum time is in seconds and must be supplied explicitly. The older
momentum blend without a relaxation time is rejected because its steady root
depends on the pseudo-time step. Pins are restricted to implicit steady solves;
unmodified physical balance defects remain visible beside selected-equation
convergence. Endpoint boundary traces have no connection to these thin-phase
policies.
