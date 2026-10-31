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
coefficient $-1.81$. Its switch at Reynolds number 2300 is sharp; this port does
not introduce a new transition correlation. `Haaland` preserves the historical
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
.venv/Scripts/python.exe -m pytest tests/test_physical_closures.py -q
```
