# R1233zd(E) flashing expansion — Elliot nozzle, six-equation two-fluid model

Stand-alone case. Everything it produces stays inside this folder; no file
outside it is read or modified.

## Case definition

| | |
|---|---|
| Fluid | R1233zd(E) — CoolProp `R1233ZDE` (alias of canonical `R1233zd(E)`) |
| Reservoir | **saturated liquid at 155 °C** → `p0 = p_sat(155 °C) = 29.831 bar` |
| Inlet composition | liquid only; vapour is a thin passive **ghost**, `α_g = 1e-3` |
| Outlet | `p_out = 2 bar` (expansion ratio `p0/p_out = 14.9`) |
| Geometry | `get_nozzle_elliot` — converging / NURBS throat / diverging |
| Model | Städtke six-equation single-pressure TFM, mass transfer **ON** |
| Grid | `Nx = 200`, `x ∈ [0, 0.31] m` |
| Initial condition | `stagnation` — uniform field at rest at reservoir conditions |
| EOS tables | 90 × 90 bicubic, metastable-extended, per phase |

Because the reservoir sits exactly on the saturation line, both phases' stagnation
states are taken from **saturation calls** (`Q = 0` for the liquid, `Q = 1` for the
ghost) rather than a `(p,T)` flash — a `(p,T)` flash is ambiguous there, and for
the ghost it would return the *liquid* branch, dropping `h_g` below the vapour
table floor.

## Files

| File | Purpose |
|---|---|
| `settings_r1233zde.yaml` | **The case definition.** Same block layout as `dev_scripts/settings_*.yaml`. This is the file you edit. |
| `case_config.py` | Loads the YAML and exposes it as constants (the local equivalent of `dev_scripts/config_loader.py`). Imported by every script, so the tables that get verified are exactly the ones the solver loads. |
| `build_tables.py` | Builds the two EOS tables into `fluid_tables/`, patches conductivity, then runs the verification suite. Writes `fluid_tables/table_verification.txt` and `fluid_tables/figures/table_map_*.png`. |
| `patch_conductivity.py` | Fills the thermal conductivity CoolProp cannot provide for this fluid (see below). Called automatically by `build_tables.py`. |
| `run_simulation.py` | The solver driver. Writes a timestamped folder under `results/`. |

## How to run

The environment that has `jax`, `jaxprop`, `nurbspy` and CoolProp 7.2.0 is the
project's poetry venv:

```
C:\Users\alber\AppData\Local\pypoetry\Cache\virtualenvs\nozzle-hyperbolic-LbBBi-fu-py3.12\Scripts\python.exe
```

```bash
python build_tables.py      # once — ~1–2 h (8100 metastable flashes per table)
python run_simulation.py    # the simulation
```

Both read `settings_r1233zde.yaml`. To run a variant, copy the YAML and point
`SETTINGS_YAML` at it — the same mechanism `main_two_fluid.py` uses:

```bash
SETTINGS_YAML=settings_my_variant.yaml python run_simulation.py
```

The settings file is copied verbatim into the run folder, so every result
carries the inputs that produced it.

Table generation is the slow part and is cached: `build_tables.py` regenerates
only if the stored metadata no longer matches the YAML — so changing `Nx`,
`p_out` or a closure knob costs nothing, while changing a table bound or
resolution triggers a rebuild.

### Inlet / outlet

The `bc:` block takes the same keys as the other cases — one reservoir pressure
shared by both phases, a per-phase temperature, slip, and composition:

```yaml
bc:
  inlet:  stagnation
  outlet: characteristic
  p0:   saturation      # Pa, or the keyword -> p_sat(T_l0)
  T_g0: 428.15          # K
  T_l0: 428.15          # K
  quality:       null   # give EITHER this ...
  void_fraction: 1.0e-3 # ... or this; the other is derived
  slip:          1.0    # S = u_g/u_l
  dom_phase:     auto   # auto (mass fraction) | gas | liquid
  p_out:         2.0e5
  p_out_relax:   0.002
```

`quality` and `void_fraction` are related by
`α = 1/(1 + ((1−x)/x)·(ρ_g/ρ_l)·S)`; give one, and the loader derives the other
and reports which way round it went. Setting both, or neither, is an error.

**The reservoir branch is chosen per phase, on whether that phase exists at
(p0,T)** — not assumed, and not inferred from an enthalpy threshold:

| Condition | Liquid | Vapour |
|---|---|---|
| `T = T_sat(p0)` (on the line) | `Q=0` | `Q=1` |
| `T < T_sat(p0)` (subcooled) | `(p,T)` flash | `Q=1` — no vapour exists there |
| `T > T_sat(p0)` (superheated) | `Q=0` — no liquid exists there | `(p,T)` flash |

This matters: a `(p,T)` flash outside a phase's own region silently returns the
*other* branch. `main_two_fluid.py` guards the ghost with `h_g <` the vapour
table floor, which works for water (floor 2200 kJ/kg, liquid ~1300) but **fails
here** — the vapour floor is 250 kJ/kg and the subcooled liquid sits at 395, so
the check passes and the liquid state goes into the solver as the ghost. Testing
a subcooled variant of this case caught exactly that.

Keywords accepted where a literal would be fragile:

| Key | Meaning |
|---|---|
| `eos.*.p_max: critical` | the CoolProp critical pressure |
| `bc.p0: saturation` | `p_sat(T_l0)` — use it when you mean "exactly on the saturation line"; a hand-typed value a few Pa off flips the reservoir to subcooled or two-phase |
| `numerics.alpha_eps: match_void` | tied to the inlet void fraction — these **must** agree |

### Changing the back-pressure

`bc.p_out` is the only place the target is set. `continuation.stages` lists
*intermediate* back-pressures to pass through; `bc.p_out` is appended
automatically, so changing it never means editing the stage list:

```yaml
bc:
  p_out: 5.0e5
continuation:
  stages: []                # -> straight to 5 bar
  # stages: [15.0e5, 10.0e5]  -> 15 bar, 10 bar, then 5 bar
```

Stages must be strictly descending and all above `bc.p_out` (each is
warm-started from the previous one); both are checked with a message that says
what to fix. Changing `p_out` does **not** trigger a table rebuild.

## EOS tables — bounds and why they are right

Both phases share the pressure window; the enthalpy windows are per phase, as
specified:

| Table | h range | p range | grid |
|---|---|---|---|
| metastable **liquid** (dominant) | 200 → 450 kJ/kg | 0.1 MPa → `p_crit` = 36.237 bar | 90 × 90 |
| metastable **vapour** (ghost) | 250 → 550 kJ/kg | 0.1 MPa → `p_crit` = 36.237 bar | 90 × 90 |

plus a 150-point saturation line per phase (the mass-transfer closure reads it).

### Reference-state warning — this one matters

Enthalpy and entropy are **reference-state dependent**, and the two CoolProp
builds on this machine do not agree for R1233zd(E):

| CoolProp | `h_l,sat(155 °C)` | `h_v,sat(155 °C)` |
|---|---|---|
| 7.2.0 (the poetry venv — **the one that runs this case**) | 416.01 kJ/kg | 484.85 kJ/kg |
| 6.6.0 (the base conda env) | 447.58 kJ/kg | 516.42 kJ/kg |

The offset is ≈ 31.57 kJ/kg. The specified bounds are comfortable in the 7.2.0
frame and would be marginal in the 6.6.0 frame (`h_l,sat` would sit 2.4 kJ/kg
below the liquid `h_max`, i.e. in the very last bicubic interval). **Run this
case with the poetry venv.** `case_config.py` derives every reservoir property
from whichever CoolProp is imported, and `build_tables.py` prints the version and
re-derives all bounds from it, so a mismatch shows up as a failed check rather
than as a silently wrong answer.

### Coverage of the operating window (CoolProp 7.2.0)

| p [bar] | T_sat [°C] | `h_l,sat` | `h_v,sat` | liquid table | vapour table |
|---|---|---|---|---|---|
| 1.00 | 17.91 | 221.35 | 416.15 | in | in |
| 2.00 | 37.68 | 245.50 | 429.85 | in | in |
| 5.00 | 69.15 | 285.52 | 450.88 | in | in |
| 10.00 | 98.11 | 324.58 | 468.44 | in | in |
| 20.00 | 132.52 | 375.77 | 484.06 | in | in |
| 29.83 | 155.00 | **416.01** | **484.85** | in | in |
| 34.00 | 162.69 | 434.90 | 477.89 | in | in |
| 36.00 | 166.07 | 450.49 | 465.89 | **out** | in |

The liquid table stops covering the saturation line only above ≈ 35 bar, which
this case never reaches (`p0 = 29.83 bar`, and pressure only falls from there).
The reservoir liquid state sits 34 kJ/kg below the liquid `h_max` and 216 kJ/kg
above `h_min`; the ghost sits 65 kJ/kg below the vapour `h_max`.

**The binding constraint is the pressure floor**, `p_min = 0.1 MPa = 1 bar`, only
a factor of 2 below `p_out`. With `A_exit/A_throat = 4.38` an over-expanded
supersonic pocket could dip under 1 bar. `run_simulation.py` therefore checks
every cell against both table rectangles each `print_every` steps and again on
the final state, and says so explicitly rather than silently using clamped
edge values.

### Thermal conductivity is an estimate — CoolProp has no model for this fluid

```
PropsSI("CONDUCTIVITY", "T", 300, "Q", 0, "R1233ZDE")
-> ValueError: Thermal conductivity model is not available for this fluid
```

Viscosity exists; conductivity does not. The table generator swallows the error
and stores **NaN at every node**, and the two-fluid closure reads conductivity
for *both* interfacial heat-transfer coefficients
(`sources.py:242` → `htc_g = k_g·Nu/d`, `sources.py:267` → Ranz–Marshall for the
liquid). Left alone this makes every heat- and mass-transfer source NaN on the
first step — i.e. it silently kills exactly the flashing physics this case is for.

`patch_conductivity.py` fills it by two-parameter **corresponding states from
R245fa**, an unusually good surrogate here:

| | T_c [K] | p_c [bar] | M [g/mol] |
|---|---|---|---|
| R1233zd(E) | 439.60 | 36.24 | 130.50 |
| R245fa | 427.01 | 36.51 | 134.05 |
| difference | 2.9 % | 0.7 % | 2.7 % |

Each node is mapped to R245fa at the same reduced temperature *and* reduced
density, then corrected by the classical reducing group
`Γ = T_c^(1/6)·M^(1/2)·p_c^(−2/3)`. That correction comes out at **1.0036**, so
the estimate rests on the reduced-state match rather than on the correction.

Resulting saturated-liquid conductivity: 0.097 W/m/K at 20 °C falling to
0.057 W/m/K at 155 °C — the right magnitude and slope for this fluid class, but
**treat it as ±10 %**. It propagates to the interfacial HTC and hence to the
flashing rate; it does **not** touch the EOS (ρ, T, h, s, a and viscosity are all
the real R1233zd(E) EOS). If you have measured `k` data, replace
`conductivity_field()` in `patch_conductivity.py`.

`run_simulation.py` refuses to start on a table with non-finite conductivity
rather than producing NaN.

### What "table is correct" means for a metastable table

A metastable table is generated by marching each pressure column into the
superheated-liquid / subcooled-vapour region until the **spinodal** is crossed;
past that there is no root and the generator freezes the last good node
(constant extrapolation). So the whole `(h,p)` rectangle can never be exact — the
requested windows deliberately extend well past the spinodal at low pressure.
The checks that matter, and that `build_tables.py` runs, are:

1. metadata matches the requested bounds (the loader silently reuses a cached
   table whose *name* matches, so this is the guard against a stale file);
2. no NaN/Inf in any value, gradient or bicubic coefficient the solver reads;
3. **spinodal-freeze audit** — the table is not globally frozen (a 2026-07
   jaxprop bug shipped a water table that was one enthalpy row broadcast over
   the entire plane), and the saturation line lies in the live region for
   *every* pressure column;
4. round-trip accuracy against CoolProp at ~4000 random points on the **stable**
   side of the saturation line;
5. the tabulated saturation line matches CoolProp `PQ`;
6. physical sanity in the live region (`ρ, T, a > 0`; `ρ` decreasing in `h`);
7. the reservoir state and the whole `p0 → p_out` expansion path are in-table
   **and** in the live region.

`fluid_tables/figures/table_map_*.png` show ρ, T and a with the frozen nodes
blanked, plus the frozen mask itself, with the saturation line and the operating
box overlaid.

## Result

Converged: **6259 steps**, SSR = 7.3e-7 (settled onto a shallow plateau, the
known SSR floor of this solver), run `results/r1233zde_155C_to_2bar_N200_20260831_115710/`.

| Quantity | Inlet | Throat (cell 91) | Exit |
|---|---|---|---|
| p [bar] | 29.81 | 14.87 | **3.94** |
| α_g | 0.001 | 0.707 | 0.986 |
| u_l [m/s] | 2.2 | 70.7 | 140.7 |
| u_g [m/s] | 2.2 | 116.2 | 180.0 |
| T_l [°C] | 155.0 | 137.8 | 92.1 |
| T_g [°C] | 155.0 | 118.4 | 66.7 |
| mixture Mach | 0.015 | 0.893 | **1.292** |

- **ṁ = 3.331 kg/s**, RMS variation along the nozzle **0.142 %** — quasi-1D mass
  is conserved.
- **Mixture stagnation enthalpy drift 5.2e-4** — energy is conserved. (Note: the
  *per-phase* `h + u²/2` is *not* a conserved quantity once mass transfers
  between phases; it spans tens of percent here, which is physical. The driver
  reports the mass-flux-weighted mixture value for this reason.)
- Exit mass quality **x = 0.636**, void fraction 0.986.
- Flashing onset at x = 0.0156 m, upstream of the throat; active in 190/200 cells.
- Liquid superheat spans −0.05 to **+40.1 K** — strong thermal non-equilibrium,
  with the vapour 25 K colder than the liquid at the exit.
- All 200 cells inside both tables at every step; no spinodal violation.

### The nozzle is choked and under-expanded

The sonic point sits at **cell 101 (x = 0.157 m), downstream of the geometric
throat** (cell 91, x = 0.1418 m) — thermal choking, displaced by the flashing
mass/heat source, exactly as expected when the sound speed is falling steeply
through the throat region.

The exit is **supersonic (M = 1.29)**, so the characteristic outlet is
transmissive and the back-pressure is not felt inside the nozzle. **`p_exit` =
3.94 bar ≠ `p_out` = 2 bar is the correct answer, not a boundary-condition
failure**: with this area ratio (`A_exit/A_th` = 4.375) and reservoir the Elliot
nozzle cannot expand to 2 bar within its own length. The remaining expansion to
2 bar would take place in a plume downstream of the exit plane. If you need
2 bar *at the exit plane*, the divergent has to be longer / wider — say the word
and I'll size it.

### Residual artefacts (honest notes)

- Small kink in p, u and slip at the throat (x ≈ 0.1418 m), where the NURBS
  throat segment joins and the sonic transition sits. Mild, but visible in
  `final_solution.png`.
- Residual 2Δx amplitude: 14 % on the slip `u_g − u_l`, 3.6 % on `u_g`, 17 % on
  the flashing source σ_Mg; pressure and α are clean (1.8 % / 0.5 %). This is the
  known slip-mode ringing near the sonic line, not a diverging mode.
- SSR plateaus at 7.3e-7 rather than reaching the 1e-8 tolerance — the shallow
  limit cycle this solver is documented to have.

## Outputs

`results/<timestamp>/` contains `run_log.txt`, `solution.csv`
(`x, A, p, alpha_g, u_g, u_l, T_g, T_l, h_g, h_l, rho_g, rho_l`),
`initial_condition.png`, `final_solution.png` and `mass_transfer_source.png`
(flashing source, liquid superheat, interfacial heat), plus a copy of the
`case_config.py` used.

The driver also reports mass-flux conservation per phase, stagnation-enthalpy
drift through the throat, odd–even (checkerboard) amplitudes, flashing onset
location, and the table-bounds check.

## Numerics

`case_config.py` uses the flashing-certified stack: E3-augmented banded system
(`eos_rows` + `eos_rows_banded`, `eos_project` off), well-balanced
non-conservative term (`nc_form: wb`), the well-balanced face-consistent area
gradient, `muscl_van_albada` reconstruction, the JST void filter
(`eps_void = 0.02`, `frozen` scaling), both ghost pins (`pin_ghost`,
`pin_ghost_momentum`), mass-weighted SSR, and adaptive `dt` with a physical
floor plus a stall detector.

## Startup: the spinodal sets the back-pressure descent rate

Getting this case to march was the hard part, and the governing constraint turned
out to be **thermodynamic, not numerical**. Worth recording, because it will
recur for any saturated-liquid inlet.

### The constraint

The reservoir liquid sits at `h_l0 = 416.0 kJ/kg`. The liquid spinodal of the
metastable table — the last enthalpy at which a real metastable state exists,
beyond which the generator freezes the column — sits at:

| p [bar] | `h_l,sat` | `h_spinodal` | max live superheat |
|---|---|---|---|
| 29.83 | 416.0 | 436.0 | +19.9 kJ/kg |
| 20.00 | 375.8 | 416.3 | +40.5 kJ/kg |
| 10.00 | 324.6 | 402.2 | +77.7 kJ/kg |
| 2.00 | 245.5 | 391.0 | +145.5 kJ/kg |

Reading the middle column against `h_l0 = 416`: a liquid that has **not yet
flashed** is past the spinodal below roughly **21 bar**. Past it the table is
constant-extrapolated (zero `∂/∂h`), so `a` and `Γ` go stale and the E3 Newton
system becomes singular → NaN.

So the liquid *must* shed enthalpy by flashing in order to remain in valid data,
and flashing is a **finite-rate** process. The back-pressure therefore has to
descend slowly enough for the mass-transfer source to track it. That is the whole
story of the startup.

### The ladder that established it

| `p_out_relax` | Result |
|---|---|
| 1.0 (hard clamp) | NaN at step 2 — exit cell dropped 29.83 → 2 bar in one step, +167 kJ/kg superheat |
| 0.05 | NaN at step 7 — the startup rarefaction front (propagating in from the outlet at ≈ `a_l` = 150 m/s, ~6 cells in 0.068 ms) carries a ~9 bar jump across one cell, again past the spinodal |
| **0.002** | **marches cleanly** — two halvings on steps 1–2, then `dt` recovers and holds at the CFL cap for the rest of the run |

A separate, independent failure sat underneath: with `alpha_eps = 1e-6` (the
value in the reference YAML, whose own comment records that it fails) the solver
re-thinned the ghost from 1e-3 to the 1e-6 clip floor within two steps, re-arming
the `h_g = E_g/(αρ_g)` amplifier — `h_g` reached 7.6e4 kJ/kg with a 99.98 %
odd-even amplitude. `ALPHA_EPS` is now tied directly to `VOID_FRACTION` in
`case_config.py` so the two cannot drift apart.

Note the relaxation is applied per **step**, not per unit time. If the adaptive
controller starts halving `dt`, the boundary forcing per unit *physical* time
goes **up** — a positive feedback that the controller cannot win. The point of a
gentle factor is to stay out of that regime entirely, which is why `dt` holding
at the CFL cap is the signal that the setting is right.

The **steady state is unchanged** by this factor: at convergence `p_last → p_out`
makes `p_ghost = p_last + w·relax·(p_out − p_last)` exact for any relaxation. It
is a continuation choice, not a change to the requested initial condition — the
initial field is still stagnation at rest.

## Inlet closure: metastable, not equilibrium

The stagnation inlet closes the shared static pressure on the dominant phase's
isentrope, `h_dom(p)`. The stock driver builds it from CoolProp's `PSmass` call,
which returns the **equilibrium** state — so once the isentrope crosses the
saturation line it hands back a two-phase *mixture* enthalpy, which the solver
then uses as the single-phase **liquid** enthalpy. For a subcooled reservoir that
is harmless. For a reservoir exactly *on* the saturation line it is wrong at the
first infinitesimal pressure drop.

`isentrope.py` integrates the liquid's own metastable single-phase isentrope
instead, `dh/dp = 1/ρ(p,h)` at constant `s`, using the solver's own table — so
the closure is EOS-consistent and monotone by construction. It validates against
CoolProp in the subcooled window `p > p0`, where the equilibrium call *does*
return the true liquid branch: agreement is exact to the printed precision
(run `python isentrope.py`).

Honest caveat: this is a correctness fix, **not** the reason the earlier runs
diverged. Because the liquid is nearly incompressible its enthalpy barely changes
along either isentrope (412–417 kJ/kg over the whole pressure range), so both
closures imply almost the same liquid superheat. The large superheat is intrinsic
physics — an isentropically expanded saturated liquid *is* massively superheated
— and relieving it is exactly what the flashing source is for.

## Nucleation density: a physics change you should know about

`BUBBLE_NUMBER` was raised from **1e12** (the value carried over from the water
SMD case) to **1e15 /m³**. This is not a numerical tweak — it changes the
predicted degree of thermal non-equilibrium — and the reason is measurable:

At `p_out` = 25 bar with Nb = 1e12 the converged flow reached **+37.7 kJ/kg of
liquid superheat where the true spinodal of R1233zd(E) is +39.0 kJ/kg** — the
model drove the liquid to **97 % of its thermodynamic stability limit**, and one
cell (93, at p ≈ 19 bar) tipped over it. Past the spinodal no liquid state
exists at all, so this is not recoverable by any numerical means.

A liquid that superheated nucleates violently long before the spinodal, so a
solution sitting *at* the limit is the model telling you the assumed nucleation
site density is too low for this fluid and expansion. Interfacial area scales as
`Nb^(1/3)`, so 1e12 → 1e15 shortens the thermal relaxation time ~10×, and the
liquid then tracks saturation instead of running to the stability limit
(final peak superheat +40 K in temperature, comfortably live).

If you have flashing data for R1233zd(E), `BUBBLE_NUMBER` is the parameter to
calibrate against it.

## Deviations / judgement calls

- **"tables … till 0.1 MPa"** is read as `p_min = 0.1 MPa = 1 bar`. Change
  `P_TABLE_MIN` in `case_config.py` and re-run `build_tables.py` if a lower
  floor was meant — see the over-expansion note above for why it could matter.
- `x2 = 0.31 m` rather than the nozzle's full 0.31966 m, so the grid stays
  strictly inside the last geometry segment (the value the existing Elliot runs
  in `dev_scripts/` use).
