# Two-Phase Supersonic Nozzle Flow — NICFD 2026

Supplementary material for:

> Bergamin et al. "A non-equilibrium time-marching model for two-phase
> supersonic nozzle flows." *Proceedings of the 6th International Seminar on
> Non-Ideal Compressible Fluid Dynamics (NICFD 2026)*, ERCOFTAC Series,
> Springer (submitted).

This repository contains the complete case specification and the solution data
for each of the six cases presented in the paper.

## Layout

```
data/<family>/<regime>/   one folder per case, holding
                            settings.yaml   the complete case specification
                            solution.csv    the converged solution
scripts/                  regenerate the paper figures from data/
figures/                  written by scripts/make_all_figures.py (not tracked)
```

`<family>` is `phase_change` or `two_component`; `<regime>` is `subsonic`,
`supersonic_shock` or `supersonic_adapted`.

## Cases

Each family has one reservoir state. Within a family, the three regimes differ
only in the imposed back-pressure. Both phases enter at the same stagnation
pressure and temperature, with no slip.

| case | fluid | p₀ (bar) | T₀ (K) | inlet composition | p_out (bar) | length (m) |
|---|---|---|---|---|---|---|
| `phase_change/subsonic` | water, flashing | 120 | 578.95 | void fraction 0.001 | 80 | 1.2 |
| `phase_change/supersonic_shock` | water, flashing | 120 | 578.95 | void fraction 0.001 | 15 | 1.2 |
| `phase_change/supersonic_adapted` | water, flashing | 120 | 578.95 | void fraction 0.001 | 0.8 | 1.2 |
| `two_component/subsonic` | nitrogen + water | 20 | 295.15 | gas mass fraction 0.5 | 16 | 0.31 |
| `two_component/supersonic_shock` | nitrogen + water | 20 | 295.15 | gas mass fraction 0.5 | 12 | 0.31 |
| `two_component/supersonic_adapted` | nitrogen + water | 20 | 295.15 | gas mass fraction 0.5 | 0.986 | 0.31 |

## Data format

Units are SI throughout — Pa, K, m, m/s, kg/m³, J/kg, kg/s. Values are written
with 13 significant digits, beyond the precision of anything downstream.
Pressures are quoted in bar in the figures and in this file, never in the data.

The first and last rows are ghost cells. In the file, their `x` is the inlet and
outlet plane (`x = 0` and `x = L`), not half a cell outside the domain. They
carry the boundary state the scheme imposes, not an interior solution. Peak
Mach, minimum pressure and similar summary statistics should be taken over the
interior rows, `[1:-1]`.

### Geometry

| column | unit | meaning |
|---|---|---|
| `x` | m | axial coordinate, cell centre |
| `A` | m² | cross-sectional area |
| `r` | m | nozzle radius, `sqrt(A/pi)` |

### Data available

In the formulas below, `alpha_l = 1 - alpha_g` is the liquid volume fraction.

| column | unit | meaning |
|---|---|---|
| `p` | Pa | pressure |
| `alpha_g` | – | void fraction |
| `u_g`, `u_l` | m/s | phase velocities |
| `T_g`, `T_l` | K | phase temperatures |
| `h_g`, `h_l` | J/kg | phase specific enthalpies |
| `rho_g`, `rho_l` | kg/m³ | phase densities |
| `a_mix` | m/s | mixture sound speed, from the model's eigenvalues (see below) |
| `u_mix` | m/s | mixture velocity, from the model's eigenvalues (see below) |
| `Ma_mix` | – | mixture Mach number, `abs(u_mix) / a_mix` |
| `quality_static` | – | gas mass fraction, `alpha_g rho_g / (alpha_g rho_g + alpha_l rho_l)` |
| `quality_flow` | – | gas mass-**flux** fraction, `alpha_g rho_g u_g / (alpha_g rho_g u_g + alpha_l rho_l u_l)` |
| `slip` | – | slip ratio, `u_g / u_l` |
| `mdot` | kg/s | total mass flow, `A (alpha_g rho_g u_g + alpha_l rho_l u_l)` |

`a_mix` and `u_mix` are the half-difference and half-sum of the two acoustic
eigenvalues of the two-fluid model, evaluated in each cell. `a_mix` is the speed
the numerical scheme chokes on. It is not a textbook mixture sound speed such as
Wood's, and `u_mix` is not the mass-averaged velocity `mdot / (rho A)`. Neither
can be rebuilt from the other columns with a closed-form formula, which is why
both are stored.

## Thermophysical properties

Fluid properties come from jaxprop 0.5.6, tabulated per phase and, for the
flashing cases, extended into the metastable region. The table bounds and
resolutions are in each `settings.yaml`.

## Regenerating the figures

To generate all the figures, run:
```
python scripts/make_all_figures.py
```

The panels are written to `figures/`. Only numpy and matplotlib are needed.

## Licence

Released under the [Creative Commons Attribution 4.0 International
licence](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0) — see
`LICENSE`. You may share and adapt this material, including commercially,
provided you give appropriate credit. If you use the data, please cite the paper
above.

## Contact

`Alberto Bergamin` — Technical University of Denmark (DTU) — `alber@dtu.dk`
