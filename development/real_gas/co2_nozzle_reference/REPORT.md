# CO2 nozzle reference results

These are continuous, semi-analytic references for steady, inviscid, adiabatic quasi-one-dimensional Euler flow. Every flow-state property is evaluated through JAXProp's equilibrium bicubic backend using 100 × 100 tables. Smooth branches use a numerical pressure-coordinate primitive and scalar roots; the internal shock satisfies the Rankine–Hugoniot conditions, and its location is iterated to match the specified backpressure. No FVM solution is included.

The density interpolant defines the mechanical EOS. Smooth segments enforce $dh/dp=1/\rho$; interpolated entropy and sound speed are retained as diagnostics. Accordingly, root accuracy and conservation accuracy do not establish the accuracy of the 100 × 100 property approximation. See [EQUATIONS.md](EQUATIONS.md) for the equations and interpretation.

The geometries are synthetic smooth nozzles with a common Lettieri-inspired converger and different exit area ratios, rather than the measured Lettieri contour. The supercritical case uses a smaller exit area ratio to keep the full path supercritical. The equilibrium condensation case does not model nucleation delay or the experimental Wilson line.

![Synthetic nozzle geometry](outputs/figures/nozzle_geometry.png)

| Case | $A_e/A_t$ | Reservoir $p_0$ [MPa] | $T_0$ [K] | Adapted $p_b$ [MPa] | Shock $p_b$ [MPa] | Shock $x_s$ [mm] | $\dot m$ [kg/s] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong real-gas, supercritical CO2 | 1.15 | 30.00000 | 400.000 | 7.581496 | 20.400000 | 41.334035 | 0.7361335 |
| Lettieri-type equilibrium condensation of CO2 | 1.69 | 8.47400 | 313.880 | 1.775685 | 7.000000 | 42.058104 | 0.2255186 |

Each adapted backpressure equals the corresponding supersonic exit pressure. For the shock case, the pressure in the table is the prescribed outlet value used by the outer position solve.

The largest local sound-speed consistency difference is 22.88% in the two-phase case near saturation. For the supercritical case it is 2.12745e-06 relative (0.000212745%). Thus the interpolation diagnostics include a substantial local acoustic mismatch even though the density-EOS reference satisfies its conservation equations tightly. This consistency check does not measure the physical accuracy of the property table.

## Strong real-gas, supercritical CO2

The shock connects Euler Mach numbers 1.37199 and 0.74074. Pressure rises from 9.566246 to 17.563198 MPa; the table entropy rise is 1.95508 J/(kg K). The raw table quality at the adapted exit is 1.

The 100 × 100 bicubic table spans $h=260$–$600$ kJ/kg and $p=4$–$32$ MPa, with uniform spacing in $h$ and $\log p$.

Over the adapted profile, the raw table quality ranges from 1 to 1, and the tabulated compressibility factor ranges from 0.459329 to 0.706641. Quality values are physically meaningful only in equilibrium two-phase states; the table can also return 1 for a supercritical state. Values outside $[0,1]$ are displayed unchanged and have no physical mixture-quality interpretation.

![Strong real-gas, supercritical CO2: expected nozzle fields](outputs/figures/supercritical_fields.png)

![Strong real-gas, supercritical CO2: interpolation diagnostics](outputs/figures/supercritical_thermodynamics.png)

The principal Mach curves use $a_\mathrm{Euler}^2=[\rho_p+\rho_h/\rho]^{-1}$, calculated by differentiating the bicubic density. Dotted Mach curves use the separately interpolated table sound speed. The entropy-drift panel resets its anchor at the shock so that it measures smooth-path interpolation drift separately from the entropy jump.

The compact verification table takes maxima over both profiles and all applicable smooth segments. Rankine–Hugoniot error is the maximum of its mass, momentum, and energy residuals. The refinement row is the maximum dimensionless change among the recorded flow and position quantities; the entropy and acoustic rows have no zero-error acceptance criterion.

| Audit metric | Value | Acceptance limit |
|---|---:|---:|
| All required checks | pass | pass |
| Maximum mass-flow relative error | 1.15741e-12 | 1e-09 |
| Maximum total-enthalpy relative error | 0 | 1e-12 |
| Maximum Rankine–Hugoniot relative error | 1.47864e-15 | 1e-09 |
| Throat Euler Mach absolute error | 8.88178e-16 | 1e-08 |
| Maximum backpressure relative error | 1.13243e-14 | 1e-08 |
| Maximum independent smooth-primitive defect / $\lvert h_0\rvert$ | 6.85095e-14 | 1e-07 |
| Maximum relative change under ODE refinement | 3.8658e-13 | 1e-06 |
| Maximum smooth entropy drift [J/(kg K)] | 4.803e-06 | diagnostic |
| Maximum sound-speed relative mismatch | 2.12745e-06 | diagnostic |
| Raw table quality range | [1, 1] | diagnostic |
| Intended phase setup | pass | pass |

Phase requirement: Every sampled state above both critical pressure and temperature. The full checks, definitions, normalization scales, and refinement settings are retained in [outputs/summary.json](outputs/summary.json).

Saved fields: [adapted CSV](outputs/supercritical_adapted.csv), [shock CSV](outputs/supercritical_shock.csv). Export figures: [fields PDF](outputs/figures/supercritical_fields.pdf), [thermodynamics PDF](outputs/figures/supercritical_thermodynamics.pdf).

## Lettieri-type equilibrium condensation of CO2

The shock connects Euler Mach numbers 1.47638 and 0.69867. Pressure rises from 2.804621 to 6.257440 MPa; the table entropy rise is 8.40781 J/(kg K). The raw table quality at the adapted exit is 0.684426.

The 100 × 100 bicubic table spans $h=250$–$520$ kJ/kg and $p=1$–$9$ MPa, with uniform spacing in $h$ and $\log p$.

Over the adapted profile, the raw table quality ranges from 0.684426 to 1.00742, and the tabulated compressibility factor ranges from 0.417734 to 0.566391. Quality values are physically meaningful only in equilibrium two-phase states; the table can also return 1 for a supercritical state. Values outside $[0,1]$ are displayed unchanged and have no physical mixture-quality interpretation.

![Lettieri-type equilibrium condensation of CO2: expected nozzle fields](outputs/figures/lettieri_equilibrium_fields.png)

![Lettieri-type equilibrium condensation of CO2: interpolation diagnostics](outputs/figures/lettieri_equilibrium_thermodynamics.png)

The principal Mach curves use $a_\mathrm{Euler}^2=[\rho_p+\rho_h/\rho]^{-1}$, calculated by differentiating the bicubic density. Dotted Mach curves use the separately interpolated table sound speed. The entropy-drift panel resets its anchor at the shock so that it measures smooth-path interpolation drift separately from the entropy jump.

The compact verification table takes maxima over both profiles and all applicable smooth segments. Rankine–Hugoniot error is the maximum of its mass, momentum, and energy residuals. The refinement row is the maximum dimensionless change among the recorded flow and position quantities; the entropy and acoustic rows have no zero-error acceptance criterion.

| Audit metric | Value | Acceptance limit |
|---|---:|---:|
| All required checks | pass | pass |
| Maximum mass-flow relative error | 4.39682e-12 | 1e-09 |
| Maximum total-enthalpy relative error | 0 | 1e-12 |
| Maximum Rankine–Hugoniot relative error | 1.01942e-15 | 1e-09 |
| Throat Euler Mach absolute error | 6.88338e-15 | 1e-08 |
| Maximum backpressure relative error | 5.55112e-15 | 1e-08 |
| Maximum independent smooth-primitive defect / $\lvert h_0\rvert$ | 4.55329e-10 | 1e-07 |
| Maximum relative change under ODE refinement | 4.24378e-09 | 1e-06 |
| Maximum smooth entropy drift [J/(kg K)] | 0.00186048 | diagnostic |
| Maximum sound-speed relative mismatch | 0.228815 | diagnostic |
| Raw table quality range | [0.684426, 1.00742] | diagnostic |
| Intended phase setup | pass | pass |

Phase requirement: Both profiles contain interior equilibrium two-phase states. The full checks, definitions, normalization scales, and refinement settings are retained in [outputs/summary.json](outputs/summary.json).

Saved fields: [adapted CSV](outputs/lettieri_equilibrium_adapted.csv), [shock CSV](outputs/lettieri_equilibrium_shock.csv). Export figures: [fields PDF](outputs/figures/lettieri_equilibrium_fields.pdf), [thermodynamics PDF](outputs/figures/lettieri_equilibrium_thermodynamics.pdf).

## Reproduction

The source settings are in [cases.json](cases.json), and the numerical values and provenance are in [outputs/summary.json](outputs/summary.json). Both shock traces occupy the same axial position in each shock CSV. The `side` column identifies the upstream and downstream values; retain both when comparing a discretized solution.

Regenerate this report and the figures from saved fields, without thermodynamic calls or flow solves:

```powershell
.\.venv\Scripts\python.exe -B -m development.real_gas.co2_nozzle_reference.scripts.plot_results
```
