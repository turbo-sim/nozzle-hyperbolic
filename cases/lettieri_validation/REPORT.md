# Lettieri HEM validation results

The updated [solver-verification study](SOLVER_COMPARISON.md) uses HLL,
MUSCL reconstruction and the van Leer limiter, sequenced through 50, 100 and
200 cells for all five cases, both with and without Haaland friction.
All 30 new FVM stages and ten resampled critical space marches converged.
At 200 cells, pressure RMS differences span 0.0164-0.6437%;
L1 remains nonmonotonic across the three grids, so grid independence is not
claimed. The report includes method/friction/grid selectors, geometry,
white-faced FVM samples, logarithmic errors and a grid-accuracy summary.
The earlier comparison and original validation tables below retain their
recorded source epochs. See the solver study for exact audits and limits.

The [interactive HTML report](REPORT.html) has been modernized with Plotly,
operating-case/mesh/attempt selectors, light/dark themes, and interpretation.
The original numerical tables and static exports below remain unchanged.
The earlier presentation-only update reran no numerical cases; the new
comparison campaign is documented separately above.

Generated from saved outputs on 2 November 2026. All cases use the shared Euler/HEM
model, zero wall friction/heating, measured contour, and equilibrium JAXProp bicubic
properties. See [README.md](README.md) for sources, units and reproduction.

5 of 5 finest-grid cases converged. Their pressure RMSE spans 3.345–4.310 bar. These errors measure agreement with the experiment; the residual tolerances measure the discrete root.

The data/geometry audit checks five 13-point pressure profiles, exact L1/L5
publication agreement, byte-exact upstream workbooks, uniform/nonuniform spline
volumes and pressure/geometry cancellation. It does not certify the flow solutions.

| Case | Cells | Stretch | Status | Max equation L2 | Pressure RMSE (bar) | Accepted updates |
|---|---|---|---|---|---|---|
| L1 | 24 | 1.0 | converged (preflight; domain unaudited) | 1.234e-10 | 3.67527 | 40 |
| L1 | 24 | 1.0 | converged | 1.234e-10 | 3.67527 | 40 |
| L1 | 48 | 1.0 | converged | 1.327e-10 | 3.45802 | 43 |
| L1 | 48 | 1.6 | converged | 6.034e-10 | 3.38110 | 52 |
| L1 | 96 | 1.0 | converged | 7.963e-10 | 3.34495 | 47 |
| L2 | 48 | 1.0 | converged | 3.055e-10 | 4.48312 | 43 |
| L2 | 96 | 1.0 | converged | 8.220e-10 | 4.30964 | 46 |
| L3 | 48 | 1.0 | converged | 6.929e-10 | 3.71216 | 43 |
| L3 | 96 | 1.0 | converged | 1.116e-10 | 3.75141 | 47 |
| L4 | 48 | 1.0 | converged | 5.493e-11 | 3.92503 | 44 |
| L4 | 96 | 1.0 | converged | 2.978e-10 | 3.75013 | 47 |
| L5 | 48 | 1.0 | converged | 7.127e-11 | 4.19025 | 44 |
| L5 | 96 | 1.0 | converged | 4.884e-10 | 3.98558 | 47 |

Pressure errors for unsuccessful runs describe unfinished states and are not
validation conclusions. Convergence requires every transported equation, boundary
equation, global budget and the physical outlet audit to pass the unchanged fixed
targets. Exact final residuals, failure reasons and all attempted steps are retained
in each `outputs/<case>/result.json`. Preparation costs/table hashes are retained in
`outputs/properties.json` and in each result, separately from solve time.

## Fresh residual and physical-budget audit

The following audit re-evaluates the saved fields without advancing them, using
the original spatial equations and independently reducing volume-weighted L2
norms. The numerical flux telescoping identity, physical inlet/outlet regime and
admissibility are checked separately. These are dimensionless fixed-scale
residuals; thresholds are $10^{-9}$, with no relaxed acceptance rule.

| Case | Cells | Fresh audit | Mass L2 | Momentum L2 | Energy L2 | BC max | Global max | Numerical flow (kg/s) | Half-channel per depth (kg/(s m)) | Outlet Mach |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 96 | passed | 1.05e-10 | 5.48e-11 | 7.96e-10 | 1.09e-15 | 1.87e-10 | 0.376010 | 29.0131 | 1.5587 |
| L2 | 96 | passed | 1.05e-10 | 4.75e-11 | 8.22e-10 | 1.19e-15 | 1.21e-10 | 0.430481 | 33.2162 | 1.5325 |
| L3 | 96 | passed | 1.37e-11 | 6.04e-12 | 1.12e-10 | 3.15e-16 | 1.71e-11 | 0.498620 | 38.4737 | 1.5190 |
| L4 | 96 | passed | 3.55e-11 | 1.50e-11 | 2.98e-10 | 1.35e-15 | 4.78e-11 | 0.562363 | 43.3922 | 1.5057 |
| L5 | 96 | passed | 5.72e-11 | 2.27e-11 | 4.88e-10 | 1.43e-15 | 7.97e-11 | 0.624718 | 48.2035 | 1.4906 |

Mass flow in this table comes from the conservative numerical face flux.
The saved cell quantity $\rho u A$ varies at finite resolution and is a separate
physical diagnostic. The per-depth column means half-channel flow per unit
depth, not the mass flow through half of the 6.48 mm hardware depth.

## Mesh sensitivity

| Case | Cells | Max pressure change at taps (bar) | Coarse RMSE (bar) | Fine RMSE (bar) |
|---|---|---|---|---|
| L1 | 48 to 96 | 0.70210 | 3.45802 | 3.34495 |
| L2 | 48 to 96 | 0.60364 | 4.48312 | 4.30964 |
| L3 | 48 to 96 | 1.98802 | 3.71216 | 3.75141 |
| L4 | 48 to 96 | 1.27911 | 3.92503 | 3.75013 |
| L5 | 48 to 96 | 1.56895 | 4.19025 | 3.98558 |

These two resolutions are a sensitivity check, not proof of grid independence.
Pressure comparisons linearly interpolate saved primitive states to the
measurement positions. The reported errors combine discretization, geometry,
equilibrium-model and measurement differences; they are not uncertainty-weighted
scores. The nonuniform L1 run is an additional geometry/solver check.

![Five-case pressure profiles](outputs/pressure_profiles.png)

## Liquid fraction and measured condensation onset

![Equilibrium liquid fraction and measured onset](outputs/liquid_fraction_profiles.png)

The dotted lines show the measured condensation locations from Appendix Table 3,
not a computed onset. The only liquid-fraction reference curves available are
L1/L5 CFD; no experimental liquid-fraction samples are invented. The HEM curve
uses the saved raw JAXProp quality $Q$ and plots $1-Q$ only for finite
$0\le Q\le1$. Single-phase sentinels such as $Q=-1$, and interpolation overshoots
outside that interval, are left as gaps rather than clipped.

L1: N=96, 1 masked samples, raw Q range 0.908071 to 1.001355; L2: N=96, 1 masked samples, raw Q range 0.862994 to 1.001305; L3: N=96, 0 masked samples, raw Q range 0.820039 to 1.000000; L4: N=96, 0 masked samples, raw Q range 0.764007 to 1.000000; L5: N=96, 0 masked samples, raw Q range 0.697291 to 1.000000.

This affects only this diagnostic plot, not the solver or EOS.
The exact per-run diagnostic counts are in `outputs/quality_diagnostics.json`.
No threshold-based numerical onset is claimed from this interpolated
equilibrium field. HEM phase appearance and delayed experimental condensation
are physically distinct; the measured onset remains an independent reference.

![Finest-grid per-equation residual histories](outputs/residual_histories.png)

The original L1/L5 Excel and reference CFD series are preserved. L2–L4 pressures
and reservoir inputs were recovered from Appendix Tables 2/3; their exact tap
coordinates, original workbooks and reference CFD files remain unavailable.
Nominal tap coordinates are explicitly approximate, with no invented uncertainty.
The 6.48 mm depth reproduces Andrea's convention and agrees with the primary
20 mm² throat area divided by the 3.0862 mm full throat height. Experimental
mass flow is not supplied. Full-channel and half-channel-per-depth
mass flows are reported separately in the machine-readable results.

The experiment measures nonequilibrium condensation. Equilibrium HEM is not
expected to reproduce delayed condensation identically. The report therefore
keeps numerical convergence separate from experimental agreement. No reference
CFD data were used to initialize the numerical solver.

The original PNG pressure/residual exports are retained. The modern HTML uses
Plotly and embeds its runtime, plotted values and original input downloads.
Structural and application checks cover selectors, navigation, themes, arrays,
reference availability and exact quality masking. The solver-comparison update
also uses an isolated Edge audit of the actual embedded Plotly figures,
including all operating cases and grids, solver visibility, marker skipping,
geometry, equations, sensitivity evidence and mobile/offline rendering.
