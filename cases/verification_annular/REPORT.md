# Annular bend verification results

The corrected NozzleX space marcher and production hyperbolic annular time
marcher agree closely for this smooth, swirling radial–axial bend. All 20
FVM solves passed their original spatial, boundary and global residual checks.
The comparison covers 50, 100, 200 and 400 cells for five closure combinations.

[Open the standalone report](REPORT.html) for geometry, plasma duct renderings,
field overlays, mesh refinement, residual histories, heat flux, closure
components and the embedded evidence download. Full setup and commands are
in [README.md](README.md); machine-readable results are in
[outputs/results.json](outputs/results.json).

| Closure case | Maximum normalized conservative L1 error, 400 cells | Inlet total Mach error |
|---|---:|---:|
| Inviscid, adiabatic | $2.8093\times10^{-6}$ | $-4.2691\times10^{-6}$ |
| Constant Fanning friction, adiabatic | $2.1485\times10^{-6}$ | $-3.5645\times10^{-6}$ |
| Full Aungier, adiabatic | $9.6527\times10^{-7}$ | $-1.7458\times10^{-6}$ |
| Full Aungier, wall at 260 K | $9.7513\times10^{-7}$ | $-1.7286\times10^{-6}$ |
| Full Aungier, wall at 340 K | $9.7302\times10^{-7}$ | $-1.7748\times10^{-6}$ |

Errors use physical-volume weights and fixed conservative scales. The
reference is sampled at the geometric cell centers; it is not an exact
volume-average projection. Errors decrease on every refinement in all five
cases and lie below the declared $10^{-3}$ finest-grid target. Inviscid and
constant-wall cases show approximately second-order refinement. Aungier
cases have less uniform observed orders as the entrance correction is
resolved; the especially large final reduction should not be interpreted
as a higher formal order.

The reference was tightened through ODE tolerances $10^{-8}$, $10^{-10}$ and
$10^{-12}$. The final refinement changes pressure by at most 0.00114 Pa,
meridional velocity by at most $1.45\times10^{-6}$ m/s and meridional Mach by
at most $3.45\times10^{-9}$ across these cases. The final reference passes
mass conservation, entropy-integration and subsonic-branch audits. Loose
references fail the strict mass-variation target in some cases; those
failures are retained in the report and their fields are not used for the
final FVM comparison.

The inlet meridional Mach is 0.4 and the flow remains meridionally subsonic
while decelerating through the bend. Aungier diffusion/curvature losses
substantially reduce pressure recovery compared with the inviscid case.
The heated and cooled walls change stagnation enthalpy in the expected
directions; adiabatic friction preserves total enthalpy. The common wall
function applies diffusion/curvature losses only meridionally and wall
friction to both velocity components.

The final campaign used unchanged source files throughout its execution;
before/after SHA-256 fingerprints are retained. Geometry evaluations agree
to $3.08\times10^{-14}$ m in radius on the mapping audit, and the full physical
endpoints are shared. The previous pilot and first complete campaign remain
archived under `outputs/`; their numerical history is not merged into the
final results. Presentation was subsequently rebuilt from saved evidence.

Validation comprises 34 targeted source/closure/adapter regression checks
and 23 existing annular physics checks, all passing. Saved-data audits check
the original residual limits, discrete budgets, refinement, field errors,
HTML/MathML, PNG payloads and the exact embedded JSON download. The report
now uses Plotly with case/grid controls, independent solver visibility,
white-face FVM markers and adjustable marker skipping. Every profile has
its matching log-scale absolute-error panel; errors and downloaded fields
retain all cell centers regardless of marker skipping. Saved numerical
values and timings are unchanged by this presentation update.

An isolated headless Edge audit exercises all five cases and four grids,
checks actual plotted arrays, solver visibility, marker thinning, errors,
themes, linked zoom, mobile layout, MathML, downloads and standalone offline
operation. The report design follows `development/real_gas/`; browser evidence
is saved separately under `outputs/browser_audit/`.

These findings verify consistency of the implemented reduced equations and
their discretizations for perfect-gas air. Shared closures are deliberately
common code, so agreement does not independently validate Aungier's empirical
accuracy or resolve boundary layers, separation and transverse pressure
gradients. The imported entrance correction retains its attribution TODO,
and the configured Reynolds transition fields remain ineffective. Timings
are single compile-inclusive observations and support no speedup claim.
