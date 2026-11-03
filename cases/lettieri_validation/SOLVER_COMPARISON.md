# Lettieri solver verification

The [offline Plotly report](REPORT.html) adds a solver-comparison page to the
existing experimental validation. The current FVM uses HLL/MUSCL/van Leer on grid-sequenced 50/100/200-cell meshes for both wall variants. Select the operating case, inviscid/Haaland
physics, FVM grid, solver visibility and marker spacing. FVM samples use white
faces above the space-marching line. Errors use the same selected physics and
grid; the spatial reference is evaluated at the actual FVM cell centers.

## Measured results: HLL / MUSCL / van Leer

All 30 new FVM grid stages converged and passed independent original-spatial-residual, boundary, conservation, admissibility and supersonic-outlet checks. All ten independent critical space marches also converged, resampled at the exact 50/100/200-cell centers. All 50 original baseline file hashes, including the cached property table, still match.

Both wall variants use the requested reconstruction on every grid. Each 50-cell run starts from the model cold Mach ramp; the converged state is linearly prolonged to 100 and then 200 cells. CFL starts again at 0.2 and residual normalization is fresh on each grid. No reference solution is used to seed the FVM.

Pressure RMS differences below are percentages relative to the space marcher, equally weighted over the physical cell centers. They measure solver differences, not experimental validation errors.

| Case | Inviscid 50 [%] | 100 [%] | 200 [%] | Haaland 50 [%] | 100 [%] | 200 [%] |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 0.6399 | 0.7162 | 0.6437 | 0.5174 | 0.6527 | 0.5904 |
| L2 | 0.4647 | 0.1831 | 0.1463 | 0.4840 | 0.1721 | 0.1405 |
| L3 | 0.3636 | 0.0748 | 0.0282 | 0.3894 | 0.0741 | 0.0298 |
| L4 | 0.3235 | 0.1074 | 0.0942 | 0.3602 | 0.1065 | 0.0912 |
| L5 | 0.3092 | 0.0763 | 0.0164 | 0.3409 | 0.0740 | 0.0170 |

At 200 cells, pressure RMS differences span **0.0164-0.6437%**, compared with 1.49-2.48% for the earlier constant-reconstruction 96-cell comparison. The finest-grid maximum pointwise pressure differences span 0.0648-2.3210%. The report also shows the earlier comparison explicitly as a different scheme/grid rather than an isolated same-grid order test.

L2-L5 pressure RMS differences decrease on each refinement for both wall models. L1 is nonmonotonic: its largest RMS difference occurs on 100 cells, and the 200-cell result remains about 0.6%. L4 also shows a smaller change from 100 to 200 cells than from 50 to 100. These observations do not establish uniform second-order convergence or grid independence. The localized L1 pressure discrepancy reaches about 2.3% despite the small residual; algebraic convergence and cross-method accuracy remain separate results.

The 200-cell conservative numerical mass-flow differences from the critical value span 0.00177-0.10021% in magnitude. They are positive for every case in this rerun. Numerical face mass flow is kept separate from cell-state density times velocity times area.

| Case | Haaland FVM, 200 cells [kg/s] | Haaland critical [kg/s] | Sonic position [mm] | Outlet Mach, space marcher |
|---|---:|---:|---:|---:|
| L1 | 0.381339 | 0.380958 | 0.1065 | 1.3927 |
| L2 | 0.435724 | 0.435716 | 0.0866 | 1.3777 |
| L3 | 0.504688 | 0.504615 | 0.0845 | 1.3704 |
| L4 | 0.570046 | 0.569938 | 0.0832 | 1.3615 |
| L5 | 0.633689 | 0.633603 | 0.0826 | 1.3507 |

Positions use the experimental coordinate. The geometric throat and inviscid sonic locations are at -0.05973 mm. Friction shifts the sonic point downstream, reduces critical mass flow and lowers exit Mach; both methods show these trends.

Across all 30 stages, the largest weighted equation RMS is 8.088e-10, largest boundary residual 1.789e-15, and largest global defect 1.095e-10. The fixed absolute, boundary and global targets remain 1e-9, together with the original relative criterion. Every grid is independently audited after convergence.

The previously retained L1/L5 reference-refinement checks use the earlier 48/96-cell observation set and the same unchanged critical calculator, geometry and thermodynamic convention. Halving sonic restart distance and quartering IVP/critical tolerances changed sampled pressure, velocity, density, temperature or Mach by at most 0.0000613%, critical mass flow by at most 1.30e-10 relative, and sonic location by at most 2.91e-14 m. Those checks are displayed at their recorded epoch; the nominal space references are resampled at the new centers for this study.

The raw space-reference mass-flow and total-enthalpy ranges remain separate diagnostics in the report. The requested original tabulated sound-speed and Gruneisen conventions are unchanged; no thermodynamic-table correction was introduced to reduce the differences.

The earlier HLL constant-reconstruction comparison is preserved in `outputs/solver_comparison/campaign_constant_48_96.json` and the HTML evidence download. The current selectors show the new MUSCL sequences; the original 13 validation attempts remain on the older report pages. Timings are single compilation-inclusive observations, not controlled performance benchmarks.

## Equations and closures

For a smooth, steady, forward solution, both formulations represent

$$
(\rho u A)'=0,\qquad
p'+\rho u u'=-\frac{P}{A}\tau_w,\qquad
\rho u h_0'=\frac{P}{A}q_w,\qquad h_0=h+\frac{u^2}{2}.
$$

$P$ is the full wetted perimeter and $q_w$ is positive into the fluid.
Stationary adiabatic friction reduces momentum and produces entropy while
preserving total enthalpy. NozzleX's HP form uses the configured tabulated
sound speed and Gruneisen identities

$$
\rho_p=\frac{1+G}{a^2},\qquad \rho_h=-\frac{\rho G}{a^2},\qquad
u p'-\rho u h'=-\frac{P}{A}(u\tau_w+q_w).
$$

The corrected NozzleX HP entropy row includes the heat term previously
omitted. The unused arbitrary heat-transfer multiplier and clipping were
removed. Heat transfer is off in this campaign. The original bicubic table,
including its tabulated $a$ and $G$, is used unchanged as requested.

Both solvers use the existing hyperbolic Haaland policy: Darcy $64/Re$ for
$Re<2300$ and $[-1.81\log_{10}((\epsilon/(3.7D_h))^{1.11}+6.9/Re)]^{-2}$
above that threshold. Here $\epsilon=0$, $D_h=4A/P$ varies with position, and
$\tau_w=f_D\rho u|u|/8$. The signed laminar limit $8\mu u/D_h$ has finite
derivatives at zero velocity. Factors, stresses, AD and production wall
source evaluation are compared directly by an optional NozzleX regression.

The measured not-a-knot cubic half-height spline is shared exactly:
$A=2by$, $P=2(b+2y)$, $b=6.48$ mm. FVM retains its exact integrated volumes
and face-area geometry source. A JAX evaluation of the same spline is passed
to NozzleX; no replacement contour is fitted.

## Critical calculation and boundary conditions

The new NozzleX `solve_choked_nozzle` calculates a regular sonic state with
$u_*=a_*$, reservoir total enthalpy and inlet entropy constraints. The sonic
compatibility equation locates the throat dynamically; friction may move it
away from the geometric minimum. For the adiabatic model it is

$$
\frac{A'}{A}=(1+G)\frac{P\tau_w}{A\rho a^2}.
$$

The custom PySolverView Newton backend solves the critical energy and actual
upstream inlet-entropy equations. The real accelerating eigenvector of the
desingularized system supplies a controlled local restart, followed by
physical-coordinate DOP853 integrations upstream and downstream. This uses
neither possible/impossible classifications nor an artificially blended
near-sonic RHS. Every failed trial and final status is retained. Requested
points inside the unresolved restart interval are gaps, except the actual
critical state; no Taylor values are presented as integrated reference data.
A restart/tolerance refinement is recorded separately.

Both methods prescribe reservoir conditions and select a choked solution with
supersonic outlet extrapolation. The FVM's $0.25p_0$ value is a cold-start
pressure guess, not an imposed backpressure. No space-marching result or old
inviscid FVM state seeds the new coarse MUSCL calculations. Both wall variants
start cold on 50 cells, then linearly prolong the converged state to 100 and
200 cells. Each stage uses HLL/MUSCL/van Leer, starts with CFL 0.2 and fresh
per-equation residual normalization, and must pass the unchanged fixed targets
before advancing to the next grid. No constant-reconstruction startup stage
is substituted for the requested 50-cell MUSCL solve.

## Reproduction and evidence

NozzleX numerical changes are committed before study execution. The study
uses its existing editable installation and the repository `.venv` (JAX x64).
The optional critical API requires PySolverView 0.8.1 or later; it reports a
clear error instead of selecting a different root backend on older versions.

```powershell
.venv/Scripts/python.exe -B -m cases.lettieri_validation.scripts.comparison --solver all
.venv/Scripts/python.exe -B -m cases.lettieri_validation.scripts.comparison --case L1 --solver space --refine
.venv/Scripts/python.exe -B -m cases.lettieri_validation.scripts.report
```

[comparison_settings.json](comparison_settings.json) defines the campaign;
the original [settings.json](settings.json) supplies unchanged FVM tolerances
and coupled implicit Euler configuration. Explicit comparison overrides select
HLL and `muscl_van_leer` on every grid.
New numerical attempts, states, original spatial-residual audits and source
hashes are under `outputs/solver_comparison/`. The report can be rebuilt from
saved data without running CFD. A 50-file baseline hash manifest protects the
original numerical/reference files and cached property table.

Pressure, velocity, density, temperature and Mach errors are
$100|f_{\rm FVM}-f_{\rm space}|/|f_{\rm space}|$. Liquid-fraction differences
are absolute percentage points, with raw quality sentinels and overshoots
left as gaps. Numerical face mass flux and cell $\rho uA$ are distinct
diagnostics. A small discrete residual establishes convergence to that
discrete problem; solver agreement, spatial refinement and agreement with
the nonequilibrium experiment are assessed separately.
