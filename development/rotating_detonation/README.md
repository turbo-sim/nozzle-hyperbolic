# Annular transition duct with periodic inlet angle

This study uses the existing quasi-1D annular Euler model with perfect-gas Air,
meridional velocity and swirl to examine a transition duct downstream of a
rotating-detonation combustor. The walls are inviscid and adiabatic. Detonation,
combustion and azimuthal wave propagation are outside this model. All study
code, evidence and reports remain inside this directory; the study does not
modify production `src/`.

## Run and inspect

Run from the `nozzle-hyperbolic` repository root using its existing environment:

```powershell
.venv/Scripts/python.exe -B -m development.rotating_detonation.run_case
.venv/Scripts/python.exe -B -m development.rotating_detonation.run_case --report-only
```

The entry point also supports direct execution:

```powershell
.venv/Scripts/python.exe -B development/rotating_detonation/run_case.py
```

The full run solves the steady grids, starts the primary transient, compares a
matched late-time window at a finer timestep, and generates `REPORT.html`. The
`--report-only` path audits retained states and rebuilds the figures, animations
and HTML without CFD time steps. Saved-state postprocessing is also available:

```powershell
.venv/Scripts/python.exe -B -m development.rotating_detonation.scripts.diffuser_postprocess
```

The current files are:

- `case.yaml`: physical conditions, geometry and numerical settings.
- `run_case.py`: module and direct-file entry point.
- `scripts/diffuser_geometry.py`: NURBS geometry, exact native mesh caches and independent steady references.
- `scripts/diffuser_model.py`: reservoir inlet and constant-pressure outlet equations.
- `scripts/diffuser_run.py`: implicit steady grid sequence, RK3 transients and every-accepted-step records.
- `scripts/diffuser_postprocess.py`: independent saved-state audit, averages and media.
- `scripts/diffuser_report.py`: standalone HTML presentation.
- `scripts/diffuser_html_audit.py` and `scripts/diffuser_javascript_audit.mjs`: report audits.
- `scripts/vendor/`: retained Plotly and MathJax assets, SVG equations, licenses and rendering provenance.
- `outputs/diffuser/`: current numerical evidence and media.
- `REPORT.html`: the self-contained report.

## Geometry and mesh

The physical tip radius stays at $R_t=0.10$ m. The first 0.10 m has constant area
and hub/tip ratio 0.8. An expansion occupies the next 0.20 m and doubles the
normal cross-sectional area. Its final hub/tip ratio is therefore

$$
\frac{R_{h,\mathrm{out}}}{R_t}
=\sqrt{1-2(1-0.8^2)}=\sqrt{0.28}\approx0.5291502622.
$$

Two 20 mm blends lie inside this expansion: the midline slope changes from zero
to its constant core value over $x=0.10$--$0.12$ m, then returns to zero over
$x=0.28$--$0.30$ m. A straight 0.10 m constant-area tail follows. The total axial
length is 0.40 m, and the final section is axial. Both outlet wall endpoints are
at $x=0.40$ m, with radii 0.10 m and approximately 0.052915 m.

The geometry API uses a NURBS midline $(x(m),r(m))$ and a channel height $b(m)$
normal to that midline, with $m$ its arclength. The height is parameterized by
normalized arclength and satisfies

$$
b(m)=\frac{2[R_t-r(m)]}{x'(m)}.
$$

Thus the physical tip radius $r+b x'/2$ stays at $R_t$. The height is represented
by a quintic NURBS with endpoint derivative constraints. Production geometry
checks give an outlet/inlet normal-area ratio of exactly 2.0 and a maximum
sampled tip-radius error of $4.16\times10^{-17}$ m. The arclength is approximately
0.400490 m. Geometry and contours use the physical normal-offset walls.

The complete-domain steady meshes contain 67, 133 and 267 cells. Adding the
constant-area tail with these counts preserves approximately the original
50/100/200-cell spacing in the 0.30 m inlet-plus-expansion region. Time-dependent
runs use the complete 267-cell mesh without transient grid sequencing. Face
areas and controlled cell-volume integrals come from the production annular
geometry evaluator. Exact native arrays are cached with geometry/evaluator
hashes and dependency versions.

`outputs/diffuser/design/geometry_metrics.json` is an analytical geometry/design
record. Its illustrative Mach 1.05 pressure reference and other candidate
pressures are distinct from the current Mach 0.95 reference selected below.

## Physical boundary conditions

The inlet reservoir has constant stagnation pressure 20 bar and temperature
1000 K. The imposed flow angle, measured from the meridional direction, is

$$
\alpha(t)=45^\circ+5^\circ\sin(2\pi f t),\qquad f=3500\ \mathrm{Hz}.
$$

The outlet static pressure remains at 1268451.145800547 Pa, or approximately
12.684511458 bar. With the new geometry and mean angle of 45 degrees, this
pressure corresponds to an isentropic steady inlet total Mach of 0.95. The YAML
Mach is a design guide; the independent reference solves its inlet Mach from
the fixed backpressure. Actual numerical Mach and mass flow are responses to
the reservoir, angle and outlet pressure. Total Mach below one is permitted and
is reported as a diagnostic. There is no prescribed inlet Mach waveform.

Characteristic directions depend on meridional Mach
$M_m=v_m/a=M\cos\alpha$. For positive meridionally subsonic flow, three
characteristics enter at the inlet and one enters at the outlet. The inlet
supplies reservoir stagnation enthalpy, entropy and flow angle; outgoing
acoustic compatibility comes from the duct. The outlet supplies static
pressure, while its outgoing acoustic, entropy and angular-momentum information
come from the interior. These trace equations use the existing boundary and
time-integration machinery through study-local callbacks.

DLR describes detonation speeds of several kilometres per second and wave
frequencies in the kilohertz range in annular chambers. See its
[rotating-detonation engine overview](https://www.dlr.de/en/ra/about-us/departments/rocket-propulsion-technology/combustion-dynamics/rotating-detonation-engines).
For this study, a wave speed of order 2 km/s divided by the circumference at an
inlet mean radius near 0.09 m motivates the representative 3.5 kHz scale.
Frequency depends on wave count, speed and combustor geometry; it is a specified
excitation here.

## Integration and operating checks

The spatial scheme uses HLL and van Leer MUSCL reconstruction. An implicit
steady grid sequence establishes converged roots on 67, 133 and 267 cells.
The initial coarse-grid guess interpolates pressure and meridional velocity
between analytical endpoint estimates while holding stagnation enthalpy and
angular momentum fixed. The analytical profile provides independent verification.

The primary physical transient starts from the exact retained, converged
267-cell numerical steady state and uses explicit SSPRK3 at CFL 0.4. It is capped
at 100 forcing cycles. For the capped comparison, a short CFL 0.2 continuation
starts from the saved primary checkpoint at cycle 90 and ends at cycle 100.
The comparison starts from the same cycle-90 state and covers the same physical
window, while retaining the primary run's earlier history. This ten-cycle
continuation measures temporal sensitivity during that interval; it does not
independently establish periodicity. The steady solve uses implicit pseudo-time
marching; physical transients in this case use RK3.

After every accepted physical timestep, the study checks all physical cell
states and both boundary traces. The required regime has positive meridional
flow and $0<M_m<1$ throughout. Outlet meridional velocity must remain below the
inlet value and below the value at the start of the expansion. These are net
meridional-deceleration checks; local gradients can vary as waves propagate.
Boundary residuals and the accumulated conservative storage/flux/source budget
are checked alongside the operating regime. Any violating state is retained
with its time and diagnostics, and the run stops with an explicit failure label.

## Periodicity and collection window

The primary run cannot certify periodic behavior before 40 forcing periods.
Each comparison uses 24 phase-aligned outputs per cycle, all physical cells and
four primitive variables divided by fixed reference scales. It must satisfy:

- Adjacent-cycle RMS changes no larger than $10^{-3}$ for each primitive.
- Maximum local normalized change no larger than $5\times10^{-3}$.
- The same RMS and local limits against the cycle ten periods earlier.
- Cycle-to-cycle changes in all four integrated conserved inventories no larger than $10^{-4}$ in their fixed content scales.

Five consecutive comparisons must pass. The run then collects ten further
complete cycles with every check still active; a failed check resets the
collection candidate. A verified periodic collection therefore requires at
least 50 periods, or approximately 14.286 ms. The primary run stops by cycle
100, corresponding to approximately 28.5714 ms, even if this verification has
not succeeded. The periodicity thresholds remain unchanged at that limit.

If a periodic collection is verified, its ten complete cycles provide the
steady-versus-periodic-mean comparison. Otherwise the report uses the finite
window from cycles 90 to 100, approximately 25.7143 to 28.5714 ms. Instantaneous
pressure, Mach, temperature, flow angle, mass flow and other nonlinear
diagnostics are averaged directly. Both the primary CFL 0.4 average and the
checkpoint-based CFL 0.2 average cover this exact ten-cycle interval.

When the periodicity tests have not passed, both records and their averages are
labeled finite-time results. Their difference from steady flow describes the
chosen window and cannot establish a persistent cyclic mean-flow shift. The
matched-window CFL comparison assesses time-discretization sensitivity of that
finite response. Any earlier operating-regime failure is retained explicitly
and limits the valid comparison interval.

The report includes the duct geometry and plasma-colored steady contours along
the physical outline. Values are uniform across each normal section, as required
by the quasi-1D model. The startup GIF uses the first three primary cycles. The
late GIF uses the final two representative cycles from the ten-cycle comparison
window, with physical time labels and fixed color limits. Without verified
periodicity, these are late transient cycles 98--100. A late animation is
labeled periodic only when the collection passes verification.

## Measured outcome

The primary run completed 100 cycles, or 28.5714 ms, with persistent modulation
and no verified periodic collection. Meridional Mach remained between 0.225020
and 0.789025, and the net expansion-deceleration checks passed. It accepted
49,746 steps with zero rejections. Saved-state numerical and flow audits passed
for both CFL runs.

Over the finite cycles 90--100 window, mean outlet-cell meridional velocity is
0.41170% below steady and mean mass flow is 0.54393% below steady. Extending the
average to the last 20 or 50 cycles lowers velocity by a further 0.02418 or
0.09097 m/s and mass flow by 0.00275 or 0.01029 kg/s, respectively, relative to
the ten-cycle mean. These differences show the conditioning on the averaging
window; they do not establish a cyclic mean-flow shift.

The largest normalized RMS difference in the matched CFL 0.4 versus 0.2
comparison is $7.8775\times10^{-6}$, below the $10^{-3}$ target. This check covers
cycles 90--100 from their common primary checkpoint and retains the coarse
run's earlier history. The fixed-pressure trace's acoustic decomposition gives
unit reflected/outgoing amplitude and a $-180^\circ$ phase difference, consistent
with pressure reflection at that boundary. Detailed evidence is retained in
`outputs/diffuser/study_summary.json` and the report.

## Offline report and audits

`REPORT.html` embeds its plotting code, images, animations and downloadable
numerical evidence. Equations are rendered by MathJax 3.2.2 into self-contained
SVG paths before packaging. Opening or rebuilding the report from the retained
SVGs requires no MathJax runtime, external fonts, CDN or network connection.
The vendor manifests retain exact LaTeX, source/archive hashes and SVG hashes.

Run the structural, numerical-download, media and MathJax-SVG audit after report
generation:

```powershell
.venv/Scripts/python.exe -B -m development.rotating_detonation.scripts.diffuser_html_audit
```

When Node is available, the JavaScript smoke audit exercises navigation, theme,
slider and downloads with a DOM stand-in:

```powershell
node development/rotating_detonation/scripts/diffuser_javascript_audit.mjs
```

Formula-editing and optional offline `.mjs` rendering instructions are retained
in `scripts/vendor/MATH_ASSETS_README.txt`. The JavaScript audit checks control
behavior; exported figures and representative GIF frames support visual review.

Saved evidence records configuration, solver settings, residuals, conservation,
regime extrema, periodicity, collection windows, timings and source fingerprints.
External source edits by other processes are permitted; fingerprints identify
which checkout produced or audited the data. Old campaigns are archived separately
and their measured outcomes do not describe this case.
