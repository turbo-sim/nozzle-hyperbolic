# Annular rotating-detonation transition-duct study

This folder contains a YAML case, a runnable Python study, retained numerical
outputs, and a standalone offline HTML report. It changes no production source.
The model is the existing quasi-1D annular Euler solver with swirl; it models the
transition duct downstream of a combustor, rather than resolving detonation or
azimuthal wave propagation.

Run from the repository root using the existing environment:

```powershell
poetry run python -B -m development.rotating_detonation.run_case
poetry run python -B -m development.rotating_detonation.run_case --report-only
```

If Poetry is not on PATH, replace `poetry run python` with
`.venv/Scripts/python.exe`. A short pilot uses `--methods implicit --repeats 1`;
the default campaign uses all three available integrators, one excluded complete
solve per method and ten measured repetitions. `--steady-only` omits transients.
`--report-only` regenerates the presentation without any numerical solves.

- `case.yaml`: physical assumptions, geometry, forcing and numerical settings.
- `run_case.py`: module entry point.
- `scripts/problem.py`: NURBS midline/normal-height construction and study-local
  time-dependent external-state inlet model.
- `scripts/run.py`: production solver calls, timings, field and provenance saving.
- `scripts/report.py`: offline HTML report generator.
- `outputs/`: retained histories, states, timings, geometry and source hashes.
- `REPORT.html`: standalone report, interactive profiles, internal navigation,
  theme control and embedded downloadable numerical evidence.

The tip radius is 0.10 m. A 0.10 m constant passage has hub/tip 0.8, followed by
a 0.20 m expansion ending at hub/tip 0.7. The midline slope blends over the first
0.02 m of the expansion because the production normal-offset geometry requires
a continuous tangent. The rest of the hub is straight. Lengths are axial midline-coordinate lengths; normal end sections have slightly offset wall endpoints. Normal height is
$b=2(R_t-r)/x'$, ensuring the normal-offset tip remains at radius $R_t$.
The height spline is parameterized by normalized arclength, as required by the
existing geometry API. Its knots retain derivative changes at blend endpoints.

The baseline has total Mach 1.2, flow angle 30 degrees from the meridional
direction, stagnation temperature 1000 K and pressure 20 bar. Perfect-gas Air
and adiabatic, inviscid walls are explicit modeling assumptions. The automatic
outlet uses 0.10 bar ambient pressure while retaining outgoing supersonic flow.
Area expansion can accelerate a supersonic stream, so this geometric expansion
need not act as an aerodynamic diffuser.

External periodic targets are $M=1.2+0.3\sin(2\pi f t)$ and
$\alpha=30^\circ+10^\circ\sin(2\pi f t)$, with $f=3500$ Hz. Eight periods are
integrated on 200 cells, without transient grid sequencing. The frequency is a
representative single-wave assumption: approximately 2 km/s divided by the
circumference at mean radius 0.09 m. It is not a combustor frequency prediction.
[DLR](https://www.dlr.de/en/ra/about-us/departments/rocket-propulsion-technology/combustion-dynamics/rotating-detonation-engines)
describes kilometre-per-second speeds and kilohertz circulation. Multiple waves
can increase the wave-passing frequency; the report includes a NASA reference.

The baseline meridional Mach is $M_m=M\cos\alpha=1.039$. The forcing crosses
$M_m=1$. The study prescribes the four targets in an external Riemann state,
which the existing HLL flux couples to the reconstructed duct-side inlet state.
Outgoing information in subsonic flow enters through the two-sided HLL flux.
The external state is not a prescribed physical interface trace in that regime.
The report plots external targets and the duct-side reconstructed state, which
can differ. Enforcing Mach, angle, $p_0$ and $T_0$ simultaneously on a physical
subsonic trace would overconstrain it. A preliminary hard-switch characteristic
boundary stalled near 0.202 ms; its failed evidence is retained under
`outputs/failed_characteristic_pilot/`, excluded from final timings.

Numerics use HLL and van Leer MUSCL throughout. Steady grid sequencing uses
50, 100 and 200 cells. Explicit Euler and SSPRK3 use CFL 0.4. Backward Euler uses
increasing steady CFL and CFL 2 for transient comparison. A second SSPRK3
transient at CFL 0.2 estimates temporal sensitivity on the same 200-cell mesh.
There is no implicit RK3 in the current package. Residual thresholds are the
unchanged production defaults. Failed cases are retained and labeled.

For timing, immutable production Operators are cached through a process-local
constructor substitution while calling the unmodified production drivers.
This reuses compiled flow-step kernels between repetitions; the production driver still creates its monitoring reducers per solve, and that setup remains included. No package files
are patched on disk. Timings include driver monitoring and rejected attempts;
geometry/model setup, output and plotting are excluded. First-use compilation
is contained in the separately retained excluded complete solve. Source hashes before and after the campaign are retained. Concurrent external
edits to Osher-related/shared-interface files occurred during this run; the
independent audit therefore also replays retained HLL timesteps under the current
checkout. This study writes only within this folder. Pre-existing and concurrent
repository changes are preserved.

Independent checks (after the numerical campaign):

```powershell
poetry run python -B -m development.rotating_detonation.scripts.audit
poetry run python -B -m development.rotating_detonation.scripts.audit_html
```

The default full run also performs the audits and generates contours/GIFs automatically. Use `--no-postprocess` to skip those steps.

The numerical audit reassembles original residuals in a separate process,
checks the source manifest, recomputes the final transient conservation budget
from retained storage and accumulated flux/source transfers, and compares steady
fields against an independent isentropic supersonic reference. It also exports
`outputs/overview.png` and `outputs/overview.pdf`. The HTML audit checks internal
navigation, embedded plot dimensions, offline dependencies and lossless embedded
evidence downloads. Browser visual verification was unavailable because computer
use stopped at its browser-URL verification check.

Plasma contours and GIF animations:

```powershell
poetry run python -B -m development.rotating_detonation.scripts.visuals
```

This adds steady Mach/pressure/angle contours inside the actual meridional duct
outline, plus `outputs/startup.gif` (first three periods) and a final two-period
GIF. The late animation is named `periodic.gif` only after two consecutive
cycle-to-cycle checks pass a fixed normalized RMS threshold of 0.001 across all
four primitives, physical cells and 24 aligned phases. Otherwise it is labeled
unverified. The RK3 trajectory can be extended in whole-period chunks separately
from the eight-period timing comparison. At 3.5 kHz the benchmark end time is
$8/f=2.285714$ ms. The report states the final animation end time and checks.

Both GIFs share fixed plasma limits and show physical time labels with slowed
playback. Values are uniform across normal height, as appropriate to the
quasi-1D model; the rendering does not resolve transverse structure. The GIFs,
steady contour image and numerical download are embedded in the standalone HTML.

The `Steady vs mean` report page compares the steady baseline with a trapezoidal
time average over the final two verified RK3 periods. It plots pressure, Mach,
temperature, angle and mass flow, with the instantaneous range and a table of
outlet and maximum local differences. Instantaneous nonlinear diagnostics are
averaged directly. The report also compares means at the two timesteps and at
12 versus 24 output phases per period. `outputs/mean_comparison.json` retains the
averages and comparison metrics. Interpretation distinguishes the nonlinear mean
inlet-flux change from other unsteady effects.

The retained run reaches the periodicity criterion by forcing cycle 20
($t=5.714286$ ms). The mean and periodic GIF use cycles 19-20
($t=5.142857$ to $5.714286$ ms). A separate half-CFL continuation also passes
the same repeatability test. Relative to the steady baseline, the periodic mean
outlet-cell pressure is 3.2909% lower, mass flow is 2.6783% lower, and angle is
1.3222 degrees lower. Mean outlet Mach changes by only -0.000348. These changes
include nonlinear inlet averaging and are not a pure measure of storage effects.

The independent HLL residual/step-replay audit passed despite concurrent external
Osher/API source edits. The steady 200-cell pressure RMS error against the
smooth-flow analytical reference is about 75 Pa. GIF frame counts, exported
contours, offline packaging and embedded mean quadrature were checked. The
JavaScript behavior smoke test (`scripts/audit_report_js.mjs`) checks six-page
navigation, theme, slider and downloads with a DOM stand-in; it is not a browser
rendering test. Browser visual review remained unavailable, while exported PNGs
and representative GIF frames were inspected directly.
