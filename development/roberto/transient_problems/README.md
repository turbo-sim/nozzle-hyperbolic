# Transient physical boundaries and numerical boundary treatment

This study compares four production methods on a common physical-time grid:
forward Euler, SSPRK3, backward Euler with coupled boundaries, and backward Euler
with segregated boundaries. All model/solver/geometry implementations come from
`src/nozzle_hyperbolic`; this directory contains configuration, numerical clients,
saved evidence and presentation code only.

The two problems are the standard Sod shock tube with transmissive endpoints,
and a fixed smooth nozzle initially at rest at $p_0=2$ bar and $T_0=300$ K. The
nozzle backpressure drops to $0.01p_0$ at time zero. Its physical finite-wave
outlet law is held fixed across all four methods; coupled/segregated selects a
numerical solution strategy for that law. The finite-wave boundary uses the
explicitly permitted perfect-gas implementation. The production default acoustic
formulation is unchanged.

Open [REPORT.html](REPORT.html) for offline animations of pressure, density,
velocity, Mach number and temperature, alongside fixed geometry, outlet regime,
boundary defects and conservation budgets. Select methods, play/pause, scrub
through computed frames and change playback speed. The report embeds all assets
and numeric data. Browsers supporting canvas capture and WebM can export the
selected animation; playback works without video libraries or FFmpeg.
Shareable computed GIFs are also available at
[`outputs/animations/sod.gif`](outputs/animations/sod.gif) and
[`outputs/animations/nozzle_step.gif`](outputs/animations/nozzle_step.gif),
with PNG/PDF keyframes in the same directory. Both GIFs are embedded in the HTML
download appendix, independently of browser video-codec support.

[VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) records measured results and
the comparison of the two implicit treatments. Full states and accepted/rejected
histories are retained once per method in
`outputs/verification/{case}__{method}.json`; `outputs/verification/summary.json` supplies
the report. Failed methods and unavailable output frames remain visible.

Run from the repository root with the existing environment:

```powershell
.venv/Scripts/python.exe -m development.roberto.transient_problems.scripts.run
.venv/Scripts/python.exe -m development.roberto.transient_problems.scripts.audit
.venv/Scripts/python.exe -m development.roberto.transient_problems.scripts.audit_production_tests
.venv/Scripts/python.exe -m development.roberto.transient_problems.scripts.render_animation
.venv/Scripts/python.exe -m development.roberto.transient_problems.scripts.report
.venv/Scripts/python.exe -m development.roberto.transient_problems.scripts.audit_report
```

The renderer can rebuild the report from saved JSON without importing the
numerical solver or rerunning simulations. Use `--input`, `--output` and
`--markdown` to render a selected dataset separately. The report validator checks
every full-resolution field frame; browser-independent JavaScript runtime checks
exercise navigation, playback, scrubbing and method controls. These checks are
distinct from visual browser inspection and actual WebM encoding.

The main campaign uses 200 physical cells, constant reconstruction with HLL,
101 common output frames, requested wave CFL 0.5, Sod end time $\tau=0.2$ and
nozzle end time $\tau=4$. Each integrator computes its own timestep from its
current wave speeds. Output clipping and rejected-trial retries can make the
actual CFL smaller than 0.5. The common output times do not force identical
internal timesteps across methods. `scripts.audit` recomputes source/boundary
checks and independently sums recorded flux/source transfers to verify the
conservation budgets.

Segregated backward Euler eliminates the nonlinear boundary trace at every
residual/trial evaluation and includes its implicit derivative in the reduced
Jacobian. It solves the same equations as coupled backward Euler. Outer Newton
counts omit nested boundary work, so they are not total nonlinear-cost counts.
Backward Euler has first-order temporal diffusion at finite CFL; its difference
from explicit methods is not by itself a boundary-law defect. No exact transient
nozzle solution is claimed. The Sod exact solution is evaluated before waves
reach the endpoints.
