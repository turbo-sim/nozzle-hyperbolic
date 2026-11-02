# Annular radial–axial bend verification

This study compares NozzleX's space-marching annular channel ODE with the
production `AnnularModel` and shared hyperbolic FVM time marcher. It imports
the original NozzleX bend configuration, cubic midline and all implemented
friction/heat policies. The two solvers call the same NozzleX
`evaluate_wall_sources` function. No empirical closure is copied into this repo.

Open [REPORT.html](REPORT.html) for the standalone offline Plotly report,
geometry, plasma duct renderings, profile comparisons, convergence and embedded
evidence downloads. [REPORT.md](REPORT.md) records the measured conclusions.

The report follows the visual layout of `development/real_gas/`. Select a
closure case and FVM grid, then show FVM only, space marching only or both.
The FVM uses white-face markers drawn over the NozzleX continuous line. Set
the number of points skipped between markers: zero shows every cell; seven
keeps every eighth cell, with the last cell retained. There is one computed
FVM method, HLL / MUSCL van Leer / implicit Euler, so that selector displays
the recorded method without offering uncomputed alternatives.

Each flow profile has its absolute-error plot directly below it, with a
logarithmic vertical axis. Errors compare every FVM cell against its saved
NozzleX value at exactly the same center, in the displayed physical units.
Skipping affects only visible FVM markers; error curves and CSV downloads
retain every cell. Zero errors remain zero in the data and are omitted on
the logarithmic axis. The paired plots support linked horizontal zoom.
The variable selector exposes all twelve common flow quantities in two groups
or together. Plotly, full plotted arrays and downloads are embedded offline.

## Physical problem

The imported `nozzlex/examples/vaneless_diffuser/case_radial_axial_bend.yaml`
defines the physical setup:

- Radial-to-axial cubic midline, from $(z,r)=(0,0.10)$ m to $(0.05,0.20)$ m.
- Inlet/outlet midline inclinations $90^\circ$ and $0^\circ$; tangent control
  distances 0.10 and 0.04 m.
- Constant full normal height $b=0.015$ m; $A=2\pi rb$, giving area ratio 2.
- Reservoir pressure 101325 Pa and temperature 300 K; total inlet Mach 0.8
  and flow angle $60^\circ$ from the meridional direction. Thus $M_{m,in}=0.4$.
- The same JAXProp perfect-gas Air object configuration and transport laws.

[case.yaml](case.yaml) imports that definition and selects five closure cases:
inviscid adiabatic; constant Fanning $C_f=0.005$ adiabatic; full Aungier
adiabatic; full Aungier with wall temperatures 260 K and 340 K.

For each case NozzleX predicts the outlet static pressure. That pressure is
the hyperbolic backpressure. The time marcher prescribes inlet reservoir
enthalpy, entropy and flow angle; its inlet Mach is an outcome. The inlet
Mach is not imposed as an extra subsonic boundary constraint.

## Equations and corrections

With meridional arclength $m$, $u=v_m$, $w=v_\theta$, $\ell=rw$ and
$h_0=h+(u^2+w^2)/2$, the common steady balances are

$$
(\rho u A)'=0,
\qquad
\rho u u'+p'=\rho w^2\frac{r'}r-\frac{2\tau_m}{b},
$$

$$
\rho u\ell'=-\frac{2r\tau_\theta}{b},
\qquad
\rho u h_0'=\frac{2q_w}{b}.
$$

NozzleX's continuity enthalpy coefficient was corrected from $-G/a^2$ to
$-\rho G/a^2$, with the usual dimensionless Grüneisen parameter. Its
total-enthalpy derivative row was corrected to use $2q_w/(b\rho u)$, and its
integrated fluid-entropy diagnostic now includes wall heat. Independent
NozzleX regression tests differentiate EOS density and check the executed
ODE's mass, torque, energy and entropy balances for heated/cooled/adiabatic flow.

The hyperbolic annular source now accepts a directional wall policy returning
signed $(\tau_m,\tau_\theta,q_w)$. Existing scalar wall policies retain their
behavior. The optional `NozzleXWall` adapter binds the external policies to a
fixed native mesh. Aungier diffusion and curvature losses act only
meridionally; wall friction also removes angular momentum. Curvature uses
its magnitude. Reynolds-analogy heating uses local stagnation temperature
and the wall-friction component alone. Stationary-wall drag does not remove
total energy.

## Geometry and numerical controls

The original cubic is obtained from NozzleX's `make_vaneless_channel_midline`.
The installed NURBSpy JAX arclength builder calls a Quadax `DirectAdjoint` API
that is unavailable in this environment. The study therefore supplies an
audited fixed-coordinate map to NozzleX's geometry evaluator; no package is
monkey-patched and no flow integrator is replaced.

A 2049-point cubic Hermite inverse of arclength is checked against the
production host inverse and used for both solvers. Full physical endpoints
are retained, avoiding NozzleX's legacy default interval shortening. The
production volume and pressure-area quadrature audits remain enabled.

NozzleX uses Dopri5 with tolerances $10^{-8}$, $10^{-10}$ and $10^{-12}$;
the tightest solution is the reference. Observations include every FVM cell
center exactly. Hyperbolic runs use HLL, MUSCL van Leer, coupled backward
Euler pseudo-time marching, colored AD and banded linear solves. Meshes have
50, 100, 200 and 400 cells with $m_i/L=(i/N)^{1.5}$ to resolve entrance losses.

The coarsest run starts from the production uniform cold guess. Subsequent
grids interpolate the preceding FVM solution; no ODE profile is used as a
warm start. Each grid receives a fresh residual normalization and CFL schedule.
Stopping requires the original fixed-scale equation RMS, boundary residual
and normalized global defects to pass absolute $10^{-9}$ limits. Backward
Euler is used to find a steady root; physical-time accuracy is not assessed.

Errors are volume-weighted comparisons against ODE point values at cell
centers, including conservative densities normalized by fixed production
scales. They are explicitly not comparisons against exact volume-averaged
reference contents. The finest-grid conservative accuracy target is $10^{-3}$.

## Run and audit

From the nozzle-hyperbolic repository root, use the existing `.venv`:

```powershell
.venv/Scripts/python.exe -m pip install --no-deps -e ../nozzlex
.venv/Scripts/python.exe -B -m cases.verification_annular.run_case
.venv/Scripts/python.exe -B -m cases.verification_annular.run_case --report-only
.venv/Scripts/python.exe -B -m cases.verification_annular.scripts.audit_browser
.venv/Scripts/python.exe -B -m cases.verification_annular.scripts.audit
.venv/Scripts/python.exe -B -m pytest tests/test_annular_nozzlex_adapter.py tests/test_annular_directional_wall.py tests/test_euler_annular_wall.py tests/test_physical_closures.py ../nozzlex/tests/test_vaneless_balances.py -q
```

NozzleX remains optional for the base hyperbolic installation. Its declared
older dependency ranges are not used to downgrade this existing environment;
the editable checkout is installed without dependency resolution. The study
records the tested versions (including JAXProp 0.5.8 and NURBSpy 1.6.0).
The required NozzleX balance fixes and reusable channel API are saved in its
commit `48ee596`; the hyperbolic directional wall adapter is in `a4d8bf6`.
Poetry was unavailable here, so the existing virtual environment was used
directly. The adapter tests skip if NozzleX is absent.

The report-only command reads saved evidence and does not import a flow solver
or evaluate the EOS. Its template, CSS and application code live under
`scripts/report_assets/`; the Plotly 2.35.2 bundle and MIT license are reused
from the repository's shared report assets and embedded into the output HTML.
Existing standalone PNG/PDF exports remain available. No simulation inputs,
states, histories or timings are changed by a presentation rebuild.

`scripts/audit_browser.py` uses the existing real-gas CDP test client and an
isolated headless Edge profile. It opens a copied HTML file, tests all five
cases and four grids, checks plotted values and every absolute-error station,
solver visibility, marker skipping, zoom, themes, mobile layout, equations,
downloads and offline operation. Screenshots and audit JSON are saved under
`outputs/browser_audit/`. It needs permission to launch Edge and local browser
pipes; it does not use an existing browser profile. The JavaScript summary
helper exports `auditReport(reportPath, outputPath)` and verifies syntax and
the current browser audit's HTML hash. This replaces the earlier DOM stand-in.

## Saved evidence and limits

`outputs/` holds raw reference profiles, initial/final FVM states, accepted and
rejected histories, settings, closure loads, face fluxes, integrated sources,
source hashes, audits, logs and PNG/PDF figures. `outputs/pilot/` preserves the
first inviscid screen; `outputs/initial_campaign/` preserves the first complete
campaign before the tighter reference and final source freeze. Failed strict
audits of the loose ODE references are retained and displayed; they are not
used as the final comparison reference.

Agreement establishes consistency between these reduced 1D equations and
discretizations. It does not independently validate the shared empirical
closures or resolve cross-channel pressure gradients, boundary layers or
separation. General-EOS equivalence was analyzed, but this campaign tests
perfect-gas air only. Imported Aungier entrance regularization and currently
ineffective transition configuration fields remain disclosed in the report.
Timings are single compile-inclusive observations, not a repeated benchmark.
