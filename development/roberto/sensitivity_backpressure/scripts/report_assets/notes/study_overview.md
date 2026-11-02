# Euler nozzle backpressure analysis

This study decreases the imposed backpressure from the reservoir pressure to
below the adapted/design pressure. It investigates steady operating regimes,
the numerical outlet condition, and agreement with perfect-gas analytical
solutions. Results are in [REPORT.md](REPORT.md); physical interpretation and
boundary details are in [PHYSICS_AND_BOUNDARIES.md](PHYSICS_AND_BOUNDARIES.md).

The [interactive HTML report](report_backpressure_sensitivity.html) consolidates the setup, motivation,
physical regimes, measured findings and limitations. It embeds Plotly, all
104 recorded states, analytical profiles and the independent audit in one
file. Open it directly in a browser; no server or network connection is
required. Use zoom/pan, trace hover, pressure/startup selection, optional
ghost/alternative-state overlays, convergence-region selection, table filters
and the embedded CSV/JSON downloads. The layout is designed for desktop and
laptop screens, with a print stylesheet for paper/PDF reading.

Rebuild the HTML from the existing saved results, without integrating again:

```powershell
poetry run python -m development.roberto.historical.backpressure_sensitivity.scripts.build_html_report
```

The renderer uses the existing analytical-reference extension and embeds a
local Plotly JavaScript bundle. It searches the rendering environment's Plotly
installation, then the neighboring TurboDash environment; alternatively pass
`--plotly-js PATH/TO/plotly.min.js`. No dependency or environment changes are
needed for the generated HTML itself. The bundled Plotly.js version is 2.35.2.
The approximately 10 MB file includes the scientific data and runtime.

Figure framing follows TurboDash's Plotly `apply_plot_theme` / `_style_axes`
and the plotting conventions of its Matplotlib `set_plot_options`: white
surfaces, mirrored axes, inward ticks and clear units. The technical HTML report
uses sans-serif typography and the sequential `Blues` colormap sampled between
0.45 and 0.95, with compatible blue-gray framing. Reference
curves, cold starts and branch probes also use different dashes/markers.

The editorial narrative is tied to the audited dataset fingerprint. If saved
results change, the builder stops until that narrative and its fingerprint
are reviewed. Renderer/input hashes are recorded in
`outputs/baseline/html_report_manifest.json`. Optional browser QA is in
`scripts/audit_html_report.py`; it requires an auditing environment with
Playwright and a Chromium browser, rather than a production dependency.

## Physical setup

Import the geometry, reservoir conditions and numerical settings from the
existing [steady EulerNozzle example](../euler_nozzle/README.md):

| Quantity | Value |
|---|---:|
| Length | 0.2 m |
| Inlet and exit area | $3.0\times10^{-4}$ m$^2$ |
| Throat area | $1.5\times10^{-4}$ m$^2$ |
| Reservoir $p_0,T_0$ | 200 kPa, 300 K |
| Fluid | `jaxprop.FluidPerfectGas(name="Air")` |
| Grid | 201 points, including two algebraic boundary/ghost points |
| Physics | Quasi-1D, inviscid, adiabatic Euler equations with area source |
| Numerical method | Regularized Städtke FVS, ordinary Van Leer MUSCL |
| Implicit method | Source backward-Euler/Newton kernel, one Newton correction per update, colored AD Jacobian and banded linear solve |

The backend supplies $R$ and $\gamma$; $\gamma$ is not hard-coded to 1.4.

## Running

From the repository root, using the existing Poetry environment:

```powershell
poetry run python -m development.roberto.historical.backpressure_sensitivity.scripts.run_sweep
# Recreate plots/reports from saved data without rerunning the numerical solves:
poetry run python -m development.roberto.historical.backpressure_sensitivity.scripts.run_sweep --plot-only
# Six-point execution check, separate from the full sweep:
poetry run python -m development.roberto.historical.backpressure_sensitivity.scripts.run_sweep --smoke --output development/roberto/historical/backpressure_sensitivity/outputs/smoke
```

The full schedule samples $p_b/p_0=1$ down to 0.02, with exact regime landmarks
and extra samples around each transition. Fourteen independent simple-start
checks compare against the descending continuation trajectory. Three deliberate
supersonic restarts probe the outlet's branch selection at higher pressures.
These additional restarts are not part of the descending sweep. `--n-points`,
`--max-updates`, `--output`, and `--no-cold-checks` allow explicit variants;
the default point budget is 5000 accepted implicit updates.

Files are organized as follows:

```text
backpressure_sensitivity/
  README.md
  PHYSICS_AND_BOUNDARIES.md
  REPORT.md
  scripts/
    run_sweep.py          source-kernel integration, continuation and diagnostics
    plot_results.py       saved-data figures and full operating-point table
    audit_results.py      independent reference/saved-field residual checks
    build_html_report.py  standalone interactive report, saved results only
    audit_html_report.py  optional offline laptop/desktop browser audit
    report_assets/        HTML template, CSS and Plotly interaction code
  report_backpressure_sensitivity.html             self-contained interactive scientific report
  outputs/baseline/       summary.json, operating_map.csv, PNG/PDF figures, REPORT.md
```

All generated output is covered by the repository `.gitignore`.

Audit references against the existing source analytical solution and reconstruct
the original spatial residual from every saved $p,T,M$ field, without advancing
any new timesteps:

```powershell
poetry run python -m development.roberto.historical.backpressure_sensitivity.scripts.audit_results
```

## What is held fixed and what changes

Only backpressure changes between primary operating points. Start at the
trivial uniform stagnant reservoir state. Each later point starts from the
last **converged numerical** field; failures are retained but are not used as
continuation seeds. No exact nozzle profile or shock position initializes a
numerical solve. The independent checks use the source `simple` initialization.
Each point is a separate steady pseudo-time solve, not a time-accurate pressure
ramp. These continuation solves are not cold-start performance benchmarks.

The source factory is called from a JIT wrapper with dynamic backpressure;
its fluxes, reconstruction, boundary maps, conservative storage and implicit
Newton algebra are unchanged. A normalized update comparison against an
ordinary static-backpressure source factory guards that wrapper. Source
hashes before/after the study are saved in the JSON.

The small experimental Python controller uses the baseline CFL/SER settings:
initial CFL 10, growth limit 1.5, SER maximum CFL $10^6$, pressure/enthalpy/
velocity increment limits 0.2/0.1/0.2 and up to eight timestep halvings. There
are two explicit differences from the production driver's stopping/safety
policy: require both its native SSR and an independent fixed-scale spatial
RMS to meet $10^{-8}$, and stop/report failure if all nine trials are rejected
instead of advancing with the production driver's final rejected trial.
The production plateau exit is not used. These are monitoring/controller
choices; no source algorithm is patched.

At exact rest the native SSR divides by vanishing momentum and is undefined
or poorly conditioned. The stagnant point is checked with the finite
fixed-scale residual and has zero accepted updates, rather than reporting
an artificial successful time integration. Non-finite JSON numbers are saved
as `null`; no residual is replaced or clipped for convergence decisions.

## Diagnostics and their meaning

- Mass flow is the **actual numerical face mass flux** used by the source
  discretization, averaged over faces; its range measures conservation.
- Exit temperature, pressure and Mach use the **last interior point**, with
  analytical values evaluated at that same coordinate. Algebraic ghost
  pressure is recorded separately. The last interior point is at 0.199 m;
  the nominal geometric exit is 0.200 m.
- The entropy diagnostic is $\overline{\dot m}(s_{N-2}-s_1)$, an advective
  entropy-flux-rise estimate between the first and last interior samples.
  At steady state with good conservation this estimates the entropy
  production within that sampled interval. It is not an implemented discrete
  entropy-stability proof or an entropy flux integrated through an external
  plume. Small numerical offsets can have either sign. Failed/unfinished
  states must not be interpreted as stationary entropy production.
- A separate analytical shock entropy loss, including a shock on the exit
  plane, records the one-sided boundary limit. A shock between the final
  interior sample and the exit is excluded from the sample-to-sample metric.
- The outlet pressure-imposition weight is the source expression evaluated
  diagnostically at the final state. Pressure/Mach errors, total-enthalpy
  error, inlet entropy error, solver histories and rejected trials are saved.

This is one-mesh verification, not a mesh-convergence study. It can expose
an operating-branch or boundary defect even when the steady residual converges.
