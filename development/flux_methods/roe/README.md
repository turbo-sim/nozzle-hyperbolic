# Roe method: production formulation and verification

The shared FVM engine supports `Scheme(flux="roe")`. Nozzle and annular Euler
supply a general-EOS construction through the existing pressure/enthalpy
property interface. Numerical source lives in `src/`; this directory contains
the final theory, verification evidence and clients of the maintained benchmark.

| File | Purpose |
|---|---|
| [roe_theory.md](roe_theory.md) | The method actually implemented: PH thermodynamics, finite-jump secant, small-jump policy, eigenstructure, entropy correction and limitations |
| [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) | Measured convergence, accuracy, real-gas and transient results, including unsuccessful checks |
| [REPORT.html](REPORT.html) | Standalone offline report with an internal **Roe theory** page, interactive results and downloadable original evidence |
| [benchmark_protocol.md](benchmark_protocol.md) | Executable four-flux comparison protocol, controls, acceptance and timing rules |
| [verification_profile.yaml](verification_profile.yaml) | Completed 160-configuration Roe-only numerical verification |
| [benchmark_profile.yaml](benchmark_profile.yaml) | Prepared 640-configuration Roe/Rusanov/HLL/original-Stadtke timing campaign |
| [evidence/](evidence/) | Final production pytest log and JUnit record |
| [scripts/](scripts/) | Verification, saved-data analysis and report-generation clients; no copied production solver |

## What has been demonstrated

- **160/160** cold steady solves converged and passed independent original
  residual, boundary/global and physical-branch checks. All **20/20** finest-grid
  MUSCL cases met their accuracy targets. Four constant-reconstruction accuracy
  misses are retained and explained in the assessment.
- **6,000/6,000** actual CO2 PH state-pair matrix checks and **42/42** tiny-jump
  probes passed. Both five-cell actual-CO2 flow cases converged; the two 25-cell
  attempts exhausted their budgets before steady convergence. This is limited
  real-gas flow coverage, not a general positivity or nonlinear entropy theorem.
- All **12 Sod runs** reached the requested time and passed numerical
  accuracy/admissibility/BC/discrete-budget checks. Three coarse-grid comparisons
  against ideal undisturbed endpoint transfer remain failed and visible.
- The retained production suite recorded **411 passed, 3 existing expected
  failures**. The full four-flux performance campaign remains unexecuted; there
  is no clean comparative timing or speed ranking.

## Rebuild the delivered report without solving again

Run commands from the repository root in the existing Poetry environment:

```powershell
poetry run python -B -m development.flux_methods.roe.scripts.restore_evidence
poetry run python -B -m development.flux_methods.roe.scripts.report --presentation-only
poetry run python -B -m development.roberto.benchmarking.scripts.audit_report --input development/flux_methods/roe/REPORT.html
```

The committed HTML intentionally embeds the final numerical evidence: it is
approximately 102 MB and can be copied and opened on its own, offline. The
restore client verifies the archive and every SHA-256 before writing the exact
original JSON/NPZ/PNG files under ignored `outputs/`. It refuses to overwrite
different evidence. The report client reads saved data only; it regenerates the
HTML, Markdown assessment, refinement audit and difficult-shock work diagnosis.
No flow solver is imported or advanced by these presentation commands.

Production source was frozen during the validation campaign. This study was
subsequently moved from `development/roberto/roe_method` to
`development/roe_method`, then to `development/flux_methods/roe`. Paths, source fingerprints and executed-command strings
inside original evidence retain their measurement-time values. The relocated
clients record their own new identities on a future run; historical campaigns
cannot be resumed across those changes. Use a new output directory for a new
campaign. Superseded planning documents, report previews and incomplete local
trial outputs have been removed; the final assessment retains their relevant
negative conclusion: no completed four-flux comparison or valid timing ranking.

## Run new verification or benchmark campaigns

The following commands **do run numerical work**. Keep new runs separate from
the delivered final evidence.

```powershell
# Real-CO2 matrix/PH/AD audit; no flow simulation.
poetry run python -B -m development.flux_methods.roe.scripts.audit_roe --output development/flux_methods/roe/outputs/new_roe_audit.json

# All 20 steady cases, four grids and both reconstructions; untimed.
poetry run python -B -u -m development.flux_methods.roe.scripts.verify_steady --roe-audit development/flux_methods/roe/outputs/new_roe_audit.json --output development/flux_methods/roe/outputs/new_steady_verification
poetry run python -B -m development.flux_methods.roe.scripts.audit_steady --input development/flux_methods/roe/outputs/new_steady_verification/summary.json

# Separate transient and actual-EOS flow verification.
poetry run python -B -m development.flux_methods.roe.scripts.verify_sod --output development/flux_methods/roe/outputs/new_sod.json
poetry run python -B -m development.flux_methods.roe.scripts.verify_real_gas --cells 25 --output development/flux_methods/roe/outputs/new_real_gas_25

# Full four-flux timing matrix: requires an idle machine and many complete solves.
poetry run python -B -u -m development.roberto.benchmarking.scripts.run --config development/flux_methods/roe/benchmark_profile.yaml --output outputs/roe_flux_v1 --wait-for-idle
```

See [benchmark_protocol.md](benchmark_protocol.md) for the exact population,
excluded warmups, repeat counts, independent checks and report command for a
new timing campaign. Physical YAML definitions and analytical references are
loaded through [the installed verification API](../../../src/nozzle_hyperbolic/verification/README.md);
they are not duplicated here. The shared runner and HTML assets remain in
[the maintained benchmark](../../roberto/benchmarking/README.md).

## Production ownership

- [Generic Roe contraction](../../../src/nozzle_hyperbolic/numerics/roe.py): arbitrary
  variable/wave counts, model-supplied spectral data, no EOS or Euler assumptions.
- [Euler Roe construction](../../../src/nozzle_hyperbolic/models/euler_roe.py): PH
  properties, centered secant projection, grouped eigenspaces and validity.
- [Nozzle](../../../src/nozzle_hyperbolic/models/euler_nozzle/model.py) and
  [annular](../../../src/nozzle_hyperbolic/models/euler_annular/model.py): physical
  face data, fixed-radius angular-momentum frame and boundary/source laws.
- [Shared operator](../../../src/nozzle_hyperbolic/solvers/fvm/operator.py):
  reconstruction, conservative area/frame assembly, trial checks and explicit
  interface-dissipation CFL bound. The source default is now `stadtke` with
  both custom corrections enabled; this comparison explicitly disables both
  flags to preserve its original FVS baseline.

The construction assumes a smooth, thermodynamically consistent PH map on the
selected branch. Invalid averages are rejected, with no sound-speed clipping or
hidden HLL fallback. Phase switching, nonconvex regimes, independently
interpolated inconsistent property tables and two-fluid systems require separate
assessment. Host-fluid derivative accuracy is limited by JAXProp's existing
finite-difference JVP bridge. The [theory note](roe_theory.md) gives precise
formulas and relates these limits to the measured checks. The sibling
knowledge-base repository was consulted read only and was not modified.

## Common report presentation

This report shares its header, typography, theme, navigation and plot styling
with the other [flux-method reports](../shared/REPORT_DESIGN.md). The main
pages are **Overview**, **Case explorer**, **Theory**, and **Protocol & evidence**.
Method-specific checks, numerical acceptance rules and saved evidence remain
explicit. Light, dark and automatic themes work offline; all offered downloads
and rendering assets are embedded in the HTML.
