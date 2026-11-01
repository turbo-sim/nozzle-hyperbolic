# Osher path-flux verification

This study selects one production flux: `Scheme(flux="osher")`. Its flat
parameters are `quadrature_points=5` and `integration="gauss"` by default;
`integration="sonic_partitioned"` selects captured-crossing panel integration.
The point count is quadrature accuracy/cost, **not spatial reconstruction order**.

| Study configuration ID | Public flux | Points | Integration |
|---|---|---:|---|
| `gauss_3` | `osher` | 3 | `gauss` |
| `gauss_5` | `osher` | 5 | `gauss` |
| `gauss_9` | `osher` | 9 | `gauss` |
| `sonic_partitioned_5` | `osher` | 5 per panel | `sonic_partitioned` |

These IDs identify experiment configurations, not flux selectors or aliases.
[THEORY.md](THEORY.md) gives the common path formula, integration policies,
differentiation assumptions and limitations. The API migration preserves the
numerical evidence through derived records and explicit origin hashes; it does
not represent new cold solves. Immutable original records remain archive-only.

The original 128 cold solves retain their recorded source/case/client SHA-256
epoch and immutable origin hashes. Historical source-guard and isolated-completion
proofs remain with the archive. The schema-2 view uses the unified public API and
separate current-source Operator evaluations; its audit checks numerical-payload
preservation, physical validity and execution/evaluation identities explicitly.
The public report includes only clean derived metadata and current-API evidence.
Generating it advances no flow and performs no numerical audit.

The [standalone Plotly report](REPORT.html) compares all four configurations for one
selected physical case at a time: pressure, velocity, temperature and total
enthalpy; full/shock views; actual residual/CFL trajectories; nonlinear work;
and analytical refinement against conservative cell projections.
[REPORT.md](REPORT.md) records saved coverage and audit status. No production
solver or physical-reference implementation is copied into this study.

## Controlled population

[verification_profile.yaml](verification_profile.yaml) requests four shock
cases (`euler_nozzle/shock_35`, `euler_nozzle/shock_70`,
`euler_annular/shock35`, `euler_annular/shock70`), four grids
(50/100/200/400), constant and Van Leer MUSCL reconstruction and all four
Osher configurations: 128 recorded independent cold solves. Requested coverage is separate
from completed coverage and numerical acceptance. Failed and incomplete
attempts remain visible; optional shifted checks are separately labeled.

Every configuration uses the same physical setup, model-owned cold initialization,
backward Euler, coupled boundary equations, Newton/linear-solver controls,
stopping thresholds and adaptive-CFL policy. Each grid starts independently;
no analytical seed, grid continuation or hidden warm start is used.

## Commands

From the repository root in the existing Poetry environment:

```powershell
# Optional new cold-solve campaign; existing derived evidence is kept separately.
poetry run python -B -u -m development.flux_methods.shared.run --study osher --config development/flux_methods/osher/verification_profile.yaml --output development/flux_methods/osher/outputs/new_verification
# Evaluate the retained canonical states with the current API.
poetry run python -B -m development.flux_methods.shared.audit --study osher --input development/flux_methods/osher/outputs/verification

# Presentation only; no flow is advanced.
poetry run python -B -m development.flux_methods.shared.report --study osher
node development/flux_methods/shared/audit_report.mjs development/flux_methods/osher/REPORT.html
```

Use `--resume` only with identical recorded source/profile/numerical identities.
Execution filters `--cells`, `--cases` and `--reconstructions` can run a smoke
subset without redefining the full matrix; complete it later without filters.

The renderer reads the schema-2 derived `outputs/verification/summary.json`,
verifies every listed row SHA, and joins current-API Operator checks by file and
hash. Recorded execution hashes remain distinct from evaluation-source hashes;
numeric-payload/origin and cold-state identities are mandatory audit checks. The HTML has no
network or server dependency. Its download includes plot arrays, compact audit
summaries, documents and the full original-file manifest. Public downloads contain clean derived data and current-API audit summaries.
Historical raw JSONs are retained only in the ignored immutable archive; no
historical API selector is exposed as a supported alias.

## What the evidence can establish

Spatial convergence, original residual/BC/global tests, admissibility, physical
branch selection and local profile shape are distinct. The complete source and
boundary assembly retains its own validity requirements. Supported model-owned
storage derivatives, stable wave labels, admissible hyperbolic paths and
captured simple roots are assumptions; finite sampling cannot certify unseen
crossings or arbitrary EOS regimes.

No unconditional entropy-stability, positivity or no-extremum theorem is
claimed. Fixed Gauss rules retain their own quadrature counterexamples.
Iteration savings do not imply a speedup: characteristic/path evaluation and
AD can cost more per face. Compilation-inclusive elapsed values are diagnostic,
not a matched performance benchmark, and provide no timing ranking.

The [shared runner](../shared/run.py), [auditor](../shared/audit.py) and
[renderer](../shared/report.py) use maintained production code. This study's
renderer configuration and findings adapter depend only on its own saved data.

Browser rendering is unverified. Structural offline checks and executable
dropdown/trace audits are retained separately from numerical verification.

## Common report presentation

This report shares its header, typography, theme, navigation and plot styling
with the other [flux-method reports](../shared/REPORT_DESIGN.md). The main
pages are **Overview**, **Case explorer**, **Theory**, and **Protocol & evidence**.
Method-specific checks, numerical acceptance rules and saved evidence remain
explicit. Light, dark and automatic themes work offline; all offered downloads
and rendering assets are embedded in the HTML.
