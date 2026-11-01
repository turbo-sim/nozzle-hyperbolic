# Stadtke FVS: original and custom corrections

This independent study calls the production finite-volume solver with
`Scheme(flux="stadtke")`. It compares the original baseline with two explicitly
custom algebraic corrections. No solver, model, geometry or analytical
reference implementation lives in this directory.

Measured results and production tests belong to the source/case/client SHA-256
epoch recorded with the campaign. They do not assert that the current checkout
is byte-identical: an external edit to `models/euler_roe.py` was preserved after
this study's recorded epoch. Rebuilding the report performs no current-tree
numerical validation and does not rewrite those historical identities.

| Saved variant | `custom_jump_correction` | `custom_endpoint_envelope` |
|---|---|---|
| `stadtke_original` | `False` | `False` |
| `stadtke_jump` | `True` | `False` |
| `stadtke_endpoint` | `False` | `True` |
| `stadtke_both` | `True` | `True` |

The flags default to `True`; explicitly disable both to request the original
baseline. **The custom terms are not the literature Stadtke method.**
Endpoint-only is a separate new test, not another name for both-on. The exact
$F^+/F^-$ split, correction terms and limits are in [THEORY.md](THEORY.md).

The [Plotly report](REPORT.html) provides one four-case dropdown, simultaneous
pressure/velocity/temperature/total-enthalpy profiles, full/shock views,
residual/CFL histories, accepted/rejected/Newton work and analytical refinement.
[REPORT.md](REPORT.md) summarizes saved coverage and audit status. Reports read
saved evidence only and make no performance ranking from compilation-inclusive
elapsed diagnostics.

## Controlled population

[verification_profile.yaml](verification_profile.yaml) requests
`euler_nozzle/shock_35`, `euler_nozzle/shock_70`,
`euler_annular/shock35` and `euler_annular/shock70`; grids 50/100/200/400;
constant and `muscl_van_leer` reconstruction; and all four variants. This is
128 independent cold solves. A requested matrix is not a completion claim.
Every saved attempt, including failures and budget stops, remains in its
summary. Optional shifted checks are labeled separately.

The shared profile specifies backward Euler, coupled boundaries, colored AD,
banded linear solves, unchanged strict Newton/residual criteria and the same
adaptive CFL policy for every variant. Each grid begins from the model-owned
cold state; no continuation or analytical initial guess is used. Physical YAML
and analytical references come from the installed verification package.

## Commands

Run from the repository root in the existing Poetry environment. The first
command performs numerical work; the next independently audits saved rows.

```powershell
poetry run python -B -u -m development.flux_methods.shared.run --study stadtke --config development/flux_methods/stadtke/verification_profile.yaml --output development/flux_methods/stadtke/outputs/verification
poetry run python -B -m development.flux_methods.shared.audit --study stadtke --input development/flux_methods/stadtke/outputs/verification
```

`--resume` is permitted only with the same recorded source, profile and
numerical identities. Execution filters `--cells`, `--cases` and
`--reconstructions` support a smoke subset without changing the requested
profile identity; rerun without filters to complete the population.

Rebuild the report without solving:

```powershell
poetry run python -B -m development.flux_methods.shared.report --study stadtke
node development/flux_methods/shared/audit_report.mjs development/flux_methods/stadtke/REPORT.html
```

The report command needs `outputs/verification/summary.json`; it verifies every
listed saved-row hash. Fresh checks in `fresh_audit.json` join by path and SHA.
The standalone HTML embeds Plotly, plot arrays, compact audit summaries,
documents and an evidence-manifest download. Full original states, histories
and audit tensors remain in `outputs/verification`. Structural and executable
application QA do not substitute for browser rendering, which is unverified.

## Ownership and interpretation

Production code belongs to
[the numerical package](../../../src/nozzle_hyperbolic/numerics/) and
[Scheme configuration](../../../src/nozzle_hyperbolic/solvers/fvm/config.py).
The [shared runner](../shared/run.py), [independent auditor](../shared/audit.py)
and [saved-data renderer](../shared/report.py) are clients. They import no
experimental solver implementation from another study.

Convergence of the original equations, admissibility, physical branch,
conservation, analytical accuracy and local shock shape are separate checks.
These corrections change the spatial residual and its Jacobian, not the Newton
algorithm. A pressure improvement may worsen total enthalpy or shock width.
No universal entropy-stability, positivity or monotonicity claim is made.

The sibling [Osher](../osher/README.md) and [Roe](../roe/README.md) studies test
different constructions. Their results do not validate a missing configuration
in this four-variant production study.

## Common report presentation

This report shares its header, typography, theme, navigation and plot styling
with the other [flux-method reports](../shared/REPORT_DESIGN.md). The main
pages are **Overview**, **Case explorer**, **Theory**, and **Protocol & evidence**.
Method-specific checks, numerical acceptance rules and saved evidence remain
explicit. Light, dark and automatic themes work offline; all offered downloads
and rendering assets are embedded in the HTML.
