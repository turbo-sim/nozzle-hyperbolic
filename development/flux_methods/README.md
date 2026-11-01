# Production numerical flux studies

These studies call the shared production `Operator` and drivers. Physical
models, packaged cases and analytical references remain in `src/`; study
scripts implement execution, assessment, provenance and presentation.

| Public flux ID | Type | Production location | Controlled study |
|---|---|---|---|
| `stadtke` | Generalized mean-state FVS, with two independent custom algebraic corrections | `numerics/characteristic.py`, shared `Operator` | [Stadtke](stadtke/README.md) |
| `rusanov` | Local Lax–Friedrichs approximate Riemann flux | `numerics/fluxes.py` | [Roe comparison profile](roe/benchmark_profile.yaml) |
| `hll` | Two-wave Harten–Lax–van Leer approximate Riemann flux | `numerics/fluxes.py` | [Roe comparison profile](roe/benchmark_profile.yaml) |
| `roe` | Model-owned finite-jump linearization and characteristic dissipation | `numerics/roe.py` | [Roe](roe/README.md) |
| `osher` | Osher-type path viscosity with selectable quadrature and integration policy | `numerics/osher.py` | [Osher](osher/README.md) |

Stadtke is the sole public name of that FVS family. Mathematical projectors,
characteristic callbacks and boundary equations retain their mathematical
names. `custom_jump_correction=True` and `custom_endpoint_envelope=True` are
the production defaults. The Städtke study explicitly tests neither, each
separately, and both together. These custom terms are not sensor blends.
All five IDs are independent of constant or Van Leer MUSCL reconstruction.
Osher uses `quadrature_points=5` by default, with 3, 5 or 9 points supported.
This count specifies integration nodes, not spatial order.
`integration="gauss"` is the default unsplit rule;
`integration="sonic_partitioned"` captures sonic/regularization cuts and applies
the selected Gauss rule on each nonempty panel. An unsplit nine-point rule is
not equivalent to sonic partitioning.

This table describes the supported shared FVM `Scheme` API. Retained low-level
Euler flux regression helpers and the deferred two-fluid solver are separate
entry points; they do not automatically acquire the shared correction flags or
Osher path support. The two-fluid implementation has its own model-specific
FVS/eigenstructure and has not been ported or benchmarked by these studies.

## Reports and controlled studies

- [Stadtke](stadtke/README.md): original FVS, each independent custom correction
  and both corrections on four nozzle/annular shock cases and both reconstructions.
- [Osher](osher/README.md): one configurable flux with explicit quadrature-point
  and integration parameters, using the same four shock cases.
- [Roe](roe/README.md): the broader steady suite, general-EOS matrix checks, Sod
  transients and retained real-gas flow attempts.

Each report records its actual coverage, failures, analytical accuracy and
execution/evaluation identities. A configured population is not a completion
claim, and verification durations do not establish matched timing rankings.

The [common presentation guide](shared/REPORT_DESIGN.md) describes the shared
Overview / Case explorer / Theory / Protocol & evidence layout, offline assets,
themes and executable presentation audits. The method-specific numerical data
and acceptance criteria remain distinct.

Shared [execution](shared/run.py), [saved-state auditing](shared/audit.py) and
[report generation](shared/report.py) use production code without importing
older prototypes. For presentation-only rebuilding from saved data:

```powershell
poetry run python -B -m development.flux_methods.shared.report --study stadtke
poetry run python -B -m development.flux_methods.shared.report --study osher
poetry run python -B -m development.flux_methods.roe.scripts.report --presentation-only
```

The generated standalone HTML files live at each study root and embed their
rendering resources and offered downloads. Raw campaign outputs retain their
recorded identities; report generation does not advance a numerical solution.

The retired shock-spike prototype's contents, inventory and backup archive have
been removed. The maintained Stadtke study retains the original method's spike
behavior and comparisons with the custom corrections.
