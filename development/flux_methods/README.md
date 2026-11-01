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

The new Stadtke and Osher studies request only nozzle/annular shocks at 35%
and 70%, on 50/100/200/400 cells, both reconstructions and four variants per
study. Each profile requests 128 independent cold solves. Actual completion,
failures and fresh original-equation checks are recorded in each report;
requested coverage alone is not evidence of completed runs. Compile-inclusive
verification durations are not repeated performance measurements.

Shared [execution](shared/run.py), [fresh auditing](shared/audit.py) and
[report generation](shared/report.py) have no runtime dependency on the older
shock-spike prototype. Each campaign preserves settings, source/client byte
snapshots, physical-case fingerprints, saved states, histories and raw balances.
Reports are standalone offline HTML with Plotly figures and a case dropdown.
The [common presentation guide](shared/REPORT_DESIGN.md) documents the shared
Overview / Case explorer / Theory / Protocol & evidence structure, theme,
accessible controls and saved-data-only rebuild commands.

The retired shock-spike prototype's contents, inventory and backup archive have
been removed. The maintained Stadtke study retains the original method's spike
behavior and comparisons with the custom corrections.

The completed [delivery check](shared/evidence/delivery_audit.json) joins
256 native converged results and 256 fresh audits to their recorded source
epoch. A concurrent Roe edit is preserved; its guarded Osher recovery used
byte-verified isolated source files, with the original 115 rows unchanged.
Accuracy tradeoffs remain explicit in each report. The delivery record predates
the requested removal of the prototype archive; its archival checks describe
that earlier state.

The [historical delivery audit](shared/audit_delivery.py) joins saved campaign
identities and production-test evidence without removing files. It requires
the original pre-migration campaign schemas, source-isolation records and test
artifacts. These prerequisites are not all retained in the current checkout,
and the current canonical campaign views use a newer schema. The saved delivery
record documents its recorded epoch; it is not a fresh validation of today's
checkout. Failed numerical runs must remain explicit and agree with their fresh
checks; a valid audit does not turn a failed solve into a converged result.
