# Migration to the shared FVM API

The supported Euler nozzle, annular Euler and scalar models use
[`nozzle_hyperbolic.solvers.fvm`](../src/nozzle_hyperbolic/solvers/fvm/__init__.py).
Physical models supply storage, flux/wave data, source/closure laws and boundary
equations. The shared engine owns numerical assembly and integration. This is
a breaking API change; historical factory signatures are not compatibility APIs.

## Entry points

| Previous usage | Supported replacement |
|---|---|
| Model-specific nozzle/annular FVM factories | `NozzleModel` or `AnnularModel`, then `Operator` and shared drivers |
| Numerical options mixed into physical dictionaries | `Scheme`, `SteadySettings`, `ResidualPolicy` and explicit transient arguments |
| `fvm_solver` compatibility imports | Canonical `models`, `numerics`, `solvers` and `geometry` modules |
| `flux="characteristic"` in the earlier shared API | `flux="stadtke"`; other current IDs are `rusanov`, `hll`, `roe` and `osher` |
| Private analytical wrappers in development studies | Model-owned `analytical/` modules and `nozzle_hyperbolic.verification` |
| Primitive-variable explicit updates | Conservative storage updates, with model-variable recovery and nonlinear boundary completion |

The installable package remains `src/nozzle_hyperbolic`, imported as
`nozzle_hyperbolic`. Use the existing Poetry environment and editable install.
Construct fixed geometry and a physical model before binding numerical choices.
See the [model interface](model_interface.md) for a complete extension contract
and the [architecture](architecture/ARCHITECTURE.md) for configuration examples.

## Numerical choices to review when migrating a script

- `solve_grid_sequence` now carries the last accepted CFL across successful
  warm grid transfers by default. Set `carry_cfl=False` to retain per-grid CFL
  reset. The new mesh determines the timestep; cold starts/retries still use
  `settings.cfl`. Residual tolerances and per-grid baselines are unchanged.
- The default Stadtke flux enables both `custom_jump_correction` and
  `custom_endpoint_envelope`. Disable both to select the original Stadtke
  contraction. An old flux name or final-field resemblance does not establish
  equivalence of the discretization or convergence trajectory.
- `Scheme.boundary_treatment` selects `auto`, `coupled` or `segregated`.
  Backward Euler supports both treatments of the same physical equations;
  explicit Euler and SSPRK3 require segregated nonlinear stage completion.
  Endpoint constraints have zero storage.
- Acoustic compatibility/entropy extrapolation is the default pressure outlet.
  `outlet_formulation="finite_wave"` retains a restricted perfect-gas alternative;
  its general-EOS extension remains open. Boundary physics belongs to the model,
  independently of the numerical boundary solve.
- Implicit flow Newton defaults to undamped corrections with cap 10. Finite
  admissibility and the configured inner convergence tests still apply.
  `newton_backtracking=True` enables Armijo damping. Failed attempts reject the
  timestep; auxiliary storage and boundary solves retain their safeguards.
- Default steady stopping requires per-equation relative reduction $10^{-6}$
  **or** absolute norm $10^{-9}$, plus independent boundary/global limits
  $10^{-9}$. These fixed-scale spatial checks differ from historical state-scaled
  SSR or update norms. See [residual monitoring](residual_monitoring.md).
- Stencil declarations belong to the physical model. Colored AD and the banded
  direct solve use those declarations; global boundary coupling requires the
  supported dense Jacobian/dense linear-solve configuration.
- Inject [wall policies](physical_closures.md) into models rather than choosing
  friction laws in numerical code. Euler numerical properties use the PH EOS
  interface; perfect-gas analytical references are separate verification tools.

## Verification and historical code

Use the [installed verification API](../src/nozzle_hyperbolic/verification/README.md)
for versioned physical YAML cases and independent analytical projections. Meshes,
fluxes, integrators, tolerances and repeat counts belong to numerical profiles.
The [steady benchmark](../development/roberto/benchmarking/README.md),
[flux studies](../development/flux_methods/README.md) and
[transient study](../development/roberto/transient_problems/README.md) call the
production source. Their reports state actual completed coverage and limitations.
The former `generic_interface_v3` directory is not a current entry point.

The frozen native implementation is retained in
[`legacy/pre_generic_fvm`](../legacy/pre_generic_fvm/README.md), with its source
manifest. Historical scripts calling retired factories need individual migration;
preserving their reports does not make their original commands supported today.

The legacy two-fluid implementation has not been ported to the shared contract.
Its own eigenstructure, nonconservative terms, storage constraints and stiff
closures require a separate physical and numerical review. Existing low-level
utilities and empirical closures are not evidence of a completed two-fluid port.
