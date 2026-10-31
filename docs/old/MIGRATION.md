# Migration to the shared FVM engine

The production Euler and scalar entry points now use
[`solvers/fvm`](../src/nozzle_hyperbolic/solvers/fvm/__init__.py). This is an
intentional breaking API change. Numerical assembly and drivers have one shared
implementation; nozzle and annular models supply physical callbacks. Read
[MODEL_INTERFACE.md](MODEL_INTERFACE.md) for the extension contract and
[PHYSICAL_CLOSURES.md](PHYSICAL_CLOSURES.md) for retained constitutive laws.

## Entry points

| Previous entry point | Current entry point |
|---|---|
| `models.euler_nozzle.fvm` factories | `NozzleModel` with `Operator` or common drivers |
| `models.euler_annular.fvm` factories/drivers | `AnnularModel` with the same common drivers |
| Model-specific numerical settings | `Scheme`, `SteadySettings`, `ResidualPolicy` |
| `fvm_solver` compatibility imports | Canonical `numerics`, `solvers`, `models.closures` imports |
| Model analytical solution packages | Independent references under `development/roberto/verification_references` |
| Perfect-gas thermodynamic recovery helpers | Model storage callbacks and JAXProp state evaluations |

`src/nozzle_hyperbolic` remains the installable package; import it as
`nozzle_hyperbolic`. Plain package/core imports do not load physical models,
JAXProp, plotting or a custom exception handler. Models are imported explicitly
or through lazy root convenience exports. No dependency/environment change is
part of this port.

Geometry lives in `geometry/`; it is constructed before numerical binding.
`SymmetricNozzleGeometry.build_mesh` and `ConstantAreaGeometry.build_mesh`
construct nozzle/tube meshes. `geometry.fvm.build_annular_mesh` adapts the physical
annular geometry into the shared mesh and the model's native geometry data.
`models.closures` is the canonical closure library. Model constructors accept
wall policies; the numerical machinery does not choose a closure.

## Deliberate numerical changes

- Default steady acceptance is per-equation relative $10^{-6}$ or absolute
  $10^{-9}$, plus independent boundary/global limits $10^{-9}$. The four
  `ResidualPolicy.criterion` choices are explicit. Old stopping defaults are
  not silently reproduced.
- Acoustic compatibility/entropy extrapolation is the default pressure outlet.
  `outlet_formulation="finite_wave"` selects the retained perfect-gas finite-wave
  alternative. General-EOS extension of that alternative remains a TODO.
- Backward Euler retains coupled zero-storage boundary equations. Explicit
  Euler and SSPRK3 complete those same equations nonlinearly at every stage.
  There is no Mach-dependent blending or velocity clipping in the new path.
- Explicit stages update conservative storage, including nonlinear EOS recovery;
  conservative filters act on transported densities. Source restrictions are
  absolute timestep bounds, separate from wave CFL. Explicit wave/source/filter
  inverse time limits add, accounting conservatively for simultaneous terms.
- Geometry/source/boundary dependency declarations belong to physical models.
  Colored/banded differentiation uses the declared reach. Nonlocal endpoint
  coupling requires dense/dense assembly.

The full default verification campaign uses backward Euler. Separate audits
exercise both outlets and all three integrators, including Sod and adapted
steady starts. This establishes tested compatibility, not unconditional explicit
stability or convergence of every possible stiff closure.

## Retired code and scripts

The pre-port source and production tests are preserved under
[`development/roberto/legacy_before_generic_fvm`](../development/roberto/legacy_before_generic_fvm/).
Duplicate nozzle/annular FVM factories, old compatibility imports, unused
perfect-gas helpers and production analytical references were retired rather
than kept as parallel implementations. Snapshot checksums are in
`source_manifest.json`; retired compatibility code is under
`retired_compatibility/`.

Historical development/case scripts that call these retired APIs require
migration before execution. They are not compatibility guarantees or the
current verification entry points. Use
[`generic_interface_v3`](../development/roberto/generic_interface_v3/README.md)
for current executable verification. It contains configuration, clients and
audits only; all numerical and physical implementations are imported from `src`.
Independent references are outside the production package and are never main
campaign numerical initial guesses.

The legacy `models/two_fluid` implementation has not been ported and is outside
the supported new FVM API. Its old `fvm_solver` import paths were retired;
low-level production tests use canonical imports. Preserving its closure
correlations does not establish compatibility with the new engine. A future
two-fluid port must review characteristic structure,
storage invertibility, source stiffness and any nonconservative products.

## Review evidence

Start with the v3 [evaluation](../development/roberto/generic_interface_v3/evaluation_report.md)
and [verification](../development/roberto/generic_interface_v3/verification_report.md)
reports. The standalone `REPORT.html` embeds field plots and downloadable data.
Generated states, histories, effective settings, audits, source fingerprints and
the production-test JUnit report live in that study's `outputs/` directory.
Coarse-grid analytical misses and all unsuccessful attempts remain recorded.
