# Package documentation

Start with the architecture for supported capabilities, then use the detailed
guides for model extensions, physical closures and convergence assessment.
Numerical source remains the authority for defaults and callback signatures.

| Guide | Purpose |
|---|---|
| [Test suite](../tests/README.md) | Organization, fast/full checks, physical regressions, saved baselines and CI |
| [Architecture](architecture/ARCHITECTURE.md) | Models, numerical fluxes, reconstruction, boundaries, integration and solver hierarchy; [offline viewer](architecture/ARCHITECTURE.html) and [SVG](architecture/ARCHITECTURE.svg) |
| [Physical model interface](model_interface.md) | Required callbacks, shapes, frames, stencil declarations and optional Stadtke/Osher/Roe capabilities |
| [Physical closures](physical_closures.md) | Wall policies, Haaland/Darcy conventions, transport properties and retained mixture correlations |
| [Acoustic backpressure derivation](backpressure_acoustic_derivation.md) | General-EOS characteristics, frozen impedance, entropy extrapolation and the relation to finite-wave outlets |
| [Residual monitoring](residual_monitoring.md) | Fixed-scale spatial, boundary and global checks, stopping criteria and physical monitors |
| [Migration](migration.md) | Breaking changes from the model-specific factories to the shared FVM API |
| [Two-fluid migration](two_fluid_migration.md) | Frozen Alberto reference, constrained cells, compatibility flux, boundaries, closures and verification limits |
| [Pseudo-time and dual time](pseudo_time_stepping.md) | Benefits, costs and distinction from physical-time integration; documentation only |
| [Continuation and restart](continuation_and_restart.md) | Explicit physical-parameter stages, constrained preparation, retained failures and portable arrays |
| [Analytical references](analytical_references.md) | Reference assumptions, invariants, source attribution and conservative cell projections |
| [Numerical studies](numerical_studies.md) | Controlled experiments, fair timing, provenance, figures and standalone report conventions |

Other material has an explicit role:

- [Proposals](proposals/README.md) contain unimplemented numerical ideas and a
  research backlog. Inclusion here is not an implementation or validation claim.
- [Historical audits](archive/README.md) retain dated branch and restructuring
  observations. Their original source locations and test counts are historical.
- [Historical Newton notes](../development/roberto/historical/newton_solve/README.md)
  preserve the stagnation analysis and earlier flux/limiter experiments beside
  the study that explains their measured evidence.
- [Annular model notes](euler_annular/) retain the original physical derivations
  and research notes. They have not been revised as part of this cleanup;
  consult the current model interface for production storage and numerical APIs.
- [Flux studies](../development/flux_methods/README.md),
  [steady benchmarks](../development/roberto/benchmarking/README.md) and
  [transient studies](../development/roberto/transient_problems/README.md)
  own their numerical profiles, saved results and report-generation scripts.

Keep active guides in this directory, proposals in `proposals/`, and dated
audits in `archive/`. Put experiment-specific evidence beside its development
study. Generated `_build/` files are disposable and ignored; do not maintain
additional guide copies there. Preserve measurement-time paths and source
fingerprints in original numerical records when reorganizing documentation.
