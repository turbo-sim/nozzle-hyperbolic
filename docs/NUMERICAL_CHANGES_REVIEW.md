# Structural refactor and uncommitted numerical changes

The structural source-layout commit uses `d2a69ef` as its pre-experiment
numerical baseline. It moves the package to `src/nozzle_hyperbolic`, adopts
the EulerNozzle/EulerAnnular names, relocates analytical references, and
updates imports and packaging. It excludes the later numerical modifications
described below. No NozzleX integration is included.

The experimental implementations remain in the working tree. Consequently,
a fresh checkout of the committed tree uses the original numerical algorithms,
while the current editable installation still sees the uncommitted experiments.
This separation preserves the work without approving its numerical changes.

## Changes deliberately excluded from the structural commit

| Working-tree file | Change | Potential effect |
|---|---|---|
| `src/nozzle_hyperbolic/models/euler_nozzle/flux_jax.py` | Replaces the public FVS implementation with conservative characteristic projectors, continuous sonic regularization, and additional sonic viscosity. Retains the old implementation as `_jax_F_fvs_stadtke_legacy`. | Changes interface fluxes and the discrete equations near the sonic regularization band. This is a numerical-method change. |
| `src/nozzle_hyperbolic/numerics/reconstruction.py` | Adds safe inactive arithmetic to the ordinary JAX Van Leer limiter and an optional smooth limiter. | The ordinary active slope formula is retained, but AD/NaN behavior changes. The optional smoothed formula changes reconstruction and finite-mesh solutions. |
| `src/nozzle_hyperbolic/models/euler_nozzle/fvm.py` | Routes fluxes through the new default `flux_variant='regularized'`, adds smooth-limiter options and validation, and uses the safe Van Leer implementation in the JAX factories. | Existing configurations can select a different flux even without adding a setting. Zero regularization widths are rejected for the corrected flux. |
| Same `fvm.py` | Adds `return_operators` and steady residual/Jacobian/Newton/storage interfaces for direct-solver experiments. | Adds an experimental API. The optional operator exposure alone is distinct from the flux/limiter changes, but all of it is excluded pending review. |

Selecting `flux_variant='legacy'` in the experimental working tree is not a
complete rollback: the shared limiter arithmetic has also changed. The
structural commit restores the complete original implementations in its Git
tree, including the original factory signatures and defaults.

The experimental direct Newton, LM, HYBR, SLSQP, continuation, and Newton/PTC
drivers remain uncommitted under `development_new/`, along with their new
tests. The existing changes to the shock-validation Python/YAML files also
remain outside the structural commit; those include a change from LUT
properties to a perfect-gas backend and predate this separation.

## What the structural commit does preserve

- EulerNozzle equations, boundary conditions, closures, state conversions,
  initialization, original fluxes, original reconstruction, and time stepping.
- TwoFluid solver, thermodynamics, sources, and closures.
- Existing analytical calculations moved beside their owning models.
- Historical `fvm_solver` imports through compatibility modules.
- Existing `dev_scripts` files moved to `development`, with analytical imports
  redirected to their model-owned implementations.

The preparation audit checks 163 package-level function/class definitions.
Existing definitions match their pre-experiment sources, including docstrings
and comments, after normalizing line endings. The one extracted sound-speed
Sod reference is checked separately against the original inline calculation
in `dev_scripts/main.py`: its mathematical statements match after argument
renaming. Moving these analytical calculations does not introduce a new
physical model or change their formulas.

The `src` migration changes installation and resource locations. It does not
change the package import name: callers still use `import nozzle_hyperbolic`.

## Reviewing the pending work

From the repository root, after the structural commit:

```powershell
git diff -- src/nozzle_hyperbolic/models/euler_nozzle/flux_jax.py
git diff -- src/nozzle_hyperbolic/numerics/reconstruction.py
git diff -- src/nozzle_hyperbolic/models/euler_nozzle/fvm.py
git status --short
```

The current reports describe experiments performed with the numerical changes,
not validation of the original algorithms in a fresh structural checkout:

- [Flux and reconstruction changes](SOLVER_REGULARITY_IMPROVEMENTS.md)
- [Historical stagnation analysis](MATHEMATICAL_STAGNATION_ANALYSIS.md)
- [Physical convergence study](PHYSICAL_CONVERGENCE_STUDY.md)
- [Newton/PTC experiment](NEWTON_PTC_EXPERIMENT.md)

The earlier [repository audit](ALBERTO_REPOSITORY_AUDIT.md) and
[JAX audit](JAX_IMPLEMENTATION_AUDIT.md) document inherited issues that remain
in the original numerical baseline. Structural preservation does not claim
that those issues are fixed.
