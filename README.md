# Nozzle-hyperbolic

## Repository layout

- [src/nozzle_hyperbolic/](src/nozzle_hyperbolic/): installable solver package.
- [demos/](demos/README.md): standalone, commented physical-model scripts with
  inline configuration, convergence plots and transient MP4s.
- [cases/](cases/README.md): physical applications and literature datasets,
  including [validation data](cases/archive/validation/notes.md).
- [development/](development/README.md): Alberto's legacy research scripts and
  Roberto's controlled verification and numerical studies.
- [tests/](tests/README.md): organized production tests, analytical checks and saved numerical regressions.
- [docs/](docs/README.md): Sphinx documentation, with earlier guides and research material in [docs/notes/](docs/notes/README.md).

## Documentation

Read the [online documentation](https://turbo-sim.github.io/nozzle-hyperbolic/).
It updates automatically when source or documentation changes reach `main`.

The [Sphinx sources](docs/source/index.md) cover getting started, the three
physical models, numerical methods, demo examples and the generated API reference.
Build a local preview with:

```bash
poetry run python docs/build_docs.py
```

Use `--no-autobuild` for a single HTML build. See [docs/README.md](docs/README.md)
for the build workflow. The original documents remain available as repository
notes outside the Sphinx source tree.

Explore the [interactive architecture map](docs/notes/architecture/ARCHITECTURE.html),
[vector diagram](docs/notes/architecture/ARCHITECTURE.svg), and
[detailed architecture guide](docs/notes/architecture/ARCHITECTURE.md).

## Shared finite-volume solver

The production Euler nozzle, annular Euler, scalar and constrained two-fluid models use the same
model-independent FVM engine in
[`solvers/fvm`](src/nozzle_hyperbolic/solvers/fvm/__init__.py). Models supply
storage, physical fluxes/eigenstructure, source/closure laws, admissibility and
boundary equations. The engine supplies reconstruction, numerical fluxes,
Newton/linear solves, explicit Euler, backward Euler, SSPRK3 and classical RK4 integration.
Numerical flux IDs are `stadtke` (the default), `rusanov`, `hll`, `roe`,
and `osher`. Osher uses `quadrature_points=5` by default (3, 5 or 9 supported)
and `integration="gauss"` or `"sonic_partitioned"`, with `"gauss"` the default.
Stadtke exposes two independent
custom corrections, `custom_jump_correction` and `custom_endpoint_envelope`,
both enabled by default; disable both to recover the original implemented FVS.
The correction names identify project modifications, not terms attributed to
Stadtke. Roe uses model-owned finite-jump data; nozzle and annular
Euler provide a general-EOS PH construction. See the
[flux studies](development/flux_methods/README.md) for equations, controlled
verification, reports and limitations.
Implicit flow Newton uses full corrections by default; select
`Scheme(newton_backtracking=True)` for Armijo damping. Invalid full trials
still reject the timestep, and strict inner convergence remains required.
`Scheme.boundary_treatment` selects coupled or segregated implicit boundaries;
explicit methods use segregated stage solves. The default `auto` preserves the
integrator's established treatment. The
[shock-tube demo](demos/shock_tube/README.md) and
[transient-nozzle demo](demos/nozzle_transient/README.md) provide standalone
examples and animations. Steady verification and
runtime regression monitoring live in
[`development/time_benchmarking/`](development/time_benchmarking/README.md).

The [two-fluid migration guide](docs/notes/two_fluid_migration.md) records Alberto's
frozen physical reference, eight-state/six-balance/two-EOS formulation,
explicit `model_compatibility` flux, closure policies and verification limits.
The [limiter comparison](development/flux_limiters/README.md) covers Minmod, Superbee
and Van Albada. [Lettieri validation](cases/lettieri_validation/README.md) uses
the separate shared Euler/HEM model with zero wall friction.

Both acoustic and finite-wave pressure outlets support general-EOS properties.
The finite-wave option computes PH Hugoniot shocks and numerical expansion
curves, retaining perfect-gas closed forms for that backend. The
[backpressure study](development/BC_backpressure/README.md) compares both laws
from stagnant and numerical adapted starts, including bicubic CO2 cases.

Versioned physical YAML cases are packaged beside the
[nozzle](src/nozzle_hyperbolic/models/euler_nozzle/cases) and
[annular](src/nozzle_hyperbolic/models/euler_annular/cases) models.
[`nozzle_hyperbolic.verification`](src/nozzle_hyperbolic/verification/README.md)
loads these definitions and builds cold physical cases with independent
analytical projections. Numerical methods, meshes, timing repetitions and
acceptance settings belong to the benchmark profile.

The production steady defaults are CFL 0.2, accepted-step growth 1.2, maximum
CFL 1e6, and ten Newton corrections. The [architecture guide](docs/notes/architecture/ARCHITECTURE.md)
describes the supported choices. The [model interface](docs/notes/model_interface.md)
and [migration guide](docs/notes/migration.md) explain extensions and breaking changes.

```powershell
poetry run python -B -m development.historical.benchmarking.scripts.run
poetry run python -B -m development.historical.benchmarking.scripts.report
```


## Cloning git repository and installing dependencies

Install Python 3.12-3.13 and Poetry, then clone the repository and install the
locked dependencies. Poetry keeps the environment in the project's `.venv/`
directory; Conda is not required.

```bash
git clone https://github.com/turbo-sim/nozzle-hyperbolic
cd nozzle-hyperbolic
poetry env use 3.13
poetry install --with dev
```

Use `poetry run` to run commands in this environment:

```bash
poetry run python -c "import nozzle_hyperbolic; print(nozzle_hyperbolic.__version__)"
poetry run pytest tests
```

Change `3.13` to another installed, supported Python version when needed. After
editing dependency requirements, run `poetry lock` and `poetry install --with dev`.
Commit `pyproject.toml` and `poetry.lock` together.

The installable package lives in `src/nozzle_hyperbolic/`. Run `poetry install`
after cloning or moving the source tree so that Python uses the editable
installation. Imports remain `import nozzle_hyperbolic`; no `PYTHONPATH` change
or package-directory entry in `sys.path` is needed.

The commented local dependencies in `pyproject.toml` support development against
the neighboring `jaxprop`, `pysolver_view`, and `nurbspy` repositories. Enable a
local entry in place of its published-version entry, then regenerate the lockfile.

`AnnularModel` solves quasi-1D curvilinear annular Euler flow with swirl,
general JAXProp thermodynamics and algebraic boundary equations. Use the
[packaged physical cases](src/nozzle_hyperbolic/verification/README.md),
[model notes](docs/notes/euler_annular) and
[NozzleX comparison](cases/verification_annular/README.md).

## Version bumping and publishing

The project uses `bump-my-version` and `.bumpversion.toml`, following TurboDash's
setup. A bump updates `pyproject.toml` and `src/nozzle_hyperbolic/__init__.py`, commits
the change, and creates a `vX.Y.Z` tag. Start from a clean working tree and preview
the operation first:

```bash
poetry run bump-my-version bump --dry-run --verbose patch
```

The release commands are documented at the end of `pyproject.toml`. Publishing is
manual until a publishing workflow is configured for this repository.


## Pushing to git repository

Commands to add changes and push to the remote repository

```bash
git add <filename>
git commit -m <commit message>
git push origin
```
