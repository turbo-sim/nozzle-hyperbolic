# Nozzle-hyperbolic

## Repository layout

- [src/nozzle_hyperbolic/](src/nozzle_hyperbolic/): installable solver package.
- [cases/](cases/README.md): physical applications and literature datasets,
  including [validation data](cases/validation/notes.md).
- [development/](development/README.md): Alberto's legacy research scripts and
  Roberto's controlled verification and numerical studies.
- [tests/](tests/): production tests.
- [docs/](docs/): architecture, derivations and numerical findings.

Explore the [interactive architecture map](docs/architecture/ARCHITECTURE.html),
[vector diagram](docs/architecture/ARCHITECTURE.svg), and
[detailed architecture guide](docs/architecture/ARCHITECTURE.md).


## Roe numerical flux

The shared FVM engine supports `Scheme(flux="roe")` for nozzle and annular
Euler, with model-owned general-EOS PH averaging. The generic numerical kernel
contains no EOS assumptions. Characteristic FVS remains the default.
See [the final Roe study](development/roe_method/README.md) for the implemented
theory, verification results, limitations and reproducible benchmark/report commands.

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

The [EulerAnnular prototype](development/roberto/historical/euler_annular/README.md) solves
quasi-1D curvilinear annular Euler flow with swirl, general JAXProp
thermodynamics and simultaneous algebraic boundary equations. Its verification
driver compares analytical, constant-reconstruction and MUSCL results. Read the
study report for the measured accuracy limits and supported boundary regimes.

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
