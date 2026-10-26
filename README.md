# Nozzle-hyperbolic


## Cloning git repository and installing dependencies

Install Python 3.11-3.13 and Poetry, then clone the repository and install the
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
poetry run python -c "import nozzle_hyperbolic"
poetry run pytest tests
```

Change `3.13` to another installed, supported Python version when needed. After
editing dependency requirements, run `poetry lock` and `poetry install --with dev`.
Commit `pyproject.toml` and `poetry.lock` together.

The commented local dependencies in `pyproject.toml` support development against
the neighboring `jaxprop`, `pysolver_view`, and `nurbspy` repositories. Enable a
local entry in place of its published-version entry, then regenerate the lockfile.

## Pushing to git repository

Commands to add changes and push to the remote repository

```bash
git add <filename>
git commit -m <commit message>
git push origin
```
