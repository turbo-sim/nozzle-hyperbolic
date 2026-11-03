# Nozzle-hyperbolic

Nozzle-hyperbolic is a Python package for simulating compressible single-phase
and two-phase flow in nozzles. It uses finite-volume methods built with JAX to
solve steady and transient problems, with support for real-fluid thermodynamics
through JAXProp.

## Installation from source

Install **Python 3.12 or 3.13** and **Poetry**, then clone the repository and
install the project with its locked dependencies:

```bash
git clone https://github.com/turbo-sim/nozzle-hyperbolic.git
cd nozzle-hyperbolic
poetry env use 3.13
poetry install --with dev
```

Use `3.12` instead of `3.13` if that is your installed Python version.
The installation includes the development and documentation tools.

Run Python commands in the project environment with `poetry run`. To check
that the package is installed:

```bash
poetry run python -c "import nozzle_hyperbolic; print(nozzle_hyperbolic.__version__)"
```

## Documentation

Read the [online documentation](https://turbo-sim.github.io/nozzle-hyperbolic/)
for usage and examples.
