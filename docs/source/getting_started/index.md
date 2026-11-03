# Getting started

## Installation

Use **Python 3.12 or 3.13**. Clone the repository and install its locked
dependencies with Poetry:

```bash
git clone https://github.com/turbo-sim/nozzle-hyperbolic.git
cd nozzle-hyperbolic
poetry env use 3.13
poetry install --with dev
poetry run python -c "import nozzle_hyperbolic; print(nozzle_hyperbolic.__version__)"
```

Choose an installed supported Python version in the `poetry env use` command.
The package uses the `src/` layout and is installed editably by Poetry; launch
scripts through `poetry run` from the repository root. The development group
includes Sphinx, its theme and MyST, so this installation also builds the docs.

For an existing Python environment, an editable installation of the runtime
package is also possible with `python -m pip install -e .`. The Poetry lockfile
is the reproducible development setup used by this repository. There is no need
to modify `PYTHONPATH`.

## A minimal working nozzle example

Save this as `minimal_nozzle.py` in the repository root, then run
`poetry run python minimal_nozzle.py`. It constructs a smooth converging-diverging
nozzle, an air property model, reservoir/pressure boundaries and a steady solve.
It uses a deliberately small mesh so the complete workflow is easy to inspect.
The first invocation includes JAX compilation.

```{literalinclude} ../_examples/minimal_nozzle.py
:language: python
```

`NozzleModel` uses SI units: pressure in Pa, temperature in K, enthalpy in J/kg,
velocity in m/s, length in m and area in m². Its iteration variables are
`(p, u, h)`. The model converts them to mass, momentum and total-energy storage;
the solver advances those conserved contents.

`solution.state` has shape `(N + 2, 3)`: the first and last rows are independent
endpoint traces, and `state[1:-1]` contains the physical cells. The example uses
`model.fields()` to obtain cell pressure, velocity, Mach number and other
thermodynamic quantities. `solution.success` and `reason` report the outcome;
the history retains accepted and rejected attempts. A small Newton correction
alone is insufficient to declare steady convergence.

Download the {download}`complete example <../_examples/minimal_nozzle.py>`.

## Change the numerical method

Reconstruction, flux and time integrator are independent choices:

```python
scheme = Scheme(
    reconstruction="muscl_van_leer",
    flux="roe",
    time_integrator="euler-implicit",
    newton_backtracking=True,
)
```

Nozzle and annular Euler support `stadtke`, `rusanov`, `hll`, `roe` and `osher`.
The [flux chapter](../numerics/fluxes.md) explains their requirements and fixes.
Two-fluid flow instead requires `Scheme(flux="model_compatibility")` and phase
fluids with the properties needed by the selected closures.

For physical-time integration, use `solve_transient` and requested output times.
Explicit Euler, SSPRK3 and classical RK4 advance conservative contents and
complete the algebraic boundary conditions at their stages. The
[time-integration chapter](../numerics/time_integration.md) explains the distinction
between physical time and steady pseudo-time.

## Next steps

Read [physical theory](../theory/index.md) for the equations and boundary laws,
[numerical solver](../numerics/index.md) for the solution process, or
[demo examples](../demos.md) for complete scripts with plots and animations.
The [API reference](../api_reference.md) lists the installed public interfaces.
