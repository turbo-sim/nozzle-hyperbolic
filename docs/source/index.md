# Nozzle-hyperbolic

**One-dimensional compressible flow, from physical equations to numerical solutions.**

Nozzle-hyperbolic solves variable-area Euler flow, annular Euler flow with swirl,
and nonequilibrium two-fluid flow. Physical models provide thermodynamics,
geometry, closures and boundary equations. A shared finite-volume engine provides
reconstruction, numerical fluxes, automatic differentiation, time integration
and nonlinear solution.

JAXProp supplies the fluid properties. Euler calculations can use a perfect-gas
backend or a general equation of state through the same model interface. The
two-fluid formulation retains separate phase velocities and enthalpies, a shared
pressure and two local EOS constraints.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install the package and run a complete air-nozzle example.
:::

:::{grid-item-card} Physical theory
:link: theory/index
:link-type: doc

Derivations, governing equations, characteristics, closures and boundaries.
:::

:::{grid-item-card} Numerical solver
:link: numerics/index
:link-type: doc

Follow a solve from face reconstruction through Newton steps and convergence.
:::

:::{grid-item-card} Demo examples
:link: demos
:link-type: doc

Steady nozzles, annular flow, two-fluid flow and transient animations.
:::

:::{grid-item-card} API reference
:link: api_reference
:link-type: doc

Browse model constructors, numerical settings, drivers and helper functions.
:::

::::

The theory pages describe the active `src/nozzle_hyperbolic` implementation.
In particular, the two-fluid model requires its explicitly selected
`model_compatibility` flux. Availability of a numerical option does not imply
that every model/closure/regime combination has been physically validated.

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
theory/index
numerics/index
demos
api_reference
```
