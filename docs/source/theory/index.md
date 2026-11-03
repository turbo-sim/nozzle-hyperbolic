# Physical theory

Each model page derives its equations, identifies the thermodynamic and closure
assumptions, analyzes its characteristics and states the implemented boundary
conditions. The [numerical chapters](../numerics/index.md) then explain how those
equations become a discrete residual.

| Model | Iteration variables | Transport balances | Cell constraints |
|---|---|---|---|
| Euler nozzle | $p,u,h$ | Mass, axial momentum, total energy | None |
| Annular Euler | $p,v_m,\ell,h$ | Mass, meridional momentum, angular momentum, total energy | None |
| Two-fluid | $p,\alpha_g,m_g,u_g,m_l,u_l,m_ge_g,m_le_l$ | Two phase masses, momenta and total energies | Two phase EOS equations |

Here $h=e+p/\rho$, $\ell=r v_\theta$ and $m_k=\alpha_k\rho_k$.
All geometries are fixed in time. The Euler models use one thermodynamic state;
the two-fluid model permits separate velocities and enthalpies with a common
pressure. A homogeneous equilibrium fluid backend in the Euler model therefore
has different physical assumptions from the nonequilibrium two-fluid model.

Throughout these pages, $a$ denotes sound speed, $s$ specific entropy, $A$ flow
area, and $Q$ a vector of conserved densities. Sources are positive when they
add to the corresponding balance. EOS data come from the configured JAXProp
fluid, rather than from an assumed constant ratio of heat capacities.

```{toctree}
:maxdepth: 2

euler_nozzle
euler_annular
two_fluid
```
