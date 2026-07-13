Acconrding to Ishii (Chapter 8, Thermo-Fluid Dynamics of Two-Phase Flow 2006), the interfacial transfer term for the averaged total energy equation for phase k can be written as:
$$
E_k = \Gamma_k \left( \widehat{h_{ki}} + \widehat{\mathbf{v}_{ki}} \cdot \widehat{\mathbf{v}_k} - \frac{\widehat{v_k}^2}{2} \right) + a_i \overline{\overline{q_{ki}''}} - \overline{\overline{p_{ki}}} \frac{\partial \alpha_k}{\partial t} + \boldsymbol{M}_{ik} \cdot \widehat{\mathbf{v}_{ki}} - \nabla \alpha_k \cdot \overline{\overline{\boldsymbol{\mathbb{T}}_{ki}}} \cdot \widehat{\mathbf{v}_{ki}} + W_{ki}^T
$$
In this case:
- the first term is connected to the energy source due to the "appearance" of new mass in phase k (mass flow) and the contribution is described in terms of enthalpy, inner product between the phase bulk velocity and the velocity with respect to the interface, and the phase velocity itself.
- the second term represets the heat  fluxes flowing from the phase to the interface.
- the third term is related to the change in volume fraction
- the forth term is related to the interfacial stress between the phases
- the fifth term is related to the total tensor of the stresses
- the last term is related to the turbulent flux of work due to drag force

The mixture energy source term can be written as:
$$
E_m = \sum_{k=1}^2 E_k = \overline{\overline{T_i}} \left( \frac{d\sigma}{dT} \right) \frac{D_i}{Dt} (a_i) + 2\overline{\overline{H_{21}}} \ \overline{\overline{\sigma}} \frac{\partial \alpha_1}{\partial t} + E_m^H
$$
In the latter, the contribution is divided into three, which are respectively, the total derivative of the interfacial are, the time derivative of the volume fraction of phase 1 and the change in mean energy of the curvature is represented by the last term.

Therefore, considering that we neglect the velocity of the interface so that $\widehat{\mathbf{v}_{ki}} = \widehat{\mathbf{v}_k}$ where it was $\widehat{\mathbf{v}_{ki}} = \left( \widehat{\mathbf{v}_i} + \frac{\Gamma_k}{\overline{\overline{\rho_{ki}}} a_i^2} \nabla \alpha_k \right)$, we neglect time dependent and turbulence dependent effects, and finally we neglect drag forces effects (same velocity between the phases). The final form of the total energy source term, according to Ishii becomes:
$$
E_k = \Gamma_k \left( \widehat{h_{ki}} + \frac{\widehat{v_k}^2}{2} \right) + a_i \overline{\overline{q_{ki}''}}
$$
And the mixture energy jump conditions simplifies to:
$$
E_m = 0
$$

Given that at the interface no energy is stored (as seen above for the mixture energy source term), it is possible to write that:
$$
\Gamma_1 \left( \widehat{h_{1i}} + \frac{\widehat{v_1}^2}{2} \right) + a_i \overline{\overline{q_{1i}''}} + \Gamma_2 \left( \widehat{h_{2i}} + \frac{\widehat{v_2}^2}{2} \right) + a_i \overline{\overline{q_{2i}''}} = 0
$$
Knowing that $\Gamma_1 = -\Gamma_2$, the mass source term can be written as:
$$
\Gamma_1  = -\Gamma_2 = - \frac{a_i \overline{\overline{q_{2i}''}} + a_i \overline{\overline{q_{1i}''}}} {\left( \widehat{h_{1i}} + \frac{\widehat{v_1}^2}{2} \right) - \left( \widehat{h_{2i}} + \frac{\widehat{v_2}^2}{2} \right)}
$$Where the enthalpy at the surface is defined as the mass flow average of the enthalpy flowing across the interfaces, namely as:
$$
\widehat{h_{ki}} \equiv \frac{\sum_{j} \frac{1}{\Delta t} \frac{1}{v_{ni}} \dot{m}_k h_k}{\sum_{j} \frac{1}{\Delta t} \frac{1}{v_{ni}} \dot{m}_k} = \frac{\sum a_{ij} \dot{m}_k h_k}{\sum a_{ij} \dot{m}_k}
$$
While the heat input per unit interfacial area is defined as:
$$
\overline{\overline{q_{ki}''}} = - \left( \sum_{j} a_{ij} \boldsymbol{n}_k \cdot \boldsymbol{q}_k \right) \frac{1}{a_i}
$$Let's consider now the interface as the region where the two fluids are in thermodynamic equilibrium (LTE, local thermodynamic equilibrium, the liquid side is considered in liquid saturation condition, same for the vapor at the interface), the source term can then be rewritten as:
$$ 
\begin{aligned}
h_{li} &= h_{sat}^l \\
h_{vi} &= h_{sat}^v \\
q_{li} &= \text{h}_{li} (T_i - T_l)\\
q_{vi} &= \text{h}_{vi} (T_i - T_v)
\end{aligned} 
$$And the mass transfer source term becomes:
$$
\Gamma_v  = -\Gamma_l = - \frac{a_i q_{li} + a_i q_{vi}} {h_{sat}^v - h_{sat}^l}
$$

And the two source terms becomes, if we neglect the energy carried by the velocities of the phases:
$$
\begin{aligned}
E_v = \Gamma_v h_{sat}^v + a_i \text{h}_{vi} (T_{sat} - T_v)\\
E_l = \Gamma_l h_{sat}^l + a_i \text{h}_{li} (T_{sat} - T_l)
\end{aligned}
$$

This formulation agree in part with CFX, where the enthalpy of the exiting phase is considered to be the actual enthalpy, and not the saturation enthalpy

## TODO
 - [ ] Where is the energy of the new phase appearing coming from? Who is losing the energy to bring the phase from liquid to vapor? 
 - [ ] What if we consider equilibrium temperature at the interface but not eequilibrium enthalpy, does it makes sense? Because how it is written now, the liquid needs to loose enthalpy, from metastable to equilibrium, in order to evaporate


