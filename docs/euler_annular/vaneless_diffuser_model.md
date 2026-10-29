## Original model

The original model describes steady, one-dimensional flow along the mean streamline of an annular channel. The governing equations are:


$$
\begin{gather}
v_m \frac{d\rho}{dm} + \rho \frac{dv_m}{dm}
  = -\frac{\rho v_m}{A} \frac{dA}{dm} \\[4pt]
\rho v_m \frac{dv_m}{dm} + \frac{dp}{dm}
  = \frac{\rho v_\theta^2}{r} \sin(\phi)
     - \frac{2 \tau}{b} \cos(\alpha) \\[4pt]
\rho v_m \frac{dv_\theta}{dm}
  = -\frac{\rho v_\theta v_m}{r} \sin(\phi)
     - \frac{2 \tau}{b} \sin(\alpha) \\[4pt]
\rho v_m \frac{dp}{dm} - \rho v_m a^2 \frac{d\rho}{dm}
  = \frac{2(\tau v + \dot{q})}{b \left( \frac{\partial e}{\partial p} \right)_\rho}
\end{gather}
$$

The first equation expresses the **mass balance** in the annular passage. The second and third equations represent the **momentum balance** in the meridional and tangential directions, respectively. The last equation represents **thermal energy balance**, incorporating viscous dissipation and wall heat transfer. The viscous shear stress is modeled using a conventional friction factor approach:
$$
\tau = \frac{1}{2} c_f \rho v^2
$$
where $c_f$ is the skin friction coefficient, obtained from empirical correlations as a function of the Reynolds number. Thermodynamic properties are evaluated from the **equation of state**, using pressure–density as input pair.

> "R. Agromayor, B. Müller, and L. O. Nord, “One-dimensional annular diffuser model for preliminary turbomachinery design,” International Journal of Turbomachinery, Propulsion and Power, vol. 4, no. 3, p. 31, 2019, doi: 10.3390/ijtpp4030031.



## New model

An equivalent formulation of the model can be derived from the **total energy equation**, leading to the following system of governing equations:
$$
\begin{gather}
\rho \frac{dv_m}{dm} + v_m  \frac{d \rho}{dm}
  = -\frac{\rho v_m}{A} \frac{dA}{dm} \\
\rho v_m \frac{dv_m}{dm} + \frac{dp}{dm}
  = \frac{\rho v_\theta^2}{r} \sin(\phi)
     - \frac{2 \tau_m}{b}  \\
\rho v_m \frac{dv_\theta}{dm}
  = -\frac{\rho v_\theta v_m}{r} \sin(\phi)
     - \frac{2 \tau_\theta}{b}  \\
v_m \frac{dv_m}{dm} + v_\theta \frac{dv_\theta}{dm} + \frac{d h}{dm} 
  = \frac{2 \dot{q}}{b}
\end{gather}
$$
To close the system, the density is expressed as a function of pressure and enthalpy:
$$
 \frac{d \rho}{dm} = \left( \frac{\partial \rho}{\partial p} \right)_h  \frac{dp}{dm} + v_m \left( \frac{\partial \rho}{\partial h} \right)_p  \frac{d h}{dm}
$$
where the partial derivatives of density can be expressed in terms of the speed of sound and the Gruneisen parameter:
$$
\begin{align}
\left(\frac{\partial \rho}{\partial h}\right)_{p} = -\frac{\rho G}{c^2} \\[0.5em]
\left(\frac{\partial \rho}{\partial p}\right)_{h} = \frac{1 + G}{c^2}
\end{align}
$$
Substituting the above thermodynamic relations into the continuity equation gives the following quasi-linear system of ordinary differential equations:
$$
\begin{gather}
\frac{\rho}{v_m} \frac{dv_m}{dm} + \frac{1 + G}{c^2} \frac{dp}{dm} - \frac{\rho G}{c^2}  \frac{d h}{dm}
  = -\frac{\rho}{A} \frac{d A}{dm} \\
\rho v_m \frac{dv_m}{dm} + \frac{dp}{dm}
  = \frac{\rho v_\theta^2}{r} \sin(\phi)
     - \frac{2 \tau_m}{b}   \\
\rho v_m \frac{dv_\theta}{dm}
  = -\frac{\rho v_\theta v_m}{r} \sin(\phi)
     - \frac{2 \tau_\theta}{b}  \\
v_m \frac{dv_m}{dm} + v_\theta \frac{dv_\theta}{dm} + \frac{d h}{dm} 
  = \frac{2 \dot{q}}{b}
\end{gather}
$$

### Characteristic determinant and singularity condition
This system of equations can be expressed in compact matrix form as:
$$
A\,\frac{dU}{dm} \;=\; S
$$
$$
A =
\begin{bmatrix}
\tfrac{\rho}{v_m} & 0 & -\,\tfrac{\rho G}{c^{2}} & \tfrac{1+G}{c^{2}} \\
\rho v_m & 0 & 0 & 1 \\
0 & \rho v_m & 0 & 0 \\
v_m & v_\theta & 1 & 0
\end{bmatrix},
\qquad
U =
\begin{bmatrix}
v_m\\
v_\theta\\
h\\
p
\end{bmatrix},
\qquad
S =
\begin{bmatrix}
-\tfrac{\rho}{A}\,\tfrac{dA}{dm} \\
\tfrac{\rho v_\theta^{2}}{r}\,\sin\phi - \tfrac{2\tau_m}{b}  \\
-\tfrac{\rho v_\theta v_m}{r}\,\sin\phi - \tfrac{2\tau_\theta}{b} \\
\tfrac{2\,\dot q_w}{b}
\end{bmatrix}.
$$
The determinant of the coefficient matrix $A$ is:
$$
\det(A) = \rho^{2} \left(1 - Ma_m^{2}\right), \qquad Ma_m = \frac{v_m}{a}
$$
This result shows that the system becomes singular when the meridional Mach number equals unity. At this condition, the flow reaches a sonic state along the mean streamline, corresponding to choking at the section of minimum area.


### Total pressure and entropy generation

To gain physical insight into aerodynamic losses, it is useful to analyze the mechanical energy equation and derive the corresponding expression for the stagnation pressure variation along the mean streamline.

Starting from the momentum equations, we multiply each by its respective velocity component:
$$
\begin{gather}
  \rho v_m \frac{d}{dm}\left(\frac{v_m^2}{2}\right) + v_m\frac{dp}{dm}
  = \frac{\rho v_\theta^2 v_m}{r} \sin(\phi)
     - \frac{2}{b} \tau_m v_m   \\[4pt]
\rho v_m \frac{d}{dm} \left(\frac{v_\theta^2}{2}\right)
  = -\frac{\rho v_\theta^2 v_m}{r} \sin(\phi)
     - \frac{2}{b} \tau_\theta v_\theta 
\end{gather}
$$
Adding both equations we obtain:
$$
v_m \left( \frac{dp}{dm} + \rho \frac{d}{dm}\left(\frac{v_m^2}{2} + \frac{v_\theta^2}{2} \right) \right)
= v_m \left( \frac{dp}{dm} + \rho v \frac{dv}{dm} \right)
= v_m \frac{dp_0}{dm}
= -\left(\frac{2}{b}\right) (\tau_m v_m + \tau_\theta v_\theta)
$$
In the special case where $\tau_m = \tau \cos \alpha$ and $\tau_\theta = \tau \sin \alpha$, this expression can be simplified to:
$$
\frac{dp_0}{dm}  = -\frac{2 \tau}{b \cos \alpha}
$$

Recalling that the meridional coordinate $m$ is related to the actual streamline arclength $s$ by
$$
dm = \cos(\alpha)\, ds \to \frac{dp_0}{ds} = -\frac{2 \tau}{b}
$$
This relation shows explicitly that the loss of stagnation pressure arises from viscous shear stress $\tau$, that is, from forces that do not perform mechanical work in the energy equation. The factor $b$ is related to the hydraulic diameter of the channel, while the $\cos \alpha$ term accounts for the projection of the streamline direction onto the meridional plane. Consequently, total pressure loss scales with the actual streamline length, implying that longer flow paths or larger flow angles lead to greater viscous dissipation.

> Note! The stagnation pressure given by integrating $dp_0=dp + v dv$ does not match exactly the stagnation pressure computed as p_0 = p(h_0,\,s)

The total pressure loss can also be expressed in terms of the pressure loss coefficient $Y$ and the Fanning skin friction factor $c_f$:
$$
\begin{gather}
  Y = -\frac{\Delta p_0}{p_0 - p} \;\;\Rightarrow\;\; dY = -\frac{dp_0}{p_0 - p} \\[4pt]
  c_f = \frac{\tau}{\tfrac{1}{2}\rho v^2} \;\equiv\; \frac{\tau}{p_0 - p}
\end{gather}
$$
Substituting the total pressure gradient relation into these definitions gives
$$
\frac{dY}{ds} = -\,\frac{2}{b}\,c_f,
$$
or, upon integration along the flow path,
$$
Y = 2\,c_f\,\left(\frac{L}{b}\right).
$$
This formulation relates the nondimensional total pressure loss directly to the wall friction coefficient and the passage geometry, where $L$ is the total streamline length and $b$ the local channel width.

The total pressure drop can also be related directly to the **entropy generation** in the channel. Starting from the Gibbs relation, the definition of total pressure, and the energy equation:
$$
T\,ds = dh - \frac{dp}{\rho}, 
\qquad
dp_0 = dp + \rho v\,dv,
\qquad
dh + v\,dv = \frac{2}{b}\dot{q},
$$
combining these gives
$$
\rho T\,ds = -\,dp_0 + \frac{2}{b}\dot{q}.
$$
For adiabatic flow $(\dot{q} = 0$), this simplifies to
$$
T\,ds = -\,\frac{dp_0}{\rho},
$$
showing that the entropy generation along the flow path is directly proportional to the loss of total (stagnation) pressure. In general, any term that appears in the momentum equations but does not contribute to mechanical work in the total energy equation represents a source of irreversibility.  



## Modeling the flow losses
Following Aungier, viscous losses in vaneless passages can be represented as the combined effect of three mechanisms:
- Skin friction loss caused by wall shear stresses that dissipate mechanical energy into heat.  
- Diffusion loss associated with adverse pressure gradients during flow deceleration, leading to boundary-layer growth and increased energy dissipation.  
- Curvature loss resulting from boundary-layer distortion and secondary flow formation in curved passages such as crossover bends or return channels.

Each of these mechanisms contributes to the overall total pressure loss and, consequently, to entropy generation within the flow passage.

#### Aungier's loss distribution

In Aungier’s formulation, skin friction losses contribute to both the meridional and tangential momentum equations, while the diffusion and curvature losses act only on the meridional component. The corresponding shear stresses are expressed as  
$$
\begin{gather}
  \tau_m = \frac{1}{2} \rho v^2 \left( c_{f,W} \cos \alpha + c_{f,D} + c_{f,C} \right), \\
  \tau_\theta = \frac{1}{2} \rho v^2 c_{f,W} \sin \alpha ,
\end{gather}
$$  
where $\tau_m$ and $\tau_\theta$ denote the meridional and tangential wall shear stresses, respectively. For adiabatic flow, the corresponding rate of entropy generation can be written as  
$$
\begin{gather}
T \dot{\sigma}_g = \rho v_m T \frac{ds}{dm} = -v_m \frac{dp_0}{dm} 
= \left(\frac{2 }{b}\right) (\tau_m v_m + \tau_\theta v_\theta), \\[4pt]
T \dot{\sigma}_g = \left(\frac{2 }{b}\right) \left(\frac{1}{2} \rho v^3 \right)
( c_{f,W} + c_{f,D}\cos \alpha + c_{f,C}\cos \alpha ).
\end{gather}
$$  

**Wall loss**

The wall friction coefficient, $c_{f,W}$, accounts for viscous drag along the flow passage walls. Aungier estimates this term using a simple boundary-layer growth model based on a 1/7th power law for the velocity profile, although no further details are provided. An effective alternative is to compute $c_{f,W}$ from established correlations for fully developed internal flows, using the hydraulic diameter of the annular channel.

In the laminar regime, the Fanning friction factor is given by the analytical result:
$$
c_{f,\text{lam}} = \frac{16}{Re}.
$$  
For turbulent flow, the Haaland correlation offers a practical explicit formulation that accurately approximates the implicit Colebrook–White equation, relating the skin friction factor to the Reynolds number and relative surface roughness:  
$$
c_{f,\text{turb}} = \frac{1}{4} 
\left[-1.8 \log_{10} \left( \frac{6.9}{Re} + \left( \frac{\epsilon / D}{3.7} \right)^{1.11} \right)\right]^{-2},
$$  
where $\epsilon$ is the absolute roughness and $D$ is the hydraulic diameter. The factor of one quarter accounts for the conversion from the Darcy to the Fanning friction factor.
To ensure a smooth transition between flow regimes, a hyperbolic tangent blending function is applied:  

$$
c_{f,W} = (1 - \alpha) \, c_{f,\text{lam}} + \alpha \, c_{f,\text{turb}},
$$  

with the blending parameter $\alpha$ defined as  

$$
\alpha = \frac{1}{2} \left[1 + \tanh\left(\frac{Re - Re_{\text{tr}}}{\Delta Re}\right)\right].
$$  

Here, $Re_{\text{tr}} = 2300$ represents the nominal transition Reynolds number, while $\Delta Re = 500$ controls the smoothness of the transition.  This formulation yields a continuous and differentiable expression for $c_{f,W}$ across all Reynolds numbers, ensuring stability and robustness in numerical implementations.

**Diffusion loss**

The diffusion loss coefficient, $c_{f,D}$, represents the additional losses associated with flow deceleration in regions where the cross-sectional area increases. It is defined as  
$$
c_{f,D} = D (1 - E),
$$  
where the diffusion parameter $D$ is formulated in analogy to the divergence parameter introduced by Reneau et al. (1967) to characterize flow regimes in planar diffusers:  
$$
D = \frac{b}{A} \frac{dA}{dm}.
$$  
Diffusion losses remain low for values of $D$ smaller than the limiting value $D_m$, which depends on the inlet geometry and arclength of the flow channel [wouldnt it be more physically consistent that the cosine terms appears in the denominator of this equation?]:  
$$
D_m = 0.4 \cos \alpha_{\text{in}} \left(\frac{b_{\text{in}}}{L_m}\right)^{0.35}.
$$  
The diffusion efficiency $E$ modulates the effective loss contribution according to the local diffusion intensity:  
$$
E =
\begin{cases}
1, & D/D_m \le 0, \\[4pt]
1 - 0.2 \left(\dfrac{D}{D_m}\right)^{2}, & 0 < D/D_m < 1, \\[6pt]
0.8 \sqrt{\dfrac{D_m}{D}}, & D/D_m \ge 1.
\end{cases}
$$  
A value of $E = 1$ corresponds to ideal diffusion (no additional losses), while decreasing $E$ indicates a rise in irreversible dissipation as the local diffusion strength increases.

**Curvature loss**

Finally, the curvature loss coefficient, $c_{f,C}$, accounts for secondary flow effects induced by streamline curvature and is given by  
$$
c_{f,C} = \frac{1}{26} \, b\,\kappa \cos \alpha,
$$  

where $\kappa$ is the local curvature of the mean streamline. Together, these three components define the total viscous loss distribution along the flow path in Aungier’s formulation.

### Alternative loss distribution

As I see it, there is no strong physical justification for the asymmetry in the loases, and it likely affects the predicted flow angle distribution within the passage. I believe that a more suitable approach is to lump the three effects into a single equivalent shear stress term $\tau = \tfrac{1}{2}c_f \rho v^2$, which consistently influences both the meridional and tangential momentum equations:
 $$
 \begin{gather}
  \tau_m = \frac{1}{2} \rho v^2 \left( c_{f,W} + c_{f,D} + c_{f,C} \right) \cos \alpha  \\
  \tau_\theta = \frac{1}{2} \rho v^2 \left( c_{f,W} + c_{f,D} + c_{f,C} \right) \sin \alpha
\end{gather}
$$
In this case, the entropy generation is given by
$$
T \dot{\sigma}_g = \left(\frac{2 }{b}\right) \tau v = \left(\frac{2 }{b}\right) \left(\frac{1}{2} \rho v^3 \right)( c_{f,W} + c_{f,D}\alpha + c_{f,C} ) 
$$
which is slightly higher than the values from Angier's approach. We should investigate this approach and validate against CFD simulations in a future work. The total loss contribution can be expressed in terms of an overall skin friction coefficient:
$$
c_f = \sum_i c_{f,i} = c_{f,W} + c_{f,D} + c_{f,C}
$$

>Note: investigate via CFD simulations if we should change the formulation of the alpha term in the diffusion loss to reflect that fact that when the flow angle is higher, the flow traverses through longer streamlines and as a result experiences a weaker diffusion per unit of meridional length? Does this argument make sense?


## Expressing the loss distribution

The pressure recovery coefficient of a diffuser can be expressed as:
$$
C_p = \frac{p_2 - p_1}{p_{01}- p_1} = \frac{(p_{01}- p_1)- (p_{02}- p_2) - (p_{01}- p_{02}) }{p_{01}- p_1} = 1 - \frac{p_{02}- p_{2}}{p_{01}- p_1} - \frac{p_{01}- p_{02}}{p_{01}- p_1} 
$$
$$
C_p = 1 - \Delta C_{p,kinetic} - \Delta C_{p,loss} = 1 - Y_{kinetic} - Y_{loss} 
$$

In an analogous way, the efficiency of a diffuser can be expressed as:
$$
\eta = \frac{h_{2s} - h_1}{h_{01}- h_1} = \frac{(h_{01}- h_1) - (h_{01}- h_{2}) - (h_{2}- h_{2s})}{h_{01}- h_1} = 1 - \frac{h_{01}- h_{2}}{h_{01}- h_1} - \frac{h_{2}- h_{2s}}{h_{01}- h_1} 
$$
$$
\eta = 1 - \Delta \eta_{kinetic} - \Delta \eta_{loss}
$$

If we formulate as a differential equation to be integrated we have that

$$
d\eta = dh_s / (h_{01}- h_1) = \frac{dp}{\rho (h_{01}- h_1)}
$$
$$
d(\Delta \eta_{kinetic}) = -dh/ (h_{01}- h_1)
$$
$$
d(\Delta \eta_{loss}) = -(dh - dp/\rho) / (h_{01}- h_1) = T ds / (h_{01}- h_1)
$$


## Thoughts about the design of vaneless diffusers

Define the diffusion factor $S$ as  
$$
S = \frac{b}{A}\,\frac{dA}{dm}.
$$

For a radial vaneless diffuser with constant width, the area variation is purely radial, giving  
$$
S = \frac{b}{r}.
$$

By analogy with plane diffusers, where  
$$
S = 2\tan\theta,
$$  
and where good performance typically occurs for diffuser half-angles $\theta$ between 4° and 8°, we can estimate practical bounds for the width-to-radius ratio at the inlet of a radial diffuser:
$$
0.14 < \left(S = \frac{b}{r}\right) < 0.28.
$$
Note the the inlet of the diffuser (where the radius is the lowest) is the region where the diffusion factor has its highest value and and separation is most likely to occur.


## Thoughts about the design of vaneless diffusers

Defining the diffuson factor $S$ as:
$$
S = \frac{b}{A}\frac{dA}{dm}
$$
for the special case of a radial vaneless diffuser of constant width gives
$$
S = b/r
$$
by analogy with plane diffusers where $S=2\tan\theta$, where we know that the region iof higher performance is a when \theta is about 4 to 8 degrees, we can estimate reasonabke bounds for the width-to-radius ratio at the inlet of the diffuser the diffusion factor is the highest and the flow is more prone to separation as
$0.14 <( S = b/r ) < 0.28$



**Observations**
- These value of pressure recovery or efficiency might be higher than unity if the flow is not adiabatic.
- It is not clear to me how to formulate the stagnation pressure differential in a way that is consistent with $p_0 = p(h_0, s)$
- Computing the efficiency using the algebraic formula or integrating the differential formulas produces different results due to the reheating effect introduced by losses (similar to the difference between polytropic and isentropic efficiencies).

## Thoughts for a paper

If there is time, we can explain the model for the 90--degree bend including first the flow equations explaining that they where modified for p-h function calls (in agremeent with look-up tables)

Then explain the geometry.

Simply phi=90 and m=r for the case of radial

for the 90-degree bent we have to define the equations of the NURBS

explain the geometric construction of the shape

3-degree bs-pline for the midline defined by parameters

Also a b-spline (linear) for the thickness distribution.

Explain the equations for the construction and arclength/coordinate reparametrization in an appendix?


- The text steps of development could be to add a 1D integral-form boundary layer analysis to get a better estimation of the wall friction loss, the blockage due to boundary layers, and also the state of the boundary layer and potential occurence of separation/stall.
