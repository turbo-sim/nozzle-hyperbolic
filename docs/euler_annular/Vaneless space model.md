
# Vaneless Channel Model

## 1. General Model

The complete model for a vaneless channel of arbitrary shape is given by the conservation equations for mass, meridional momentum, tangential momentum, and energy along the meridional coordinate $m$:

$$ \begin{gather} \rho \frac{dv_m}{dm} + v_m \frac{d\rho}{dm} = -\frac{\rho v_m}{A} \frac{dA}{dm} \\ \rho v_m \frac{dv_m}{dm} + \frac{dp}{dm} = \frac{\rho v_\theta^2}{r}\sin(\phi) - \frac{2\tau_m}{b} \\ \rho v_m \frac{dv_\theta}{dm} = -\frac{\rho v_\theta v_m}{r}\sin(\phi) - \frac{2\tau_\theta}{b} \\\frac{dh}{dm} +  v_m \frac{dv_m}{dm} + v_\theta \frac{dv_\theta}{dm}  = \frac{2\dot{q}}{b} \end{gather} $$

where $\phi$ is the angle between the meridional direction and the axis of rotation, $b$ is the local channel width, and $A$ is the cross-sectional flow area. These equations are derived in:

>R. Agromayor, B. Müller, and L. O. Nord, “One-dimensional annular diffuser model for preliminary turbomachinery design,” _International Journal of Turbomachinery, Propulsion and Power_, vol. 4, no. 3, p. 31, 2019, doi: [10.3390/ijtpp4030031](https://doi.org/10.3390/ijtpp4030031).

## 2. Purely Radial Interspace

For a purely radial channel, $\phi = \pi/2$, so $\sin(\phi) = 1$. The system reduces to:

$$ \begin{gather} \rho \frac{dv_m}{dm} + v_m \frac{d\rho}{dm} = -\frac{\rho v_m}{A} \frac{dA}{dm} \\ \rho v_m \frac{dv_m}{dm} + \frac{dp}{dm} = \frac{\rho v_\theta^2}{r} - \frac{2\tau_m}{b} \\ \rho v_m \frac{dv_\theta}{dm} = -\frac{\rho v_\theta v_m}{r} - \frac{2\tau_\theta}{b} \\ v_m \frac{dv_m}{dm} + v_\theta \frac{dv_\theta}{dm} + \frac{dh}{dm} = 0 \end{gather} $$

## 3. Loss Coefficient and Skin Friction

The total pressure loss can be expressed in terms of the loss coefficient $Y$ and the Fanning skin friction factor $c_f$:
$$ Y = -\frac{\Delta p_0}{p_0 - p} \quad\Rightarrow\quad dY = -\frac{dp_0}{p_0 - p} $$
$$ c_f = \frac{\tau}{\tfrac{1}{2}\rho v^2} \equiv \frac{\tau}{p_0 - p} $$
Substituting the total pressure gradient relation into these definitions gives
$$ \frac{dY}{ds} = -\frac{2}{b}\,c_f, $$

or, integrated along the flow path,

$$ Y = 2\,c_f\,\frac{L}{b}. $$

This relates the nondimensional total pressure loss directly to the wall friction coefficient and the passage geometry, where $L$ is the total streamline length and $b$ is the local channel width. In other words, the longer and narrower the flow passage is, the higher the total pressure drop due to friction.

For a flow with a **constant flow angle** $\alpha$ (measured from the radial direction), the streamline is a logarithmic spiral. Its arclength between $r_{in}$ and $r_{out}$ is
$$ L = \frac{r_{out} - r_{in}}{\cos(\alpha)}. $$
Substituting into the loss coefficient expression yields
$$ \boxed{Y = \frac{2 c_f (r_{out} - r_{in})}{b_{mean}\cos(\alpha_{mean})},}. $$
## 5. Derivation: Arclength of a Logarithmic Spiral

The result $L = \Delta r / \cos(\alpha)$ deserves a derivation, since it follows directly from integrating the streamline equation in polar coordinates under the constant-flow-angle assumption.

### 5.1 Streamline equation in polar coordinates
In polar coordinates $(r, \theta)$, an infinitesimal arc length element is

$$ ds^2 = dr^2 + r^2  d\theta^2. $$
The flow angle $\alpha$ is defined as the angle between the velocity vector and the radial (meridional) direction:
$$ \tan(\alpha) = \frac{v_\theta}{v_r} = \frac{rd\theta}{dr}. $$
For a **constant** flow angle $\alpha$, this gives a separable ODE:

$$ \frac{d\theta}{dr} = \frac{\tan(\alpha)}{r}. $$
### 5.2 Streamline shape — the logarithmic spiral

Integrating from $(r_{in}, \theta_{in})$ to $(r, \theta)$:

$$ \theta - \theta_{in} = \tan(\alpha)\ln\left(\frac{r}{r_{in}}\right), $$

or equivalently,

$$ r(\theta) = r_{in} e^{\big[(\theta - \theta_{in})\cot(\alpha)\big]} $$

This is the equation of a **logarithmic (equiangular) spiral**, characterized by the property that the curve crosses every radial line at the same angle $\alpha$.

### 5.3 Arclength integration

Using $ds^2 = dr^2 + r^2d\theta^2$ and substituting $d\theta = \tan(\alpha)dr/r$:

$$ ds^2 = dr^2 + r^2\left(\tan(\alpha)\frac{dr}{r}\right)^2 = dr^2\left(1 + \tan^2(\alpha)\right) = \frac{dr^2}{\cos^2(\alpha)}. $$
Taking the positive root,
$$ ds = \frac{dr}{\cos(\alpha)}. $$
Since $\alpha$ is constant, integrating from $r_{in}$ to $r_{out}$ gives

$$ L = \int_{r_{in}}^{r_{out}} \frac{dr}{\cos(\alpha)} = \frac{r_{out} - r_{in}}{\cos(\alpha)}. $$

### 5.4 Geometric interpretation
The result has a clean geometric meaning: along a logarithmic spiral with constant flow angle $\alpha$, the radial coordinate plays the role of a "projected" coordinate, and the actual path length is stretched by a factor $1/\cos(\alpha)$ relative to the radial span. As $\alpha \to 0$ the streamline becomes purely radial and $L \to \Delta r$; as $\alpha \to \pi/2$ the streamline approaches a circular path and $L \to \infty$, consistent with the diverging $1/\cos(\alpha)$ factor.


## 6. Change of Angular Momentum Due to Friction

### 6.1 Tangential momentum vs. angular momentum

Before continuing, it is worth being precise about two distinct quantities that are easily confused:

- **Tangential velocity** $v_\theta$ — the swirl component of the velocity. It is _not_ a conserved quantity in a frictionless radial flow; it varies with radius even when no torque acts on the fluid.
- **Angular momentum (per unit mass)** $r v_\theta$ — the moment of the tangential momentum about the axis. This is the quantity governed by the angular-momentum (torque) balance, and it is what is conserved in the absence of friction.

The tangential momentum equation for the purely radial interspace is the starting point:
$$ \rho v_m \frac{dv_\theta}{dm} = -\frac{\rho v_\theta v_m}{r} - \frac{2\tau_\theta}{b} $$
Dividing through by $\rho v_m$ gives the evolution of the tangential velocity along the streamline:
$$ \frac{dv_\theta}{dm} = -\frac{v_\theta}{r} - \frac{2\tau_\theta}{b\rho v_m} $$
The first term on the right is _not_ a loss term — it is a purely kinematic effect of moving in cylindrical coordinates. The second term is the frictional effect.

### 6.2 Frictionless limit: conservation of angular momentum

Setting $\tau_\theta = 0$ removes the torque, leaving
$$ \frac{dv_\theta}{dm} = -\frac{v_\theta}{r} $$
For a radial channel $dm = dr$, so the equation separates:
$$ \frac{dv_\theta}{v_\theta} = -\frac{dm}{r} = -\frac{dr}{r} $$
Integrating between stations 1 and 2:

$$ \begin{gather}  \ln\left(\frac{v_{\theta,2}}{v_{\theta,1}}\right) = - \ln\left(\frac{r_2}{r_1}\right) =  \ln\left(\frac{r_1}{r_2}\right) \\ \frac{v_{\theta,2}}{v_{\theta,1}} = \frac{r_1}{r_2} \\ v_{\theta,2},r_2 = v_{\theta,1},r_1 \end{gather} $$

The final line is exactly the **conservation of angular momentum**, $r v_\theta = \text{const}$. Note how this differs from the tangential velocity itself: $v_\theta$ falls off as $1/r$, but the product $r v_\theta$ stays fixed. This is the free-vortex behavior of an ideal vaneless space.

### 6.3 With friction: the modified balance
Reintroducing the wall shear, the tangential component of the wall stress is
$$ \tau_\theta = \tfrac{1}{2} c_f \rho v^2 \sin(\alpha), $$
where $\alpha$ is the flow angle (from radial) and $v$ the total velocity. Substituting into the tangential momentum equation and dividing by $v_\theta$:
$$ \frac{1}{v_\theta}\frac{dv_\theta}{dm} = -\frac{1}{r} - \frac{2\cdot\tfrac{1}{2} c_f \rho v^2 \sin(\alpha)}{b\rho v_m\, v_\theta} $$
Using $v_m = v\cos(\alpha)$ and $v_\theta = v\sin(\alpha)$, the friction term simplifies cleanly:
$$ \frac{c_f v^2 \sin(\alpha)}{b, v\cos(\alpha), v\sin(\alpha)} = \frac{c_f}{b\cos(\alpha)} \equiv K $$
so that
$$ \frac{1}{v_\theta}\frac{dv_\theta}{dm} = -\frac{1}{r} - K $$
With $dm = dr$ this becomes
$$ \frac{dv_\theta}{v_\theta} = -\frac{dr}{r} - Kdr $$

### 6.4 Integrated result with friction

Integrating between stations 1 and 2 (treating $K$, i.e. $c_f$ and $\alpha$, as constant):

$$ \begin{gather}  \ln\left(\frac{v_{\theta,2}}{v_{\theta,1}}\right) = - \ln\left(\frac{r_2}{r_1}\right) - K(r_2 - r_1) \\ \frac{v_{\theta,2}}{v_{\theta,1}} = \frac{r_1}{r_2}e^{-K(r_2 - r_1)} \\ v_{\theta,2}r_2 = v_{\theta,1}r_1 e^{-K(r_2 - r_1)} \end{gather} $$

Substituting $K = c_f/(b\cos\alpha)$ gives

$$\boxed{v_{\theta,2}\,r_2 = v_{\theta,1}\,r_1 \cdot \exp\left[-\frac{c_f(r_2 - r_1)}{b\cos(\alpha)}\right]}$$

Recalling the loss coefficient derived earlier, $Y = \dfrac{2c_f(r_{2}-r_{1})}{b \cos(\alpha)}$, the exponent is simply $Y/2$, so:
$$\boxed{v_{\theta,2}\,r_2 = v_{\theta,1}\,r_1 \cdot e^{-Y/2}}$$
This shows that friction causes the angular momentum $r v_\theta$ to **decay exponentially** with the loss coefficient, rather than remaining constant as in the ideal case. The decay rate is set by the friction factor $c_f$, the channel height $b$, and the flow angle through $\cos(\alpha)$. As $c_f \to 0$ the exponential collapses to unity and angular-momentum conservation is recovered.

## 7. Justification of the Constant Flow Angle Assumption
The integration in Section 6.4 (and the logarithmic spiral of Section 5) assumed a constant flow angle $\alpha$. This is self-consistent in an important special case. Consider the frictionless free-vortex result together with mass conservation. The continuity equation for a radial annular channel of height $b$ is
$$ \dot{m} = \rho v_m A = \rho v_m(2\pi r b), $$
so the meridional (radial) velocity scales as
$$ v_m = \frac{\dot{m}}{2\pi r b\rho} \propto \frac{1}{r b\rho}. $$
The free-vortex condition gives the tangential velocity as
$$ v_\theta \propto \frac{1}{r}. $$
The flow angle is then

$$ \tan(\alpha) = \frac{v_\theta}{v_m} = \frac{1/r}{\dot{m}/(2\pi r b,\rho)} = \frac{2\pi b\rho}{\dot{m}}. $$
For **constant channel height $b$ and constant density $\rho$**, the right-hand side is independent of $r$. Hence $\tan(\alpha)$ — and therefore $\alpha$ itself — is **constant** throughout the interspace, and the radial dependence cancels exactly.

This is the justification for treating $\alpha$ as constant: in the idealized incompressible, constant-height vaneless space the flow angle is genuinely invariant with radius, so assuming a constant $\alpha$ (and hence a logarithmic-spiral streamline) is a reasonable approximation even when mild compressibility or small height variations are present.

