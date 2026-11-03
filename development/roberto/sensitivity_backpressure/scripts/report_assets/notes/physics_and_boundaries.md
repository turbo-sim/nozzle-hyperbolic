# Physical regimes and the numerical outlet

## What the continuum model predicts

Let $p_b$ be the downstream ambient pressure, and $p_0,T_0$ the reservoir
stagnation state. Decreasing $p_b$ at fixed reservoir state and area ratio
produces the familiar sequence below. The
[NASA converging-diverging nozzle verification case](https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html)
uses the same inviscid verification framework, with different geometry.
The [Virginia Tech nozzle operating-regime explanation](https://www.engapplets.vt.edu/fluids/CDnozzle/cdinfo.html)
also distinguishes internal normal shocks from exterior jet adjustment.

| Backpressure range | Inside an ideal nozzle | Outside the modeled domain |
|---|---|---|
| $p_b=p_0$ | Stagnant, zero mass flow | No jet |
| First reduction, before choking | Entirely subsonic; mass flow increases | Subsonic discharge adjusts to ambient |
| First critical backpressure | Sonic throat, subsonic limiting branch downstream | Subsonic discharge |
| Between first critical and shock-at-exit backpressure | Choked throat, supersonic pocket, normal shock, subsonic exit | Downstream adjustment |
| Shock-at-exit backpressure | Supersonic expansion up to the exit, normal shock on the plane | Subsonic state immediately after the shock |
| Below shock-at-exit but above adapted pressure | Supersonic divergent section, overexpanded discharge | Compression/oblique shock structure adjusts to higher ambient pressure |
| Adapted/design pressure | Supersonic divergent section; $p_e=p_b$ | Ideal matching, no pressure-mismatch adjustment |
| Below adapted pressure | Same ideal internal supersonic solution; $p_e>p_b$ | Expansion fans and subsequent jet adjustment |

Once the normal shock is outside the modeled nozzle, the exact internal
mass flow, exit temperature and isentropic pressure/Mach profiles cease to
depend on ambient pressure. The adapted point does not introduce another
internal flow branch. Exterior entropy production is **not** predicted by
this nozzle-only model. The ideal internal entropy generation is zero on
the shock-free branch; numerical dissipation can give a small nonzero value.

Real overexpanded jets can separate and interact with nozzle walls and
boundary layers. That requires physics absent from this inviscid quasi-1D
problem. Neither analytical agreement nor a small residual establishes
the behavior of a real separated nozzle or an exterior shock-cell pattern.

## Analytical reference and three distinct critical pressures

For a calorically perfect gas,

$$
\frac{A}{A^*}=\frac1M\left[\frac{2}{\gamma+1}
\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}},
\qquad \frac{p}{p_0}=\left(1+\frac{\gamma-1}{2}M^2\right)^{-\gamma/(\gamma-1)}.
$$

At choking, $A^*=A_t$. The **subsonic** exit root gives the first critical
backpressure; the **supersonic** exit root gives the adapted pressure.
Multiplying that supersonic exit pressure by the normal-shock pressure jump
gives the shock-at-exit backpressure:

$$
\frac{p_2}{p_1}=1+\frac{2\gamma}{\gamma+1}(M_1^2-1).
$$

The sonic **throat static pressure** is different from all three exit/
backpressure landmarks. In particular, the familiar sonic pressure ratio
near 0.528 is not the first choking backpressure of this area-ratio-two
converging-diverging nozzle.

For an unchoked case, infer exit Mach from $p_b/p_0$, then the virtual
$A^*<A_t$ and the entirely subsonic area-Mach profile. For an internal shock,
solve the normal-shock position so that the downstream subsonic branch
matches $p_b$. This scalar reference calculation never enters the numerical
initialization. Across the shock, total enthalpy and mass flow are conserved,
but stagnation pressure decreases and entropy increases:

$$
\Delta s=-R\ln\frac{p_{02}}{p_{01}},\qquad
\dot S_{\mathrm{gen}}=\dot m\,\Delta s.
$$

As the shock moves downstream it becomes stronger. Its entropy increase
grows until it leaves the nozzle. The **interior-domain** entropy production
then drops toward zero, although the exterior jet may still produce entropy.
At the exit-plane limit, specify whether the integration domain includes
the shock. The report distinguishes shock loss including the exit plane
from entropy-flux rise between interior sampling points.

## What the current source boundary conditions do

The active implementation is nested in
`make_jax_stadtke_step` (retired factory),
not solely in the standalone boundaries module. Its eliminated ghost states
are recomputed within every residual evaluation and implicit Newton update.

The inlet extrapolates velocity from the first two interior points and sets
$h_b=h_0-u_b^2/2$, subject to source bounds. It makes three Newton pressure
updates to enforce $s(h_b,p_b)=s_0$, with the implemented pressure bounds.
The inlet diagnostic records the remaining entropy error; clipped or finite
values alone are not proof of satisfying the reservoir equations.

The outlet uses the last interior Mach number:

$$
w=\frac12\left[1+\tanh\left(\frac{1-|M_{N-2}|}{0.004}\right)\right],
\qquad
p_g=w\,p_b+(1-w)(2p_{N-2}-p_{N-3}),
$$

followed by source pressure bounds. Enthalpy is linearly extrapolated and
bounded; velocity comes from area-weighted density/momentum extrapolation.
Only the **pressure** uses this Mach blend; the function's prose about
"all variables extrapolated linearly" is an abbreviated description.

When the last interior point is subsonic, $w\approx1$ and the ghost pressure
tracks the specified backpressure. When it is supersonic, $w\approx0$ and
the imposed pressure effectively disappears from the discrete equations.
This matches the characteristic argument that all waves leave a supersonic
outflow and no pressure datum should propagate upstream from ambient.
The tanh transition is numerical, not an exterior jet model.

An important limitation is **branch selection**: a supersonic-outlet steady
root can be insensitive to backpressure even when another physical startup
would put a shock inside. A local Mach switch does not by itself model shock
entry from the external domain. Descending continuation and independent
simple starts therefore need to be distinguished. Agreement along one path
does not prove uniqueness or hysteresis-free response to pressure increases.

## What the sweep can establish

Record residual convergence, physical/reference accuracy, and conservation
as separate checks. Near choking, the existing baseline already has a
finite-mesh discrepancy in throat Mach; solving the algebraic system more
accurately cannot remove that discrepancy. Near shock exit, finite shock
thickness, ghost closure and the Mach switch can shift the apparent
transition. Measure that shift instead of labeling every converged state
as the correct analytical operating point.

For underexpanded and exterior-overexpanded operation, unchanged internal
profiles are the expected result **if** the numerical exit has become
supersonic. They do not mean that the real external flow has stopped changing.
