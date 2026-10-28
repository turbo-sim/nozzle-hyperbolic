# Current and pressure-wave outlet boundary conditions

The report's [Theory & equations page](#theory) presents the complete
equations as continuous typeset blocks. This note summarizes the implemented
model and controls without changing the imported source.

## Common finite-volume equations

For primitive variables $(p,u,h)$ and cell volume $V_i=A_i\Delta x$,

$$
\begin{aligned}
Q&=(\rho,\rho u,\rho E)^T,& E&=h+u^2/2-p/\rho,\\
F&=(\rho u,\rho u^2+p,\rho u(h+u^2/2))^T,\\
\mathcal R_i&=V_i(Q_i^{n+1}-Q_i^n)\\
&\quad+\Delta t\left(A_{i+1/2}\widehat F_{i+1/2}
-A_{i-1/2}\widehat F_{i-1/2}-V_iS_i\right).
\end{aligned}
$$

Both methods reuse the same inlet, conservative storage, area source, ordinary
MUSCL Van Leer reconstruction and corrected FVS interior flux formulas. New
ghost values can change near-outlet reconstructed values. Ghosts have no storage.

## Current outlet

For the last physical cell $i$, upstream neighbor $i-1$, and ghost $g$,

$$
\begin{aligned}
w&=\frac12\left[1+\tanh\left(\frac{1-|u_i|/a_i}{0.004}\right)\right],\\
p_g&=wp_b+(1-w)\operatorname{clip}(2p_i-p_{i-1},p_{\min},p_{\max}),\\
h_g&=\operatorname{clip}(2h_i-h_{i-1},h_{\min},h_{\max}),\\
\rho_e&=\operatorname{clip}\left(
\frac{2\rho_iA_i-\rho_{i-1}A_{i-1}}{A_g},10^{-3},10^4\right),\\
u_g&=\frac{2\rho_i u_iA_i-\rho_{i-1}u_{i-1}A_{i-1}}{A_g\rho_e}.
\end{aligned}
$$

Here $\rho_e$ is the auxiliary extrapolated density used for velocity; the
ghost's thermodynamic density is evaluated from $(p_g,h_g)$. The completed state
passes through the usual reconstructed FVS flux. At the
adapted exit, $M\approx2.19$ and $w=0$ in double precision. Increasing $p_b$
leaves the adapted field a low-residual supersonic root of the current equations.

## Pressure-wave boundary

Use the perfect-gas relations and define the pressure ratio:

$$
\rho_i=\frac{\gamma p_i}{(\gamma-1)h_i},\qquad
a_i=\sqrt{(\gamma-1)h_i},\qquad
q=\frac{p_b}{p_i},\qquad \beta=\frac{\gamma-1}{\gamma+1}.
$$

For compression ($q>1$), the incoming acoustic-family shock curve is

$$
\begin{aligned}
\rho_*&=\rho_i\frac{q+\beta}{\beta q+1},\\
u_*&=u_i-(p_b-p_i)
\sqrt{\frac{2}{(\gamma+1)\rho_i(p_b+\beta p_i)}},\\
h_*&=\frac{\gamma p_b}{(\gamma-1)\rho_*},\\
S&=u_i-a_i\sqrt{\frac{\gamma+1}{2\gamma}q+
\frac{\gamma-1}{2\gamma}}.
\end{aligned}
$$

Use this trace when $S<0$; otherwise retain outgoing extrapolation. Finite wave
strength determines whether downstream pressure information enters.

For expansion ($q\le1$), preserve entropy and the outgoing acoustic invariant:

$$
\begin{aligned}
a_*&=a_iq^{(\gamma-1)/(2\gamma)},&
u_*&=u_i+\frac{2(a_i-a_*)}{\gamma-1},\\
\rho_*&=\rho_iq^{1/\gamma},& h_*&=a_*^2/(\gamma-1).
\end{aligned}
$$

The fan head/tail speeds are $u_i-a_i$ and $u_*-a_*$. An outgoing fan retains
extrapolation; a fully incoming fan uses the prescribed-pressure trace. When
the fan straddles zero, use $u_s=a_s=[2a_i+(\gamma-1)u_i]/(\gamma+1)$ and the
isentropic sonic pressure. Outgoing pressure uses pure extrapolation.

If the candidate demands reverse trace velocity while $u_i\ge0$, use a
consistent reflecting-wall trace rather than clipping velocity:

$$
\begin{aligned}
K&=\frac{\gamma(\gamma+1)}2,\\
q_w&=1+\frac K2M_i^2+
\frac{M_i}{2}\sqrt{K^2M_i^2+4K(1+\beta)},\\
p_w&=p_iq_w,& \rho_w&=\rho_i\frac{q_w+\beta}{\beta q_w+1},\\
u_w&=0,& h_w&=\frac{\gamma p_w}{(\gamma-1)\rho_w},\\
F_w&=(0,p_w,0)^T.
\end{aligned}
$$

This is a one-way-outflow safeguard during steady iteration, not an exterior
reservoir backflow model. Negative interior flow remains unsupported. The final
solutions must have positive outflow, target pressure and an inactive wall.

For incoming, sonic or wall traces, apply the physical trace flux directly at
the last face. A second averaged-characteristic FVS split can discard an
incoming finite compression. The conservative correction is

$$
\begin{aligned}
\mathcal R_{\mathrm{last}}^{\mathrm{new}}
={}&\mathcal R_{\mathrm{last}}^{\mathrm{base}}\\
&+\Delta t\,A_{\mathrm{out}}
\left[F(Q_{\mathrm{trace}})-\widehat F_{\mathrm{FVS,out}}\right].
\end{aligned}
$$

The mass monitor uses the same correction. The dependency remains local, and
colored AD is checked against dense AD. Regime changes are piecewise smooth;
global differentiability through shock entry is not claimed. Incoming traces
use the last cell-center state while outgoing flux uses source reconstruction,
so their threshold limits can differ by boundary discretization error.

## Implicit iterations and CFL control

Both methods use the unchanged full-step nozzle Newton engine, colored AD and
banded solves. Following the conservative recommendation in
[the annular CFL study](../euler_annular/cfl_investigation/REPORT.md), the
study-local controller starts at CFL 0.2, multiplies accepted CFL by 1.2,
halves it after failure and caps it at $10^6$. It allows eight Newton corrections
and twelve retries. Unlike the earlier study, no SER ramp is used.

$$
\begin{aligned}
\lambda_n&=\max_{\mathrm{physical}}(|u_i^n|+a_i^n),\\
\Delta t_n&=C_n\Delta x/\lambda_n,\qquad C_0=0.2,\\
C_{\mathrm{trial}}&\leftarrow0.5 C_{\mathrm{trial}}\quad\text{after rejection},\\
C_{n+1}&=\min(1.2C_{\mathrm{accepted},n},10^6).
\end{aligned}
$$

The cap is a maximum. A relative Newton update below $10^{-10}$ and an
independently evaluated implicit residual RMS below $10^{-10}$ are both
required. Fixed conservative storage scales normalize that residual:

$$
D_i=V_i\rho_0(1,a_0,h_0)^T,\qquad
r_{\mathrm{BE}}=\sqrt{\operatorname{mean}[(R_{\mathrm{BE}}/D)^2]}.
$$

The annular inner backtracking engine is not substituted for the nozzle's
full-step Newton. The recommended CFL/growth/shrink/cap parameters are
transferred; their applicability is tested here, not assumed universal.
Domain and full-state increment checks retain bounds of 0.2 for pressure,
0.1 for enthalpy and 0.2 for velocity scaled by $|u|+a$. Failed trials retain
the preceding accepted state. CFL uses physical cells; increment checks
include ghosts. Every accepted CFL is plotted. Initial index zero has no CFL.
The source SSR is diagnostic only. The outer stopping target remains the
original fixed-scale spatial residual RMS $\le10^{-8}$.

## Initialization comparison

The first page starts every physical cell at $(p_0,0,h_0)$: uniform stagnation
at 200 kPa and 300 K. The second starts from the source implicit solver's
numerically converged adapted expansion. Both variants receive identical
interior arrays within each comparison; algebraic ghosts follow each law.
Analytical shocked profiles are references only. The adapted seed cost is
recorded separately, so these pages are not a timing comparison of end-to-end
initialization strategies.

## Verification

The common mesh has 201 points; the update cap is 5000. Acceptance requires
spatial RMS $\le10^{-8}$, normalized mean errors in pressure/density/Mach each
below 0.03, and relative mass imbalance below $10^{-4}$. Physical checks also
require a captured internal supersonic-to-subsonic transition, positive subsonic
exit, relative boundary-pressure error below $10^{-6}$ and inactive wall.

The current outlet takes zero updates from the adapted root, so its absent CFL
curve is meaningful. The updated report retains all outcomes for both initialization strategies. This is one
mesh and a perfect-gas forward-outflow prototype, not a general-EOS or true
unsteady-backflow validation. Imported source remains unchanged.

## References

- [Clawpack: Euler wave curves and Riemann solutions](https://www.clawpack.org/riemann_book/html/Euler.html).
- [Cart3D boundary-condition paper](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2018-0334.pdf#page=4).
- [Fluent: pressure-outlet behavior](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/flu_ug/flu_ug_sec_compressible.html).
