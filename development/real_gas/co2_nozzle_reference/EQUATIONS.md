# Equations defining the CO2 reference

## Model and exactness

Assume steady, inviscid, adiabatic quasi-one-dimensional flow in a stationary
nozzle. There is no wall friction, heat transfer, gravity, shaft work, slip or
surface-energy contribution. Across a shock the area is continuous and the
shock has zero thickness. The reference solves the continuous Euler equations
through numerical thermodynamic evaluations, one smooth-path ODE and scalar
bracketed roots. It is a semi-analytic numerical reference, not a second FVM.

All evaluated states come from JAXProp `FluidBicubic("CO2", backend="HEOS")`
with `metastable_phase=None`, 100 enthalpy nodes and 100 logarithmic-pressure
nodes. The HEOS source is used internally by JAXProp to generate the tables.
There are no direct HEOS/CoolProp state calls in these scripts. Reservoir
enthalpy is obtained by bracketed inversion of **bicubic** $T(h,p_0)=T_0$.

In the two-phase region the underlying homogeneous equilibrium model uses
vapor mass fraction $q$ and saturated phases at common $p,T$:

$$
\frac{1}{\rho}=(1-q)\frac{1}{\rho_l}+q\frac{1}{\rho_v},\qquad
h=(1-q)h_l+q h_v,\qquad s=(1-q)s_l+q s_v.
$$

JAXProp handles these properties inside its table generator; the scripts do
not reconstruct saturation states, mix EOS backends or impose an ideal-gas law.
The tabulated quality and phase flag are retained raw. Values near the phase
boundary can be smeared or overshoot by interpolation; phase fractions are
interpreted only in the interior of the equilibrium two-phase region.

## Why the smooth path uses density, rather than constant table entropy

For a thermodynamically consistent EOS the Gibbs identity gives

$$
T\,ds=dh-\frac{dp}{\rho},\qquad
ds=0\ \Longrightarrow\ \frac{dh}{dp}=\frac{1}{\rho}.
$$

Bicubic density, entropy and sound speed are **independently interpolated**.
Consequently $s_p+s_h/\rho$ need not vanish, where derivatives are with
respect to the interpolated fields. Solving $s_{\rm table}=\mathrm{constant}$
would then violate smooth Euler momentum by a small amount. To give an Euler
FVM a conservation-based baseline, define smooth branches by

$$
\boxed{\frac{dh}{d\ell}=\frac{p}{\rho_{\rm bicubic}(p,h)},\qquad
\ell=\log p.}
$$

This is the mechanical isentrope of the density-defined EOS. Table entropy is
a diagnostic, not a second, inconsistent constraint. For an exact equilibrium
EOS it becomes the physical isentrope. The observed entropy drift measures
thermodynamic interpolation inconsistency; it is not modeled irreversible
entropy production in the smooth nozzle.

The scripts retain two acoustic quantities:

$$
a_{\rm Euler}^2=
\left[\left.\frac{\partial\rho}{\partial p}\right|_h+
\frac{1}{\rho}\left.\frac{\partial\rho}{\partial h}\right|_p\right]^{-1},
\qquad a_{\rm table}=\text{JAXProp's interpolated speed of sound}.
$$

The first is the characteristic acoustic speed of the actual conservative
Euler equations with closure $\rho(p,h)$. Its derivatives are computed by JAX
automatic differentiation of the density interpolant, not by interpolating
JAXProp's stored derivative properties. In a consistent EOS the two speeds
agree. `M_euler` is used for branch/admissibility checks; `M_table` is also saved
and plotted. A later FVM comparison that uses independently interpolated
acoustics must account for this distinction.

## Steady conservation laws

With area $A(x)$, forward velocity $u>0$ and total enthalpy $h_0$:

$$
\frac{d(\rho u A)}{dx}=0,\qquad
\frac{d[A(\rho u^2+p)]}{dx}=p\frac{dA}{dx},\qquad
\frac{d(\dot m h_0)}{dx}=0,
$$

$$
\dot m=\rho uA,\qquad h_0=h+\frac{u^2}{2}.
$$

On a smooth branch, these reduce to

$$
dp+\rho u\,du=0,\qquad dh+u\,du=0,
\qquad dh=\frac{dp}{\rho}.
$$

Integrate the boxed ODE from the reservoir $(p_0,h_0)$ down to the lower
pressure search bound. At any pressure on this path:

$$
u(p)=\sqrt{2[h_0-h(p)]},\qquad G(p)=\rho[p,h(p)]u(p).
$$

`DOP853` dense output evaluates $h(p)$ during scalar root solves. It is not
interpolation of a sampled nozzle profile. Table domains are explicitly
checked; no out-of-bounds property extrapolation is accepted.

## Choking, branches and adapted expansion

The choked mass flux is the maximum on the reservoir path:

$$
G_*=\max_p G(p),\qquad \dot m=A_tG_*.
$$

Differentiating the mass flux along $dh/dp=1/\rho$ gives

$$
\frac{dG}{dp}=\frac{1}{u}
\left[u^2\left(\rho_p+\frac{\rho_h}{\rho}\right)-1\right].
$$

Thus a smooth interior maximum satisfies $u_*=a_{\rm Euler,*}$. The code scans
1201 pressure stations for extrema, refines the unique acoustic root with
Brent's method, checks the maximum against the search endpoints, and checks
monotonicity of both limbs. It fails on multiple extrema or unsupported
branches instead of silently selecting one. This finite scan is evidence for
these cases, not a general proof of uniqueness for arbitrary nonconvex fluids.

At each section solve

$$
G(p)=\frac{\dot m}{A(x)}.
$$

Choose the high-pressure/subsonic limb before the throat and the
low-pressure/supersonic limb in the divergent section. The throat itself uses
$p_*$. For **adapted expansion**, the required backpressure is the resulting
exit pressure:

$$
p_{b,\rm adapted}=p_e.
$$

Equilibrium acoustic speed may be discontinuous at the saturation boundary of
the untabulated physical EOS, and saturation choking can occur; see
[Hammer et al., Appendix A](https://doi.org/10.1016/j.ijmultiphaseflow.2022.104201).
Here the chosen bicubic density interpolant is smooth and each selected
choking state is checked as an interior mass-flux maximum. No claim is made
that every physical two-phase nozzle must have a smooth sonic point.

## Inner iteration: Rankine–Hugoniot conditions

For a trial shock position $x_s$, determine state 1 from the upstream
supersonic branch. At the same local area $A_s$, solve

$$
\rho_1u_1=\rho_2u_2=j=\frac{\dot m}{A_s},
$$

$$
p_1+\rho_1u_1^2=p_2+\rho_2u_2^2,\qquad
h_1+\frac{u_1^2}{2}=h_2+\frac{u_2^2}{2}=h_0.
$$

Introduce compression $r=\rho_2/\rho_1>1$. Mass, momentum and energy give

$$
u_2=\frac{u_1}{r},\quad
p_2=p_1+\rho_1u_1^2\left(1-\frac{1}{r}\right),\quad
h_2=h_0-\frac{u_1^2}{2r^2}.
$$

Only the scalar EOS closure remains:

$$
F(r)=\frac{\rho_{\rm bicubic}[p_2(r),h_2(r)]}{\rho_1}-r=0.
$$

The trivial root $r=1$ is excluded. Scan 401 logarithmically spaced values of
$r-1$ from $10^{-5}$ to 29 (so $r\le30$), bracket every sign change inside the
table domain and refine using Brent's method. Exactly one admissible root is
required, with

$$
p_2>p_1,\quad\rho_2>\rho_1,\quad
s_{{\rm table},2}>s_{{\rm table},1},\quad
M_{{\rm Euler},1}>1>M_{{\rm Euler},2}.
$$

The moderate shocks chosen here have entropy jumps larger than the measured
smooth entropy drift. The interpolated entropy criterion alone is not reliable
for arbitrarily weak shocks. The finite compression scan does not prove
uniqueness among all possible nonclassical wave patterns.

## Downstream path and outer shock-position iteration

Anchor a new smooth mechanical path at $(p_2,h_2)$ and integrate toward
increasing pressure on the subsonic limb. Its total enthalpy and mass flow
remain $h_0$ and $\dot m$. Terminate the pressure integration at stagnation
$h=h_0$, or the upper table-pressure bound if reached first; solve the exit
mass-flux equation within that interval.

For the **prescribed** backpressure, the outer scalar residual is

$$
R(x_s)=p_e(x_s)-p_b=0.
$$

Each evaluation performs a new inner RH solve and downstream path integration.
Brent's method brackets the shock between 15% and 95% of the divergent length.
The complete history of $(x_s,p_e,R)$ is saved. The final pressure comes from
the computed exit state, not an overwritten outlet value. Backpressures are
fixed in `cases.json`; no final shock location is supplied as input.

## Geometry

Let $x_t=0$, $x_i=-0.03201$ m, $x_e=0.06636$ m and
$A_t=\pi(0.0015\ \mathrm{m})^2$. The synthetic $C^1$ area law is

$$
\frac{A(x)}{A_t}=\begin{cases}
1+(17.1-1)(x/x_i)^2,&x\le0,\\
1+(R_e-1)(x/x_e)^2,&x\ge0.
\end{cases}
$$

$R_e=1.15$ for the supercritical case and $R_e=1.69$ for the condensation
case. The smaller first expansion keeps the entire sampled flow supercritical,
including the adapted exit, while retaining strong nonideality. Lengths and
the second case's area ratios are inspired by the repository's Lettieri case;
the analytic contour is intentionally self-contained and is not an experimental
geometry fit. Area is exact at the throat and shock.

## Verification and reproducibility

The driver persists all audit results before rejecting an unsuccessful case.
Required checks include:

| Check | Normalization / threshold |
|---|---|
| Mass flow | $\max|\rho uA/\dot m-1|<10^{-9}$ |
| Total enthalpy | $\max|h+u^2/2-h_0|/\max(|h_0|,1\ \mathrm{J/kg})<10^{-12}$ |
| RH jumps | Difference divided by upstream flux or total enthalpy, floor 1 in SI; each $<10^{-9}$ |
| Choking | $|M_{\rm Euler,*}-1|<10^{-8}$, correct high/low-pressure limbs |
| Backpressure | $|p_e/p_b-1|<10^{-8}$, checked on root and sampled profile |
| Independent smooth-path quadrature | Sum of absolute local defects of $\Delta h-\int p/\rho\,d\log p$, divided by $|h_0|$, $<10^{-7}$ |
| Tighter ODE repeat | Relative mass-flow/pressure changes and $|\Delta x_s|/x_e<10^{-6}$ |

Quadrature uses eight-point Gauss–Legendre rules on 256 independent logarithmic
pressure subintervals of each delivered smooth branch. The default ODE uses
`rtol=2e-11`, absolute enthalpy tolerance $2\times10^{-6}$ J/kg and maximum
$\Delta\log p=0.008$. Refinement uses `rtol=2e-12` and maximum step 0.003.
Pressure roots use absolute tolerance $10^{-5}$ Pa and relative tolerance
$2\times10^{-13}$; the position root uses absolute tolerance $2\times10^{-12}$ m.
The table size remains 100 by 100 in this refinement: it verifies numerical
integration/root sensitivity, **not convergence of the thermodynamic table**.

The report separately records raw phase indicators, entropy drift and
$\max|a_{\rm table}/a_{\rm Euler}-1|$ without silently asserting that these
interpolated properties satisfy exact identities. Table-generation success
must be 10000/10000 when building a fresh table; relevant node/coefficient
arrays must be finite. Saved table hashes and installed package versions
identify the baseline. No `src/` changes or package dependency changes are
needed; before/after source manifests verify this for each execution.

## Interpretation and sources

The reduction from conservation laws, acoustic derivative and scalar RH
compression equation above are derived here for this benchmark. The physical
equilibrium model excludes nucleation delay, metastable Wilson-line behavior,
finite-rate mass transfer, phase slip and multidimensional boundary layers.

The chosen condensation reservoir uses Case 5, Table 3 of
[Lettieri, Paxson, Spakovszky and Bryanston-Cross, *Characterization of
Nonequilibrium Condensation of Supercritical Carbon Dioxide in a de Laval
Nozzle* (2018)](https://doi.org/10.1115/1.4038082), also available through the
[MIT repository](https://dspace.mit.edu/entities/publication/530f98ea-dfaa-427a-884c-78788fe511ac).
That experiment studied nonequilibrium condensation; the present equilibrium
calculation is a verification case, not an experimental validation claim.

[Hammer et al., *Experiments and modelling of choked flow of CO2 in orifices
and nozzles* (2022)](https://doi.org/10.1016/j.ijmultiphaseflow.2022.104201)
describes HEM, delayed phase change and saturation choking. The property
implementation is [JAXProp](https://github.com/turbo-sim/jaxprop); exact installed
versions and table settings are saved with the results rather than inferred
from the current online package.
