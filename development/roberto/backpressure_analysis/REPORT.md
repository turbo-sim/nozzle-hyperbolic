# Backpressure sweep: converged roots, physical branches and outlet artifacts

The [interactive HTML edition](report_backpressure_sensitivity.html) consolidates this study with
zoomable Plotly figures, selectable operating-point profiles, data filters
and embedded downloads. It opens offline as a single file.

The expected operating regimes appear, but **a small steady residual does not
guarantee the correct response to ambient pressure**. This study exposes delayed
numerical choking, substantial shock retention at the outlet, and dependence
on the initial flow branch. Once the exit is supersonic, the internal numerical
solution is independent of further backpressure changes, as expected for the
nozzle-only model.

## Conditions and evidence

Run on 2026-10-28 using the [shared baseline](../euler_nozzle/README.md):
perfect-gas Air, $p_0=200$ kPa, $T_0=300$ K, 0.2 m symmetric nozzle,
$A_e/A_t=2$, 201 points, ordinary Van Leer MUSCL, regularized FVS, and the
unchanged source implicit Newton kernel. [README.md](README.md) gives the
complete protocol and its controller/stopping differences from the driver.

There are **87 descending-continuation points, 14 independent simple starts,
and three supersonic-restart branch probes**. All 104 final states met the
spatial residual check. The maximum independently recomputed fixed-scale RMS
was $9.78\times10^{-9}$. Twenty-one descending points required zero updates:
the initial stagnant equilibrium and points whose existing supersonic-outlet
solution already satisfied the new discrete equations. The three deliberately
supersonic restarts also required zero updates.

Source hashes were unchanged. The dynamic-pressure wrapper and the ordinary
static-pressure source factory agreed to $4.1\times10^{-17}$ in normalized
one-step state differences. The new arbitrary-backpressure analytical reference
agreed with the existing subsonic reference to about $2\times10^{-14}$ in Mach
and with the general-EOS shocked reference to $3.9\times10^{-9}$ in Mach.
These checks verify the reference extension and kernel reuse; they do not
certify the physical accuracy of every converged discrete state.

The [raw summary](outputs/baseline/summary.json),
[operating-point CSV](outputs/baseline/operating_map.csv),
[independent audit](outputs/baseline/audit.json), and
[complete point table](outputs/baseline/REPORT.md) retain the evidence. The
preliminary 68-point sweep remains in `outputs/coarse/`; it prompted refinement
around the two observed numerical transitions. No production code was edited.

## Analytical landmarks

The backend gives $\gamma=1.40169049$ and $R=287.04739$ J/(kg K).

| Landmark | $p_b/p_0$ | Backpressure [kPa] | Interpretation |
|---|---:|---:|---|
| First critical | 0.93711174 | 187.422 | Sonic throat, subsonic limiting exit |
| Normal shock at exit | 0.51314809 | 102.630 | Supersonic state just inside the exit; normal shock on its plane |
| Adapted/design | 0.09373883 | 18.748 | Supersonic exit static pressure matches ambient |

Exact choked mass flow is 0.07003015 kg/s. Below the shock-at-exit threshold,
the inviscid reference inside the nozzle is the same supersonic expansion.
External overexpansion adjustment persists until the adapted point; below
adapted pressure, external expansion is needed. Those exterior waves are
outside this domain. See [PHYSICS_AND_BOUNDARIES.md](PHYSICS_AND_BOUNDARIES.md)
for the equations and the exit-plane entropy convention.

## Mass flow, entropy and exit temperature

![Operating map](outputs/baseline/operating_map.png)

Solid lines follow the descending numerical path; dashed lines show the exact
interior references. Open circles are independent simple starts. Triangles
are the separate supersonic-restart probes, not another descending sweep.
The horizontal axis follows decreasing backpressure from left to right.

Mass flow rises from zero and reaches a numerical plateau near 0.06995144 kg/s,
about **0.1124% below** the exact choked value. This plateau alone does not
identify whether the shock/outlet state is correct: nearly the same mass flow
is retained through the problematic outlet regime.

The interior entropy-flux-rise estimate grows as the shock strengthens and
moves downstream. It reaches about 8.76 W/K before the outlet becomes strongly
affected. It then decreases along the boundary-retained shock branch and drops
near zero after the shock disappears. The numerical drop occurs much later
than the analytical interior-domain drop. The dotted analytical curve also
records the shock loss when the exit-plane shock is included; the dashed
sample-to-sample reference excludes shocks beyond the last interior sample.

After numerical shock departure, the last interior temperature is 152.944 K,
versus 153.003 K analytically at the same coordinate. It remains unchanged
through adapted and underexpanded operation. At the geometric exit $x=L$ the
analytical Mach is 2.19844; the **last interior** Mach is about 2.1871. These
are different coordinates, not two estimates at the same station.

## Near rest and choking: convergence does not remove finite-mesh error

![Transition close-ups](outputs/baseline/transition_zooms.png)

At $p_b/p_0=0.9999$, computed mass flow is 0.0013560 kg/s versus 0.0028914 kg/s
analytically: a **53.1% relative deficit**, although the absolute flow is small
and the residual is converged. At 0.97 the deficit is 3.52%. This documents
weak-driving/low-Mach accuracy loss on the chosen discretization. Its cause
has not been isolated between flux dissipation, geometry discretization and
boundary treatment; the sweep does not establish a specific repair.

At the exact first critical backpressure, numerical throat Mach is **0.8464**
instead of 1, and mass flow is 2.23% below the analytical value. This reproduces
the known sonic-limit baseline discrepancy. Along this descending path the
numerical throat reaches its choked plateau between **0.9342 and 0.9340**:
at 0.9342 its Mach is 0.9941, while at 0.9340 it is 1.0000145 with a short
supersonic region. These are sampled numerical brackets on one mesh, not
precisely located continuum bifurcations.

## Shock exit: a stationary boundary artifact persists too long

For well-contained internal shocks, the profiles broadly track the exact
solution. The situation changes as the captured shock enters the last few
interior points. At the analytical shock-at-exit ratio 0.51314809, the last
interior Mach is still 0.5421 and its pressure is $0.53305p_0$, while the ghost
pressure equals the imposed $0.51314809p_0$.

Reducing backpressure further does not immediately release the captured shock.
It remains near $x\approx0.1975$ m on this grid. At $p_b/p_0=0.45$, last-point
Mach is 0.5401 and temperature is 279.91 K, whereas the correct shock-free
interior reference has temperature 153.00 K there. At 0.425, last-point pressure
has **increased to $0.65746p_0$ despite the lower imposed pressure**.

Along descending continuation, the switch to supersonic outflow finally occurs
between **0.425 and 0.420**. This delay is much larger than the distinction
between an exit-plane shock and sampling one grid point upstream.

![Outlet close-up](outputs/baseline/outlet_closeup.png)

Squares mark algebraic ghost states. The exact curves in this figure are
supersonic in the interior. The numerical pressure overshoot and entropy rise
followed by a large **entropy decrease between the final interior points**
are not the smooth post-shock isentropic diffuser predicted by the continuum
reference. Maximum interior total-enthalpy error reaches about **4.03%** at
0.425 even though the mass flux is conserved and the spatial residual is tiny.

This is evidence of a discrete outlet/shock interaction, not evidence of an
exterior oblique shock being resolved. The experiment does not yet separate
the effects of extrapolated ghost enthalpy/velocity, MUSCL near the boundary,
and the particular FVS dissipation. No mesh, reconstruction or boundary-law
modification was made to attribute the discrepancy to one component.

## Initial conditions can select different converged branches

At the **same** $p_b/p_0=0.42$:

| Initial state | Last interior Mach | Last interior $p/p_0$ | Last interior temperature [K] | Spatial residual target |
|---|---:|---:|---:|---|
| Descending numerical continuation | 2.1871 | 0.09529 | 152.94 | Met |
| Source simple cold start | 0.5008 | 0.67656 | 271.07 | Met |

The cold-start state still retains a shock at the boundary. It is shown as a
dotted curve in the outlet close-up. Thus this is more than a minor difference
in nonlinear convergence speed: different steady fields satisfy the implemented
discrete boundary problem at this pressure.

The three additional probes start with the already-converged supersonic state
from the end of the descending sweep and change imposed backpressure to
0.48, 0.70 and 0.95. All three remain steady without an update. At 0.70 the
classical operating map requires an internal shock, and at 0.95 it requires
entirely subsonic flow. The local supersonic outlet has **dropped ambient
pressure from the equations**, so those pressure changes cannot initiate an
incoming shock from an exterior domain that is not present.

This is consistent with small-disturbance supersonic outflow causality, but
it is insufficient for selecting the classical ambient-pressure operating
branch after a large pressure increase. It does not demonstrate physical
hysteresis of a real nozzle. Coupling the same boundary equations into the
global Newton vector would retain these same steady roots; that algebraic
reorganization alone would not supply the missing exterior interaction.

![Boundary diagnostics](outputs/baseline/boundary_behavior.png)

The pressure-imposition weight remains nearly one on the retained-shock branch
because the last interior Mach is subsonic. Once it becomes supersonic, the
weight is zero to floating-point precision. This explains both the continued
pressure forcing before release and the complete insensitivity afterward.

## Adapted and below-adapted operation

After the descending path reaches the supersonic-outlet branch, all sampled
pressures from 0.415 to 0.02 leave the interior field unchanged. In particular,
there is **no extra internal transition at the adapted pressure**. Below it,
the solver extrapolates its own exit pressure instead of forcing the lower
ambient pressure onto a supersonic boundary.

This is the expected ideal internal behavior. It predicts neither the external
expansion fans below adapted pressure nor the exterior compression pattern
above it. Interior entropy-flux rise on this branch is approximately
$-0.00228$ W/K, a small numerical entropy deficit rather than physical negative
entropy production. The exact inviscid shock-free value is zero.

## Profiles, convergence and conservation

![Pressure profiles](outputs/baseline/pressure_profiles.png)

![Mach profiles](outputs/baseline/mach_profiles.png)

![Convergence histories](outputs/baseline/convergence_histories.png)

The panels group operating regimes so that the near-rest, choking and
shock-exit histories remain readable. Each curve is an actual operating point,
with plasma colors in the requested 0.2–0.8 range. The plotted RMS has fixed
reservoir scales and remains defined at rest. Native SSR and every rejected
trial are retained in the JSON. There were 724 rejected trial updates across
the 104 operating-point checks; none exhausted the final safety policy.

![Work and conservation](outputs/baseline/solver_work_and_conservation.png)

The maximum relative face-mass-flux range over non-stagnant descending points
was $3.37\times10^{-6}$. Good conservation and small residuals therefore coexist
with the outlet artifacts; conservation alone is not an accuracy check.
Iteration counts describe numerical continuation versus separate cold starts,
not a repeated timing benchmark or a physical transient duration.

## What this study establishes and what remains open

The model gives the expected choked mass-flow plateau and supersonic
ambient-pressure independence. On this mesh it does **not** reproduce choking
or shock departure accurately near their limiting pressures, and its local
outlet can admit different pressure-insensitive branches depending on startup.
Exit-only and entropy diagnostics reveal discrepancies that domain-averaged
pressure errors can largely hide.

A focused next investigation would hold the setup fixed while varying the
mesh and reconstruction around the last few cells, then compare alternative
outlet formulations against the same branch/entropy checks. That would
distinguish resolution and flux effects from the boundary map. An external
domain or suitable nonlinear boundary interaction is a separate modeling
question if pressure increases and shock entry must also be predicted. These
are follow-up proposals, not changes implemented by this study.
