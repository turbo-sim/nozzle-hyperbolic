# EulerAnnular design questionnaire

This version records the answers supplied in the pasted questionnaire and
reconciles them with the explicit requirements already given in the chat.
Resolved choices are written as decisions, not unanswered blocking questions.
The remaining transonic-benchmark choice is identified at the end. The answers
below remain the accepted design record. Following the user's authorization,
the general solver and geometry implementation have been added, with
verification reported separately.

Implementation status: NURBSpy 1.6.0 is installed and its explicit NumPy backend
preprocesses fixed geometry through the common NURBS pipeline. JAXProp 0.5.8
retains JAX 0.6.2 for the numerical solver; NURBSpy 1.6's optional JAX extra
requires JAX 0.12 and is not selected. Geometry/control-point AD is outside the
initial scope. Coordinated JAX migration is optional future work if geometry AD
is requested. This status update does not change the user answers,
boundary/EOS decisions or selected channel-width laws.

See the [implementation plan](IMPLEMENTATION_PLAN.md),
[physics and benchmarks](../REPORT.html#theory-overview),
[geometry specification](../REPORT.html#use-geometry) and
[boundary block specification](../REPORT.html#boundary-equations).

## Physical problem and verification cases

### Q1. What does `general and curvilinear` mean here? [resolved]

**Answer:** A quasi-1D solver. Describe geometry using NURBS. Particular cases,
such as a radial duct, are simple NURBS curves that are preprocessed and passed
through the same general geometry path.

**Decision:** Four axisymmetric balances with swirl on a general meridional
midline. Use NURBSpy B-spline/NURBS midline and normal-height curves; radial
and conical constructors create these definitions. They do not introduce
separate metric or FVM implementations. The latest reviewed local NURBSpy
version, 1.6.0, is installed with the fixed-geometry NumPy backend described
above. A 2D solver or resolved rotating detonation is outside this scope.

### Q2. Which channel-width law should verification use? [resolved]

**Answer:** Use constant channel width $b$ for the test/verification cases,
except for the conical channel, where $b$ varies according to the
semi-divergence and cant angles.

**Decision:** Initial radial and general curved verification cases use constant
normal width. Straight-wall conical cases derive their varying normal-width
spline from the prescribed angles. The general solver still accepts arbitrary
spline height distributions; this restriction applies to verification fixtures.

The earlier variable-height radial C-D default is superseded by this answer.
Constant-width radial inflow can accelerate subsonic flow to a sonic exit/limit,
but cannot support a smooth internal subsonic-to-supersonic transition and the
associated downstream internal-shock family. The earlier five-regime radial
C-D case is therefore not selected unchanged. A compatible additional
transonic fixture is the remaining case choice at the end of this file.

### Q3. Which `supersonic diffusion` cases should be verified? [resolved]

**Answer:** Consider both cases as distinct verifications.

**Decision:** Include smooth swirl-dominated diffusion with total Mach above
one and meridional Mach below one, and meridionally supersonic flow with a
shock followed by subsonic diffusion. Keep their states, boundary conditions
and analytical references separate. Smooth outward meridionally supersonic
expansion remains an optional physical sign check.

### Q4. What is the EOS and verification-backend policy? [resolved]

**Answer:** Use JAXProp for the numerical implementation. Perfect-gas analytical
solutions are for verification only.

**Decision:** The FVM supports general real-gas JAXProp fluid objects and uses
EOS-provided properties, sound speed and the Gruneisen parameter. No
$\gamma=c_p/c_v$, perfect-gas closure or gamma-based wave/boundary formulas
belong in numerical FVM paths. A perfect-gas JAXProp object exercises the same
numerical implementation during initial testing.

Retain `jaxprop.FluidPerfectGas(name="Air")`, $p_0=200$ kPa and $T_0=300$ K as
proposed starting fixture values, not newly supplied numerical data. Record
its actual constants for the independent analytical references; the reviewed
Air has $\gamma\approx1.40169$. Initial cases are adiabatic and frictionless.
Real-fluid application testing and empirical loss/heat closures are deferred.

### Q5. Which inlet swirl law stays fixed as backpressure changes? [resolved]

**Answer:** Prescribe a constant inlet angle $\alpha_{in}$ during backpressure
sweeps.

**Decision:** Use $(p_0,T_0,\alpha_{in})$ as subsonic inlet data for an operating
family. Do not simultaneously prescribe inlet angular momentum, mass flow or
Mach as another independent condition.

The solver transports $\ell=r v_\theta$. It is conserved along each inviscid
solution, but its inlet value can change between backpressure operating points:
$\ell_{in}=r_{in}v_{m,in}\tan\alpha_{in}$. Fixed-$\ell_{in}$ sweeps are an
optional alternative, not the selected default. Distinguish total and meridional
Mach in all case data.

## Numerical and API decisions

### Q6. Angular-momentum storage and nonlinear variables? [resolved]

**Answer:** Agreed with the recommendation.

**Decision:** Conserve $(\rho,\rho v_m,\rho\ell,\rho E)$ with
$\ell=r v_\theta$, and use $W=(p,v_m,\ell,h)$ as nonlinear/reconstruction
variables. Recover $v_\theta=\ell/r$ and include both velocity components in
total energy. Use `v_m` and `v_theta` as velocity names in code, interfaces and
outputs, and $v_m,v_\theta$ in equations.

### Q7. First numerical flux and reconstruction? [resolved]

**Answer:** Agreed with the recommendation.

**Decision:** Generalize the existing regularized characteristic/Stadtke flux
to four variables, sharing its dimension-independent regularization and
contraction kernels. Derive projectors from general-EOS sound speed and
Gruneisen data. Implement constant reconstruction first, then shared MUSCL
Van Leer; every completed verification comparison includes both.

HLL/Rusanov remains an optional diagnostic if needed, not the selected
baseline. Add low-Mach corrections only when measured errors justify them.

### Q8. Shared-code extraction and backward compatibility? [resolved]

**Answer:** Agreed with the recommendation.

**Decision:** Share dimension-parameterized coloring, banded linear solves,
implicit stepping and integrated FVM assembly. Preserve the EulerNozzle API,
defaults and numerical behavior during a separately verified extraction.
Do not migrate TwoFluid or integrate NozzleX in this task. Reuse numerical
infrastructure rather than copying complete solver factories.

### Q9. Boundary formulation and operating-map direction? [resolved]

**Pasted answer:** `Agreed with your recommendation.`

**Reconciled decision:** Use independent boundary states and algebraic BC
equations as blocks in the overall residual, solved together with the physical
cells by one Newton process. Boundary rows have zero storage. No nested
nonlinear BC solve, finite-count boundary inversion or eliminated annular
default is permitted.

The pasted agreement was attached to an earlier recommendation that mentioned
eliminated boundaries. This merged version retains the separately specified
coupled-block requirement and applies the operating-map recommendation:
start with descending backpressure from an admissible subsonic cold state.
Keep cold starts and continuation distinct, with independent BC diagnostics.
Ascending-pressure external shock entry/unstart and reverse-flow boundary
physics are deferred unless separately requested.

### Q10. Time integration and mesh scope? [resolved]

**Answer:** Focus on steady cases for now, using implicit time integration to
reach steady state. The solver should also support transient cases.

**Decision:** Use the shared backward-Euler/Newton/colored-banded infrastructure
for steady pseudo-time marching. Retain physical storage and global-time
integration for transients, with stopping rules separate from steady convergence.
A full transient application campaign is not the initial focus.

Use the same general NURBSpy geometry path on uniform meridional meshes first,
then verify nonuniform meshes. Standard geometry input remains a midline curve
plus an independent normal-height spline.

## Verification and reporting

### Q11. Cp normalization and acceptance targets? [resolved]

**Answer:** Agreed with the recommendation.

**Decision:** Use $C_p=(p-p_{in})/(p_{0,in}-p_{in})$ as the primary coefficient.
Label dynamic-pressure normalization separately. Use exact compressible
perfect-gas references at finite Mach and the incompressible formula as a
limiting check.

Retain the initial targets: fixed-scale spatial RMS, BC and conservation defects
below $10^{-8}$; finest-grid smooth Cp error below $10^{-3}$; demonstrated mesh
refinement; and shock position within two finest-grid cells. These are
acceptance targets, not achieved results. Report failures, branch-selection
errors and sonic-limit discrepancies.

### Q12. Which cases, comparisons and outputs take priority? [resolved with case qualification]

**Answer:** Agreed with the recommendation. No comparison with NozzleX is
needed yet; compare against well-known analytical solutions.

**Decision:** Prioritize radial low-Mach pressure recovery, constant-width
radial-inflow acceleration/choking, and the two distinct outward diffusion
cases in Q3. Preserve the intent to verify an adapted/internal-shock/subsonic
operating family, but qualify its fixture according to Q2: the earlier
variable-height radial C-D proposal is not approved by this generic agreement.

Use 50/100/200/400 physical-cell studies, JSON/CSV data, a Markdown report and
PNG/PDF figures. Every primary flow-field comparison shows, together and in
this order:

1. `Analytical reference`.
2. `FVM (constant reconstruction)`.
3. `FVM (MUSCL reconstruction)`.

Hold other settings fixed across the two reconstruction runs, audit each
separately and retain failures. Timing follows correctness. NozzleX code/notes
remain review provenance only, not a required comparison or validation target.

## Recorded decisions and remaining case choice

The selected solver is quasi-1D, general-EOS Euler with swirl, NURBSpy geometry,
angular-momentum storage, primitive $W=(p,v_m,\ell,h)$ unknowns, the regularized
characteristic flux, shared implicit infrastructure and coupled algebraic BC
blocks. Initial verification uses a perfect-gas backend and independent
analytical solutions. Inlet angle stays fixed across pressure sweeps.
Verification width is constant except for the straight-wall conical law.

**Remaining transonic fixture choice:** a constant-width general NURBS channel
with an internal minimum radius could retain the five-regime analytical
verification without varying $b$. It would be a separate curvilinear channel,
not purely radial inflow. Alternatively, a variable-height radial C-D fixture
would require an explicit exception to Q2. Neither alternative is recorded as
selected by the pasted answers. Until this is resolved, selected purely radial
inflow verification is subsonic acceleration and the sonic-exit limit. This case
choice does not block implementation of the agreed general solver and feasible
initial benchmarks.

Answer for this remaining case choice:

Exact dimensional geometry, conical angles and individual backpressure values
remain case-configuration details. Preserve proposed starting values where
compatible with the decisions above; do not restore a superseded width law or
inlet swirl law implicitly.
