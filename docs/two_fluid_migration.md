# Alberto two-fluid migration

This port adapts Alberto's physical model to the shared physical-model/FVM
architecture. It does not import his Newton loop, time controller, normal
equations or complete solver. Roberto's retained two-fluid implementation was
excluded as a physical and numerical reference. Alberto should perform final
physical verification; implementation checks are listed below with their limits.

## Frozen references

`git fetch origin alberto-branch andrea-branch` completed on 2 November 2026.
The fetched remote revisions are:

| Reference | Exact revision |
|---|---|
| Alberto, sole two-fluid physical/discrete reference | `faafff704e54fd866abff95fcb88fc0a5217405c` |
| Andrea, selected numerical methods and validation assets | `bcfe6ae1ec8abd71528e57b218ac8291dc9461fa` |
| Roberto HEAD at start | `5f602c782ee8cc8ffb4814ae1bc010aec94808b0` |

There are **no Alberto commits after the assessment's `faafff7` revision** in
the fetched remote. Local branches with the same names were not used. The
initial working tree already contained general-EOS outlet, local-derivative,
benchmark and documentation work; it was preserved rather than reset.

Pinned source: [Alberto physical modules](https://github.com/turbo-sim/nozzle-hyperbolic/tree/faafff704e54fd866abff95fcb88fc0a5217405c/nozzle_hyperbolic/fvm_solver/two_fluid),
[driver defaults](https://github.com/turbo-sim/nozzle-hyperbolic/blob/faafff704e54fd866abff95fcb88fc0a5217405c/dev_scripts/settings/defaults.yaml),
[closure laws](https://github.com/turbo-sim/nozzle-hyperbolic/blob/faafff704e54fd866abff95fcb88fc0a5217405c/nozzle_hyperbolic/fvm_solver/source_two_fluid.py).

## Old-to-new map

| Frozen Alberto source | New responsibility/location |
|---|---|
| `two_fluid/matrices.py` | `models/two_fluid/_alberto_matrices.py`: physical matrices, eigenstructure and EOS derivative policy |
| `two_fluid/flux.py` | `models/two_fluid/_alberto_flux.py`: named compatibility FVS and NC action, with explicit singular-state handling |
| `two_fluid/sources.py` | `models/two_fluid/_alberto_sources.py`: source transformation and corrected pressure work |
| `source_two_fluid.py` | `models/two_fluid/_alberto_closures.py`: implemented constitutive correlations |
| `solver.py` state conversion | `models/two_fluid/constrained_state.py`: eight-state storage, primitives and EOS constraints |
| `solver.py` physical boundaries, filters/pins | `models/two_fluid/model.py` and `configuration.py`: model-owned equations and policies |
| `solver.py` Newton/AD/integration | Existing `solvers/fvm/`, `solvers/time_marching/`, `numerics/`: generic assembly and solution |
| Case/grid/parameter ladder and restart plumbing | `solvers/fvm/continuation.py`, existing grid driver, explicit case clients |

The older files still present under `models/two_fluid/` are retained historical
code. The public package exports the new `TwoFluidModel`; its imports use only
the frozen-reference modules above. None of the historical solver factories is
part of the new public route.

## Variables, storage and constraints

Iteration states retain Alberto's ordering:

$$
W=(p,\alpha_g,m_g,u_g,m_l,u_l,m_ge_g,m_le_l),\qquad m_k=\alpha_k\rho_k.
$$

The transported contents are phase mass, momentum and total energy, in this
order: gas mass, liquid mass, gas momentum, liquid momentum, gas energy, liquid
energy. Six primitive variables used for constitutive evaluation and MUSCL are
$U_h=(p,u_g,u_l,\alpha_g,h_g,h_l)$. This reconstruction representation does not
reduce the eight unknowns in each cell.

Two intrinsic-density EOS equations complete the cell system:

$$
g_g=m_g/\alpha_g-\rho_g(p,h_g)=0,\qquad
g_l=m_l/(1-\alpha_g)-\rho_l(p,h_l)=0.
$$

These rows have zero storage and fixed reference-density scales. Using intrinsic
density errors prevents a small phase fraction from hiding an EOS defect. The
six transport scales likewise remain fixed, without evolving phase weights.
The engine recovers eight variables from six target contents plus these two
constraints at every explicit stage. Backward Euler and steady Newton append
the two algebraic rows directly; neither $dt$ nor cell volume multiplies them.
See [the model contract](model_interface.md) and [residual monitoring](residual_monitoring.md).

## Compatibility discretization and geometry

Select `Scheme(flux="model_compatibility")`. Other fluxes fail explicitly for
this model. The port preserves the frozen FVS decomposition, low-Mach acoustic
factor and zero-slip/full-slip blend. It does not substitute a Rusanov, Roe,
HLL or path-conservative two-fluid method.

The ordering is: reconstruct primitives, compute the unfiltered numerical flux,
evaluate Alberto's $XK^{-1}\Delta\widehat F$ correction from that flux, then add
the independently selected conservative filters. The engine applies face area
once. The model supplies integrated geometry/source terms and the NC correction.

`CompatibilityConfig.nc_form="cell"` preserves the active upstream default:
$A_iX_iK_i^{-1}(\widehat F_R-\widehat F_L)$. Upstream `wb` and `staedtke`
pairings remain explicit alternatives. In `wb`, the area-weighted flux change
subtracts $\Delta A F(U_i)$ before the NC action. The area source uses the same
face-area difference as the divergence, supporting uniform and nonuniform
grids. Exact stationary pressure/void/area balance is checked for `wb`; that
does not establish identical balance for the default `cell` scheme.

The latest source pressure-work terms are retained: gas energy receives
$-pD_\alpha$ and liquid energy $+pD_\alpha$, both for interphase sources and
the separate area source. The source test checks the full primitive-to-content
transformation, not just a cancelling sum. At finite slip, the sum of energy NC
rows is generally nonzero. Inspect actual NC work in the discrete budget; do
not label the two energy rows independently conservative.

At literal rest, $K$ is singular. The compatibility implementation uses a
documented reduced solve with its known inactive entropy columns, verifies
range consistency and rejects incompatible right-hand sides. It applies no
velocity floor or arbitrary pseudoinverse. Pure phases $\alpha=0,1$ cannot
carry eight independent variables and are rejected; open phase limits are
tested. This does not make the compatibility scheme valid for every sonic or
singular state. Such failures remain failures rather than alternate physics.
The algebraic range audit uses
$|Ky-b|\le10^{-10}\max(|K||y|+|b|,1)$ componentwise; this is a linear-solve
consistency test, not a relaxed nonlinear convergence criterion. Finite AD of
the literal-rest branch does not prove differentiability across every rank change.

## Boundaries

`BoundaryConfig` owns physical inlet/outlet equations once. Stagnation inlet
data include phase total enthalpies, dominant-phase reservoir entropy,
vapor mass-flow quality and slip. The quality/slip/EOS relation determines void
fraction; cross multiplication avoids a division by zero inlet velocity.
Total-state, prescribed-state, upstream `subsonic_inflow` and transmissive
inlet options remain explicit.
`subsonic_inflow` prescribes the two velocities, void fraction and two phase
enthalpies, while taking pressure from the adjacent interior cell.
Outlet choices include prescribed pressure and the upstream characteristic
correction, including its Mach transition and pressure-relaxation settings.
Both trace states include two EOS rows, giving eight equations per endpoint.

For implicit solves, **coupled is the recommended initial default**: it solves
all cells and trace equations together and avoids nested trace solves near a
singular or changing branch. `Scheme(boundary_treatment="segregated")` solves
the same trace equations locally and differentiates the completed root with
the existing implicit-function JVP. Explicit schemes require segregated stage
completion. Physical endpoint traces are distinct from a vanishing phase
inside a physical cell.

## Properties, closures and defaults

Supply the configured JAXProp vapor and liquid fluids, retaining the selected
bicubic PH and saturation tables. There is no EOS redesign or density-derived
thermodynamic experiment. The upstream physical matrices retain
$\rho_p=(1+\Gamma)/a^2$ and $\rho_h=-\rho\Gamma/a^2$ from the selected property
data. Their equality to AD derivatives is tested with a perfect gas; no claim
is made that independently interpolated bicubic properties satisfy that identity.
Domain, density, temperature, sound-speed and phase-admissibility checks remain
explicit. Invalid states are rejected, without property substitutions.

`ClosureConfig` wires all implemented upstream selectors:

| Physics | Selections and meaning |
|---|---|
| Interfacial area | `original`, `polynomial`, `parabola`, `segments`; correlation parameters passed through by name |
| Drag | `schiller_naumann`, `ishii_zuber`, independently for bubbly/droplet regimes |
| Liquid bubbly HTC | `ranz_marshall`, `jakob`, `max`, `liao` |
| Liquid droplet HTC | `internal`, `jakob`, `max` |
| Regime treatment | `regime_htc`, `linear`/`smoothstep` blend between `alpha_b` and `alpha_d` |
| Mass/heat exchange | One-component mass transfer, evaporation gates, optional pressure cutoff, or two-component heat/drag without mass transfer |

Number densities `Nb,Nd` are in m$^{-3}$, diameters in m, HTC in
W/(m$^2$ K), pressure cutoff `p_mt_max` in Pa and slip controls in m/s.
The polynomial/parabolic/segmented area laws do not use the number densities.
Existing approximations, regime limits and correlation bounds are retained in
the closure source docstrings. Upstream wall friction was TODO; no new
two-fluid wall-friction physics has been invented.
The [closure guide](physical_closures.md) lists each retained correlation bound.
Drag's $1/Re$ singularity is cancelled analytically before evaluation, restoring
the finite Stokes derivative at literal rest without a Reynolds or viscosity
floor. This preserves the nonzero-slip correlation and intentionally fixes the
upstream zero-slip derivative. Named area parameters also take precedence over
upstream call-site defaults that previously shadowed some overrides.

| Choice | Frozen driver | New default / intentional difference |
|---|---|---|
| Virtual mass | $k=0.5$ | Same |
| NC form | `cell` | Same |
| Acoustic factor / entropy width | `lowmach_diss=1`, `delta_factor=1e-6` | Same, in `CompatibilityConfig` |
| Slip blend | 5–20 m/s | Same |
| Reconstruction | regularized Van Albada with global normalization | Generic `Scheme` defaults to constant; selected generic limiters use local safe formulas, an intentional higher-order numerical difference |
| Void filter | `eps_void=0.05` | Off until explicitly selected |
| Phase clip | `alpha_eps=1e-6` | No clipping; optional admissibility threshold, default open interval |
| Pins | Off | Off; independently selectable policies are equation changes, not acceleration |
| Closure activation | Case-dependent | `closures=None` disables exchange explicitly; pass `ClosureConfig(...)` for physics |
| BC | Stagnation/characteristic with case inputs | Safe generic construction uses transmissive; physical cases supply their reservoir/outlet data |
| Newton/CFL/stopping | Alberto controller and SSR | Existing generic controller and fixed per-equation/constraint/boundary checks |

Example configuration (the fluids, mesh and physical inlet values come from
the chosen case):

```python
from nozzle_hyperbolic.models.two_fluid import (
    TwoFluidModel, BoundaryConfig, ClosureConfig, CompatibilityConfig,
    StabilizationConfig, PhasePinConfig,
)
from nozzle_hyperbolic.solvers.fvm import Scheme, solve_single_grid

model = TwoFluidModel(
    mesh, vapor, liquid, initial_primitives=initial_primitives,
    boundaries=BoundaryConfig(inlet="stagnation", outlet="characteristic",
        quality=quality, slip=slip, H_g0=H_g0, H_l0=H_l0,
        s0=s0, back_pressure=p_out),
    closures=ClosureConfig(Nb=1e10, Nd=1e10, area_law="polynomial",
        drag_bubbly="schiller_naumann", drag_droplet="ishii_zuber"),
    compatibility=CompatibilityConfig(nc_form="cell"),
    stabilization=StabilizationConfig(),
)
solution = solve_single_grid(model, Scheme(flux="model_compatibility"))
```

`StabilizationConfig(void_strength=0.05)` explicitly enables the upstream
background fourth-difference filter. Thin-phase filtering and the phase-mass
upwind policy have independent controls. To select the finite-time momentum
pin, pass `pins=PhasePinConfig(momentum=True, momentum_alpha=1e-4,
momentum_time=1e-6)` to the model; `momentum_time` is in seconds.
`saturation_energy=True` independently selects the binary evaporation-gated
gas-energy pin and requires valid saturation data. Pins are restricted to
implicit steady solves. The deprecated momentum blend with no finite
relaxation time is rejected because its steady root depends on pseudo-time step.
The optional mass-upwind policy retains the upstream smooth direction selector
$\tanh(\bar u/(10^{-3}\,\mathrm{m\,s^{-1}}))$ and a 0.25-decade phase-fraction
transition. These are switch widths, not floors on physical velocities. They
are inactive unless the mass-upwind cutoff is explicitly selected.

## Preparation, continuation and inspection

Prepare the selected property tables once and record their bounds, backend and
preparation cost. `solve_grid_sequence` recovers constrained contents after
mesh interpolation. `solve_parameter_sequence` accepts explicit physical
parameter mappings with one fixed numerical scheme. For each numerically
attempted stage after successful setup, it records preparation costs,
unsuccessful warm attempts and cold retries; an earlier success
never replaces the final requested stage. `save_restart`/`load_restart` store
portable arrays and metadata without pickling EOS objects. Apply
`prepare_continuation_state` before reusing the state with a new model.
See [continuation and restart](continuation_and_restart.md) for the full API.

Density interpolation is an initialization policy, not a conservative
physical-time remap; source and target inventories are recorded. Preparation
includes content recovery and trace completion, with independent audits.
Configuration, model-construction and operator-setup errors raise exceptions;
recovery and trace-completion exceptions are retained in preparation records.

Inspect `solution.success`, `reason`, all six `equation_norm` entries,
`constraint_norm`, `boundary_equation_max`, `global_defect`, and retained
accepted/rejected history. `operator.residual_contributions(state)` separates
flux divergence, stabilization, NC work, sources and equation-policy changes.
When a pin is enabled, original unmodified balance defects are also reported;
convergence of the changed equations does not prove convergence of the
unmodified physical balances.

## Deferred policies and verification

Alberto's optional adaptive local Newton scaling, Jacobian lag/reuse across
iterations and normal-equation policies are deliberately not ported. Adaptive
scaling can help disparate phase magnitudes; lagged Jacobians can reduce cost
when consecutive linearizations are close; normal equations can solve
overdetermined least-squares steps but square conditioning and alter the linear
problem. Reconsider these only after profiling and diagnosing inadequate
robustness of the existing generic square constrained solve. Existing exact
local derivatives are recomputed at each linearization; this is not a cache
across Newton steps.

The optional chunked compiled steady driver was explained and explicitly
**deferred by the user**. Current measurements do not isolate a compelling
host-loop bottleneck. HLLC, local pseudo-time stepping, global LF, Andrea's
advertised implicit update and a numerical-method sequence driver were excluded.
Classical RK4 and the selected generic limiters are separate supported options;
see [the limiter study](../development/flux_limiters/README.md) and
[pseudo-time/dual-time discussion](pseudo_time_stepping.md).

Implementation verification includes conservative regressions; generic
constrained recovery/zero-storage/AD/stencil checks; source and pressure-work
identities; rest, reversal, open phase limits and stationary varying-area void
contacts; and frozen-source flux/NC fixtures. The separate
[Lettieri HEM validation](../cases/lettieri_validation/REPORT.md) uses the shared
one-mixture Euler/HEM model, not this nonequilibrium model.

## Recorded verification results

All checks use the existing Poetry `.venv`, JAX float64 and the current source.
Installed versions: Python 3.13.2, JAX/JAXlib 0.6.2, JAXProp 0.5.8,
NumPy 2.4.6, SciPy 1.17.1 and pytest 8.4.2. Dependencies were not changed.
Timings below are execution records, not performance comparisons; independent
work ran concurrently.

| Check | Recorded outcome |
|---|---|
| Before/after generic change: `test_generic_fvm`, `test_fvm_models`, `test_boundary_treatment` | 81 passed before and 81 passed after |
| Broad regression: `pytest tests --ignore=tests/test_two_fluid_port.py` | 635 passed, 3 expected failures; one Windows temporary-directory setup error, resolved by rerunning that restart roundtrip with access to the existing temp directory (1 passed) |
| Latest constrained/mapped checks | 15 passed, including nonlinear EOS recovery, all integrators, nonuniform meshes, dense/colored and directional derivatives, mapped reconstruction and rectangular sonic-partitioned Osher |
| Additional final policy/timestep checks | 2 passed: selected equations retain unmodified/invariant defects; explicit wave/source limits use recovered algebraic state |
| Explicit unfinished restart opt-in | 1 passed; source remains unsuccessful in metadata and automatic sequences still reject it |
| Frozen-reference two-fluid physics/contract suite | 50 passed: matrix/source/pressure-work identities, frozen flux/NC fixtures, simultaneous and individual phase rest, reversal/open limits, every closure selector, parameter/domain validation, bicubic stencil masks, trace treatment agreement and nonuniform mapped MUSCL Jacobians |
| Small frozen two-fluid cases | Three gas solves converged, including a nonuniform 4-to-8-cell portable restart; liquid baseline and both targeted robustness diagnostics remain unfinished |
| Generic limiter study | All 8 cold shock-35 solves (24/48 cells) converged; at 48 cells pressure L1/$p_0$ was $6.03\times10^{-4}$ for Van Albada versus $7.25\times10^{-4}$ Van Leer, $8.05\times10^{-4}$ Minmod and $1.48\times10^{-3}$ Superbee |
| Lettieri equilibrium HEM | All 5 cases at 48/96 cells, plus L1 24-cell/nonuniform checks, converged; all 13 saved-field audits passed. Finest pressure RMSE 3.345–4.310 bar; no grid-independence or uncertainty-weighted validation claim |
| Documentation/report checks | New Markdown links and math delimiters checked; architecture SVG/HTML regenerated and browser navigation, deep links, downloads and math rendering checked |

The former small frozen-reference case client was removed at the user's request
after the NICFD study became the maintained physical comparison. Its recorded
gas-dominant case converged with constant and nonuniform
MUSCL reconstruction. These coarse cases test solver behavior, not grid accuracy.
The liquid case stagnated. Increasing the inner Newton iteration cap from 12
to 40, with the same equations and tolerances, also stagnated (105 accepted,
52 rejected updates; maximum equation L2 $1.62748\times10^{-3}$). A separate
cold run with the source-selected filters stagnated at
$1.23259\times10^{-3}$ (137 accepted, 59 rejected). Small EOS and boundary
defects do not compensate for these unfinished transport balances. The removed
client's local run records are no longer retained. Full flashing,
shock-path/filter refinement and agreement with Alberto's archived converged
profiles remain unverified. The upstream archived restarts were not distributed
with the frozen source; cold guesses and explicit unfinished diagnostic
restarts are identified. These limitations should guide Alberto's next physical
checks rather than being hidden by changes to tolerances or properties.

The subsequent [NICFD nitrogen–water reproduction](../cases/nicfd_validation/README.md)
uses the authors' published settings and profiles at supplementary revision
`2dc04ad9d474afb727857139df89537fcadb5769`. All three backpressure regimes
converged at 498 physical cells with a Newton cap of 20 and unchanged generic
tolerances. These close fine-grid comparisons used explicitly declared reference
seeds. An independent cold-to-fine subsonic attempt converged numerically but
selected a different root with a supersonic pocket and failed the intended-regime
audit. Selecting the
historical number-density interfacial-area law is essential: the later polynomial
default is not the NICFD closure. The missing YAML selector was resolved against
September's `e185f571f8ea0512a7a16ce53a4b2458f72c33c2` source and audited.
See the [NICFD results](../cases/nicfd_validation/RESULTS.md) for errors, numerical
differences and unsuccessful coarse/default-law attempts. This adds a realistic
two-component check; it does not establish the flashing-water reproduction.

The independent NICFD comparison uses 50, 100, 200 and 400 physical cells with
the same numerical settings as the previous 498-cell comparison. All twelve
regime/mesh combinations were run independently from published-profile seeds.
Fresh residual, constraint, boundary, budget and regime checks verify eight:
all four subsonic meshes and the 100/400-cell shock and adapted meshes. The
50/200-cell shock and adapted runs remain unfinished. At 400 cells, pressure
RMSE against the published profiles is 0.0268 bar (subsonic), 0.0446 bar (shock)
and 0.0125 bar (adapted). The standalone [Plotly report](../cases/nicfd_validation/REPORT.html)
provides published/new/both visibility, mesh and regime selectors, signed
differences, convergence histories and exports. Earlier attempts remain archived;
these reference-seeded results do not establish independent cold-start recovery
or grid independence.

The active timing campaign starts over at 100 cells and sequences through 200
and 400 cells, using MUSCL Van Albada throughout. Only its first mesh uses a
published-profile seed. Separate cold/warm executions distinguish compilation
from a repeat with identical initial/controller state and reused kernels.
The former 50-cell strict sequences and explicitly authorized unfinished-coarse
diagnostic remain historical evidence, outside these new totals. The NICFD
report owns the measured times and independent final-state audits.

Lettieri L2–L4 original Excel/CFD, precise tap coordinates and uncertainty
metadata were not recovered. Published pressure tables are included with
provenance and approximate coordinate labels. Equilibrium liquid fractions are
plotted only where raw quality is in $[0,1]$; out-of-range samples are gaps,
not clipped values. Predicted condensation onset is not claimed from those
interpolated samples. All five measured onset markers remain available.
