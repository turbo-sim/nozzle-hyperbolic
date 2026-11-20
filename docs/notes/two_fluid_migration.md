# Alberto two-fluid model: current implementation and review guide

The supported two-fluid implementation is `TwoFluidModel`, using the shared
finite-volume engine. The deprecated standalone solver and its duplicate state,
matrix, flux, source and closure implementations have been removed. There is one
maintained implementation of the two-fluid physics in `src/`.

This guide describes the current layout and the retained physical and numerical
choices. It replaces the earlier migration diary. The documentation now lives in
`docs/notes/`; the previous `docs/two_fluid_migration.md` location has moved here.

## Source reference and what this cleanup changes

The physical/discrete port is based on Alberto's immutable source revision
`faafff704e54fd866abff95fcb88fc0a5217405c`. This identifies the source used for
the port; it is not a claim about the current head of any remote branch.
`TwoFluidModel.reference_revision` retains that identity, and the model name
remains `two_fluid_alberto` for existing restart/provenance records.

The cleanup promotes the active `_alberto_matrices.py`, `_alberto_flux.py` and
`_alberto_sources.py` kernels to the familiar `matrices.py`, `flux.py` and
`sources.py` names. It groups the active `_alberto_closures.py` laws by mechanism
and extracts complete model policy methods into small model-owned modules.
Function names, state ordering, arithmetic, defaults and JAX branching remain
unchanged. The historical spelling `D_VE_algebric` is deliberately retained.

Alberto's matrix identities, sparse-entry explanations, projector construction,
source pressure-work derivation and closure-regime comments remain beside the
calculations. Stale comments about unused functions, heat/drag signs, historical
experiments and safe differentiation through eagerly evaluated singular branches
have been corrected or removed. Old commented-out alternatives are not runtime
options; Git history retains the previous implementation.

## Where to review the code

All paths below are relative to
[`src/nozzle_hyperbolic/models/two_fluid/`](../../src/nozzle_hyperbolic/models/two_fluid).

| Current module | Responsibility and familiar entry points |
|---|---|
| [`model.py`](../../src/nozzle_hyperbolic/models/two_fluid/model.py) | `TwoFluidModel`: initialization/scales and the shared-engine adapter; face flux, integrated NC action and integrated source assembly |
| [`configuration.py`](../../src/nozzle_hyperbolic/models/two_fluid/configuration.py) | Validated `BoundaryConfig`, `ClosureConfig`, `CompatibilityConfig`, `StabilizationConfig`, `PhasePinConfig`; docstrings describe units and selectors |
| [`constrained_state.py`](../../src/nozzle_hyperbolic/models/two_fluid/constrained_state.py) | `conserved`, `primitives`, `from_primitives`, `eos_constraints`; explicit maps between Alberto's eight unknowns and six physical balances |
| [`matrices.py`](../../src/nozzle_hyperbolic/models/two_fluid/matrices.py) | `phase_props`, `common_quantities`, `G_h_matrix`, `Phi_inv_matrix`, `J_matrix`, `K_matrix`, `X_matrix`, `flux_vector`, `eigenstructure` |
| [`flux.py`](../../src/nozzle_hyperbolic/models/two_fluid/flux.py) | `fvs_interface_flux`, `compatible_flux_solve`, `nonconservative_correction`; Alberto's compatibility discretization |
| [`sources.py`](../../src/nozzle_hyperbolic/models/two_fluid/sources.py) | `TwoFluidClosure`, `evaporation_weight`, `two_fluid_closure`, active `D_VE_algebric` / `D_VE_algebric_two_component`, separate `D_VE_area`, independent verification transformations |
| [`boundaries.py`](../../src/nozzle_hyperbolic/models/two_fluid/boundaries.py) | Physical inlet/outlet trace equations, extracted from the model; numerical boundary treatment remains generic |
| [`thermodynamics.py`](../../src/nozzle_hyperbolic/models/two_fluid/thermodynamics.py) | One-time `table_stencil_mask` audit and JAX-compatible `cell_admissible` checks |
| [`stabilization.py`](../../src/nozzle_hyperbolic/models/two_fluid/stabilization.py) | Optional conservative filters, equation-changing phase pins and explicit local rate bound |
| [`closures/`](../../src/nozzle_hyperbolic/models/two_fluid/closures) | `regimes.py`, `mass_transfer.py`, `drag.py`, `heat_transfer.py`, `interfacial_area.py`; package initializer exports the active laws |

The original solver's Newton loop, normal equations, timestep controller and
case-specific ramping are not retained as a second numerical engine. Their
supported responsibilities belong to `solvers/fvm/`, `solvers/time_marching/`
and `numerics/`. Configuration dataclasses describe physics and selected model
policies; `Scheme` describes the numerical solution.

## Variables, storage and thermodynamics

Alberto's iteration and enthalpy-primitive orderings are retained:

$$
W=(p,\alpha_g,m_g,u_g,m_l,u_l,m_ge_g,m_le_l),\qquad
U_h=(p,u_g,u_l,\alpha_g,h_g,h_l).
$$

Here $m_i=\alpha_i\rho_i$, $\alpha_l=1-\alpha_g$ and
$e_i=h_i-p/\rho_i$. The six transported contents are

$$
q(W)=\left(m_g,m_l,m_gu_g,m_lu_l,
m_ge_g+\tfrac12m_gu_g^2,m_le_l+\tfrac12m_lu_l^2\right).
$$

The algebraic maps $W\mapsto q$ and $W\mapsto U_h$ need no EOS solve:
$h_g=(m_ge_g+p\alpha_g)/m_g$, with the analogous liquid expression.
Two additional equations compare the intrinsic densities $m_g/\alpha_g$ and
$m_l/\alpha_l$ with their configured phase EOS densities. Their fixed reference
scales do not vanish with local phase fraction. Pure-phase states are outside
this eight-variable model; admissibility rejects them without clipping.

`phase_props` retains Alberto's sound-speed/Gruneisen derivative policy:

$$
\left(\frac{\partial\rho}{\partial p}\right)_h
=\frac{1+\Gamma}{a^2},\qquad
\left(\frac{\partial\rho}{\partial h}\right)_p
=-\frac{\rho\Gamma}{a^2}.
$$

Independently interpolated bicubic properties need not have those exact AD
derivatives. This policy has not been replaced during the cleanup. Accepted
states must have positive physical properties, lie in the phase PH domains and
use populated finite table stencils. Transport and saturation checks apply when
the selected closures require them. The host stencil audit rejects copied
metastable rows; it does not repair tables or substitute properties.

## Discretization and JAX choices

Select `Scheme(flux="model_compatibility", ...)`. Generic `stadtke`, HLL and HLLC
flux selections are not supported by `TwoFluidModel`. The compatibility flux is
Alberto's flux-space FVS, followed by the cell action $XK^{-1}\Delta F$.
The generic residual has the sign convention

$$
R_i=A_{i+1/2}\widehat F_{i+1/2}-A_{i-1/2}\widehat F_{i-1/2}
+N_i-S_i.
$$

`CompatibilityConfig.nc_form` preserves the three explicit choices:

| Choice | Flux difference supplied to the NC action |
|---|---|
| `cell` (default) | $A_i(\widehat F_R-\widehat F_L)$ |
| `staedtke` | $A_R\widehat F_R-A_L\widehat F_L$ |
| `wb` | $A_R\widehat F_R-A_L\widehat F_L-(A_R-A_L)F(U_{h,i})$ |

At equal slip, the two void eigenvectors coalesce. The shared-velocity branch
uses the combined projector $P_{12}=I-P_3-P_4-P_5-P_6$. Scalar `lax.cond`
inside `lax.map` avoids evaluating the singular inactive full decomposition
during AD. Keep this structure when reviewing vectorization changes: `vmap`
over a conditional can evaluate both branches. The low-level eager
`single_branch=True` alternative remains available but is not used by the model.

At exactly zero phase velocity, `compatible_flux_solve` handles the known null
entropy columns, checks that the right-hand side is in the original matrix
range and rejects incompatible solves with NaNs. It uses row/column
equilibration, without a velocity floor or pseudoinverse. These are existing
active-port treatments, not numerical changes introduced by this cleanup.

The model explicitly passes its configuration to the kernels. In particular,
`CompatibilityConfig` defaults are `delta_factor=1e-6`, `lowmach_diss=1`,
and slip blend thresholds 5 and 20 m/s. The historical low-level function
defaults differ and remain unchanged; direct kernel callers must choose their
parameters deliberately. Sonic expansion crossings activate face-dependent
smoothing, with a small robustness band elsewhere.

`CompatibilityConfig` also provides two default-off numerical options, added
after this cleanup. With the defaults the operator above is unchanged bit for bit.

| Option | Default | Effect |
|---|---|---|
| `k_free` | `False` | K-free face flux and straight-path NC integral; `nc_form` is not used |
| `smooth` | `False` | Continuous Harten magnitude on every wave and on $u_1$, without the sonic-crossing test; applies to either flux |
| `smooth_width` | `0.02` | Harten width $\delta$ as a fraction of $a_{mix}$ |

With `k_free=True` neither kernel solves with $K$. `k_free_interface_flux`
keeps the signed weights, low-Mach factor and slip blend of the native flux, but
applies its viscosity to the primitive jump,

$$
\widehat F=\tfrac12(F_L+F_R)
-\tfrac12 J(\overline U_h)\sum_k z_k m_k r_k\,
\frac{l_k\cdot(U_{h,R}-U_{h,L})}{D_k},
$$

with $m_k=\lambda_k s_k$, or the Harten magnitude $m_k=H_\delta(\lambda_k)$ with
`smooth`. The model then selects the operator's `nonconservative_mode="path"`.
`k_free_path_nonconservative` integrates $X\,dU_h$ along straight primitive
paths with two Gauss points, and the cell assembly is

$$
N_i=\tfrac12A_{i+1/2}\mathcal P_{i+1/2}+\tfrac12A_{i-1/2}\mathcal P_{i-1/2}
+A_i\mathcal P_i ,
$$

where $\mathcal P_{i\pm1/2}$ are the face-jump paths and $\mathcal P_i$ is the
within-cell path, zero for constant reconstruction. A uniform state gives exactly
zero, so no `wb` subtraction is needed. `k_free` is refused together with
`mass_upwind_cutoff`. The four switch combinations reproduce the four arms of the
[v2 study](../../development/two_fluid_test_v2/README.md) to roundoff; the
derivation is in [nonconservative terms](nonconservative_terms.md) and the port
evidence in the [K-free port study](../../development/two-fluid/README.md).
On the NICFD nitrogen-water cases, K-free converged on every tested grid
(100-800 cells), whereas the native operator stagnated at 200 cells for the
shocked and adapted flows. The native operator also keeps a grid-independent
void-fraction bump at the $\det K=0$ point; near that point the two operators
converge to different limits. These are tested-case observations, not a
general convergence result.

Interphase source densities are multiplied by cell volume and `source_scale`.
The separate `D_VE_area` receives the exact face-area difference $A_R-A_L$ and
already returns an integrated source. It must not receive a second cell-volume
factor. Its argument name `dlnA_dx` comes from the original density-form kernel.
The wall-pressure force and opposite phase-energy pressure-work terms are kept.

## Closures, boundaries and stabilization

`closures=None` disables interphase exchange. `ClosureConfig()` enables Alberto's
one-component mass/heat/drag formulation; `mass_transfer=False` selects the active
two-component heat/drag formulation. The latter is not unused historical code.
The full selector table is in the [physical closure guide](physical_closures.md).

Important conventions retained for review:

- Positive `sigma_Mg` means evaporation; `sigma_Ml=-sigma_Mg`. The two interfacial
  heat sources and mass source receive the same binary gate. Liquid superheat
  must pass both enthalpy and temperature thresholds, or gas fraction must exceed
  0.05. The optional pressure cutoff takes precedence. `gate_dT`/`gate_dh` remain
  signature-compatible historical smooth-gate arguments and do not control this
  binary law; `dh_gate`/`dT_gate` are the low-level hard thresholds.
- The drag kernel returns resistance with the sign of $u_g-u_l$; source assembly
  negates it for force on gas. The zero-slip Stokes factor is canceled
  analytically. Ranz-Marshall retains its $Re\ge10^{-12}$ square-root guard.
- Mass transfer uses donor-phase velocity and separate saturated gas/liquid
  enthalpies. Both transfer directions use the same saturation latent heat.
  Without mass transfer, phase HTCs act in series with equal/opposite heat sources.
- Area laws retain their different calibrations. Polynomial defaults fit
  historical segment points, not the current segment defaults. The original
  number-density law scales diameter inversely with `beta_ai`, including its
  original floor values; the other area laws reapply their diameter floor.
- `D_s_primitive` and `D_VE_conservative` remain independent source-identity
  verification routes. Production uses the algebraic source functions. General
  external force/wall heat fields exist in the verification container but are
  not supported by the active algebraic source/configuration. No two-fluid
  wall-friction closure is supplied.

`BoundaryConfig` supports transmissive endpoints, prescribed/subsonic/total-state
inlets and reservoir stagnation conditions. The stagnation inlet keeps Alberto's
quality, slip, separate total enthalpies and dominant-phase entropy. Pressure
outlets can use the incoming acoustic eigenvector or the shared-velocity pressure
target. `characteristic_old` is a retained policy selector with Mach blending;
it does not refer to a retained legacy solver. Coupled/segregated treatment is a
`Scheme` choice. `solution_audit` separately checks the prescribed forward-flow
reservoir/outlet branch; signed interior velocities remain admissible.

`TwoFluidModel(..., runtime_inputs=True)` declares run-time inputs (default
False keeps the configured constants in the compiled kernels). With closures,
per-cell fields `beta_ai`, `beta_cd`, `beta_htc_g` and `beta_htc_l` of shape
(N,) multiply the corresponding `ClosureConfig` factors; with a pressure outlet
and numeric back pressure, a scalar `back_pressure` in Pa replaces the configured
value. All default to the configuration, are validated positive, and are passed
to the solve rather than to the model, so a back-pressure sweep or new fields
reuse the compiled kernels:

```python
from nozzle_hyperbolic.solvers.fvm import SteadyWorkspace

workspace = SteadyWorkspace(model, scheme, settings.residual)  # model with runtime_inputs=True
for pressure in (16e5, 15e5):
    inputs = dict(back_pressure=pressure, beta_cd=np.full(mesh.n_cells, 1.2))
    solution = hyp.solve_single_grid(model, scheme, settings, workspace=workspace, parameters=inputs)
```

With default values the results equal the configured model to roundoff. These
inputs and the closure factors reproduce the upstream run-time `p_out` and
per-cell correction columns of its `step` function (`ai_scale` is now `beta_ai`):
residuals agree to $3.5\times10^{-10}$ of the flux term with fields, and the
upstream converged solutions satisfy this model's interior equations; the
remaining solution differences come from the boundary closure (ghost cells versus
trace equations). See the [port study](../../development/two-fluid/README.md).

Filters and phase pins are independent and disabled by default. Filters add
conservative face fluxes without phase clipping. Pins change the equations and
are restricted to implicit steady solutions. The saturation-energy pin uses the
same gate as phase change; the momentum pin requires a finite relaxation time
so its steady root does not depend on pseudo-time step. Original unpinned
physical defects remain separate from convergence of the selected pinned rows.

## Using the supported API

The following small verification example is entirely constructed in Python. It
uses two perfect-gas phase fixtures to check the API and constant-state balance;
it is not a physical liquid-vapor validation case.

```python
import jax

jax.config.update("jax_enable_x64", True)
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry

mesh = ConstantAreaGeometry().build_mesh(8)
fluid = jxp.FluidPerfectGas(name="Air")
model = hyp.TwoFluidModel(
    mesh, fluid, fluid,
    initial_primitives=[2e5, 25.0, 25.0, 0.4, 3e5, 2.8e5],
    closures=None,
)
scheme = hyp.Scheme(
    flux="model_compatibility",
    reconstruction="muscl_van_albada",
    time_integrator="euler-implicit",
    boundary_treatment="coupled",
    newton_backtracking=True,
)
solution = hyp.solve_single_grid(model, scheme, hyp.SteadySettings(max_steps=10))
print(solution.success, solution.reason)
assert solution.success, solution.reason
```

For real nitrogen-water subsonic, shocked and adapted flows, use the documented
single-file [two-fluid demo](../../demos/two_fluid/README.md). It imports
configuration objects from `nozzle_hyperbolic.models.two_fluid`; those public
imports and the package-root `TwoFluidModel` export remain unchanged. Imports of
the private `_alberto_*` modules must now use the canonical modules above.
`solver.py`, `state.py` and `closures/source_two_fluid.py` have no compatibility
shims; use the model and shared drivers.

## Verification and remaining physical limits

The maintained [two-fluid tests](../../tests/models/two_fluid) cover matrix and
source identities, intrinsic-density constraints, area balance, phase-rest and
reverse-flow AD, closure selectors, table rejection, phase pins, characteristic
boundaries, Jacobian assembly and coupled/segregated implicit consistency.
`test_k_free.py` covers the default-off `k_free` and `smooth` options, and
`test_closure_corrections.py` the closure scales, correction factors and
run-time inputs.
The pinned [upstream fixture](../../tests/regression/test_upstream_compatibility.py)
and [saved-solution regression](../../tests/regression/test_saved_solutions.py)
provide independent compatibility checks. Snapshot arrays, source archives and
historical provenance paths were not regenerated for this cleanup. Coverage now
includes all active two-fluid modules; obsolete exclusions have been removed.

Cleanup checks on 3 November 2026 used the existing Poetry environment. The
same 51 focused two-fluid/upstream-compatibility tests passed before and after
relocation. An independent AST comparison verified identical executable bodies,
signatures and defaults for 82 definitions/constants, allowing only module
relocation and the extracted helpers' `self`/`model` naming. All 46 module-level
kernels have parseable NumPy docstrings covering their arguments. An isolated
Sphinx/numpydoc API build passed with warnings treated as errors, the example
above converged, and the built wheel includes the current modules without the
deleted implementations. The full repository suite passed all **746 tests**.
A final 51-test focused run measured **83.7% combined Python statement/branch
coverage** of the active two-fluid package. Python tracing coverage does not
establish that every compiled XLA branch or physical regime was exercised.
These checks establish structural compatibility; they do not substitute for
the physical validation described next.

The [NICFD reproduction](../../cases/NICFD2026/README.md) is the authority for
saved physical campaign results. Its saved three nitrogen-water cases converge
on the intended branches and meet the declared profile criteria. The three
flashing-water cases are rejected before CFD because the installed JAXProp
metastable-liquid table generator supplies copied liquid stencils over the
required hot-liquid domain. Repairing and independently validating that
dependency remains necessary. This structural cleanup does not rerun or
reinterpret that campaign. Published-profile initialization and successful grid
transfers are disclosed in its reports; those results do not establish
independent cold-start recovery or grid independence.

Historical development scripts remain under `development/alberto/` and
`cases/archive/`. Some still import the removed `nozzle_hyperbolic.fvm_solver`
namespace, including the two-fluid driver and sensitivity studies. They are
research records requiring an explicit driver migration, not supported clients
or production tests. Their status is recorded in
[Alberto's development index](../../development/alberto/README.md).

| Historical caller | Remaining dependency |
|---|---|
| `development/alberto/two_fluid/main_two_fluid.py` and its grid-sequence wrapper | Retired `fvm_solver.two_fluid` factories, state helpers and sources |
| `development/alberto/sensitivity/sensitivity_study.py`, `sensitivity_closure.py` | Retired solver API and `closures.source_two_fluid`; the active laws now live in the model-local `closures/` package |
| Two-fluid scripts under `cases/archive/` | Historical `fvm_solver` imports and old configuration/driver assumptions |

Run maintained checks from the repository root in the existing environment:

```powershell
poetry run python -B -m pytest tests/models/two_fluid tests/regression/test_upstream_compatibility.py -q
poetry run python -B -m pytest tests -q --cov=nozzle_hyperbolic
```

Matching historical results does not validate every empirical correlation.
Alberto's physical review remains useful, especially for source/energy balances,
closure calibration and the limits of phase pins.

The model equations do not conserve mixture total energy. The phase-energy NC
rows do not cancel between the phases, and the source vector carries
interface-related inviscid terms in addition to the closures. The current
explanation is that the interfacial forces are written so that they do not
increase phase entropy; the imbalance is then energy exchanged with the
interface. It belongs to the formulation, not to a discretization: the native
and K-free operators give the same budget. On the NICFD nitrogen-water cases at
400 cells the mixture total enthalpy per unit mass rises from inlet to outlet by
3.6% (subsonic), 6.8% (shocked) and 9.8% (adapted), and the outlet gas
temperature exceeds the 295.15 K inlet temperature (300 K subsonic, 304 K
shocked). Acknowledge this property when reporting results. The budget is in
the [K-free port study](../../development/two-fluid/README.md); the per-phase
entropy verification of the explanation is still open. Numerical-policy changes should
be reviewed separately from this relocation so their effects remain attributable.
