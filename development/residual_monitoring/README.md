# Shared residual monitoring for nozzle and annular flow

Date: 2026-10-29. Status: shared design, with an opt-in annular implementation.

Implementation update: the generic residual reducers, accepted-sample monitor
windows, convergence table and JSON exporter now live under
`src/nozzle_hyperbolic/solvers/`. EulerAnnular connects them through
`solve_steady_monitored`; YAML verification inputs and a single-case runner are
documented in [the annular study](../roberto/euler_annular/README.md#yaml-cases-and-a-single-case-runner).
This implementation uses fixed physical scaling, volume-weighted per-equation
RMS, nine orders of initial-relative reduction for active equations, a separate
absolute safeguard, boundary/global checks and optional monitor stationarity.
Initially roundoff-level equations use an explicit absolute-only rule. The
additional initial normalization was requested after the design below was
written. EulerNozzle and the legacy annular drivers have not migrated; the
broader shared migration plan below remains a proposal.

This note records the proposed replacement for the current nozzle steady-state
residual (SSR) monitoring, and its generalization to the annular nozzle/diffuser.
It describes the inspected legacy implementation, mathematical definitions, convergence
policy, history and plotting requirements, compatibility implications, and a
phased implementation plan. Writing this note does not authorize or implement a
production numerical change.

The requested location, `development/residual_monitoring/`, holds this shared
design discussion. Any later experiments should keep implementation in
`scripts/` and generated data in `outputs/`. The implemented annular validation
remains in its existing study rather than creating a second solver here.

## 1. Recommendation and scope

Introduce shared residual monitoring with:

- The original signed spatial conservation residual as its input.
- Fixed, positive physical reference scales, independent of the evolving state,
  numerical timestep, CFL, and initial residual magnitude.
- Mean absolute (`l1`) and root-mean-square (`l2`) residuals for each equation.
- An explicit weighting convention for nonuniform meshes.
- A convergence requirement for every equation, with common or individual
  tolerances.
- Separate boundary-equation and global conservation checks.
- Histories containing all equation residuals, and both individual-equation and
  overall convergence plots.
- A versioned definition and a staged migration from legacy monitoring.

RMS is the proposed default norm. The important changes are scaling, equation-wise
monitoring, and convergence policy: the current JAX nozzle SSR already uses an
RMS, so this is not a change from a non-norm metric to a norm.

This work should initially change diagnostics and outer steady stopping only.
Changes to fluxes, reconstruction, sources, boundary laws, inner Newton equations,
linear solvers, admissibility, or trial acceptance are separate numerical changes.
Residual-based CFL controllers require explicit treatment during migration because
changing their input metric can change the entire iteration trajectory.

## 2. Current implementation and terminology

The source inspection for this note used the local `roberto-branch` checkout,
which already contained unrelated tracked modifications and untracked additions.
These observations describe the inspected working tree, not an immutable release
or a clean commit snapshot. A numerical implementation study must record its own
source fingerprints.

### 2.1 Nozzle model

The canonical implementation is
[`models/euler_nozzle/fvm.py`](../../src/nozzle_hyperbolic/models/euler_nozzle/fvm.py).
The `fvm_solver/` modules are compatibility imports, not an additional solver to
modify.

For the implicit JAX path, `make_jax_stadtke_step` computes the integrated spatial
residual after an update by evaluating its implicit residual with identical new
and old states and `dt=1`. Storage therefore cancels. The reported SSR is

$$
\mathrm{SSR}_{\mathrm{legacy}}
=\sqrt{\frac{1}{3N}
\sum_{i=1}^{N}\sum_{k=1}^{3}
\left[
\frac{t_{\mathrm{char}}r_{i,k}}
{\Omega_i|q_{i,k}|+\varepsilon}
\right]^2},
\qquad
t_{\mathrm{char}}=
\frac{\Delta x}{\max_j(|u_j|+a_j)+\varepsilon}.
$$

Here $N$ counts physical cells, $\Omega_i$ is cell volume, and
$\mathbf q=(\rho,\rho u,\rho E)$ contains conservative quantities per unit volume.
The maximum wave speed in the current implicit SSR includes the full state
array. The actual numerical timestep is not its normalization time.

The JAX explicit path uses a related RMS of the spatial right-hand side,
normalized by current local conservative quantities and a mesh crossing time.
It currently evaluates its metric at the state before the update. A future common
history must resolve that difference in evaluation timing. Also, the
Lax-Friedrichs flux accepts the numerical timestep: removing timestep factors
from monitoring does not remove timestep dependence intrinsic to that selected
spatial flux.

The older NumPy path uses relative differences between successive conservative
states, taking an RMS per equation and then a maximum. That is an update metric,
not the same spatial residual definition.

The implicit driver currently reads a scalar `parameters['residuals']`. Its
optional `convergence_history` contains `ssr` and `stopping_ssr`, rather than all
equation residuals. The driver can stop at a plateau below three times the stated
threshold after warmup. It also uses SSR in its optional SER CFL ramp. Some
docstrings still describe older return signatures or call SSR a relative change;
the executable definitions take precedence.

The inner nozzle `newton_tol` is a separate maximum scaled primitive-variable
correction test. It must not be relabeled as an outer conservation residual.

### 2.2 Annular model

The canonical implementation is
[`models/euler_annular/fvm.py`](../../src/nozzle_hyperbolic/models/euler_annular/fvm.py).
Its four conservative quantities are

$$
\mathbf q=(\rho,\rho v_m,\rho\ell,\rho E),
\qquad \ell=r v_\theta.
$$

Its current `spatial_rms` first divides each integrated cell residual by

$$
s_{i,k}=\Omega_i q_{\mathrm{ref},k}\frac{a_{\mathrm{ref}}}{L},
$$

then takes an unweighted RMS over cells and equations. The fixed scales are

$$
\mathbf q_{\mathrm{ref}}=
\left(\rho_{\mathrm{ref}},\rho_{\mathrm{ref}}a_{\mathrm{ref}},
\rho_{\mathrm{ref}}r_{\mathrm{ref}}a_{\mathrm{ref}},
\rho_{\mathrm{ref}}a_{\mathrm{ref}}^2\right).
$$

The annular solver therefore already has the proposed type of physical scaling.
It does not yet expose the proposed complete per-equation history and selectable
norm policy. Its current steady driver requires the maximum of `spatial_rms`,
`boundary_max`, and `conservation_max` to be below a single tolerance, and also
checks admissibility and the selected boundary regimes before reporting success.

The report's four conservation curves are different diagnostics. The
[`conservation recorder`](../roberto/euler_annular/scripts/conservation_history.py)
and the study's
[`compiled marcher`](../roberto/euler_annular/scripts/steady_march.py)
record signed global balances, plotted as absolute values. The
[`recommended controller`](../roberto/euler_annular/scripts/recommended_controller.py)
also independently audits final convergence. Those study consumers must be
included in a migration inventory; changing the production driver alone would
leave inconsistent stopping definitions.

### 2.3 Three distinct quantities

| Quantity | Question answered | Cancellation between cells? |
|---|---|---|
| Local spatial residual norm | Do individual cells satisfy the steady equations? | No, after absolute values or squaring |
| Global conservation defect | Do inlet/outlet fluxes and integrated sources balance over the domain? | Yes |
| Inner implicit residual or update | Has the current timestep's nonlinear system been solved? | Depends on the inner criterion; not a steady convergence certificate |

Use explicit names for all three. SSR means **steady-state residual** in the
existing code; it is not a universally standardized formula.

## 3. Common mathematical definition

### 3.1 Preserve the signed governing equations

For physical cell $i$ and equation $k$, define

$$
r_{i,k}(\mathbf q)=
\Phi_{i+1/2,k}-\Phi_{i-1/2,k}
-S^{\mathrm{integrated}}_{i,k}.
$$

$\Phi$ is the actual numerical face flux including area weighting. The source
term is integrated over the cell using the same discretization as the solver.
For a steady discrete root, $r_{i,k}=0$ for every physical row.

Use the actual reconstructed fluxes, geometry sources, wall closures, and any
enabled conservative numerical filter. Replacing them with analytical fluxes or
cell-center values such as $\rho u A$ would monitor a different equation.
For the annular implementation, geometric pressure and centrifugal sources enter
the meridional momentum balance; angular momentum and total energy currently
have no geometric sources. Future sources must be included consistently.

Only physical cells participate in these norms. Algebraic boundary/ghost rows
belong to a separate check, even if a coupled Newton vector packs them alongside
the conservation rows.

The backward-Euler system has the form

$$
\Omega_i(\mathbf q_i^{n+1}-\mathbf q_i^n)
+\Delta t\,\mathbf r_i(\mathbf q^{n+1})=0.
$$

The outer steady monitor uses $\mathbf r$, not this full timestep residual. The
signed residual used by Newton and automatic differentiation remains unchanged.
Absolute values, norms, and tolerances are applied only by the diagnostic layer.

### 3.2 Fixed physical scaling

Choose positive reference length $L_{\mathrm{ref}}$, speed $U_{\mathrm{ref}}$, and
conservative-variable scales $q_{\mathrm{ref},k}$ at setup. Define

$$
s_{i,k}=\Omega_i q_{\mathrm{ref},k}
\frac{U_{\mathrm{ref}}}{L_{\mathrm{ref}}},
\qquad
z_{i,k}=\frac{r_{i,k}}{s_{i,k}}.
$$

The raw $r_{i,k}$ has units of integrated conservative quantity per time. The
scale $s_{i,k}$ has the same units; $z_{i,k}$ is dimensionless.

Equivalently,

$$
z_{i,k}=
\frac{(L_{\mathrm{ref}}/U_{\mathrm{ref}})(r_{i,k}/\Omega_i)}
{q_{\mathrm{ref},k}}.
$$

The physical time $L_{\mathrm{ref}}/U_{\mathrm{ref}}$ converts an equation rate into
a dimensionless measure. It is not the numerical timestep and does not shrink
when CFL is reduced. It should use a fixed domain-scale length, not cell width.
This removes the explicit mesh crossing-time factor in the old nozzle SSR.

A common default is $U_{\mathrm{ref}}=a_{\mathrm{ref}}$, with:

| Equation | Nozzle scale | Annular scale |
|---|---|---|
| Mass | $\rho_{\mathrm{ref}}$ | $\rho_{\mathrm{ref}}$ |
| Axial/meridional momentum | $\rho_{\mathrm{ref}}a_{\mathrm{ref}}$ | $\rho_{\mathrm{ref}}a_{\mathrm{ref}}$ |
| Angular momentum | Not present | $\rho_{\mathrm{ref}}r_{\mathrm{ref}}a_{\mathrm{ref}}$ |
| Total energy | $\rho_{\mathrm{ref}}a_{\mathrm{ref}}^2$ | $\rho_{\mathrm{ref}}a_{\mathrm{ref}}^2$ |

For annular flow, $r_{\mathrm{ref}}$ may default to the inlet radius, matching the
current convention. Model setup selects a valid reference thermodynamic state
and a physical duct length; the generic reducer must not infer either from
array values. Positive acoustic scales remain defined at rest and at zero swirl.

Reference scales should be constant during each solve and reproducible across
comparisons. Save their resolved numerical values and their origin. Do not
recompute them from the evolving maximum velocity, local momentum, or initial
residual. Explicit overrides should accept physically justified positive values,
not silently replace invalid values with tiny denominators.

Meaningful scaling is not an error estimator. Acoustic scaling can understate
errors relative to small convective effects at very low Mach number. Energy
reference conventions and EOS conditioning affect attainable numerical floors.
Shocks and mesh changes affect the residual distribution. These are reasons to
retain physical accuracy checks and allow documented overrides, not reasons to
promise a universal tolerance for every problem.

### 3.3 Norms and mesh weighting

For positive weights $w_i$ with $\sum_iw_i=1$, calculate each equation separately:

$$
R_{1,k}=\sum_i w_i|z_{i,k}|,
\qquad
R_{2,k}=\sqrt{\sum_i w_i z_{i,k}^2},
\qquad
R_{\infty,k}=\max_i|z_{i,k}|.
$$

Configuration names `l1` and `l2` refer to these normalized measures. A 2-norm
includes the square root; an unrooted mean square has a different threshold
interpretation. Raw $\sum_i|z_i|$ and $\sqrt{\sum_i z_i^2}$ are not the recommended
defaults because their magnitudes grow with cell count for a uniform field.

Proposed default: volume weighting,

$$
w_i=\frac{\Omega_i}{\sum_j\Omega_j}.
$$

This approximates a volume average of the normalized residual density and avoids
giving a region more influence just because more cells were placed there. It
does not make results exactly mesh independent. A localized defect in a tiny
volume can still be small in an average; record $R_{\infty,k}$ and its cell
location to expose this.

Equal cell weighting, $w_i=1/N$, is a useful explicit alternative and matches the
annular model's current RMS reduction convention. It agrees with volume
weighting only for equal-volume cells. Equal axial spacing in a variable-area
nozzle does not imply equal-volume cells. Weighting is therefore a numerical
definition choice that must be settled and versioned, not a plotting preference.

For identical weights and scaling, $R_{1,k}\leq R_{2,k}\leq R_{\infty,k}$.
Consequently, the same numerical tolerance is generally more demanding with RMS
than mean absolute residuals. Changing norm changes the acceptance criterion;
there is no field-independent conversion that makes the two equivalent.

For example, with equal weights, $(1,1,1,1)$ gives $R_1=R_2=1$, whereas
$(0,0,0,4)$ gives $R_1=1$ and $R_2=2$. RMS emphasizes concentrated large defects;
neither norm permits cancellation of positive and negative cell imbalances.

### 3.4 Global conservation remains a separate diagnostic

For each equation, retain the signed whole-domain balance

$$
B_k=\Phi_{\mathrm{out},k}-\Phi_{\mathrm{in},k}
-\sum_i S^{\mathrm{integrated}}_{i,k}
=\sum_i r_{i,k}.
$$

The equality follows by telescoping the actual numerical face fluxes. Compute a
direct boundary/source audit as well when verifying this assembly identity.
Opposite local residuals can cancel: two cells with $+1$ and $-1$ have zero global
balance but nonzero local norms.

Use fixed positive global flux scales, for example

$$
F_{\mathrm{ref},k}=A_{\mathrm{ref}}U_{\mathrm{ref}}q_{\mathrm{ref},k},
\qquad G_k=\frac{|B_k|}{F_{\mathrm{ref},k}}.
$$

This reproduces the current annular report convention when $A_{\mathrm{ref}}$ is
inlet area and $U_{\mathrm{ref}}=a_{\mathrm{ref}}$. Store signed $B_k/F_{\mathrm{ref},k}$
for diagnostics and plot absolute values on logarithmic axes. Its size is a
fraction of the reference flux, not necessarily a fraction of actual inlet flux.

Do not call $G_k$ a local residual or use it alone to establish convergence.
Nozzle support should use the actual discrete geometry, friction, and enabled
filter contributions; deriving a second approximate flux path for the monitor
would defeat this check.

## 4. Convergence policy

### 4.1 Every equation must pass

Let $R_k$ be the selected per-equation norm. Accept a common positive tolerance
or a mapping from stable equation names to individual tolerances $\tau_k$.
Require

$$
R_k\leq\tau_k\quad\text{for every equation},
\qquad
C_{\mathrm{equations}}=\max_k\frac{R_k}{\tau_k}.
$$

$C_{\mathrm{equations}}\leq1$ means every equation has passed. Use one documented
comparison convention consistently in JAX kernels, host audits, and plots.

Default to monitoring all conservation equations. Unknown names, missing entries
in an explicit tolerance mapping, nonpositive tolerances, nonfinite scales, and
invalid volumes are setup errors. If an equation is deliberately excluded in a
future advanced interface, record that choice and avoid claiming full equation
convergence.

An RMS over the equation norms can be a secondary summary, but must not determine
success: averaging can hide a failing equation. For comparison, the current
annular all-equation RMS equals
$\sqrt{\sum_kR_{2,k}^2/M}$ with equal cell weights and $M$ equations. Moving to a
maximum over equation norms is stricter at a common threshold even before any
change in weighting or scaling.

### 4.2 Boundary and global checks

Retain boundary equations with their physical scaling and their own tolerance
policy. Do not append them to the cell RMS: they have different meanings and
would be diluted as the physical mesh grows. For the annular model, preserve the
current boundary/regime and global conservation requirements during migration.

For an overall stopping display, define separate tolerance ratios for equations,
boundary constraints, and any required global conservation checks:

$$
C_{\mathrm{stop}}=
\max(C_{\mathrm{equations}},C_{\mathrm{boundary}},C_{\mathrm{global}}).
$$

Include only applicable checks, and explicitly state which are required. The
model still owns admissibility and regime validity. Success requires those
conditions, finite required diagnostics, and $C_{\mathrm{stop}}\leq1$ on a fresh
audit of the returned state.

The nozzle eliminates its native ghost values through boundary maps; it cannot
automatically expose the annular model's independent coupled boundary rows.
Audit the applied boundary conditions where supported, using the existing
physical definitions rather than copying annular equations or inventing a zero
boundary residual. Unavailable diagnostics must be marked as unavailable.

### 4.3 Other stopping conditions

- Reaching an iteration cap, retry cap, or a plateau above tolerance is a
  nonconverged exit, not successful convergence under the new definition.
- The legacy nozzle plateau rule may remain only in explicit legacy mode.
- Nonfinite required residuals fail or reject a trial according to the existing
  controller policy. Do not drop cells, use NaN-ignoring reductions, or clip
  residuals to claim success.
- A maximum-cell residual tolerance can be an optional additional requirement.
  It should not silently become a new mandatory criterion during diagnostics
  migration.
- Relative reduction from the initial residual may be reported. It is not the
  primary physical scaling or a substitute for the absolute dimensionless target.
- Small updates, a small inner Newton residual, and a small global defect alone
  do not certify a steady local root.

### 4.4 Physical-time integration

Do not apply steady stopping automatically to transient calculations. A correct
time-dependent solution can have a substantial spatial residual balanced by
time storage. A transient inner check must use the complete discretized temporal
equation, with its own documented scaling and time-discretization metadata.

The shared norm utilities can reduce either kind of residual. The driver must
identify `steady_spatial` versus `time_discrete` residuals explicitly. Preserve
physical final-time termination and current inner-solve semantics in the initial
implementation scope.

## 5. Shared architecture and proposed interface

### 5.1 Ownership

| Component | Responsibility |
|---|---|
| Model operator | Signed integrated physical residual; exact numerical fluxes and sources |
| Model setup adapter | Equation names/order, volumes, reference scales, boundary/global diagnostics |
| Shared residual reducer | Dimensionless local scaling; weighted `l1`, `l2`, `linf`; finite flags and worst-cell indices |
| Shared convergence policy | Tolerance validation, equation ratios, applicable block ratios, convergence result |
| Driver/controller | Trial acceptance, timesteps, accepted/rejected history, stop reason, final audit |
| Reporting helpers | Equation curves, overall curve, global balance plots, labels and metadata |

Proposed production location:
`src/nozzle_hyperbolic/solvers/residuals.py`. Use the existing
[`finite-volume assembly`](../../src/nozzle_hyperbolic/solvers/finite_volume.py)
and model operators as inputs. This does not require a common state vector,
common boundary treatment, or a rewrite of either time-marching driver.

### 5.2 Conceptual contract

The following describes a contract, not a finalized public Python API:

```text
setup:
  equation_names: fixed ordered tuple of M stable identifiers
  volumes: positive array (N,)
  conservative_scales: positive array (M,)
  reference_length, reference_speed: positive scalars
  weighting: volume | cell
  convergence_norm: l2 | l1
  tolerances: scalar | complete equation-name mapping

reduce(integrated_spatial_residual[N, M]) ->
  l1[M], l2[M], linf[M], worst_cell[M], finite

assess(selected_equation_norms[M], applicable_constraint_metrics) ->
  equation_ratios[M], equation_max_ratio, overall_max_ratio,
  all_equations_passed, all_required_checks_passed
```

Use stable identifiers such as `mass`, `axial_momentum`, `meridional_momentum`,
`angular_momentum`, and `energy`. Do not label the primitive pressure unknown as
the continuity equation. Equation identifiers and display labels are separate.

History and plot settings should select the norm used for stopping, while both
norms remain available for comparison. Do not silently change convergence by
changing a plot preference. If supporting a combined RMS summary, give it an
explicit name distinct from the maximum tolerance ratio.

### 5.3 JAX and performance requirements

- Reduce arrays shaped `(N, M)` on the device. Keep equation ordering, shapes,
  norm selection, and weighting fixed for each compiled solver configuration.
- Resolve names, mappings, reference states, and validation on the host before
  JIT. Avoid traced Python branching, host conversions, or dynamic string keys
  inside iteration kernels.
- Store fixed-size metric arrays in compiled histories and attach human-readable
  metadata outside the compiled loop.
- Preserve the solver precision policy, including current 64-bit verification.
  Accumulate norms with stable arithmetic if extreme ranges require it.
- Do not introduce norms or absolute values into the residual differentiated by
  Newton. The monitoring layer need not be differentiable at zero.
- Reuse an existing residual evaluation when it corresponds to exactly the same
  accepted state. The explicit pre-update right-hand side cannot be labeled as a
  post-update residual without reevaluation.
- If post-update evaluation needs extra flux/EOS work, measure and disclose that
  cost. Cheap norm reductions do not imply free residual evaluations.
- Keep scalar legacy step return signatures intact during the additive phase;
  expose diagnostics through an explicit operator or opt-in extended interface.
- Do not store every cell residual at every iteration by default. Per-equation
  reductions need only order $M$ storage per history sample. Full fields can be
  saved for selected snapshots and final audits.

## 6. History and plotting

### 6.1 Run metadata

Store a versioned residual definition, independently of the package version:

- Definition identifier, for example `steady_spatial_fixed_scale_v1`.
- Equation identifiers and their order, display labels, and raw residual units.
- Norm used for convergence, available norms, weighting convention, and resolved
  absolute tolerances for all required blocks.
- Numerical reference values, reference EOS state/backend, reference-selection
  rule, geometry length/area/radius conventions, and cell-volume information.
- Model, flux, reconstruction, sources/filter settings, dtype, source revision
  and relevant working-tree fingerprints for numerical studies.
- Whether the data are steady spatial or time-discrete residuals, and whether
  each sample describes an initial, pre-update, or post-update state.
- Legacy diagnostics, controller input metric, and any optional monitoring or
  convergence exclusions.

The suggested identifier names the future definition; it is not an implemented
schema or an assertion that the final weighting policy has been approved.

### 6.2 Samples and event semantics

Record the initial state as accepted-update index zero. Record post-update
diagnostics for every accepted update when history is enabled. Include:

- Accepted-update index, attempt index, physical or pseudo-time with an explicit
  label, timestep, CFL, and elapsed time where measured.
- Per-equation `l1`, `l2`, `linf`, worst-cell indices, selected norm values and
  tolerance ratios.
- Applicable boundary metrics, signed scaled global balances, and their maxima.
- `equation_max_ratio`, `overall_max_ratio`, admissibility, and pass/fail flags.
- Available inner Newton iterations, residual/update measures, status, trial
  rejections, and the outer acceptance outcome.

Retain rejected attempts in a separate event stream or clearly flagged records.
They must not be plotted as accepted progress. If a rejected candidate was not
evaluated, mark its outer diagnostics unavailable; do not present a copy of the
last accepted metrics as measurements of that candidate. Preserve nonfinite
failure information explicitly when exporting strict JSON.

Record a precise final reason such as `converged`, `max_iterations`, `plateau`,
`retry_limit`, `nonfinite`, `regime_mismatch`, or `final_time_reached`.
Histories and the final result must refer to the same audited returned state.

### 6.3 Two primary residual views and a conservation view

1. **Equation view:** one curve per equation for the selected dimensionless norm.
   Show each tolerance, or optionally plot all equation tolerance ratios with a
   common target of one. Label norm, weighting, and definition.
2. **Overall view:** plot `equation_max_ratio` for the worst conservation equation,
   or `overall_max_ratio` when boundary/global constraints are included. State
   which curve is shown. A value at or below one has a direct stopping meaning.
3. **Conservation audit:** retain the existing four annular global balance
   curves, clearly labeled as global defects rather than cell residual norms.

Use accepted updates, elapsed time, or an explicitly labeled physical/pseudo-time
as the horizontal axis. Show actual trajectories, not best-so-far envelopes.
On logarithmic axes, annotate or omit exact zeros while preserving raw values;
do not change the stored metrics to a plotting floor. Keep model-independent
report helpers separate from model-specific conservation labels.

## 7. Compatibility and migration

Changing the default is numerically backward incompatible even if Python
signatures remain unchanged. Existing tolerances, iteration counts, final states,
controller trajectories, history consumers, and report comparisons can change.

There is no universal conversion from old SSR thresholds to new thresholds:
the old local state normalization changes throughout a run, the time scale
differs, and the proposed equation reduction and weighting also differ.
Switching between `l1` and `l2` likewise changes the meaning of a fixed threshold.

Use a staged policy:

| Stage | Legacy behavior | New behavior |
|---|---|---|
| Additive diagnostics | Existing defaults, tuples, `ssr` fields, controller input, and stopping preserved | Opt-in metrics evaluated alongside legacy behavior |
| Opt-in stopping | Explicit legacy mode available | New norm, scaling, and per-equation checks selected explicitly |
| Default migration | Deprecated with clear metadata and warnings in configuration/documentation | New default announced as a numerical breaking change |
| Later removal, if desired | Separate announced decision | No silent reinterpretation of old data |

Do not populate `ssr` with a new quantity while implying its historical formula
is unchanged. Keep `legacy_ssr` or the unchanged old field during transition and
publish new fields under distinct names. Likewise, the annular `spatial_rms`
cannot silently acquire volume weighting or become a maximum over equations.

Old saved histories lacking per-equation data cannot reconstruct these metrics
from one scalar. Regenerate them from saved full states and original operators
where possible, or retain their legacy labels. Existing published reports should
not be cosmetically relabeled as new-definition results.

For controllers, first keep the old metric as the input while recording the new
one. Then make controller input explicit. A later switch of the SER ramp or any
residual-driven growth policy is a separate measured behavior change. Do not
compare timing or iteration counts without documenting that difference.

## 8. Implementation plan

### Phase 0: freeze the specification and inventory consumers

1. Confirm default RMS, fixed physical scaling, proposed volume weighting, and
   whether equal-cell weighting is included in the first public interface.
2. Confirm scalar and per-equation tolerance semantics, separate boundary/global
   tolerances, and which optional maximum-cell checks are exposed initially.
3. Define a reference-state selection rule for each model, overrides, equation
   identifiers, history version, and legacy-mode policy. Do not guess universal
   default tolerances from the old SSR values.
4. Inventory production and development consumers of `ssr`, `stopping_ssr`,
   `steady_residual_norm`, `spatial_rms`, `conservation_max`, `parameters['residuals']`,
   plateau checks, SER/CFL ramps, callbacks, step tuples, and saved JSON/CSV.
5. Pin the numerical baseline to source fingerprints and controlled physical
   setups. Distinguish in-progress source changes from this monitoring work.

Deliverable: agreed formulas, compatibility matrix, and consumer list before
changing production defaults.

### Phase 1: implement the shared reducers and policy

1. Add the shared module with setup validation, dimensionless scaling, weighted
   `l1`, `l2`, `linf`, and worst-cell locations.
2. Add complete per-equation tolerance resolution, finite checks, and maximum
   tolerance ratios for equations and applicable constraint blocks.
3. Keep the functions independent of EOS, primitive-state ordering, flux method,
   and number of equations. Separate host metadata from JAX-compatible arrays.
4. Add focused production tests for mathematical invariants and convergence
   decisions, including cases described in Section 9.

Deliverable: tested generic utilities; neither model's default behavior changed.

### Phase 2: add model adapters and parallel diagnostics

1. Nozzle implicit: use `steady_residual` or an equivalent canonical spatial
   evaluation, reshape interior rows to `(N, 3)`, and supply exact cell volumes,
   fixed scales, and names. Preserve existing public step returns.
2. Nozzle explicit and NumPy: identify the actual residual representation and
   area/volume convention before converting to integrated rows. Reuse existing
   flux/source assembly; do not multiply area twice. Define post-update sampling
   and explicitly account for the Lax-Friedrichs timestep-dependent flux.
3. Annular: reuse `spatial_residual`, volumes, reference quantities, boundary
   operator, and direct global balance audit. Keep the existing combined RMS
   and its equal-cell weighting under their legacy definition.
4. Record new metrics alongside old ones on identical accepted states, keeping
   stopping, trial acceptance, Newton work, and controller inputs unchanged.
5. Implement a common history serializer and initial/final audit semantics.

Deliverable: both models expose matching diagnostic contracts; an unchanged
legacy configuration remains reproducible within the existing numerical policy.

### Phase 3: add opt-in per-equation stopping

1. Introduce an explicit configuration block for the new definition, norm,
   weighting, scales, and tolerances. Keep the old scalar setting in legacy mode.
   Reject ambiguous configurations that specify conflicting policies.
2. Update host and compiled loops to require all equations and applicable
   constraints to pass. Expose both equation and complete stopping ratios.
3. Disable plateau-as-success in new mode; distinguish stall, cap, invalid state,
   boundary regime failure, and convergence in result status.
4. Reevaluate the final returned state with the original spatial/boundary
   operators, independent of the loop's cached stop flag.
5. Keep physical-time termination and inner Newton rules unchanged. Make any
   residual-driven controller migration explicit rather than incidental.

Deliverable: selectable new convergence policy with meaningful result status.

### Phase 4: migrate histories and reports

1. Add reusable plotting helpers for per-equation norms/ratios and the overall
   maximum ratio, preserving the distinct global conservation plots.
2. Update the annular native driver, recommended compiled controller, verification
   drivers, and report/audit readers to consume the versioned definitions.
3. Update the nozzle verification and residual-dependent numerical studies with
   explicit metric selection and scales. Historical datasets retain their labels.
4. Export metadata with histories and document unavailable legacy fields.
   Presentation changes should regenerate reports from saved compatible data;
   rerun numerical solves only when the required data were never saved.

Deliverable: plots and final success checks agree with the actual configured
convergence policy, and old results remain interpretable.

### Phase 5: validate and choose defaults

1. Run controlled cases for both models, including a deliberate localized defect,
   smooth/subsonic flow, a shocked/choked case, and nonuniform meshes.
2. First compare metrics on the same saved states. Then compare stopping policies
   on matched solves. Finally study controller changes separately, if requested.
3. Audit local residuals, boundary equations, global conservation, intended
   physical branch, and reference accuracy as distinct outcomes.
4. Check whether proposed tolerances are attainable above roundoff/property
   recovery floors and adequate for relevant physical outputs.
5. Measure additional monitoring cost with synchronized device work and
   equivalent history/audit overhead. Use the repository's repeated-benchmark
   protocol only if a timing claim is needed.
6. Document selected defaults and migration examples. Change defaults only in a
   clearly announced numerical breaking release; do not imply that this design
   note has already validated those choices.

Deliverable: evidence supporting the new defaults, plus an explicit legacy route.

## 9. Verification requirements

Production tests belong under `tests/`; numerical experiment drivers and
independent audits belong under this study's future `scripts/` directory.

| Check | Required evidence |
|---|---|
| Norm definitions | Small arrays with independently calculated mean absolute, RMS, and maximum results |
| Cancellation | Opposite signed cell residuals give zero global sum and nonzero local norms |
| Equation isolation | One failing equation prevents convergence even when the average over equations is small |
| Nonuniform weights | Subdividing a cell while preserving volume and residual density preserves the volume-weighted norms |
| Scale consistency | Simultaneous consistent rescaling of raw residual units and their scales leaves dimensionless results unchanged |
| Timestep normalization | A fixed spatial residual array produces identical metrics for changes in numerical timestep/CFL; flux schemes with intrinsic timestep dependence are identified separately |
| Rest/zero swirl | Positive scales and finite metrics without division by evolving zero momentum |
| Boundary separation | Boundary defects are not diluted by increasing the number of physical cells |
| Global identity | Direct boundary-flux-minus-source balance matches the signed sum of cell residuals within floating-point accuracy |
| Failure semantics | NaNs, infinities, invalid scales, missing equation tolerances, plateau, and caps cannot produce false success |
| History alignment | Stored metrics match independently reevaluated initial and accepted states, especially explicit versus implicit sampling |
| JAX behavior | Eager/JIT agreement and no change to signed residual/Jacobian construction or default step tuple contracts |
| Legacy behavior | Focused regression checks preserve old diagnostics and decisions when legacy mode is selected |
| Returned solution | Fresh local, boundary, and conservation audits agree with the reported final convergence status |

Existing relevant tests include
[`test_euler_steady_operator.py`](../../tests/test_euler_steady_operator.py),
[`test_euler_annular_fvm.py`](../../tests/test_euler_annular_fvm.py), and
[`test_shared_fvm.py`](../../tests/test_shared_fvm.py).
Retain their established contract checks rather than treating a changed test
expectation as proof that numerical compatibility was preserved.

For physical verification, reuse the established
[Euler nozzle cases](../roberto/euler_nozzle/README.md) and
[annular cases](../roberto/euler_annular/README.md). Smaller residuals do not prove
smaller analytic error or correct branch selection. Save failed runs and actual
final fields; do not relax thresholds to manufacture a successful comparison.

## 10. Outstanding decisions before coding

The following remain proposals rather than settled public interface choices:

1. Volume weighting as the new default, versus retaining equal-cell weighting
   initially to reduce the annular migration scope.
2. Exact default reference-state rules and per-model overrides, especially for
   general EOS and low-Mach applications.
3. New absolute tolerances, including whether boundary/global checks use a common
   default or explicitly separate values.
4. Whether the first implementation includes all nozzle backends or initially
   supports the implicit JAX and annular paths with other paths explicitly
   unsupported in new mode.
5. The opt-in API and legacy deprecation schedule, including low-level step tuple
   compatibility and residual-driven controller selection.

These choices should be resolved against the specification and validation cases,
without broadening the work into changes to the physical equations or nonlinear
solver algorithms.

## 11. CFD context and references

Fluent is a useful example of equation-wise residual monitoring, not a definition
that this project should copy exactly. Its pressure-based global scaling uses a
sum of absolute cell equation imbalances divided by an equation-dependent global
scale; continuity has special scaling. Its density-based solver reports RMS
equation-rate residuals with separate scaling options. Neither should be confused
with the absolute signed whole-domain conservation defect.

- [Ansys Fluent: Monitoring Solution Convergence, 2025 R1](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/flu_ug/flu_ug_sec_solve_monitor.html)
  documents residual definitions and scaling options.
- [Ansys Fluent: Judging Convergence, 2025 R2](https://ansyshelp.ansys.com/public/views/secured/corp/v252/en/flu_ug/flu_ug_sec_solve_conv_stab.html)
  explains why thresholds are not universal and why integrated physical quantities
  should accompany residual checks.

The proposed shared design follows the same broad practice of monitoring every
equation while specifying its own fixed physical scales, normalized norms,
history schema, and convergence policy. A numerical threshold from Fluent or
legacy SSR must not be transferred without checking those definitions.
