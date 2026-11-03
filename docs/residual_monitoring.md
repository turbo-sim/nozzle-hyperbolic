# Residual monitoring and steady convergence

This document describes the shared production FVM implementation checked on
31 October 2026. It replaces the earlier design note in
`development/residual_monitoring/`. The monitoring machinery is implemented and
used by nozzle, annular and compatible user-programmed models; it is no longer
an opt-in annular migration proposal.

For the surrounding package structure, see the
[architecture guide](architecture/ARCHITECTURE.md).

## Responsibilities and the three residual checks

The physical model supplies conserved-variable scales, a fixed reference time,
global flux scales, equation names and scaled boundary equations. The shared
operator supplies the actual discrete spatial residual. The numerical driver
reduces those rows and applies the selected convergence policy, without knowing
the EOS or the meaning of the model variables.

Four quantities answer different questions:

| Quantity | Meaning | Role |
|---|---|---|
| Per-equation spatial norm | Do the physical cells satisfy the steady discrete balances? | Every equation must pass its selected relative/absolute criterion |
| Boundary defect | Do the endpoint traces satisfy the physical boundary equations? | Independent absolute check, for either coupled or segregated treatment |
| Cell constraint defect | Do all cell-local algebraic equations hold? | Independent fixed-scale absolute maximum, with no initial-relative reduction or phase weights |
| Global conservation defect | Do the domain's numerical boundary fluxes and integrated sources balance? | Independent absolute check for every conservation equation |

The inner Newton residual and the size of a timestep update are additional
diagnostics. Neither is a substitute for the original spatial residual when
deciding that a steady solution has converged.

## Spatial residual and fixed physical scaling

For physical cell $i$ and conservation equation $k$, the operator assembles

$$
r_{i,k}=\Phi_{i+1/2,k}-\Phi_{i-1/2,k}+N_{i,k}^{\mathrm{integrated}}-S_{i,k}^{\mathrm{integrated}}.
$$

$\Phi$ includes the actual numerical face flux, frame conversion and face area.
$S^{\mathrm{integrated}}$ includes the model's cell-integrated sources. Enabled
conservative filters are part of the discrete operator. Monitoring does not
replace those fluxes with an analytical or cell-centred approximation. Only
physical cell rows enter the spatial norms; endpoint constraints are checked
separately.

For a nonconservative model, $N$ is the actual discrete nonconservative work,
evaluated using the model's declared compatibility formulation. Summed rows
remain a **discrete balance defect**, not a claim that every phase momentum or
energy is independently conserved. Inspect the model's exchange/work identities
and invariant projections separately. `Operator.residual_contributions(state)`
returns the signed conservative, stabilization, NC and source arrays.

Cell constraints are monitored as
`max(abs(cell_constraints/constraint_scales), axis=cells)`, one value per
constraint in `constraint_norm`; `constraint_max` is their maximum.
`ResidualPolicy.constraint_tolerance` defaults to $10^{-9}$. Every constraint
must pass this absolute test for steady convergence and transient snapshots.
They never enter the transported-equation relative baseline. Endpoint residuals
also expose `boundary_equation_max` in addition to the overall `boundary_max`.

A model may declare `invariant_projections`, `invariant_names` and positive
fixed `invariant_flux_scales`. `invariant_defect` reports the signed projected
sum of the **unmodified** physical spatial rows, including discrete NC work.
Declared invariant budgets also gate steady convergence. The two-fluid model
currently declares only total mass; it does not assume that summed phase-energy
slots form a conserved invariant. Physical-time snapshots instead use the
integrator-consistent content-change budget, since a transient spatial sum need
not vanish.

The dimensionless cell residual is

$$
z_{i,k}=\frac{t_{\mathrm{ref}}r_{i,k}}{V_i q_{\mathrm{ref},k}},
\qquad t_{\mathrm{ref}}=\frac{L_{\mathrm{ref}}}{U_{\mathrm{ref}}}.
$$

Volumes, conservative-variable scales and reference time must be positive and
finite. They remain fixed during a solve. The scaling does not use the numerical
timestep, CFL, evolving local momentum or initial residual magnitude. Relative
reduction from the initial norm is a separate convergence test.

The current Euler models use a fixed acoustic speed and density from their
reference fluid state. Nozzle conservative scales are
$\rho_{\mathrm{ref}}(1,a_{\mathrm{ref}},a_{\mathrm{ref}}^2)$; annular scales are
$\rho_{\mathrm{ref}}(1,a_{\mathrm{ref}},r_{\mathrm{ref}}a_{\mathrm{ref}},a_{\mathrm{ref}}^2)$,
with inlet radius as the annular radius reference. The reference time uses the
fixed mesh extent divided by that acoustic speed. These are physical-model
choices, not assumptions inside the generic reducer. Acoustic scaling remains
defined at rest and at zero swirl.

Fixed scaling makes the norm reproducible, but does not make it an error
estimator. Low-Mach effects, EOS conditioning, shocks and roundoff still require
appropriate physical accuracy checks.

## Norms and nonuniform meshes

For normalized positive cell weights $w_i$, the reducer returns, for each
equation,

$$
R_{1,k}=\sum_i w_i|z_{i,k}|,\qquad
R_{2,k}=\sqrt{\sum_i w_i z_{i,k}^2},\qquad
R_{\infty,k}=\max_i|z_{i,k}|.
$$

It also returns the zero-based physical-cell index of each maximum. The RMS
calculation rescales before squaring to reduce overflow for large finite inputs.

| Setting | Choices | Default |
|---|---|---|
| `ResidualPolicy.norm` | `"l1"` or `"l2"` for stopping | `"l2"` |
| `ResidualPolicy.weighting` | `"volume"`: $w_i=V_i/\sum_j V_j$; `"cell"`: $w_i=1/N$ | `"volume"` |

Both L1 and L2, plus L∞ and worst-cell indices, are retained regardless of the
selected stopping norm. L∞ is diagnostic; it is not currently a selectable
stopping norm.

Equal axial spacing in a variable-area nozzle does not imply equal-volume
cells. Volume weighting avoids giving a region extra influence solely because
it contains more cells, but a small localized defect can still be diluted in an
average. Inspect L∞ as well. Opposite signs cannot cancel in any of these norms.
For the same weights, $R_{1,k}\leq R_{2,k}\leq R_{\infty,k}$, so changing norm
also changes the meaning of a fixed numerical tolerance.

## Relative and absolute stopping

Let $R_k$ be the selected dimensionless norm and $R_k^0$ its value at the start
of this solve. For an active equation, define

$$
\widehat R_k=R_k/R_k^0,\qquad
a_k=R_k/\varepsilon_{\mathrm{abs}},\qquad
b_k=\widehat R_k/\varepsilon_{\mathrm{rel}}.
$$

The criterion applies to each equation individually, not to an average over
equations:

| `criterion` | Each active equation passes when | Its tolerance ratio |
|---|---|---|
| `"either"` | Relative OR absolute target is met | $\min(a_k,b_k)$ |
| `"both"` | Relative AND absolute targets are met | $\max(a_k,b_k)$ |
| `"relative"` | Relative target is met | $b_k$ |
| `"absolute"` | Absolute target is met | $a_k$ |

`equation_max_ratio` is the maximum of these ratios over all equations. A value
at or below one means every equation passes. Nonfinite or negative norm inputs
fail the assessment.

An equation with $R_k^0\leq\texttt{initial_floor}$ uses the absolute target only,
including in `"relative"` mode. This avoids demanding a reduction from a zero or
roundoff-level initial residual. `relative_active` records that decision. A zero
reported relative value for an initially zero equation must not be read as proof
of convergence; its absolute check still applies.

The current defaults are:

| Field | Default |
|---|---|
| `criterion` | `"either"` |
| `relative_tolerance` | `1e-6`: six orders of reduction from the initial norm |
| `absolute_tolerance` | `1e-9`: dimensionless, physically scaled norm |
| `boundary_tolerance` | `1e-9` |
| `global_tolerance` | `1e-9` |
| `initial_floor` | `1e-14` |

Tolerance fields are positive finite scalars shared across equations.
Equation-name-to-tolerance mappings are not implemented. The absolute threshold
is not a tolerance on a raw residual in SI units.

Each grid-sequence stage, and each independent retry solve, establishes its own
initial norms. Starting from an adapted solution therefore changes the relative
baseline; it does not change the fixed physical scales.

## Boundary checks, conservation and final success

The boundary maximum is the largest absolute entry of the model's scaled
boundary residual. These endpoint rows have zero storage and are not included
in the physical-cell RMS. Increasing the number of cells cannot dilute the
boundary check.

For a conservative assembly, interior face fluxes telescope:

$$
B_k=\sum_i r_{i,k}
=\Phi_{\mathrm{out},k}-\Phi_{\mathrm{in},k}
-\sum_i S_{i,k}^{\mathrm{integrated}},\qquad
G_k=|B_k|/F_{\mathrm{ref},k}.
$$

The model supplies fixed positive $F_{\mathrm{ref},k}$ through
`global_flux_scales`. The Euler models use inlet area times reference acoustic
speed times their conservative scales. `global_defect` stores the absolute
dimensionless vector $G_k$, not signed balances and not percentages of the
instantaneous inlet flux. A zero global defect can coexist with large local
defects of opposite sign.

Steady success requires finite diagnostics, an admissible state, every equation
passing, the boundary check passing, every global check passing, and any required
engineering monitor windows passing. The driver freshly evaluates spatial,
boundary and global metrics at the returned state. If provided, the model's
`solution_audit` must also pass before success is reported.

`max_steps`, `retry_limit`, `invalid_timestep`, `invalid_initial_state`,
`stagnation` and `solution_audit_failed` are unsuccessful exits. Stagnation here
means repeated exactly unchanged accepted states while convergence remains
unsatisfied; it is not a residual plateau accepted as success. An inner solve
that fails its own convergence/admissibility checks causes a rejected attempt
and timestep retry, rather than an accepted steady update.

## Configure the shared solver

Supply an already constructed physical `model`. The following configuration
works with any model satisfying the shared interface; it defines no physical
case or initial condition:

```python
import jax

jax.config.update("jax_enable_x64", True)

from nozzle_hyperbolic.solvers.fvm import (
    ResidualPolicy, Scheme, SteadySettings, solve_single_grid,
)

policy = ResidualPolicy(
    norm="l2",
    weighting="volume",
    relative_tolerance=1e-6,
    absolute_tolerance=1e-9,
    boundary_tolerance=1e-9,
    global_tolerance=1e-9,
    criterion="either",  # or "both", "relative", "absolute"
)
scheme = Scheme(time_integrator="euler-implicit")
settings = SteadySettings(residual=policy)

def solve_model(model):
    return solve_single_grid(model, scheme, settings)
```

The same steady stopping policy is used with explicit Euler and SSPRK3. Selecting
an integrator does not change the definitions of the spatial norms.

## Optional engineering monitors

`MonitorSpec` describes a named physical quantity; a supplied evaluator computes
it from `(full_state, numerical_face_flux)`. The shared `MonitorTracker` owns the
accepted-sample window and stationarity test, rather than a formula for mass flow
or pressure coefficient.

For each complete window, stationarity means

$$
\max_j m_j-\min_j m_j
\leq\varepsilon_{m,\mathrm{abs}}
+\varepsilon_{m,\mathrm{rel}}m_{\mathrm{ref}}.
$$

The reference is fixed and positive, not the evolving monitor value. Tolerances
have the units implied by the evaluated quantity and its reference. The initial
state supplies the first sample; only accepted updates add further samples.
Rejected attempts do not advance a monitor window. `convergence=True` adds the
complete, stable window as a requirement; otherwise the monitor is diagnostic.
All configured monitor values must still be finite.

For an existing annular `model`, the package supplies a physical evaluator:

```python
from nozzle_hyperbolic.models.euler_annular import AnnularQuantities
from nozzle_hyperbolic.solvers.monitoring import MonitorSpec

monitors = (
    MonitorSpec(
        name="outlet_mass_flow",
        quantity="mass_flow",
        reduction="outlet",
        convergence=True,
        window=20,
        absolute_tolerance=1e-8,  # kg/s; choose for the physical case
    ),
)

def solve_monitored_annular(model):
    return solve_single_grid(
        model, scheme, settings,
        monitor_specs=monitors,
        quantity_evaluator=AnnularQuantities(model, monitors),
    )
```

Annular mass-flow monitors use actual numerical face fluxes. The evaluator also
provides pressure-coefficient monitors with explicitly resolved reference data.
Other models can supply their own evaluator without modifying the driver.
Stationary engineering quantities supplement the conservation checks; they do
not replace them.

## History, export and plots

For an admissible initial state, `Solution.history` starts with `step=0`,
`phase="initial"`, `accepted=True` and `dt=0`. An invalid initial state returns an
unsuccessful solution with an empty history. Accepted rows contain post-update
metrics. Rejected rows are explicitly
marked and carry available inner-solve information, without copied metrics from
the previous accepted state. `step` identifies the proposed accepted-update
number; retries can share it, with `retry` distinguishing those attempts.

| Fields | Interpretation |
|---|---|
| `equation_norm`, `equation_l1`, `equation_l2`, `equation_linf`, `worst_cell` | Selected and diagnostic per-equation norms and locations |
| `equation_relative`, `relative_active`, `absolute_satisfied`, `relative_satisfied` | Initial-relative reductions, applicability and individual target checks |
| `equation_max_ratio`, `all_equations_passed`, `criterion` | Equation stopping assessment |
| `boundary_max`, `global_defect`, `boundary_passed`, `global_passed` | Separate boundary/global assessment |
| `finite`, `admissible`, `monitors` | State validity and monitor-window diagnostics |
| `cfl`, `requested_cfl`, `dt`, `inner_iterations`, `inner_status`, `inner_residual`, `rejected_newton_trials` | Attempt/controller and inner-solve information when available |

`overall_relative` is the largest initial-relative norm among active equations,
or zero when none are active. It is not an overall tolerance ratio: under
`"either"`, an equation can pass absolutely without reaching its relative target.
The driver does not currently return an `overall_max_ratio` combining all blocks.
If plotted, such a residual-only ratio can be constructed as

$$
C=\max\left(C_{\mathrm{equations}},
\frac{\texttt{boundary_max}}{\varepsilon_{\mathrm{BC}}},
\frac{\max_k G_k}{\varepsilon_{\mathrm{global}}}\right).
$$

Monitor windows and model audits remain additional checks outside this ratio.

The library defines `DEFINITION="steady_spatial_fixed_scale_equation_v1"` and
provides `strict_json()` for arrays and classified nonfinite diagnostics. It does
not automatically attach a complete versioned metadata envelope to each
`Solution`. Applications should save the definition, policy, equation order,
resolved scales and mesh measures explicitly, along with settings and provenance.
For example, with a returned `solution`, `scheme` and `settings`:

```python
from dataclasses import asdict
from pathlib import Path
from nozzle_hyperbolic.solvers.convergence_output import strict_json
from nozzle_hyperbolic.solvers.residuals import DEFINITION

def save_convergence(path, solution, scheme, settings):
    model = solution.operator.model
    payload = {
        "definition": DEFINITION,
        "equation_names": model.equation_names,
        "conservative_scales": model.conservative_scales,
        "reference_time": model.reference_time,
        "global_flux_scales": model.global_flux_scales,
        "volumes": model.mesh.volumes,
        "scheme": asdict(scheme),
        "settings": asdict(settings),
        "success": solution.success,
        "reason": solution.reason,
        "metrics": solution.metrics,
        "history": solution.history,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json(payload), encoding="utf-8")
```

This is an application export recipe, not an enforced file schema. Numerical
studies should additionally retain source fingerprints, backend/precision,
physical setup, reference-state origin and measurement-time provenance.
`strict_json()` converts nonfinite values to JSON `null` and records their paths
and classifications; it does not alter the solver's arrays or convergence tests.

For plots, show equation norms or tolerance ratios, separate boundary/global
defects, and accepted updates or clearly labelled elapsed/pseudo-time. Mark
rejections separately. Preserve exact zeros in saved data even if logarithmic
plots must omit or annotate them. A common plot-label/metadata interface is not
currently part of the shared solver.

## Known limits and reporting mismatch

- `ResidualPolicy` provides scalar tolerances, not per-equation tolerance maps.
  L∞ is recorded but has no separate built-in stopping threshold.
- `global_defect` retains absolute values; signed global histories require
  additional recording. Full per-cell residual fields are not retained at every
  step by default.
- Versioned export metadata and a combined stopping ratio require explicit
  application handling, as described above.
- `ConvergenceTable` currently reads `conservation_max`, while the shared driver
  supplies `global_defect`. Passing native rows directly therefore prints `--`
  in the G column. A display callback can supply
  `conservation_max=max(row["global_defect"])` for rows containing that vector.
  This is a reporting mismatch, not a missing conservation check. It remains
  open; this documentation migration does not change production code.

## Transients and historical results

`solve_transient` advances to requested physical output times. A valid unsteady
state can have a nonzero spatial residual balanced by conservative storage, so
the steady convergence criteria are not its termination rule. Inner solves
check the time-discrete equations, while transient conservation budgets compare
changes in total contents against integrator-consistent boundary/source
transfers. Keep those budgets distinct from steady global defects.

Historical state-normalized nozzle SSR, combined annular RMS and update norms
are different definitions. There is no universal conversion from their old
thresholds to this policy. Preserve old report labels and settings; a scalar
legacy history cannot reconstruct missing per-equation norms. Converging the
discrete root, conserving integrated quantities, matching an analytical solution
and reaching the intended physical branch remain separate verification tasks.

The old opt-in/staged migration plan has been superseded by the shared FVM
interface and its breaking API migration. The deleted development note remains
available through Git history.

## Source and verification map

| Source | Responsibility |
|---|---|
| [solvers/residuals.py](../src/nozzle_hyperbolic/solvers/residuals.py) | Fixed-scale reductions, validation, definition identifier and `ResidualPolicy` |
| [solvers/fvm/drivers.py](../src/nozzle_hyperbolic/solvers/fvm/drivers.py) | Sampling, initial baselines, histories, final assessment and monitor integration |
| [solvers/fvm/config.py](../src/nozzle_hyperbolic/solvers/fvm/config.py) | `SteadySettings` and `Solution` |
| [solvers/monitoring.py](../src/nozzle_hyperbolic/solvers/monitoring.py) | `MonitorSpec` and accepted-sample stationarity windows |
| [solvers/convergence_output.py](../src/nozzle_hyperbolic/solvers/convergence_output.py) | Console table and strict JSON serialization |
| [NozzleModel](../src/nozzle_hyperbolic/models/euler_nozzle/model.py), [AnnularModel](../src/nozzle_hyperbolic/models/euler_annular/model.py) | Physical reference scales, admissibility and boundary equations |
| [annular diagnostics](../src/nozzle_hyperbolic/models/euler_annular/diagnostics.py) | Physical engineering-quantity evaluation |
| [test_residual_monitoring.py](../tests/test_residual_monitoring.py) | Independently calculated norms, weighting, cancellation, initial-zero rules, finite checks and monitor windows |
| [test_generic_fvm.py](../tests/test_generic_fvm.py) | Criterion semantics and shared-driver contracts for user models |
| [test_euler_annular_monitoring.py](../tests/test_euler_annular_monitoring.py) | Original-row scaling, physical quantities, rejected history and final-state checks |

The documentation review ran 22 focused residual/stopping tests successfully.
That result covers those tests, not every physical model or numerical
configuration. From the repository root, the same focused check is:

```powershell
$env:JAX_ENABLE_X64 = "true"
poetry run python -B -m pytest -q -p no:cacheprovider tests/test_residual_monitoring.py tests/test_generic_fvm.py -k "residual or stopping or initially_zero_equations"
```
