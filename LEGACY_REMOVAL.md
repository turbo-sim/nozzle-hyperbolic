# Frozen archive removal and remaining references

Removed on 3 November 2026 at the user's request.

## Removed

- The entire repository-root `legacy/` directory: 154 files, including the frozen
  implementation, its tests/manifests and the original-suite backup created during
  the preceding review. That backup is no longer present.
- `tests/legacy/`: three files containing 29 parametrized historical cases,
  including the three expected failures of the deprecated mass-transfer closure.
- The Euler test comparing the regularized flux against the private old flux.
  The continuity test retains its independent convergence assertions and no longer
  calls the old implementation. Rest/sonic JIT and AD checks remain.
- `cases/archive/validation/test_friction_factors.py`: an archived diagnostic
  importing the retired `nozzle_hyperbolic.fvm_solver` namespace.
- The obsolete pytest marker, default exclusion, historical CI invocation,
  archive-specific ignore rule and maintained documentation links to deleted files.

The seven numerical baselines under `tests/regression/data/physical_v1/` remain.
They use local arrays and their own source-provenance archive, and do not load the
deleted implementation. The upstream two-fluid JSON fixture also remains an
independent data comparison.

## Direct references to the removed directory

No remaining Python file in `src/`, `tests/`, `demos/`, `development/` or `cases/`
references the deleted archive path. No maintained test imports a deprecated
two-fluid module or the retired `fvm_solver` namespace.

The earlier [test review](TEST_SUITE_REVIEW.md) retains the removed path as plain
text to explain what was saved and subsequently deleted. It is explicitly marked
as a historical measurement; it no longer links to a nonexistent archive. The
[architecture review](ARCHITECTURE_REVIEW.md) also retains historical outcomes.
Saved logs, JSON/XML results and cached reports under `outputs/`, `htmlcov/` and
development evidence may still mention old paths or test names. Those records
are historical evidence and are not imported by maintained tests or solvers.

Concrete retained records include `outputs/test_suite_audit/full_results.json`,
`current_collection.json` and `structure_audit.json`. The flux-study records
`development/flux_methods/stadtke/outputs/verification/settings.json`,
`development/flux_methods/osher/outputs/current_api_smoke/settings.json`,
`development/flux_methods/osher/outputs/archives/recorded_execution/settings.json`
and the corresponding settings inside the shared frozen-source output also name
the old archive. In those settings files, the matches are recorded Git-status
entries, not paths used to load a solver. They were preserved as provenance.

## Deprecated implementations still present outside that directory

Removing the frozen archive does not remove the parallel deprecated implementation
inside the installable package. These files are still present:

| Location under `src/nozzle_hyperbolic/` | Status |
|---|---|
| `models/two_fluid/solver.py` | Deprecated separate solver; active model uses shared FVM drivers |
| `models/two_fluid/state.py` | Deprecated state helpers |
| `models/two_fluid/matrices.py` | Deprecated matrix implementation |
| `models/two_fluid/flux.py` | Deprecated flux implementation |
| `models/two_fluid/sources.py` | Deprecated source implementation |
| `models/two_fluid/boundaries.py` | Deprecated boundary implementation |
| `models/two_fluid/closures/` | Deprecated closure package |
| `models/euler_nozzle/flux_jax.py` | Still defines private `_jax_F_fvs_stadtke_legacy`; maintained tests no longer import it |

These are separate source-removal candidates. Their removal requires retiring or
migrating the research callers listed below. The active `TwoFluidModel` and its
`_alberto_*` physics modules implement the current constrained model and remain
covered by the maintained suite.

## Research scripts using retired imports

These scripts reference `nozzle_hyperbolic.fvm_solver`, which is not a supported
current namespace, or explicitly import a deprecated closure. They do not access
the deleted root archive, but need migration before use with the current API.
They were retained as research/application history.

| File | Relevant line(s) |
|---|---|
| [Alberto nozzle driver](development/alberto/euler_nozzle/main.py) | 115 |
| [Alberto shock-tube driver](development/alberto/euler_nozzle/shock_tube_fvs.py) | 24 |
| [Alberto two-fluid driver](development/alberto/two_fluid/main_two_fluid.py) | 31–43, 909 |
| [Closure sensitivity](development/alberto/sensitivity/sensitivity_closure.py) | 61–64, 606 |
| [Sensitivity study](development/alberto/sensitivity/sensitivity_study.py) | 92–101 |
| [Archived cold start](cases/archive/data_assimilation/main_cold_start.py) | 115 |
| [Archived interfacial-area processing](cases/archive/data_assimilation/postprocess_interfacial_area.py) | 10 |
| [Archived inlet sensitivity](cases/archive/r1233zde_inlet_sensitivity/run_simulation.py) | 62–71 |
| [Archived FVS validation](cases/archive/validation_shock_fvs/validation_fvs_shock_nozzle.py) | 56 |
| [Bernoulli comparison](cases/archive/nicfd/nicfd_figures/plot_bernoulli_compare.py) | 36 |
| [Bernoulli journal figure](cases/archive/nicfd/nicfd_figures/plot_bernoulli_journal.py) | 65 |
| [Bubble-number figure](cases/archive/nicfd/nicfd_figures/plot_bubble_number_hem.py) | 51–52 |
| [Gas-dominated comparison](cases/archive/nicfd/nicfd_figures/plot_gasdom_compare.py) | 33 |
| [Phase-change journal figure](cases/archive/nicfd/nicfd_figures/plot_phase_change_journal.py) | 43–44 |
| [Stacked phase-change figure](cases/archive/nicfd/nicfd_figures/plot_phase_change_stacked.py) | 42 |
| [Two-component journal figure](cases/archive/nicfd/nicfd_figures/plot_two_component_journal.py) | 43 |
| [Stacked two-component figure](cases/archive/nicfd/nicfd_figures/plot_two_component_stacked.py) | 37 |

Historical configuration-loader comments also name the retired geometry namespace
in Alberto's two-fluid driver and the archived data-assimilation/NICFD applications.
These comments are not additional executable imports.

## Other stale benchmark references discovered

The old `development/roberto/benchmarking/` directory was already absent before
this removal. These remaining references are pre-existing and separate from the
root archive deletion:

| File | Remaining dependency |
|---|---|
| [Roe steady verification](development/flux_methods/roe/scripts/verify_steady.py) | Imports its worker, assessment and common helpers at lines 22–24 |
| [Roe steady audit](development/flux_methods/roe/scripts/audit_steady.py) | Imports audit/common helpers at lines 9–10 |
| [Roe report](development/flux_methods/roe/scripts/report.py) | Imports report helpers at line 14 |
| [Roe summary](development/flux_methods/roe/scripts/summarize.py) | Generates commands for the removed harness around lines 455–528 |
| [Real-gas report](development/real_gas/scripts/report.py) | Plotly asset path at line 22 points into the removed harness |
| [Flux API audit](development/flux_methods/shared/audit_api.py) | Settings path at line 34 points into the removed harness |

Update these clients against `development/benchmark_production/` or extract the
needed supported helpers after checking callback/data contracts. Blindly renaming
imports could change study behavior; no numerical study code was migrated here.

## Names that remain intentionally

- `VerificationProblem.legacy_payload` is a compatibility name for the same plain
  mapping as `resolved`; it does not load any archived code. The verification test
  and Roe worker client still use that name. Its documentation now states this.
- The negative API test passing `Scheme(flux="legacy")` verifies rejection of an
  unsupported option. It has no dependency on the removed implementation.
- `legacy_full_steps` in the implicit integrator names an existing numerical mode;
  it is not an archived-module import.
- `legacy_refinement`, `defaults_legacy.yaml`, `UnsupportedLegacyCase` and older
  theme names in research code are labels/settings, not paths to the deleted folder.

## Verification

- Both removed directories are absent; deletion targets were resolved and checked
  within the repository before recursive removal.
- All **745 maintained cases collect successfully**, down from 775 after removing
  29 historical cases and one old-flux comparison. The removed archived diagnostic
  was outside production pytest discovery.
- AST import inspection found no maintained-test imports of retired modules, and
  a path scan found no remaining Python references to the removed archive.
- **438 fast-selection tests passed**, with 307 expensive/optional cases deselected,
  in 316.25 seconds. There were no skips or expected failures in that run.
- **13 focused numerical/API/data checks passed** in 11.69 seconds, including the
  revised flux tests, rejection of the retired flux option, pinned source-archive
  integrity and upstream two-fluid fixture agreement.
- Black passes for all 86 remaining test Python files. The Poetry lock check
  passes; dependencies were not changed.
- The complete slow CFD suite was not rerun for this archive/configuration cleanup.
  All remaining modules were collected and the affected behavior was exercised.

This is archive/test/configuration cleanup. Production numerical function bodies
were not changed. A verification-property docstring was clarified. The previous
full CFD campaign and coverage percentages in `TEST_SUITE_REVIEW.md` predate these
deletions; they are not presented as a new full-suite or coverage measurement.
