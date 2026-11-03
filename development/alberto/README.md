# Alberto: legacy development scripts

**Current API status:** this is a historical research collection. Several
drivers and sensitivity studies still import the retired
`nozzle_hyperbolic.fvm_solver` namespace. The sensitivity scripts also reference
the removed `models.two_fluid.closures.source_two_fluid` module. Their commands
and measurements below describe the earlier execution audit; they do not
establish compatibility with today's package. The duplicate packaged two-fluid
solver has now been deleted. Use the [current two-fluid guide](../../docs/notes/two_fluid_migration.md),
[standalone demo](../../demos/two_fluid/README.md) and
[NICFD case client](../../cases/NICFD2026/README.md) for maintained entry points.
Migrating these historical drivers requires an explicit update to their solver
API and configuration handling, rather than restoring the removed solver.

These files predate the controlled studies in `development/roberto/`. They have
been grouped by topic without changing their numerical algorithms or original
configuration values. Start with the [execution audit](RUN_REPORT.md), which
lists every program/helper and all configurations, including failures, time
limits and the difference between execution and numerical verification.

```text
development/alberto/
  README.md, RUN_REPORT.md      Human-facing navigation and findings
  euler_nozzle/               Legacy Euler/HEM nozzle and Sod/FVS programs
    settings/                 Five original Euler YAML inputs
  two_fluid/                  Driver, grid sequencing, config loader, initial states
    settings/                 Cases, two Python wrappers and seven defaults fragments
    OPEN_QUESTIONS.md          Historical numerical questions
    SPEEDUP_REPORT.md         Historical two-fluid performance notes
  sensitivity/                Closure AD studies and plotting helper
    settings/                 Six finite-difference perturbation inputs
  support/                    CPU-affinity and run-provenance utilities
  hpc/                        Archived LSF cluster launcher
  audit/scripts/              Bounded headless execution checks (not pytest tests)
  outputs/                    Generated runs, audit evidence and archived data
```

The original single-phase speed report is beside the Euler scripts. Historical
Markdown notes were preserved byte-for-byte: their old `dev_scripts/` commands
are historical. All original Python/configuration basenames are retained.
`settings_single_phase_nozzle.yaml` is in `two_fluid/settings/` because it
uses the two-fluid configuration schema despite its name.

## Historical launch commands from the execution audit

Use the existing Poetry environment and module commands, without `sys.path`
workarounds:

```powershell
poetry run python -m development.alberto.euler_nozzle.main
poetry run python -m development.alberto.euler_nozzle.shock_tube_fvs

poetry run python -m development.alberto.two_fluid.main_two_fluid
poetry run python -m development.alberto.two_fluid.main_two_fluid_grid_sequence settings/settings_flashing_nitrogen_simoneau.yaml

poetry run python -m development.alberto.sensitivity.sensitivity_study
poetry run python -m development.alberto.sensitivity.sensitivity_closure C:/absolute/path/to/a/converged/run
```

For the two-fluid driver, the existing `SETTINGS_YAML` environment variable
selects another case. Relative settings paths are resolved beside the two-fluid
driver; an absolute path also works. For example:

```powershell
$env:SETTINGS_YAML = 'settings/settings_two_fluid_gas_isentropic.yaml'
poetry run python -m development.alberto.two_fluid.main_two_fluid
Remove-Item Env:SETTINGS_YAML
```

The sensitivity-study `--config` path is relative to `development/alberto/`, defaulting
to `two_fluid/settings/settings_flashing_smd_space_marching_comparison.yaml`.
Keep an absolute path for saved-run analysis to avoid ambiguity. Its
`--help` lists smaller execution controls. The plotting helper is a module of
functions, not an independent simulation.

Outputs now go under `development/alberto/outputs/`. The three pre-existing Roe NPZs
are preserved in `outputs/archived_results/paper_reproduction/`. Old external
case-study wrappers referencing `dev_scripts/` are not covered by this cleanup.
The LSF launcher requires a suitable Linux/cluster environment; do not interpret
its syntax-check pass as an executed cluster campaign.

## Important audit results

The Sod program and helper checks complete. Full default real-fluid campaigns
were not certified: property-table generation exceeded the audit windows.
An existing ideal-gas two-fluid case advances finite timesteps, and a separate
21-point/three-step smoke completed through solution and figure export, but
neither observation establishes a converged default flashing solution.
The closure sensitivity script needs a compatible converged saved run.

All 35 YAML files parse; 23 complete two-fluid/finite-difference configurations
pass the typed loader. Original solver/closure equations, settings, and `src`
were left unchanged. Read [RUN_REPORT.md](RUN_REPORT.md) for the full table and
coverage definitions before deciding which code to reuse.

The collection was subsequently nested under `development/alberto/`; imports,
root anchors and launch commands were updated statically. No scripts were rerun
for that directory move. Saved audit records preserve their original paths.
